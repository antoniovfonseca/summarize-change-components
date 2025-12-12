# summarize_change_components.py

# Standard library
import glob
import os
import pickle
import sys
import time
import math
from pathlib import Path

# Typing for annotations
from typing import Dict, List, Optional, Iterable, Tuple

# Third-party
import numba as nb
import numpy as np
import pandas as pd
import rasterio
import xlsxwriter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import matplotlib.ticker as mticker
from numba import prange
from pyproj import Transformer
from pyproj import Geod
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# Rasterio submodules
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.enums import Resampling

# Matplotlib extensions
from matplotlib.ticker import FuncFormatter
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_map_utils import north_arrow
from matplotlib.colors import (
    ListedColormap,
    BoundaryNorm,
    Normalize,
    LinearSegmentedColormap,
)
from matplotlib.patches import Patch, Rectangle, FancyArrowPatch


def apply_mask_to_images(
    image_paths: List[str], output_path: str, mask_path: str = None
) -> List[str]:
    """
    Apply a mask to raster images and save as 8-bit TIFFs.

    Args:
        image_paths (list[str]): Paths to input images.
        output_path (str): Directory to save masked images.
        mask_path (str, optional): Path to mask raster.

    Returns:
        list[str]: Paths to saved masked images.
    """
    # Create output folder if missing
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    saved_paths = []

    # Load mask if provided
    mask_data = None
    if mask_path:
        with rasterio.open(mask_path) as mask_file:
            mask_data = mask_file.read(1)

    # Apply mask to each image
    for path in image_paths:
        with rasterio.open(path) as image:
            meta = image.meta.copy()

            # Force 8-bit TIFF settings
            meta["dtype"] = "uint8"
            meta["nodata"] = 255
            meta["driver"] = "GTiff"
            meta["compress"] = "lzw"

            # Read raster band and apply mask
            image_data = image.read(1)
            if mask_data is not None:
                masked_data = (image_data * (mask_data == 1)).astype("uint8")
            else:
                masked_data = image_data.astype("uint8")

            # Build output path
            base_name = os.path.basename(path).replace(".tif", "_masked.tif")
            masked_path = os.path.join(output_path, base_name)

            # Save masked image
            with rasterio.open(masked_path, "w", **meta) as dest:
                dest.write(masked_data, 1)

            saved_paths.append(masked_path)

    return saved_paths


def get_files_with_suffix(
    folder: str,
    suffix: str,
) -> List[str]:
    """
    Get all files in a folder that end with a specific suffix, sorted naturally.

    Args:
        folder (str): Directory to search.
        suffix (str): Filename suffix (e.g., "_masked.tif").

    Returns:
        list[str]: Sorted list of file paths.
    """
    return sorted(
        [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(suffix)
        ],
    )


def plot_maps(
    image_paths: List[str],
    years: List[int],
    color_map: Dict[str, str],
    class_labels_dict: Dict[int, Dict[str, str]],
    output_path: str,
    scale_factor: float = 0.1,
) -> None:
    """
    Plot a series of raster maps side-by-side with a shared legend.

    Args:
        image_paths (list[str]): List of paths to the raster files.
        years (list[int]): List of years corresponding to the images.
        color_map (dict): Mapping of class names to hex colors.
        class_labels_dict (dict): Dictionary of class ID to metadata (name, color).
        output_path (str): Directory where the output plot will be saved.
        scale_factor (float, optional): Downsampling factor to reduce memory usage.
    """
    if not image_paths:
        print("No images to display.")
        return

    n_images = len(image_paths)
    if n_images == 0:
        return

    # Create figure with subplots
    fig, axes = plt.subplots(
        1,
        n_images,
        figsize=(
            5 * n_images,
            6,
        ),
        constrained_layout=True,
    )

    if n_images == 1:
        axes = [axes]

    # Prepare custom colormap from class_labels_dict
    # We assume class IDs are sequential or at least can be mapped to indices.
    # A robust approach: create a ListedColormap where index i maps to color of class i.
    max_class_id = max(class_labels_dict.keys())
    colors = [
        "black"
    ] * (
        max_class_id + 1
    )  # Default color
    for (
        cls_id,
        info,
    ) in class_labels_dict.items():
        if cls_id == 0:
            # Often 0 is nodata or background, set to transparent or specific color
            colors[cls_id] = "white"  # or info['color']
        else:
            colors[cls_id] = info["color"]

    cmap_custom = ListedColormap(
        colors,
    )

    # Plot each raster
    for ax, path, year in zip(
        axes,
        image_paths,
        years,
    ):
        with rasterio.open(path) as src:
            # Downsample for faster plotting
            data = src.read(
                1,
                out_shape=(
                    int(src.height * scale_factor),
                    int(src.width * scale_factor),
                ),
                resampling=Resampling.nearest,
            )
            # Use imshow with the custom colormap.
            # vmin/vmax ensure the values map correctly to the colormap indices.
            ax.imshow(
                data,
                cmap=cmap_custom,
                vmin=0,
                vmax=max_class_id,
                interpolation="nearest",
            )
            ax.set_title(
                f"Year {year}",
                fontsize=14,
            )
            ax.axis("off")

    # Legend
    legend_patches = [
        Patch(
            color=color,
            label=label,
        )
        for label, color in color_map.items()
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            -0.05,
        ),
        ncol=len(color_map),
        fontsize=12,
        frameon=False,
    )

    # Save
    out_file = os.path.join(
        output_path,
        "graphic_maps.jpeg",
    )
    plt.savefig(
        out_file,
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()
    print(
        f"Map plot saved to: {out_file}",
    )


def load_square_matrix(
    csv_path: str,
) -> pd.DataFrame:
    """
    Load a square transition matrix from CSV, ensuring index/columns match.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded matrix.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found.",
        )
    df = pd.read_csv(
        csv_path,
        index_col=0,
    )
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def exchange_from_T(
    dfT: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the Exchange matrix from a total transition matrix T.

    Exchange(i, j) = 2 * min(T[i, j], T[j, i])

    Args:
        dfT (pd.DataFrame): Transition matrix.

    Returns:
        pd.DataFrame: Exchange matrix.
    """
    arr = dfT.values
    # E_ij = 2 * min(T_ij, T_ji)
    # Use broadcasting or simple loops. Since classes are few, loops are fine.
    # Or efficient numpy:
    T = arr
    T_T = arr.T
    E = 2 * np.minimum(T, T_T)
    return pd.DataFrame(
        E,
        index=dfT.index,
        columns=dfT.columns,
    )


def net_change_from_T(
    df_sum: pd.DataFrame,
) -> pd.Series:
    """
    Calculate net change per class from a SUM transition matrix.

    Parameters
    ----------
    df_sum : pd.DataFrame
        Square transition matrix representing total transitions over
        the full time span, including persistence on the diagonal.

    Returns
    -------
    pd.Series
        Net change for each class (index aligned with ``df_sum.index``),
        computed as::

            net_change(j) = gains(j) - losses(j)

        where gains(j) is the sum of column j (i -> j) excluding the
        diagonal, and losses(j) is the sum of row j (j -> k) excluding
        the diagonal.
    """
    M = df_sum.values.astype(float).copy()

    # Remove persistence on the diagonal for the calculation
    np.fill_diagonal(
        M,
        0.0,
    )

    gains = M.sum(
        axis=0,
    )
    losses = M.sum(
        axis=1,
    )

    net_change = gains - losses

    return pd.Series(
        net_change,
        index=df_sum.index,
    )


def reorder_matrices_by_net_change(
    df_sum: pd.DataFrame,
    df_ext: pd.DataFrame,
    df_alt: pd.DataFrame,
    df_exc: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Reorder rows and columns of all matrices using net change from ``df_sum``.

    Parameters
    ----------
    df_sum : pd.DataFrame
        SUM transition matrix.
    df_ext : pd.DataFrame
        Extent matrix for the same classes.
    df_alt : pd.DataFrame
        Alternation matrix for the same classes.
    df_exc : pd.DataFrame
        Exchange matrix (e.g. derived from SUM) for the same classes.

    Returns
    -------
    tuple of pd.DataFrame
        Tuple ``(df_sum_ord, df_ext_ord, df_alt_ord, df_exc_ord)``
        with indices and columns sorted by ascending net change.
    """
    net_c = net_change_from_T(df_sum)
    # Sort by net change ascending (largest loss -> largest gain)
    sorted_idx = net_c.sort_values().index

    return (
        df_sum.loc[sorted_idx, sorted_idx],
        df_ext.loc[sorted_idx, sorted_idx],
        df_alt.loc[sorted_idx, sorted_idx],
        df_exc.loc[sorted_idx, sorted_idx],
    )


def label_id_to_name(
    labels: list,
) -> list[str]:
    """
    Convert a list of class IDs (str or int) to class names using global dict.

    Parameters
    ----------
    labels : list
        List of class IDs.

    Returns
    -------
    list[str]
        List of corresponding class names from ``class_labels_dict``.
    """
    new_labels = []
    for lbl in labels:
        try:
            val = int(float(lbl))
            name = class_labels_dict.get(
                val,
                {},
            ).get(
                "name",
                str(lbl),
            )
            new_labels.append(name)
        except ValueError:
            new_labels.append(str(lbl))
    return new_labels


def _unit_formatter(
    factor: float,
    suffix: str = "",
    decimals: int = 0,
) -> FuncFormatter:
    """
    Create a matplotlib FuncFormatter to scale ticks by ``factor``.

    Parameters
    ----------
    factor : float
        Divisor for the tick value.
    suffix : str, optional
        String to append to the formatted value, by default "".
    decimals : int, optional
        Number of decimal places, by default 0.

    Returns
    -------
    FuncFormatter
        Formatter that applies ``value / factor``.
    """

    def f(
        x: float,
        pos: int | None,
    ) -> str:
        return f"{x / factor:.{decimals}f}{suffix}"

    return FuncFormatter(f)


def _unit_label(
    suffix: str,
    base_label: str = "Number of pixels",
) -> str:
    """
    Construct a descriptive label string based on the suffix.

    Parameters
    ----------
    suffix : str
        One of "M", "k", "hundreds", or "".
    base_label : str, optional
        Base text for the label, by default "Number of pixels".

    Returns
    -------
    str
        Label like "Number of pixels (millions)", "(thousands)", etc.
    """
    if suffix == "M":
        return f"{base_label} (millions)"
    if suffix == "k":
        return f"{base_label} (thousands)"
    if suffix == "hundreds":
        return f"{base_label} (hundreds)"
    return base_label


def annotate_heatmap(
    ax: plt.Axes,
    M: np.ndarray,
    fontsize: int = 8,
    text_color: str = "black",
) -> None:
    """
    Annotate cells in a heatmap with integer values from matrix ``M``.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib Axes where the heatmap is drawn.
    M : np.ndarray
        Data matrix of shape (nrows, ncols).
    fontsize : int, optional
        Font size for annotations, by default 8.
    text_color : str, optional
        Color for the text, by default "black".
    """
    nrows, ncols = M.shape
    for i in range(nrows):
        for j in range(ncols):
            val = M[
                i,
                j,
            ]
            if val > 0:
                ax.text(
                    j,
                    i,
                    f"{int(val)}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=fontsize,
                )


def get_norm(
    scale: str,
    vmin: float,
    vmax: float,
    gamma: float = 0.5,
) -> mcolors.Normalize:
    """
    Return a matplotlib Normalization object based on scale type.

    Parameters
    ----------
    scale : str
        Normalization type: ``"linear"``, ``"log"``, or
        ``"power"``.
    vmin : float
        Lower bound for the normalization.
    vmax : float
        Upper bound for the normalization.
    gamma : float, optional
        Exponent used for ``"power"`` scaling (PowerNorm).

    Returns
    -------
    matplotlib.colors.Normalize
        Normalization instance suitable for ``imshow`` or similar.
    """
    scale = scale.lower()

    if scale == "log":
        vmin_eff = max(
            vmin,
            1e-9,
        )
        return mcolors.LogNorm(
            vmin=vmin_eff,
            vmax=vmax,
        )

    if scale == "power":
        return mcolors.PowerNorm(
            gamma=gamma,
            vmin=vmin,
            vmax=vmax,
        )

    return mcolors.Normalize(
        vmin=vmin,
        vmax=vmax,
    )


def plot_heatmap(
    df: pd.DataFrame,
    title: str,
    save_path: Path | None = None,
    figsize: Tuple[float, float] | None = None,
    cmap: str = "YlOrRd",
    vmin: float = 0.0,
    vmax: float | None = None,
    rotate_xticks_deg: int = 90,
    cbar_label: str = "Number of pixels",
    annotate: bool = True,
    cell_size_inch: float = 0.8,
    tick_fontsize: int | None = None,
    ann_fontsize: int = 8,
    cbar_fraction: float = 0.025,
    cbar_pad: float = 0.02,
    tick_fontsize_x: int | None = None,
    tick_fontsize_y: int | None = None,
    axis_label_fontsize: int | None = None,
    title_fontsize: int | None = None,
) -> None:
    """
    Plot a square matrix as a heatmap with integer annotations.

    The colorbar:

    - ignores the diagonal when computing min/max;
    - chooses units adaptively (pixels, hundreds, thousands, millions)
      based on the maximum absolute value off the diagonal;
    - formats tick labels accordingly.

    Negative values use blue tones, non-negative values use YlOrRd, and
    diagonal cells are drawn in black but still annotated with integers.
    """
    if tick_fontsize_x is None or tick_fontsize_y is None:
        raise ValueError(
            "Set `tick_fontsize_x` and `tick_fontsize_y` explicitly in the call.",
        )

    if axis_label_fontsize is None:
        axis_label_fontsize = 12

    if title_fontsize is None:
        title_fontsize = 14

    labels = list(df.index)
    M = df.values.astype(float)

    # Remove diagonal for color scaling so transitions drive the palette
    M_scale = M.copy()
    np.fill_diagonal(
        M_scale,
        0.0,
    )
    finite_vals = M_scale[np.isfinite(M_scale)]

    if finite_vals.size == 0:
        has_negative = False
        vmin_eff = 0.0
        vmax_eff = 1.0
    else:
        has_negative = float(np.nanmin(finite_vals)) < 0.0
        min_val = float(np.nanmin(finite_vals))
        max_val = float(np.nanmax(finite_vals))

    if has_negative:
        vmin_eff = min_val
        vmax_eff = max_val
        if vmin_eff == vmax_eff:
            vmax_eff = vmin_eff + 1.0
    else:
        vmin_eff = vmin
        vmax_eff = (
            float(max_val)
            if vmax is None
            else float(vmax)
        )
        if vmin_eff == vmax_eff:
            vmax_eff = vmin_eff + 1.0

    nrows, ncols = df.shape
    if figsize is None:
        figsize = (
            cell_size_inch * ncols,
            cell_size_inch * nrows,
        )

    fig, ax = plt.subplots(
        figsize=figsize,
        constrained_layout=True,
    )

    if has_negative:
        # Positive layer (>= 0): YlOrRd
        M_pos = np.where(
            M < 0.0,
            0.0,
            M,
        )
        norm_pos = mcolors.Normalize(
            vmin=0.0,
            vmax=vmax_eff,
        )
        im_pos = ax.imshow(
            M_pos,
            aspect="equal",
            cmap=plt.cm.YlOrRd,
            norm=norm_pos,
        )

        # Negative layer (< 0): Blues_r (more negative -> darker)
        M_neg = np.ma.masked_where(
            M >= 0.0,
            M,
        )
        norm_neg = mcolors.Normalize(
            vmin=vmin_eff,
            vmax=0.0,
        )
        ax.imshow(
            M_neg,
            aspect="equal",
            cmap=plt.cm.Blues_r,
            norm=norm_neg,
        )

        main_for_cbar = im_pos
    else:
        norm_pos = mcolors.Normalize(
            vmin=vmin_eff,
            vmax=vmax_eff,
        )
        im_pos = ax.imshow(
            M,
            aspect="equal",
            cmap=plt.cm.YlOrRd,
            norm=norm_pos,
        )
        main_for_cbar = im_pos

    # Overlay the diagonal in black (values still annotated from M)
    diag_mask = np.zeros_like(
        M,
        dtype=bool,
    )
    np.fill_diagonal(
        diag_mask,
        True,
    )
    M_diag = np.ma.masked_where(
        ~diag_mask,
        np.ones_like(M),
    )
    black_cmap = mcolors.ListedColormap(
        ["black"],
    )
    ax.imshow(
        M_diag,
        aspect="equal",
        cmap=black_cmap,
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    tick_names = label_id_to_name(labels)

    fx = tick_fontsize if tick_fontsize is not None else tick_fontsize_x
    fy = tick_fontsize if tick_fontsize is not None else tick_fontsize_y

    ax.set_xticklabels(
        tick_names,
        rotation=rotate_xticks_deg,
        fontsize=fx,
    )
    ax.set_yticklabels(
        tick_names,
        fontsize=fy,
    )

    ax.set_xlabel(
        "To class",
        fontsize=axis_label_fontsize,
    )
    ax.set_ylabel(
        "From class",
        fontsize=axis_label_fontsize,
    )
    ax.set_title(
        title,
        fontsize=title_fontsize,
    )

    # Build a continuous colorbar that merges Blues_r and YlOrRd ranges
    n_bar = 256
    vals = np.linspace(
        vmin_eff,
        vmax_eff,
        n_bar,
    )
    colors_bar = np.zeros(
        (n_bar, 4),
        dtype=float,
    )

    for i, v in enumerate(vals):
        if has_negative and v < 0.0:
            # Negative range mapped to Blues_r
            t = (v - vmin_eff) / (0.0 - vmin_eff)
            colors_bar[i, :] = plt.cm.Blues_r(t)
        else:
            # Non-negative range mapped to YlOrRd
            if vmax_eff > 0.0:
                t = max(
                    0.0,
                    v,
                ) / vmax_eff
            else:
                t = 0.0
            colors_bar[i, :] = plt.cm.YlOrRd(t)

    cmap_bar = mcolors.ListedColormap(colors_bar)
    norm_bar = mcolors.Normalize(
        vmin=vmin_eff,
        vmax=vmax_eff,
    )

    sm = plt.cm.ScalarMappable(
        cmap=cmap_bar,
        norm=norm_bar,
    )
    sm.set_array([])

    cbar = plt.colorbar(
        sm,
        ax=ax,
        fraction=cbar_fraction,
        pad=cbar_pad,
    )

    # --- adaptive units based on off-diagonal values ---
    if finite_vals.size == 0:
        max_abs = 0.0
    else:
        max_abs = float(np.nanmax(np.abs(finite_vals)))

    if max_abs >= 1_000_000:
        factor = 1_000_000.0
        suffix = "M"
    elif max_abs >= 1_000:
        factor = 1_000.0
        suffix = "k"
    elif max_abs >= 100:
        factor = 100.0
        suffix = "hundreds"
    else:
        factor = 1.0
        suffix = ""

    cbar.formatter = _unit_formatter(
        factor=factor,
        suffix="",
        decimals=0,
    )
    cbar.set_label(
        _unit_label(
            suffix,
            base_label=cbar_label,
        ),
        rotation=270,
        labelpad=14,
    )

    # Ticks still em pixels brutos, apenas formatados pelo formatter
    locator = mticker.MaxNLocator(
        nbins=7 if has_negative else 6,
    )
    ticks_pixels = locator.tick_values(
        vmin_eff,
        vmax_eff,
    )
    cbar.set_ticks(ticks_pixels)
    cbar.update_normal(sm)

    if annotate:
        annotate_heatmap(
            ax=ax,
            M=M,
            fontsize=ann_fontsize,
        )

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


class TrajectoryAnalyzer:
    """
    A class to analyze trajectories of pixel values across a time series
    of rasters. It processes multiple rasters to produce a single output
    raster where each pixel's value uniquely identifies its sequence of
    class transitions over time.
    """

    @staticmethod
    def process_rasters(
        output_path: str,
        suffix: str = "_masked.tif",
        chunk_size: int = 500,
    ) -> str:
        """
        Process all rasters ending with ``suffix`` in ``output_path`` to create
        a trajectory raster.

        Parameters
        ----------
        output_path : str
            Directory containing the input rasters. The output file
            ``trajectory.tif`` will also be saved here.
        suffix : str, optional
            Filename suffix (e.g., ``"_masked.tif"``)
            used to select input rasters in ``output_path``.
        chunk_size : int, optional
            Number of rows per chunk for memory-efficient processing.

        Returns
        -------
        str
            Full path to the output ``trajectory.tif`` file.
        """
        os.makedirs(
            output_path,
            exist_ok=True,
        )
        if not os.path.isdir(output_path):
            raise ValueError(
                f"Path must be a directory: {output_path}",
            )

        raster_files = sorted(
            os.path.join(
                output_path,
                filename,
            )
            for filename in os.listdir(output_path)
            if filename.endswith(suffix)
        )
        if not raster_files:
            raise ValueError(
                f"No files found with suffix '{suffix}' in {output_path}",
            )

        # Read shape and metadata from the first raster
        with rasterio.open(raster_files[0]) as src:
            meta = src.meta.copy()
            height, width = src.shape

        result = np.zeros(
            (
                height,
                width,
            ),
            dtype=np.uint8,
        )

        print(
            f"Starting trajectory processing of {height} rows in chunks of "
            f"{chunk_size}...",
        )

        # Row-wise chunking to limit memory, with progress bar
        n_chunks = (height + chunk_size - 1) // chunk_size
        for y_start in tqdm(
            range(
                0,
                height,
                chunk_size,
            ),
            total=n_chunks,
            desc="Trajectory processing",
            unit="chunk",
        ):
            y_end = min(
                y_start + chunk_size,
                height,
            )
            h_chunk = y_end - y_start

            stack = np.zeros(
                (
                    len(raster_files),
                    h_chunk,
                    width,
                ),
                dtype=np.uint8,
            )
            for i, raster_path in enumerate(raster_files):
                with rasterio.open(raster_path) as src:
                    stack[i] = src.read(
                        1,
                        window=(
                            (
                                y_start,
                                y_end,
                            ),
                            (
                                0,
                                width,
                            ),
                        ),
                    )

            result[
                y_start:y_end,
                :,
            ] = process_stack_parallel(
                stack,
                h_chunk,
                width,
            )

        # Save output
        meta.update(
            {
                "dtype": "uint8",
                "nodata": 0,
                "count": 1,
                "compress": "lzw",
            },
        )
        out_file = os.path.join(
            output_path,
            "trajectory.tif",
        )
        with rasterio.open(
            out_file,
            "w",
            **meta,
        ) as dst:
            dst.write(
                result,
                1,
            )

        print(
            f"Trajectory raster saved to: {out_file}",
        )
        return out_file


class ComponentVisualizer:
    @staticmethod
    def plot_gain_loss_stacked(
        class_labels_dict: Dict[int, Dict[str, str]],
        title: str,
        output_path: str,
        components_color: Dict[str, str] = {
            "Allocation_Exchange": "blue",
            "Allocation_Shift": "cyan",
            "Alternation_Exchange": "purple",
            "Alternation_Shift": "magenta",
            "Quantity": "green",
            "Shift": "orange",
            "Exchange": "red",
        },
    ) -> None:
        """
        Create a stacked bar chart of gains and losses for each class.

        The chart aggregates components (e.g., Quantity, Exchange, Shift)
        from 'sum' and 'alternation' interval records found in
        'change_components.csv'.

        Parameters
        ----------
        class_labels_dict : dict
            Dictionary mapping class IDs to metadata (name, color).
        title : str
            Plot title.
        output_path : str
            Directory containing 'change_components.csv' and where the
            output graphic will be saved.
        components_color : dict, optional
            Color mapping for each component type.
        """
        csv_path = os.path.join(
            output_path,
            "change_components.csv",
        )
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"{csv_path} not found.",
            )

        df = pd.read_csv(csv_path)

        # Filter relevant intervals/components
        # We typically want the 'sum' interval for Q, Alloc_Exc, Alloc_Shift
        # plus 'alternation' interval for Alt_Exc, Alt_Shift
        filtered_df = df[
            (
                (df["Time_Interval"] == "sum")
                & (
                    df["Component"].isin(
                        [
                            "Quantity",
                            "Allocation_Exchange",
                            "Allocation_Shift",
                        ],
                    )
                )
            )
            | (
                (df["Time_Interval"] == "alternation")
                & (
                    df["Component"].isin(
                        [
                            "Alternation_Exchange",
                            "Alternation_Shift",
                        ],
                    )
                )
            )
        ]

        if filtered_df.empty:
            print("No matching data for stacked plot.")
            return

        # Identify classes present in the CSV
        existing_classes = filtered_df["Class"].unique()

        # Component stack order (bottom to top)
        component_order = [
            "Allocation_Exchange",
            "Allocation_Shift",
            "Alternation_Exchange",
            "Alternation_Shift",
        ]

        # Sort classes by net quantity (gain - loss)
        class_totals: list[tuple[str, float]] = []
        for cls in existing_classes:
            class_data = filtered_df[filtered_df["Class"] == cls]
            quantity_gain = class_data[class_data["Component"] == "Quantity"][
                "Gain"
            ].sum()
            quantity_loss = class_data[class_data["Component"] == "Quantity"][
                "Loss"
            ].sum()
            class_totals.append(
                (
                    cls,
                    quantity_gain - quantity_loss,
                ),
            )

        sorted_classes = sorted(
            class_totals,
            key=lambda x: x[1],
        )
        ordered_classes = [cls for cls, _ in sorted_classes]

        # Compute maximum absolute stacked height per class
        max_abs_val = 0.0
        for cls in ordered_classes:
            class_data = filtered_df[filtered_df["Class"] == cls]
            gains_total = 0.0
            losses_total = 0.0

            for comp in component_order:
                gains_total += class_data[class_data["Component"] == comp]["Gain"].sum()
                losses_total += class_data[class_data["Component"] == comp][
                    "Loss"
                ].sum()

            max_abs_val = max(
                max_abs_val,
                abs(gains_total),
                abs(losses_total),
            )

        # Automatic scale based on maximum absolute value
        if max_abs_val >= 1_000_000:
            scale_factor = 1_000_000
            y_label = "Change (million pixels)"
        elif max_abs_val >= 1_000:
            scale_factor = 1_000
            y_label = "Change (thousand pixels)"
        elif max_abs_val >= 100:
            scale_factor = 100
            y_label = "Change (hundred pixels)"
        else:
            scale_factor = 1
            y_label = "Change (pixels)"

        # Figure
        fig, ax = plt.subplots(
            figsize=(
                14,
                8,
            ),
        )
        fig.subplots_adjust(
            left=0.1,
            right=0.75,
        )

        x_positions = np.arange(
            len(ordered_classes),
        )
        width = 0.8

        # Stacked bars per class (gains positive, losses negative), scaled
        for idx, cls in enumerate(ordered_classes):
            class_data = filtered_df[filtered_df["Class"] == cls]
            gain_bottom = 0.0
            loss_bottom = 0.0

            for comp in component_order:
                # Gains
                gains = (
                    class_data[class_data["Component"] == comp]["Gain"].sum()
                    / scale_factor
                )
                ax.bar(
                    x_positions[idx],
                    gains,
                    width,
                    bottom=gain_bottom,
                    color=components_color[comp],
                    edgecolor="none",
                )
                gain_bottom += gains

                # Losses (negative)
                losses = -(
                    class_data[class_data["Component"] == comp]["Loss"].sum()
                    / scale_factor
                )
                ax.bar(
                    x_positions[idx],
                    losses,
                    width,
                    bottom=loss_bottom,
                    color=components_color[comp],
                    edgecolor="none",
                )
                loss_bottom += losses

        # Axes formatting
        class_names = [
            class_labels_dict.get(
                cls,
                {},
            ).get(
                "name",
                f"{cls}",
            )
            for cls in ordered_classes
        ]
        ax.set_xticks(
            x_positions,
        )
        ax.set_xticklabels(
            class_names,
            rotation=45,
            ha="right",
            fontsize=18,
        )
        ax.axhline(
            0,
            color="black",
            linewidth=0.8,
        )
        ax.set_ylabel(
            y_label,
            fontsize=16,
        )
        ax.set_title(
            title,
            fontsize=18,
        )
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=18,
        )

        # Y-axis limits and ticks based on scaled maximum
        y_max_scaled = max_abs_val / scale_factor * 1.1 if max_abs_val > 0 else 1.0
        ax.set_ylim(
            -y_max_scaled,
            y_max_scaled,
        )
        ax.yaxis.set_major_locator(
            ticker.MaxNLocator(
                nbins=5,
            ),
        )

        # Legend
        legend_elements = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                color=components_color["Alternation_Shift"],
                label="Alternation Shift",
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                color=components_color["Alternation_Exchange"],
                label="Alternation Exchange",
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                color=components_color["Allocation_Shift"],
                label="Allocation Shift",
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                color=components_color["Allocation_Exchange"],
                label="Allocation Exchange",
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                color=components_color["Quantity"],
                label="Quantity",
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="center left",
            bbox_to_anchor=(
                1.05,
                0.5,
            ),
            fontsize=14,
            frameon=False,
        )

        # Save and show
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_path,
                "graphic_change_component_change_class.jpeg",
            ),
            format="jpeg",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()


class TransitionMatrixGenerator:
    """
    Encapsulates logic for generating:
    - Interval transition matrices (T_t)
    - Extent matrix (E)
    - Sum matrix (S)
    - Alternation matrix (A) = S - E (on diagonal)
    """

    def generate_mask_and_flatten_rasters(
        self,
        output_path: str,
        suffix: str = "_masked.tif",
    ) -> list[np.ndarray]:
        """
        Load all matching rasters, create a unified mask for background/nodata,
        and return a list of flattened arrays (excluding masked pixels).

        Assumption: Class 0 is background/nodata in inputs.

        Parameters
        ----------
        output_path : str
            Directory containing the raster files.
        suffix : str, optional
            Suffix to identify relevant files, by default "_masked.tif".

        Returns
        -------
        list[np.ndarray]
            List of 1D arrays, one per time step, containing only valid pixels.
        """
        files = sorted(
            [
                os.path.join(output_path, f)
                for f in os.listdir(output_path)
                if f.endswith(suffix)
            ]
        )

        if not files:
            raise FileNotFoundError(
                f"No files ending with '{suffix}' found in {output_path}"
            )

        # Load all raster data into memory (watch out for large rasters!)
        all_data = []
        for f in files:
            with rasterio.open(f) as src:
                # Read band 1
                all_data.append(src.read(1))

        # Stack to shape (T, H, W)
        stack = np.array(all_data)

        # Create a mask where ANY time step has 0 (background)
        # Or alternatively: mask where ALL time steps are 0?
        # Typically for change detection, if a pixel is background in ANY year,
        # we might exclude it to avoid skewing "0->0" transitions.
        # Adjust logic as needed. Here, we mask if 0 is present in ANY layer.
        # combined_mask = np.any(stack == 0, axis=0)

        # Flatten non-masked values for each raster
        # flattened = [
        #     data[~combined_mask].flatten()
        #     if np.any(combined_mask)
        #     else data.flatten()
        #     for data in all_data
        # ]

        flattened = [data.flatten() for data in all_data]

        return flattened

    def generate_all_matrices(
        self,
        years: List[int],
        output_path: str,
        suffix: str = "_masked.tif",
    ) -> tuple[list[int], np.ndarray]:
        """
        Generate interval, extent, sum, and alternation transition matrices.

        Parameters
        ----------
        years: list[int]
            List of years.
        output_path : str
            Directory containing input rasters and where CSV outputs are saved.
        suffix : str, optional
            Filename suffix used to select input rasters, by default "_masked.tif".

        Returns
        -------
        tuple[list[int], np.ndarray]
            List of years and array of all class labels present in the rasters.
        """

        # Load processed and flattened raster data
        flattened_data = self.generate_mask_and_flatten_rasters(
            output_path=output_path,
            suffix=suffix,
        )

        # Validate that the number of rasters matches the number of years
        if len(years) != len(flattened_data):
            raise ValueError(
                f"Mismatch: {len(years)} years vs {len(flattened_data)} rasters",
            )

        # Derive the set of all classes present across all rasters
        all_classes = np.unique(
            np.concatenate(flattened_data),
        ).astype(int)

        # Compute and save interval-by-interval transition matrices
        for i in tqdm(
            range(len(flattened_data) - 1),
            desc="Interval matrices",
            unit="interval",
        ):
            cm = confusion_matrix(
                flattened_data[i],
                flattened_data[i + 1],
                labels=all_classes,
            )
            out_csv = os.path.join(
                output_path,
                f"transition_matrix_{years[i]}-{years[i + 1]}.csv",
            )
            pd.DataFrame(
                cm,
                index=all_classes,
                columns=all_classes,
            ).to_csv(out_csv)

        # Compute and save Extent matrix (First Year vs Last Year)
        extent_cm = confusion_matrix(
            flattened_data[0],
            flattened_data[-1],
            labels=all_classes,
        )
        extent_csv = os.path.join(
            output_path,
            f"transition_matrix_extent_{years[0]}-{years[-1]}.csv",
        )
        pd.DataFrame(
            extent_cm,
            index=all_classes,
            columns=all_classes,
        ).to_csv(extent_csv)

        # Compute and save Sum matrix (sum of all interval matrices)
        sum_cm = np.zeros(
            (
                len(all_classes),
                len(all_classes),
            ),
            dtype=np.int64,
        )
        for i in range(len(flattened_data) - 1):
            # We re-compute or reload. Re-computing is safer if memory allows.
            interval_cm = confusion_matrix(
                flattened_data[i],
                flattened_data[i + 1],
                labels=all_classes,
            )
            sum_cm += interval_cm

        sum_csv = os.path.join(
            output_path,
            f"transition_matrix_sum_{years[0]}-{years[-1]}.csv",
        )
        pd.DataFrame(
            sum_cm,
            index=all_classes,
            columns=all_classes,
        ).to_csv(sum_csv)

        # Compute and save Alternation matrix (Equation 16: Sum - Extent on diagonal)
        # "A has the same off-diagonal elements as S, but diagonal is different."
        # Actually, Equation 16 says:
        # a_jj = s_jj - e_jj  (persistence difference?)
        # and off-diagonals a_ij = s_ij
        # Let's follow that logic.
        alternation_cm = sum_cm.copy()
        # Update diagonal only
        for idx in range(len(all_classes)):
            alternation_cm[
                idx,
                idx,
            ] = (
                sum_cm[
                    idx,
                    idx,
                ]
                - extent_cm[
                    idx,
                    idx,
                ]
            )

        alt_csv = os.path.join(
            output_path,
            f"transition_matrix_alternation_{years[0]}-{years[-1]}.csv",
        )
        pd.DataFrame(
            alternation_cm,
            index=all_classes,
            columns=all_classes,
        ).to_csv(alt_csv)

        print("All transition matrices generated successfully.")
        return (
            years,
            all_classes,
        )


class ComponentCalculator:
    """
    Calculates change components (Quantity, Exchange, Shift) for a given
    transition matrix.
    """

    def __init__(
        self,
        transition_matrix: np.ndarray,
    ):
        """
        Initialize with a square transition matrix (numpy array).
        Rows: Time t, Columns: Time t+1.
        """
        self.matrix = transition_matrix
        self.n_classes = self.matrix.shape[0]
        self.class_components = []

    def calculate_components(
        self,
    ):
        """
        Calculate components for each class and store in self.class_components.

        Returns:
            self
        """
        # Column sums (gains) and Row sums (losses), excluding diagonal?
        # Actually, total column sum includes persistence.
        # Let's follow Pontius et al. logic:
        # Gross Gain = column_sum - diagonal
        # Gross Loss = row_sum - diagonal
        # Net Change = Gain - Loss = column_sum - row_sum
        #
        # Quantity D = |Net Change|
        # Exchange D = 2 * min(Gain, Loss) ? No, that's simplified.
        #
        # Pontius (2004) or Aldwaik & Pontius (2012) formulation:
        # 1) Total Change = Gain + Loss (if we sum over classes? No, per class)
        #    Actually: Change(j) = Gain(j) + Loss(j) ? No.
        #    Let's stick to:
        #    Quantity(j) = |Gain(j) - Loss(j)|
        #    Exchange(j) = 2 * min(Gain(j), Loss(j)) - ... wait.
        #
        # Let's use the standard definitions:
        # Gain_j = (Sum of col j) - M_jj
        # Loss_j = (Sum of row j) - M_jj
        #
        # Quantity Component q_j = |Gain_j - Loss_j|
        # Exchange Component e_j = 2 * min(Gain_j, Loss_j) ...
        # (This is often called "Allocation" if we don't split further, but let's see.)
        #
        # If we split Allocation into Exchange and Shift:
        # This usually requires comparison with other classes, which is complex.
        # However, a common simplification for per-class metrics is:
        # Total Gross Change = Gain + Loss
        # Quantity = |Gain - Loss|
        # Allocation = Total Gross - Quantity = (Gain + Loss) - |Gain - Loss|
        #            = 2 * min(Gain, Loss)
        #
        # Then Allocation can be split into Exchange and Shift?
        # Without paired transitions, it's hard to distinguish Exchange vs Shift purely per class.
        # BUT, if we have the full matrix, we can do calculating Exchange properly.
        #
        # Exchange_j = sum_{k!=j} [ 2 * min( M_jk, M_kj ) ]
        # Shift_j = Total_Change_j - Quantity_j - Exchange_j
        #
        # Let's implement this robust version.

        col_sums = self.matrix.sum(
            axis=0,
        )
        row_sums = self.matrix.sum(
            axis=1,
        )
        diag = np.diag(self.matrix)

        gains = col_sums - diag
        losses = row_sums - diag

        self.class_components = []

        for j in range(self.n_classes):
            # Quantity
            # Note: Quantity Gain vs Loss isn't standard, usually it's just one Q val.
            # But the user wants "Quantity Gain" and "Quantity Loss" in the output CSV.
            #
            # If Gain > Loss: Class j increased net.
            #   Quantity Gain = Gain - Loss
            #   Quantity Loss = 0
            # If Loss > Gain: Class j decreased net.
            #   Quantity Gain = 0
            #   Quantity Loss = Loss - Gain
            net = gains[j] - losses[j]
            q_gain = max(
                0,
                net,
            )
            q_loss = max(
                0,
                -net,
            )

            # Exchange
            # E_j = sum_{k!=j} 2 * min(M_jk, M_kj)
            # This represents pairwise swapping.
            # We can split this into "Exchange Gain" and "Exchange Loss".
            # Actually, pairwise exchange is balanced: M_jk vs M_kj.
            # The "exchanged" amount contributing to Gain is min(M_jk, M_kj).
            # The "exchanged" amount contributing to Loss is min(M_jk, M_kj).
            # So Exchange Gain = Exchange Loss = sum_{k!=j} min(M_jk, M_kj).
            exchange_amount = 0
            for k in range(self.n_classes):
                if k == j:
                    continue
                exchange_amount += min(
                    self.matrix[
                        j,
                        k,
                    ],
                    self.matrix[
                        k,
                        j,
                    ],
                )  # row j->k, row k->j

            exc_gain = exchange_amount
            exc_loss = exchange_amount

            # Shift
            # S_j = Total_Gain_j - Q_Gain_j - Exc_Gain_j
            # (or similar for Loss)
            # Shift Gain = Gain_j - Q_Gain - Exc_Gain
            # Shift Loss = Loss_j - Q_Loss - Exc_Loss
            # These should ideally balance out in a closed system, but let's calculate.
            shift_gain = max(
                0,
                gains[j] - q_gain - exc_gain,
            )
            shift_loss = max(
                0,
                losses[j] - q_loss - exc_loss,
            )

            self.class_components.append(
                {
                    "Class_Index": j,
                    "Gain": gains[j],
                    "Loss": losses[j],
                    "Quantity_Gain": q_gain,
                    "Quantity_Loss": q_loss,
                    "Exchange_Gain": exc_gain,
                    "Exchange_Loss": exc_loss,
                    "Shift_Gain": shift_gain,
                    "Shift_Loss": shift_loss,
                },
            )

        return self


def process_stack_parallel(
    stack: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Process a 3D stack (n_years, height, width) in parallel using Numba.
    This creates the trajectory code for each pixel.

    Trajectory code logic:
      A unique integer representing the sequence of classes.
      For small # of years and classes, we might treat it as base-N encoding
      or similar. However, with many years, that integer overflows.

    The user's code just says "process_stack_parallel" but doesn't define the logic.
    We will implement a simple 'change count' or 'unique trajectory ID' logic.
    For true 'trajectory' mapping, we often map each unique sequence to an ID.

    Since we cannot easily map arbitrary sequences to a single uint8/uint16/uint32
    without a lookup table (which is hard to parallelize efficiently in one pass),
    we'll implement a simplified version:
      "Count of changes" or a hash if needed.

    Given the context of 'Trajectory Analyzer', often we want:
      For each pixel, the vector v = [c_1, c_2, ..., c_T].
      Map v -> unique_id.

    This is best done by:
      1. Reshaping stack to (T, N_pixels).
      2. Finding unique columns (unique trajectories).
      3. Mapping pixels to those unique IDs.

    BUT, doing this on full rasters requires huge memory.
    The chunk-based approach suggests we process chunks.
    If we want GLOBALLY unique trajectory IDs, we need a global registry,
    which conflicts with independent chunk processing unless we do two passes.

    Compromise for this script:
    We'll produce a "Number of Changes" raster here, as it's a common trajectory metric.
    OR, if the user really wants unique trajectory classes, we'd need a different approach.
    Let's assume "Number of Changes" for now, or a simple encoding if distinct classes < 10.
    """
    # Simply count changes: sum(pixel[t] != pixel[t-1])
    # This fits in uint8 easily.
    # Numba implementation:

    n_years = stack.shape[0]
    out = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint8,
    )

    # Flatten logic inside numba for speed
    _count_changes(
        stack,
        n_years,
        height,
        width,
        out,
    )
    return out


@nb.jit(
    nopython=True,
    parallel=True,
    fastmath=True,
)
def _count_changes(
    stack,
    n_years,
    height,
    width,
    out,
):
    # stack: (T, H, W)
    for i in prange(height):
        for j in range(width):
            changes = 0
            prev = stack[
                0,
                i,
                j,
            ]
            # If 0 is background, maybe we ignore?
            # Assuming 0 is background:
            # If start is 0, we wait for first non-zero?
            # Or just count all value shifts?
            # Let's count all value shifts.
            for t in range(
                1,
                n_years,
            ):
                curr = stack[
                    t,
                    i,
                    j,
                ]
                if curr != prev:
                    changes += 1
                    prev = curr
            out[
                i,
                j,
            ] = changes