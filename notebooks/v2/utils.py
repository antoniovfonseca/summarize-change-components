"""
utils.py

This module contains utility functions and classes for summarizing changes among
classes during a time series, following the methodology presented by
Pontius Jr. and da Fonseca (2025).

It includes:
1. Helper functions (Scale bar, North arrow).
2. Visualization of classified maps.
3. Component calculation logic (Quantity, Exchange, Shift).
4. Data processing for transition matrices (reading CSVs and formatting).
5. Advanced plotting for transition matrices and change components.
6. Trajectory analysis (Number of Changes).
"""

import os
import re
import math
import glob
from typing import List, Dict, Optional, Tuple, Union, Iterable

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Rectangle
import matplotlib.ticker as ticker
import matplotlib.ticker as mticker
from matplotlib_scalebar.scalebar import ScaleBar
from pyproj import Transformer, Geod
import numba as nb
from numba import prange
from tqdm import tqdm

# ------------------------------------------------------------------------------
# 1. Helper Functions
# ------------------------------------------------------------------------------

def north_arrow(
    ax: plt.Axes,
    location: str = "upper left",
    shadow: bool = False,
    rotation: Optional[Dict] = None
) -> None:
    """
    Adds a North Arrow to the map (Simplified wrapper for consistency).
    
    Parameters
    ----------
    ax : plt.Axes
        The axes to draw on.
    location : str, optional
        Location code (e.g., 'upper left').
    shadow : bool, optional
        Whether to add a shadow effect.
    rotation : dict, optional
        Rotation parameters.
    """
    x, y = 0.95, 0.95
    if location == "upper left":
        x, y = 0.05, 0.95
    
    ax.annotate(
        'N',
        xy=(x, y),
        xytext=(x, y-0.1),
        arrowprops=dict(
            facecolor='black',
            width=5,
            headwidth=15
        ),
        ha='center',
        va='center',
        fontsize=12,
        xycoords=ax.transAxes,
        textcoords=ax.transAxes
    )


def compute_display_pixel_size_km(
    raster_path: str,
    downsample_divisor: int,
) -> float:
    """
    Compute horizontal resolution in kilometers per displayed pixel.

    This function uses geodesic calculations to determine the real-world
    width of the raster in kilometers and divides it by the number of
    displayed pixels to get the scale.

    Parameters
    ----------
    raster_path : str
        Path to a raster file used to derive spatial extent and CRS.
    downsample_divisor : int
        Integer factor used to downsample the raster width for display.

    Returns
    -------
    float
        Pixel size in kilometers for the downsampled display grid.
    """
    with rasterio.open(raster_path) as src:
        left, bottom, right, top = src.bounds
        lat_mid_src = (top + bottom) / 2.0

        to_ll = Transformer.from_crs(
            src.crs,
            "EPSG:4326",
            always_xy=True,
        )
        lon_l, lat_mid = to_ll.transform(
            left,
            lat_mid_src,
        )
        lon_r, _ = to_ll.transform(
            right,
            lat_mid_src,
        )

        geod = Geod(
            ellps="WGS84",
        )
        _, _, width_m = geod.inv(
            lon_l,
            lat_mid,
            lon_r,
            lat_mid,
        )

        cols_disp = max(
            1,
            src.width // downsample_divisor,
        )

        return (width_m / cols_disp) / 1_000


# ------------------------------------------------------------------------------
# 2. Visualization of Input Data
# ------------------------------------------------------------------------------

def plot_classified_images(
    class_map: Dict[int, Dict[str, str]],
    years: List[int],
    output_path: str,
    image_paths: List[str],
    cols_disp: int = 4, # Kept for signature compatibility, though logic uses max_cols
    downsample_divisor: int = 1,
    panel_size: tuple = (4.0, 6.0),
    dx_km: Optional[float] = None,
    resampling_method: Resampling = Resampling.bilinear,
) -> None:
    """
    Visualizes the spatial distribution of categories across multiple time points.
    
    Replicates the logic from the original notebook to ensure accurate aspect ratios,
    scales, and geodetic measurements.

    Parameters
    ----------
    class_map : Dict[int, Dict[str, str]]
        Mapping from class ID to metadata (name, color).
    years : List[int]
        List of years corresponding to the images.
    output_path : str
        Directory to save the figure.
    image_paths : List[str]
        List of absolute paths to raster files.
    cols_disp : int, optional
        Deprecated in this logic, but kept for compatibility. Function calculates ncols based on max_cols.
    downsample_divisor : int, optional
        Factor to downsample raster for plotting speed (default 1).
    panel_size : tuple, optional
        Size (width, height) of each panel in inches.
    dx_km : float, optional
        Pixel size in km. If None, computed automatically.
    resampling_method : Resampling, optional
        Resampling method for reading data.

    Returns
    -------
    None
    """
    # Resolve input raster paths based on argument or fallback logic
    if not image_paths:
        # Fallback: try to find files in 'input_rasters' subfolder if explicit list not provided
        input_dir = os.path.join(output_path, "input_rasters")
        if os.path.exists(input_dir):
            image_paths = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
        else:
            print("Error: No image paths provided and input_rasters folder not found.")
            return

    # Configure subplot grid with a maximum number of columns.
    n_images = len(image_paths)
    max_cols = 10
    ncols = min(
        max_cols,
        n_images,
    )
    nrows = math.ceil(n_images / max_cols)

    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(
            panel_size[0] * ncols,
            panel_size[1] * nrows,
        ),
        sharey=True,
        constrained_layout=False,
    )
    
    # Specific adjustments to match the desired layout spacing
    plt.subplots_adjust(
        left=0.02,
        right=0.85,
        top=0.95,
        bottom=0.05,
        wspace=0.04,
        hspace=0.04,
    )

    if isinstance(axs, np.ndarray):
        axes = axs.ravel()
    else:
        axes = [axs]

    # Build colormap and normalization for class IDs.
    class_ids_sorted = sorted(class_map.keys())
    cmap = mcolors.ListedColormap(
        [
            class_map[k]["color"]
            for k in class_ids_sorted
        ],
    )
    norm = mcolors.BoundaryNorm(
        class_ids_sorted + [class_ids_sorted[-1] + 1],
        cmap.N,
    )

    # Derive km per displayed pixel when not provided.
    if dx_km is None:
        dx_km = compute_display_pixel_size_km(
            raster_path=image_paths[0],
            downsample_divisor=downsample_divisor,
        )

    # Plot each classified raster in its subplot.
    for i, (path, year) in enumerate(
        zip(
            image_paths,
            years,
        ),
    ):
        ax = axes[i]

        with rasterio.open(path) as src:
            h = max(
                1,
                src.height // downsample_divisor,
            )
            w = max(
                1,
                src.width // downsample_divisor,
            )
            data = src.read(
                1,
                out_shape=(
                    h,
                    w,
                ),
                resampling=resampling_method,
            )

        ax.imshow(
            data,
            cmap=cmap,
            norm=norm,
        )
        ax.set_title(
            f"{year}",
            fontweight="bold",
            fontsize=24,
        )
        ax.axis("off")

    # Disable unused axes when the grid is larger than the number of images.
    for j in range(
        n_images,
        len(axes),
    ):
        axes[j].axis("off")

    # Create legend using all class IDs on the last active axis.
    legend_elements = [
        Rectangle(
            (0, 0),
            1,
            1,
            color=class_map[k]["color"],
            label=class_map[k]["name"],
        )
        for k in sorted(class_map.keys())
    ]

    last_ax = axes[n_images - 1]

    # Legend placement logic from original notebook
    last_ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(
            1.02,
            0.5,
        ),
        frameon=False,
        fontsize=24,
    )

    # Add scale bar to the last subplot.
    scalebar = ScaleBar(
        dx=dx_km,
        units="km",
        length_fraction=0.35,
        location="lower right",
        scale_loc="bottom",
        color="black",
        box_alpha=0,
    )
    last_ax.add_artist(
        scalebar,
    )

    # Add north arrow to the last subplot.
    north_arrow(
        last_ax,
        location="upper left",
        shadow=False,
        rotation={
            "degrees": 0,
        },
    )

    # Save figure to disk.
    out_fig = os.path.join(
        output_path,
        "input_raster_maps.png",
    )
    plt.savefig(
        out_fig,
        format="png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()
    plt.close()
    print(f"Classified maps saved to: {out_fig}")

# ------------------------------------------------------------------------------
# 3. Pixel Count Analysis
# ------------------------------------------------------------------------------

def plot_pixel_counts(
    image_paths: List[str],
    years: List[int],
    class_labels_dict: Dict[int, Dict[str, str]],
    output_path: str,
) -> None:
    """
    Counts pixels per class for each year, plots a stacked bar chart,
    and exports the counts to a CSV file.

    Parameters
    ----------
    image_paths : List[str]
        List of file paths to the raster images (must be ordered by year).
    years : List[int]
        List of years corresponding to the images.
    class_labels_dict : Dict[int, Dict[str, str]]
        Mapping of class IDs to metadata (name and color).
    output_path : str
        Directory where the CSV and plot figure will be saved.

    Returns
    -------
    None
        Saves 'pixels_per_class_per_year.csv' and 'pixels_per_class_bar.png'.
    """
    # 1. Verify Input Data
    if len(image_paths) != len(years):
        raise ValueError(
            f"Mismatch: Found {len(image_paths)} rasters but len(years) == {len(years)}."
        )

    records: List[Dict] = []

    # 2. Count Pixels per Class
    for year, path in zip(years, image_paths):
        with rasterio.open(path) as src:
            data = src.read(
                1
            )

        values, counts = np.unique(
            data,
            return_counts=True
        )

        for value, count in zip(values, counts):
            value = int(
                value
            )
            if value not in class_labels_dict:
                continue

            records.append({
                "Year": year,
                "ClassID": value,
                "ClassName": class_labels_dict[value]["name"],
                "Pixels": int(count),
            })

    # 3. Aggregate Data (Pivot)
    df_pixels = pd.DataFrame(
        records
    )

    pivot_pixels = (
        df_pixels.pivot_table(
            index="Year",
            columns="ClassName",
            values="Pixels",
            aggfunc="sum",
        )
        .fillna(
            0.0
        )
        .astype(
            float
        )
    )

    # 4. Export CSV
    csv_out = os.path.join(
        output_path,
        "pixels_per_class_per_year.csv"
    )
    pivot_pixels.to_csv(
        csv_out,
        index_label="Year"
    )
    print(
        f"Pixel counts table saved to: {csv_out}"
    )

    # 5. Prepare Data for Plotting (Scale & Sort)
    years_array = pivot_pixels.index.values
    max_val = pivot_pixels.to_numpy().max()

    # Determine Scale
    if max_val >= 1_000_000:
        scale_factor = 1_000_000
        y_label = "Area (million pixels)"
    elif max_val >= 1_000:
        scale_factor = 1_000
        y_label = "Area (thousand pixels)"
    elif max_val >= 100:
        scale_factor = 100
        y_label = "Area (hundred pixels)"
    else:
        scale_factor = 1
        y_label = "Area (pixels)"

    pivot_scaled = pivot_pixels / scale_factor

    # Sorting Logic (Net Change Descending -> Class ID Descending)
    first_year = years_array[0]
    last_year = years_array[-1]
    
    net_change = (
        pivot_scaled.loc[last_year] - pivot_scaled.loc[first_year]
    )
    
    name_to_id = {v["name"]: k for k, v in class_labels_dict.items()}
    color_map = {v["name"]: v["color"] for v in class_labels_dict.values()}

    df_sort = net_change.to_frame(
        name="net_change"
    )
    df_sort["class_id"] = df_sort.index.map(
        name_to_id
    )

    classes_for_stack = list(
        df_sort.sort_values(
            by=[
                "net_change",
                "class_id"
            ],
            ascending=[
                False,
                False
            ]
        ).index
    )
    
    # Legend order is reversed stack order (Losers on top)
    classes_for_legend = list(
        reversed(
            classes_for_stack
        )
    )

    # 6. Generate Stacked Bar Chart
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )
    
    x = np.arange(
        len(years_array)
    )
    width = 0.9
    base = np.zeros(
        len(years_array),
        dtype=float
    )
    patches_by_class = {}

    for cls in classes_for_stack:
        if cls not in pivot_scaled.columns:
            continue
            
        vals = pivot_scaled[cls].reindex(
            years_array,
            fill_value=0.0
        ).values
        
        bars = ax.bar(
            x, 
            vals, 
            bottom=base, 
            width=width, 
            label=cls, 
            color=color_map.get(
                cls,
                "gray"
            ),
            edgecolor="white",
            linewidth=0.5
        )
        patches_by_class[cls] = bars[0]
        base += vals

    # Formatting Axes
    ax.set_xticks(
        x
    )
    ax.set_xticklabels(
        years_array
    )
    
    n_labels = len(
        years_array
    )
    if n_labels <= 6:
        rotation, ha = 0, "center"
    elif n_labels <= 12:
        rotation, ha = 45, "right"
    else:
        rotation, ha = 90, "center"
        
    plt.setp(
        ax.get_xticklabels(),
        rotation=rotation,
        ha=ha
    )
    ax.tick_params(
        axis="both",
        labelsize=14
    )
    
    ax.set_ylabel(
        y_label,
        fontsize=18
    )
    ax.set_xlabel(
        "Time points",
        fontsize=18
    )
    ax.set_title(
        "Number of pixels per class",
        fontsize=20
    )

    # Formatting Y-Axis
    y_max_scaled = base.max() * 1.1 if base.max() > 0 else 1.0
    ax.set_ylim(
        0,
        y_max_scaled
    )
    
    ax.yaxis.set_major_locator(
        ticker.MaxNLocator(
            nbins=5,
            integer=True
        )
    )
    ax.yaxis.set_major_formatter(
        ticker.FormatStrFormatter(
            "%d"
        )
    )

    # 7. Add Legend and Save
    handles = [
        patches_by_class[c] 
        for c in classes_for_legend 
        if c in patches_by_class
    ]
    labels = [
        c 
        for c in classes_for_legend 
        if c in patches_by_class
    ]
    
    ax.legend(
        handles, 
        labels, 
        bbox_to_anchor=(
            1.05,
            1.0
        ), 
        loc="upper left", 
        frameon=False, 
        fontsize=12
    )

    plt.tight_layout()
    
    out_fig = os.path.join(
        output_path,
        "pixels_per_class_bar.png"
    )
    plt.savefig(
        out_fig,
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()
    
    print(
        f"Pixel counts plot saved to: {out_fig}"
    )

# ------------------------------------------------------------------------------
# 4. Transition Matrix Generation
# ------------------------------------------------------------------------------

def generate_mask_and_flatten_rasters(
    output_path: str,
    suffix: str = ".tif",
    image_paths_arg: Optional[List[str]] = None,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Read rasters from an explicit list or scan folder, mask nodata, 
    and return flattened arrays of valid pixels along with sorted years.

    Parameters
    ----------
    output_path : str
        Directory to scan for rasters if image_paths_arg is not provided.
    suffix : str, optional
        File extension to filter by (default: ".tif").
    image_paths_arg : Optional[List[str]], optional
        Explicit list of file paths to process. If provided, overrides directory scan.

    Returns
    -------
    Tuple[List[np.ndarray], List[int]]
        A tuple containing:
        1. List of flattened 1D numpy arrays (one per time point) with invalid pixels removed.
        2. List of integers representing the years sorted chronologically.
    """
    # 1. Resolve Paths
    if image_paths_arg and isinstance(image_paths_arg, (list, tuple)) and image_paths_arg:
        # Use the provided list (e.g., filtered 1989-2021)
        paths = list(
            image_paths_arg
        )
    else:
        # Fallback: scan the entire folder
        search_pattern = os.path.join(
            output_path, 
            f"*{suffix}"
        )
        paths = sorted(
            glob.glob(
                search_pattern
            )
        )

    if not paths:
        raise FileNotFoundError(
            "No input raster paths provided or found."
        )

    # 2. Extract Years from Filenames
    path_year_pairs: List[Tuple[int, str]] = []
    for p in paths:
        basename = os.path.basename(
            p
        )
        # Try finding 19xx or 20xx
        m = re.search(
            r"(19|20)\d{2}", 
            basename
        )
        if m:
            year = int(
                m.group(0)
            )
            path_year_pairs.append(
                (year, p)
            )
        else:
            # Fallback: any 4 digits
            m2 = re.search(
                r"\d{4}", 
                basename
            )
            if m2:
                year = int(
                    m2.group(0)
                )
                path_year_pairs.append(
                    (year, p)
                )

    # Sort by year
    path_year_pairs.sort(
        key=lambda x: x[0]
    )
    ordered_paths = [
        p 
        for _, p in path_year_pairs
    ]
    ordered_years = [
        y 
        for y, _ in path_year_pairs
    ]

    # 3. Read and Mask
    all_data: List[np.ndarray] = []
    all_masks: List[np.ndarray] = []
    
    for path in tqdm(ordered_paths, desc="Reading rasters", unit="raster"):
        with rasterio.open(path) as src:
            data = src.read(
                1
            )
            nod = getattr(
                src, 
                "nodata", 
                None
            )
            
            # Create mask for NoData (255, defined nodata, or NaN)
            mask = (
                data == 255
            )
            if nod is not None:
                mask = mask | (
                    data == nod
                )
            
            # Handle float NaNs if applicable
            if np.issubdtype(data.dtype, np.floating):
                mask = mask | np.isnan(
                    data
                )
                
            all_masks.append(
                mask
            )
            all_data.append(
                data
            )

    if all_masks:
        combined_mask = np.any(
            all_masks, 
            axis=0
        )
    else:
        combined_mask = np.zeros_like(
            all_data[0], 
            dtype=bool
        )

    flattened = [
        data[~combined_mask].flatten() if np.any(combined_mask) else data.flatten()
        for data in all_data
    ]

    return flattened, ordered_years


def generate_all_matrices(
    input_path: str,
    output_path: str,
    suffix: str = ".tif",
    image_paths_arg: Optional[List[str]] = None,
) -> Tuple[List[int], np.ndarray]:
    """
    Generate transition matrices using Hybrid Decomposition logic.
    
    Logic:
    1. Extent: Decomposed Aggregately (Pontius Slide 4).
       Captures swaps between different pixels (e.g., A->C and C->A).
    2. Alternation: Decomposed Per-Pixel (Pontius Slide 9 & 10).
       Preserves trajectory identity (e.g., A->B->C is Shift, not Exchange).

    Parameters
    ----------
    input_path : str
        Directory containing the input raster images (used as fallback).
    output_path : str
        Directory where the generated CSV matrices will be saved.
    suffix : str, optional
        File extension to identify input rasters (default: ".tif").
    image_paths_arg : List[str], optional
        Explicit list of paths to process (overrides folder scan).

    Returns
    -------
    Tuple[List[int], np.ndarray]
        A tuple containing:
        1. List of years processed.
        2. Numpy array of unique class IDs found in the data.
    """
    # 1. Load Data
    # Pass the explicit list (image_paths_arg) to the loader
    flattened_data, years = generate_mask_and_flatten_rasters(
        output_path=input_path, 
        suffix=suffix,
        image_paths_arg=image_paths_arg
    )
    
    if not flattened_data:
        raise ValueError(
            "No data found to process."
        )

    all_classes = np.unique(
        np.concatenate(
            flattened_data
        )
    ).astype(int)
    
    n = len(
        all_classes
    )
    class_to_idx = {
        cls: i 
        for i, cls in enumerate(all_classes)
    }

    # 2. Initialize Matrices
    mat_sum = np.zeros(
        (n, n)
    )
    mat_ext = np.zeros(
        (n, n)
    )
    mat_ax = np.zeros(
        (n, n)
    )
    mat_as = np.zeros(
        (n, n)
    )
    
    interval_matrices = {
        f"{years[t]}-{years[t+1]}": np.zeros((n, n)) 
        for t in range(len(years) - 1)
    }

    # 3. Pixel-level Processing
    num_pixels = len(
        flattened_data[0]
    )
    
    # Pre-convert all data to indices for speed (vectorized lookup)
    mapped_data = []
    vectorized_map = np.vectorize(
        class_to_idx.get
    )
    
    for t_data in flattened_data:
        mapped_data.append(
            vectorized_map(
                t_data
            )
        )
    
    print(
        f"Processing {num_pixels} pixels for transitions..."
    )
    
    # Iterate over pixels
    for r in tqdm(range(num_pixels), desc="Processing Pixels"):
        traj = [
            mapped_data[t][r] 
            for t in range(len(mapped_data))
        ]
        
        start_idx = traj[0]
        end_idx = traj[-1]

        # A) Individual Interval Matrices
        m_r = np.zeros(
            (n, n)
        )
        for t in range(len(traj) - 1):
            s = traj[t]
            e = traj[t+1]
            m_r[
                s, 
                e
            ] += 1
            interval_matrices[
                f"{years[t]}-{years[t+1]}"
            ][
                s, 
                e
            ] += 1
        
        # B) Extent Matrix (First vs Last)
        e_r = np.zeros(
            (n, n)
        )
        e_r[
            start_idx, 
            end_idx
        ] += 1
        
        # Accumulate Aggregates
        mat_sum += m_r
        mat_ext += e_r

        # C) Alternation Decomposition (Per Pixel)
        # Eq 4: A_r = M_r - E_r
        a_r = m_r - e_r
        
        # Eq 5: Ax_r = min(A_r, A_r.T) (clamped >= 0)
        ax_r = np.maximum(
            0, 
            np.minimum(
                a_r, 
                a_r.T
            )
        )
        
        # Eq 6: As_r = A_r - Ax_r
        as_r = a_r - ax_r
        
        mat_ax += ax_r
        mat_as += as_r

    # 4. Extent Decomposition (Aggregate Level)
    # Applied on the total Extent matrix to capture inter-pixel swaps
    mat_ex = np.maximum(
        0, 
        np.minimum(
            mat_ext, 
            mat_ext.T
        )
    )
    mat_es = mat_ext - mat_ex

    # 5. Save Matrices to CSV
    
    # Save Intervals
    for interval_name, mat in interval_matrices.items():
        fname = f"transition_matrix_{interval_name}.csv"
        out_csv = os.path.join(
            output_path, 
            fname
        )
        pd.DataFrame(
            mat, 
            index=all_classes, 
            columns=all_classes
        ).to_csv(
            out_csv
        )

    # Save Aggregates
    interval_str = f"{years[0]}-{years[-1]}"
    aggregated = {
        "sum": mat_sum,
        "extent": mat_ext,
        "extent_exchange": mat_ex,
        "extent_shift": mat_es,
        "alternation_exchange": mat_ax,
        "alternation_shift": mat_as,
    }
    
    for name, mat in aggregated.items():
        fname = f"transition_matrix_{name}_{interval_str}.csv"
        out_csv = os.path.join(
            output_path, 
            fname
        )
        pd.DataFrame(
            mat, 
            index=all_classes, 
            columns=all_classes
        ).to_csv(
            out_csv
        )

    return years, all_classes

# ------------------------------------------------------------------------------
# 5. Transition Matrix Visualization (Decomposed Components)
# ------------------------------------------------------------------------------

def _extract_year_str(
    val: int,
) -> str:
    """
    Helper function to extract year digits from an integer or string.

    Parameters
    ----------
    val : int or str
        The input value containing the year (e.g., 1990, "year_2000").

    Returns
    -------
    str
        The extracted numeric year string.
    """
    match = re.search(
        r"(\d+)",
        str(val),
    )
    return match.group(1) if match else str(val)


def load_square_matrix(
    csv_path: str,
) -> pd.DataFrame:
    """
    Load a square transition matrix from a CSV file and align row/column labels.

    This function ensures that the resulting DataFrame is perfectly square,
    handling cases where rows and columns might differ due to missing transitions.
    It uses a robust sorting method to handle numeric strings consistently.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the transition matrix.

    Returns
    -------
    pd.DataFrame
        A square DataFrame with consistent row and column indices, 
        where missing values are filled with 0.0.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the specified path.
    ValueError
        If the matrix cannot be squared after alignment.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Matrix file not found: {csv_path}"
        )

    df = pd.read_csv(
        csv_path,
        index_col=0,
    )
    df.index = df.index.map(str)
    df.columns = df.columns.map(str)

    # Ensure square shape by union of indices/columns
    if list(df.index) != list(df.columns):
        # Robust sort handles "1" and "1.0" consistently
        labels = sorted(
            set(df.index).union(df.columns),
            key=lambda x: int(float(x)) if x.replace('.', '', 1).isdigit() else x,
        )
        df = df.reindex(
            index=labels,
            columns=labels,
        ).fillna(0.0)

    if df.shape[0] != df.shape[1]:
        raise ValueError(
            f"Matrix not square after alignment: {csv_path}"
        )

    return df


def compute_net_change_from_sum(
    df_sum: pd.DataFrame,
) -> pd.Series:
    """
    Compute the net change per class from a Sum transition matrix.

    Net Change is calculated as: Gains - Losses.
    Diagonal elements (persistence) are excluded from this calculation.

    Parameters
    ----------
    df_sum : pd.DataFrame
        The square transition matrix representing the sum of transitions.

    Returns
    -------
    pd.Series
        A pandas Series containing the net change value for each class,
        indexed by class ID.
    """
    M = df_sum.values.astype(float).copy()
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
    return pd.Series(
        gains - losses,
        index=df_sum.index,
    )


def label_id_to_name(
    labels: Iterable[str],
    class_labels_dict: Dict[int, Dict[str, str]],
) -> List[str]:
    """
    Map class ID strings to human-readable names using the provided dictionary.

    This function attempts to match class IDs (even if formatted as floats
    like "1.0") to their corresponding names defined in `class_labels_dict`.
    It prioritizes the 'rename' key if available, otherwise uses 'name'.

    Parameters
    ----------
    labels : Iterable[str]
        A sequence of class IDs (as strings) to be mapped.
    class_labels_dict : Dict[int, Dict[str, str]]
        A dictionary containing class metadata, including "name" and optional "rename".

    Returns
    -------
    List[str]
        A list of mapped human-readable class names.
    """
    id_to_name = {
        int(k): v.get(
            "rename",
            v.get(
                "name",
                str(k),
            ),
        )
        for k, v in class_labels_dict.items()
    }

    names: List[str] = []
    for lab in labels:
        try:
            # Handle potential float strings like "1.0"
            cid = int(float(lab))
            names.append(
                id_to_name.get(
                    cid,
                    str(lab),
                )
            )
        except Exception:
            names.append(
                str(lab)
            )

    return names


def _unit_label(
    suffix: str,
    base_label: str = "Number of pixels",
) -> str:
    """
    Build a descriptive label for the colorbar based on the unit suffix.

    Parameters
    ----------
    suffix : str
        The unit suffix (e.g., "M", "k", "B").
    base_label : str, optional
        The base text for the label (default: "Number of pixels").

    Returns
    -------
    str
        The formatted label string (e.g., "Millions of pixels").
    """
    mapping = {
        "": base_label,
        "k": "Thousands of pixels",
        "M": "Millions of pixels",
        "B": "Billions of pixels",
        "T": "Trillions of pixels",
    }
    return mapping.get(
        suffix,
        f"{base_label} ({suffix})",
    )


def _unit_formatter(
    factor: float,
    suffix: str,
    decimals: int = 1,
) -> mticker.FuncFormatter:
    """
    Build a matplotlib tick formatter that scales values and appends a suffix.

    Parameters
    ----------
    factor : float
        The factor to divide values by (e.g., 1_000_000 for Millions).
    suffix : str
        The string suffix to append (e.g., "M").
    decimals : int, optional
        Number of decimal places to display (default: 1).

    Returns
    -------
    mticker.FuncFormatter
        A formatter object compatible with matplotlib axes.
    """
    fmt = f"{{:.{decimals}f}}{suffix}"

    def _fmt(
        x: float,
        pos: int,
    ) -> str:
        return fmt.format(x / factor)

    return mticker.FuncFormatter(_fmt)


def annotate_heatmap(
    ax: plt.Axes,
    M: np.ndarray,
    fontsize: int = 8,
) -> None:
    """
    Annotate a heatmap with integer cell values, using adaptive text color.

    This function iterates through the matrix cells and places text labels.
    It automatically chooses between white or black text to ensure readability
    against the background color (White for dark cells, Black for light cells).
    Crucially, it does NOT skip zero values.

    Parameters
    ----------
    ax : plt.Axes
        The matplotlib axes object to annotate.
    M : np.ndarray
        The 2D data matrix containing the values to display.
    fontsize : int, optional
        Font size for the annotations (default: 8).
    """
    # 1. Check if matrix is empty
    if M.size == 0:
        return

    # 2. Prepare data for threshold calculation (ignoring diagonal)
    M_off = M.copy()
    np.fill_diagonal(
        M_off,
        np.nan,
    )
    data_off = M_off[np.isfinite(M_off)]

    # 3. Check for positive and negative ranges
    has_pos = np.any(data_off > 0)
    has_neg = np.any(data_off < 0)

    # 4. Calculate adaptive thresholds
    # Values exceeding 50% of the max (or min) will get white text
    max_pos = float(np.nanmax(data_off[data_off > 0])) if has_pos else 0.0
    min_neg = float(np.nanmin(data_off[data_off < 0])) if has_neg else 0.0

    thresh_pos = 0.5 * max_pos if has_pos else np.inf
    thresh_neg = 0.5 * min_neg if has_neg else -np.inf

    # 5. Iterate over matrix cells
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            # 5a. Skip diagonal annotation (persistence)
            if i == j:
                continue

            v = float(M[i, j])
            txt = f"{int(round(v))}"

            # 5b. Determine text color based on background intensity
            # Note: Zeros fall into the 'else' block (Black text)
            if (has_pos and v >= thresh_pos) or (has_neg and v <= thresh_neg):
                color = "white"
            else:
                color = "black"

            # 5c. Place the text on the plot
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=fontsize,
                color=color,
                clip_on=True,
            )


def plot_heatmap(
    df: pd.DataFrame,
    title: str,
    class_labels_dict: Dict[int, Dict[str, str]],
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "YlOrRd",
    vmin: float = 0.0,
    vmax: Optional[float] = None,
    rotate_xticks_deg: int = 90,
    cbar_label: str = "Number of pixels",
    annotate: bool = True,
    cell_size_inch: float = 0.8,
    tick_fontsize_x: Optional[int] = None,
    tick_fontsize_y: Optional[int] = None,
    axis_label_fontsize: Optional[int] = None,
    title_fontsize: Optional[int] = None,
    ann_fontsize: int = 8,
    cbar_fraction: float = 0.025,
    cbar_pad: float = 0.02,
) -> None:
    """
    Plot a square matrix as a heatmap with adaptive integer colorbar and dual scaling.

    This function supports plotting negative values (used in Alternation Shift)
    by applying a divergent color map (Blues for negative, YlOrRd for positive)
    and creating a composite colorbar.

    Parameters
    ----------
    df : pd.DataFrame
        The square transition matrix to plot.
    title : str
        The title of the plot.
    class_labels_dict : Dict
        Dictionary mapping class IDs to human-readable names.
    save_path : Optional[str]
        File path to save the generated image.
    figsize : Tuple[float, float], optional
        Figure size in inches. If None, calculated from `cell_size_inch`.
    cmap : str, optional
        Colormap for positive values (default: "YlOrRd").
    vmin : float, optional
        Minimum value for normalization (default: 0.0).
    vmax : float, optional
        Maximum value for normalization. If None, inferred from data.
    rotate_xticks_deg : int, optional
        Rotation angle for x-axis ticks (default: 90).
    cbar_label : str, optional
        Label for the colorbar (default: "Number of pixels").
    annotate : bool, optional
        Whether to write values inside cells (default: True).
    cell_size_inch : float, optional
        Size of each cell in inches, used to calculate figsize (default: 0.8).
    tick_fontsize_x : int, optional
        Font size for x-axis ticks.
    tick_fontsize_y : int, optional
        Font size for y-axis ticks.
    axis_label_fontsize : int, optional
        Font size for axis labels.
    title_fontsize : int, optional
        Font size for the title.
    ann_fontsize : int, optional
        Font size for cell annotations (default: 8).
    cbar_fraction : float, optional
        Fraction of original axes to use for colorbar (default: 0.025).
    cbar_pad : float, optional
        Padding between axes and colorbar (default: 0.02).
    """
    # 0. Validate and set default font sizes
    if tick_fontsize_x is None:
        tick_fontsize_x = 12
    if tick_fontsize_y is None:
        tick_fontsize_y = 12
    if axis_label_fontsize is None:
        axis_label_fontsize = 12
    if title_fontsize is None:
        title_fontsize = 14

    # 1. Prepare Data and Labels
    labels = list(df.index)
    matrix_values = df.values.astype(float)

    # 2. Determine Scale (masking diagonal to focus on transitions)
    matrix_scale = matrix_values.copy()
    np.fill_diagonal(
        matrix_scale,
        0.0,
    )
    finite_vals = matrix_scale[np.isfinite(matrix_scale)]

    # 3. Calculate effective vmin/vmax for color scaling
    if finite_vals.size == 0:
        has_negative = False
        vmin_eff, vmax_eff = 0.0, 1.0
    else:
        has_negative = float(np.nanmin(finite_vals)) < 0.0
        min_val = float(np.nanmin(finite_vals))
        max_val = float(np.nanmax(finite_vals))

        if has_negative:
            # Use data bounds for divergent scaling
            vmin_eff, vmax_eff = min_val, max_val
        else:
            vmin_eff = vmin
            vmax_eff = float(max_val) if vmax is None else float(vmax)

        if vmin_eff == vmax_eff:
            vmax_eff += 1.0

    # 4. Setup Figure dimensions
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

    # 5. Plot Heatmap Layers
    if has_negative:
        # 5a. Positive values layer (YlOrRd)
        matrix_pos = np.where(
            matrix_values < 0.0,
            0.0,
            matrix_values,
        )
        norm_pos = mcolors.Normalize(
            vmin=0.0,
            vmax=vmax_eff,
        )
        ax.imshow(
            matrix_pos,
            aspect="equal",
            cmap=plt.cm.YlOrRd,
            norm=norm_pos,
        )
        # 5b. Negative values layer (Blues_r)
        matrix_neg = np.ma.masked_where(
            matrix_values >= 0.0,
            matrix_values,
        )
        norm_neg = mcolors.Normalize(
            vmin=vmin_eff,
            vmax=0.0,
        )
        ax.imshow(
            matrix_neg,
            aspect="equal",
            cmap=plt.cm.Blues_r,
            norm=norm_neg,
        )
    else:
        # 5c. Standard positive layer
        norm_pos = mcolors.Normalize(
            vmin=vmin_eff,
            vmax=vmax_eff,
        )
        ax.imshow(
            matrix_values,
            aspect="equal",
            cmap=plt.cm.YlOrRd,
            norm=norm_pos,
        )

    # 6. Overlay Black Diagonal
    diag_mask = np.eye(
        nrows,
        dtype=bool,
    )
    matrix_diag = np.ma.masked_where(
        ~diag_mask,
        np.ones_like(matrix_values),
    )
    ax.imshow(
        matrix_diag,
        aspect="equal",
        cmap=mcolors.ListedColormap(["black"]),
        vmin=0,
        vmax=1,
    )

    # 7. Configure Axes Ticks and Labels
    ax.set_xticks(
        range(len(labels))
    )
    ax.set_yticks(
        range(len(labels))
    )
    
    tick_names = label_id_to_name(
        labels,
        class_labels_dict,
    )

    ax.set_xticklabels(
        tick_names,
        rotation=rotate_xticks_deg,
        fontsize=tick_fontsize_x,
    )
    ax.set_yticklabels(
        tick_names,
        fontsize=tick_fontsize_y,
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

    # 8. Construct Colorbar
    # Build continuous gradient for colorbar
    n_bar = 256
    vals = np.linspace(
        vmin_eff,
        vmax_eff,
        n_bar,
    )
    colors_bar = []
    for v in vals:
        if has_negative and v < 0.0:
            t = (v - vmin_eff) / (0.0 - vmin_eff) if vmin_eff < 0 else 0
            colors_bar.append(
                plt.cm.Blues_r(t)
            )
        else:
            t = max(0.0, v) / vmax_eff if vmax_eff > 0.0 else 0.0
            colors_bar.append(
                plt.cm.YlOrRd(t)
            )

    cmap_bar = mcolors.ListedColormap(colors_bar)
    norm_bar = mcolors.Normalize(
        vmin=vmin_eff,
        vmax=vmax_eff,
    )
    scalar_mappable = plt.cm.ScalarMappable(
        cmap=cmap_bar,
        norm=norm_bar,
    )
    scalar_mappable.set_array([])

    cbar = fig.colorbar(
        scalar_mappable,
        ax=ax,
        fraction=cbar_fraction,
        pad=cbar_pad,
    )

    # 9. Format Colorbar Units
    max_abs = float(np.nanmax(np.abs(finite_vals))) if finite_vals.size > 0 else 0.0
    if max_abs >= 1_000_000:
        factor, suffix = 1_000_000.0, "M"
    elif max_abs >= 1_000:
        factor, suffix = 1_000.0, "k"
    elif max_abs >= 100:
        factor, suffix = 100.0, "hundreds"
    else:
        factor, suffix = 1.0, ""

    cbar.locator = mticker.MaxNLocator(
        nbins=5,
        integer=True,
        steps=[1, 2, 5, 10],
    )
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
        labelpad=15,
        fontsize=12,
    )
    cbar.update_ticks()

    # 10. Add Annotations
    if annotate:
        annotate_heatmap(
            ax=ax,
            M=matrix_values,
            fontsize=ann_fontsize,
        )

    # 11. Save and Display
    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Saved plot: {save_path}")

    plt.show()
    plt.close()


def plot_decomposed_transitions(
    output_path: str,
    years: List[int],
    class_labels_dict: Dict[int, Dict[str, str]],
) -> None:
    """
    Orchestrate the loading, sorting, and plotting of the decomposed matrices.

    This function performs the following steps:
    1. Identifies the relevant CSV files based on years.
    2. Loads the 'Sum' matrix to calculate Net Change.
    3. Sorts classes from 'Largest Loser' to 'Largest Gainer'.
    4. Loads, reorders, and plots the Extent and Alternation matrices.

    Parameters
    ----------
    output_path : str
        Directory containing the transition matrix CSV files.
    years : List[int]
        List of years included in the analysis range (used for filenames).
    class_labels_dict : Dict[int, Dict[str, str]]
        Metadata for mapping class IDs to names.
    """
    # 1. Define Paths and Years
    str_y0 = _extract_year_str(
        years[0]
    )
    str_y1 = _extract_year_str(
        years[-1]
    )
    interval = f"{str_y0}-{str_y1}"

    files = {
        "Sum": f"transition_matrix_sum_{interval}.csv",
        "Ext_Exc": f"transition_matrix_extent_exchange_{interval}.csv",
        "Ext_Shift": f"transition_matrix_extent_shift_{interval}.csv",
        "Alt_Exc": f"transition_matrix_alternation_exchange_{interval}.csv",
        "Alt_Shift": f"transition_matrix_alternation_shift_{interval}.csv",
    }

    # 2. Load Sum Matrix & Determine Sort Order
    path_sum = os.path.join(
        output_path,
        files["Sum"],
    )
    try:
        df_sum = load_square_matrix(
            path_sum
        )
    except FileNotFoundError:
        print(f"Critical: Sum matrix not found at {path_sum}. Cannot sort.")
        return

    net_change = compute_net_change_from_sum(
        df_sum
    )
    sorted_classes = net_change.sort_values(
        ascending=True
    ).index.tolist()

    # 3. Process Each Decomposed Matrix
    matrices_to_plot = [
        ("Ext_Exc", f"Allocation Exchange {str_y0}-{str_y1}"),
        ("Ext_Shift", f"Allocation Shift {str_y0}-{str_y1}"),
        ("Alt_Exc", f"Alternation Exchange {str_y0}...{str_y1}"),
        ("Alt_Shift", f"Alternation Shift {str_y0}...{str_y1}"),
    ]

    for key, title in matrices_to_plot:
        path = os.path.join(
            output_path,
            files[key],
        )
        try:
            df = load_square_matrix(
                path
            )
        except FileNotFoundError:
            print(f"Skipping {title}: File not found ({files[key]})")
            continue

        # Reorder to match Sum matrix order
        df_sorted = df.reindex(
            index=sorted_classes,
            columns=sorted_classes,
        ).fillna(0.0)

        # Plot using specific parameters from the notebook example
        out_name = f"heatmap_{title.replace(' ', '_').upper()}.png"

        plot_heatmap(
            df=df_sorted,
            title=title,
            class_labels_dict=class_labels_dict,
            save_path=os.path.join(
                output_path,
                out_name,
            ),
            tick_fontsize_x=12,
            tick_fontsize_y=12,
            axis_label_fontsize=16,
            title_fontsize=16,
            ann_fontsize=9,
            cbar_fraction=0.025,
            cbar_pad=0.02,
        )

# ------------------------------------------------------------------------------
# 6. Component Calculation Logic
# ------------------------------------------------------------------------------

class ComponentCalculator:
    """
    Decomposes the difference between two maps into components of Quantity,
    Exchange, and Shift.

    Following Pontius Jr. and da Fonseca (2025), this class analyzes a square
    contingency matrix (Transition Matrix) to quantify:
    1. Quantity: The net difference in the size of the category.
    2. Exchange: The difference component attributable to simultaneous gain and
       loss.
    3. Shift: The difference component that is neither Quantity nor Exchange.
    """

    def __init__(
        self,
        transition_matrix: np.ndarray
    ) -> None:
        """
        Initializes the calculator.

        Parameters
        ----------
        transition_matrix : np.ndarray
            A square numpy array where rows represent the categories at the
            Initial Time Point and columns represent the categories at the
            Final Time Point.
        """
        self.matrix = transition_matrix.astype(
            float
        )
        self.num_classes = transition_matrix.shape[0]
        self.class_components: List[Dict] = []

    def calculate_components(
        self,
        force_component: Optional[str] = None
    ) -> "ComponentCalculator":
        """
        Executes the calculation of change components for all categories.

        Parameters
        ----------
        force_component : str, optional
            A string to override standard calculation. If "Exchange", treats
            matrix as pure exchange. If "Shift", treats matrix as pure shift.
            Used for processing Alternation matrices. Default is None.

        Returns
        -------
        ComponentCalculator
            Returns the instance itself to allow for method chaining.
        """
        for class_idx in range(self.num_classes):
            # Calculate standard sums
            gain_sum = np.sum(
                self.matrix[:, class_idx]
            )
            loss_sum = np.sum(
                self.matrix[class_idx, :]
            )

            # Standard net change calculation
            q_gain = max(
                0.0,
                gain_sum - loss_sum
            )
            q_loss = max(
                0.0,
                loss_sum - gain_sum
            )

            if force_component == "Exchange":
                # Matrix content is treated purely as exchange
                exchange = loss_sum - self.matrix[class_idx, class_idx]
                shift = 0.0
                q_gain = gain_sum - loss_sum
                q_loss = loss_sum - gain_sum
                
            elif force_component == "Shift":
                # Matrix content is treated purely as shift
                exchange = 0.0
                shift = loss_sum - self.matrix[class_idx, class_idx]
                q_gain = 0.0
                q_loss = 0.0
                
            else:
                # Standard Pontius decomposition
                mutual = np.sum(
                    np.minimum(
                        self.matrix[class_idx, :],
                        self.matrix[:, class_idx]
                    )
                )
                exchange = mutual - self.matrix[class_idx, class_idx]
                total_trans = loss_sum - self.matrix[class_idx, class_idx]
                shift = total_trans - q_loss - exchange

            self.class_components.append({
                "Quantity_Gain": q_gain,
                "Quantity_Loss": q_loss,
                "Exchange_Gain": exchange,
                "Exchange_Loss": exchange,
                "Shift_Gain": shift,
                "Shift_Loss": shift,
            })
        return self


def process_matrix_file(
    matrix_type: str,
    output_path: str,
    time_points: List[int],
    class_map: Dict[int, Dict[str, str]],
    start_time: Optional[int] = None,
    end_time: Optional[int] = None
) -> List[Dict]:
    """
    Reads a specific transition matrix CSV, calculates components, and formats.

    Uses a robust search logic to find files that match both the requested years
    and the specific matrix type (interval, sum, extent, alternation), avoiding
    ambiguity and warning if files are missing.

    Parameters
    ----------
    matrix_type : str
        Type of matrix ('interval', 'extent', 'sum', 'alternation_exchange', etc).
    output_path : str
        Directory containing the CSV files.
    time_points : List[int]
        List of all time points in the study.
    class_map : Dict[int, Dict[str, str]]
        Dictionary mapping class IDs to names.
    start_time : int, optional
        Start year for interval matrices. Default is None.
    end_time : int, optional
        End year for interval matrices. Default is None.

    Returns
    -------
    List[Dict]
        A list of dictionaries representing the component data. Returns empty
        list if file is not found.
    """
    results = []
    
    # 1. Determine the Year String pattern to search for
    if matrix_type == "interval":
        year_pattern = f"{start_time}-{end_time}"
        label_time = year_pattern
    else:
        y0_str = str(
            time_points[0]
        )
        yN_str = str(
            time_points[-1]
        )
        year_pattern = f"{y0_str}-{yN_str}"
        label_time = matrix_type

    # 2. Robust File Search
    # Instead of guessing the full name, we filter by years AND type keyword.
    full_path = None
    
    if os.path.exists(output_path):
        all_files = [
            f for f in os.listdir(output_path) 
            if f.endswith(".csv") and year_pattern in f
        ]
        
        # Filter by matrix type strictly to avoid ambiguity
        candidates = []
        for f in all_files:
            if matrix_type == "interval":
                # Intervals should NOT contain aggregation keywords
                if "sum" not in f and "extent" not in f and "alternation" not in f:
                    candidates.append(
                        f
                    )
            elif matrix_type == "sum":
                if "sum" in f:
                    candidates.append(
                        f
                    )
            elif matrix_type == "extent":
                if "extent" in f:
                    candidates.append(
                        f
                    )
            elif "alternation" in matrix_type:
                # Matches alternation_exchange or alternation_shift
                if matrix_type in f:
                    candidates.append(
                        f
                    )
        
        if candidates:
            # Pick the first match
            full_path = os.path.join(
                output_path,
                candidates[0]
            )
    
    # Fail Fast / Warning System
    if not full_path or not os.path.exists(full_path):
        print(
            f"Warning: Could not find CSV for Type='{matrix_type}' and Years='{year_pattern}' in {output_path}"
        )
        return []

    # 3. Process components
    force_comp = None
    if "exchange" in matrix_type:
        force_comp = "Exchange"
    elif "shift" in matrix_type:
        force_comp = "Shift"
    
    df_mat = pd.read_csv(
        full_path,
        index_col=0
    )
    
    # Calculate components
    calc = ComponentCalculator(
        df_mat.values
    ).calculate_components(
        force_component=force_comp
    )

    # Format results
    for idx, class_id in enumerate([int(c) for c in df_mat.index]):
        cls_name = class_map.get(
            class_id,
            {}
        ).get(
            "name",
            f"Class {class_id}"
        )
        
        comp_vals = calc.class_components[idx]
        
        for comp_name in ["Quantity", "Exchange", "Shift"]:
            # Standardize component labels.
            # Quantity remains "Quantity".
            # Exchange/Shift become "Allocation_" or "Alternation_" based on type.
            label_comp = comp_name
            
            if comp_name == "Quantity":
                label_comp = "Quantity"
            else:
                if matrix_type in ["extent", "sum"]:
                    label_comp = f"Allocation_{comp_name}"
                elif "alternation" in matrix_type:
                    label_comp = f"Alternation_{comp_name}"

            results.append({
                "Time_Interval": label_time,
                "Class": cls_name,
                "Component": label_comp,
                "Gain": comp_vals[f"{comp_name}_Gain"],
                "Loss": comp_vals[f"{comp_name}_Loss"],
            })
            
    return results


def generate_component_summary(
    output_path: str,
    years: List[int],
    class_labels_dict: Dict[int, Dict[str, str]]
) -> pd.DataFrame:
    """
    Generates a consolidated DataFrame of all change components with adjustments.

    Iterates through all time intervals and aggregated matrix types, reads the
    CSVs, aggregates results, and applies the logic to adjust Alternation
    Exchange and Alternation Shift values before saving.

    Parameters
    ----------
    output_path : str
        Directory containing the transition matrix CSVs.
    time_points : List[int]
        List of time points (years).
    class_labels_dict : Dict[int, Dict[str, str]]
        Mapping of class IDs to metadata.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns: Time_Interval, Class, Component, Gain, Loss.
    """
    all_results = []

    # 1. Process Annual Intervals
    for i in range(len(years) - 1):
        interval_results = process_matrix_file(
            matrix_type="interval",
            output_path=output_path,
            time_points=years,
            class_map=class_labels_dict,
            start_time=years[i],
            end_time=years[i+1]
        )
        all_results.extend(
            interval_results
        )
    
    # 2. Process Aggregated Matrices
    agg_types = [
        "extent",
        "sum",
        "alternation_exchange",
        "alternation_shift"
    ]
    
    for mtype in agg_types:
        agg_results = process_matrix_file(
            matrix_type=mtype,
            output_path=output_path,
            time_points=years,
            class_map=class_labels_dict
        )
        all_results.extend(
            agg_results
        )

    df_raw = pd.DataFrame(
        all_results
    )
    
    if df_raw.empty:
        print(
            "Warning: No component data found. Check your file paths."
        )
        return pd.DataFrame()

    # 3. Apply Business Logic: Adjust Alternation Components
    # We iterate by class to adjust Exchange vs Shift within Alternation
    adjusted_rows = []
    classes = df_raw["Class"].unique()

    for cls in classes:
        cls_data = df_raw[df_raw["Class"] == cls].copy()
        
        # Extract raw Alternation values
        row_exc = cls_data[cls_data["Component"] == "Alternation_Exchange"]
        row_shf = cls_data[cls_data["Component"] == "Alternation_Shift"]
        
        # Default values
        gain_exc = row_exc["Gain"].sum() if not row_exc.empty else 0.0
        loss_exc = row_exc["Loss"].sum() if not row_exc.empty else 0.0
        gain_shf = row_shf["Gain"].sum() if not row_shf.empty else 0.0
        loss_shf = row_shf["Loss"].sum() if not row_shf.empty else 0.0

        # Logic for Gains
        net_alt_gain = gain_exc + gain_shf
        adj_gain_exc = 0.0
        adj_gain_shf = 0.0
        
        if net_alt_gain > 0.0001:
            if gain_exc > 0:
                adj_gain_exc = gain_exc
                adj_gain_shf = max(0, net_alt_gain - gain_exc)
            else:
                adj_gain_exc = 0
                adj_gain_shf = net_alt_gain
        
        # Logic for Losses
        net_alt_loss = loss_exc + loss_shf
        adj_loss_exc = 0.0
        adj_loss_shf = 0.0
        
        if net_alt_loss > 0.0001:
            if loss_exc > 0:
                adj_loss_exc = loss_exc
                adj_loss_shf = max(0, net_alt_loss - loss_exc)
            else:
                adj_loss_exc = 0
                adj_loss_shf = net_alt_loss

        # Update or Append adjusted rows
        # We keep non-alternation rows as is
        non_alt = cls_data[~cls_data["Component"].str.contains("Alternation")]
        adjusted_rows.extend(
            non_alt.to_dict('records')
        )
        
        # Add adjusted Alternation rows (Exchange and Shift)
        if not row_exc.empty:
            rec = row_exc.iloc[0].to_dict()
            rec["Gain"] = adj_gain_exc
            rec["Loss"] = adj_loss_exc
            adjusted_rows.append(
                rec
            )
            
        if not row_shf.empty:
            rec = row_shf.iloc[0].to_dict()
            rec["Gain"] = adj_gain_shf
            rec["Loss"] = adj_loss_shf
            adjusted_rows.append(
                rec
            )
            
        # Include Alternation Quantity if present (legacy support)
        # It's kept as "Quantity" per the new logic but filtered here if labeled otherwise in raw
        row_qty = cls_data[cls_data["Component"] == "Alternation_Quantity"]
        if not row_qty.empty:
            adjusted_rows.extend(
                row_qty.to_dict('records')
            )

    df_out = pd.DataFrame(
        adjusted_rows
    )
    
    # Save the FINAL, ADJUSTED values
    output_file = os.path.join(
        output_path,
        "change_components_summary.csv"
    )
    df_out.to_csv(
        output_file,
        index=False
    )
    print(
        f"Summary components (adjusted) saved to: {output_file}"
    )
    
    return df_out


def compute_net_change_from_sum(
    df_sum: pd.DataFrame
) -> pd.Series:
    """
    Computes the Net Change (Quantity) for each category from a summary table.

    Net Change corresponds to the difference between the size of the category
    at the Final Time Point and the Initial Time Point.

    Parameters
    ----------
    df_sum : pd.DataFrame
        A DataFrame representing the transition matrix. Rows denote the
        Initial Time Point and columns denote the Final Time Point.

    Returns
    -------
    pd.Series
        A pandas Series where the index corresponds to the category and values
        represent the Net Change (Column Sums - Row Sums).
    """
    col_sums = df_sum.sum(
        axis=0
    )
    row_sums = df_sum.sum(
        axis=1
    )
    return col_sums - row_sums


def get_ordered_transition_matrices(
    df_sum: pd.DataFrame,
    df_ext: pd.DataFrame,
    df_alt: pd.DataFrame,
    df_exc: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reorders matrices based on Net Change (from largest loss to largest gain).

    This ordering facilitates the visual interpretation of the patterns in the
    transition matrices.

    Parameters
    ----------
    df_sum : pd.DataFrame
        The summary matrix (Total or Gross Change).
    df_ext : pd.DataFrame
        The Exchange/Transition matrix.
    df_alt : pd.DataFrame
        The Alternation matrix.
    df_exc : pd.DataFrame
        The Exchange component matrix.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing the reordered DataFrames in the same order as input.
    """
    net_change = compute_net_change_from_sum(
        df_sum
    )
    # Sort labels: Largest Loss (negative) -> Largest Gain (positive)
    order_labels = net_change.sort_values(
        ascending=True
    ).index.tolist()

    def _reorder(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        return df.reindex(
            index=order_labels,
            columns=order_labels
        )

    return (
        _reorder(df_sum),
        _reorder(df_ext),
        _reorder(df_alt),
        _reorder(df_exc),
    )

# ------------------------------------------------------------------------------
# 5. Component Visualization (Bar Charts)
# ------------------------------------------------------------------------------

def plot_change_components(
    components_df: pd.DataFrame,
    class_labels: Dict[int, str],
    output_path: str,
    filename: str = "change_components_bar.png"
) -> None:
    """
    Plots stacked bar charts for Gain and Loss components.

    Visualizes Quantity, Allocation Exchange, Allocation Shift,
    Alternation Exchange, and Alternation Shift using a strict color scheme.

    Parameters
    ----------
    components_df : pd.DataFrame
        A DataFrame containing component data with columns 'Class', 'Gain',
        'Loss', and 'Component'.
    class_labels : Dict[int, str]
        A dictionary mapping class IDs to class names.
    output_path : str
        The directory path where the output plot will be saved.
    filename : str, optional
        The name of the output file. Default is "change_components_bar.png".

    Returns
    -------
    None
        The function saves the bar chart to disk.
    """
    # Strict color definition as provided.
    comp_colors = {
        "Quantity": "#1f77b4",
        "Allocation_Exchange": "#ffd700",
        "Alternation_Exchange": "#ff8080",
        "Allocation_Shift": "#2ca02c",
        "Alternation_Shift": "#990099",
    }
    
    # Strict stack order as provided.
    component_order = [
        "Quantity",
        "Allocation_Shift",
        "Allocation_Exchange",
        "Alternation_Shift",
        "Alternation_Exchange"
    ]

    # Group classes
    classes = sorted(
        components_df["Class"].unique()
    )
    
    fig, ax = plt.subplots(
        figsize=(14, 8)
    )
    
    bar_width = 0.4
    indices = np.arange(
        len(classes)
    )
    
    for i, cls in enumerate(classes):
        cls_data = components_df[components_df["Class"] == cls]
        
        current_bottom_gain = 0.0
        current_bottom_loss = 0.0
        
        for comp_name in component_order:
            row = cls_data[cls_data["Component"] == comp_name]
            
            # Skip if component not present in data for this class
            # (e.g., Annual intervals might not have Allocation_ prefix if raw)
            # However, with process_matrix_file fixed, legacy 'Exchange' should be handled.
            
            # Fallback handling for legacy non-prefixed components (Annual intervals)
            # If "Allocation_Exchange" is requested but only "Exchange" exists, we map it.
            if row.empty and comp_name == "Allocation_Exchange":
                row = cls_data[cls_data["Component"] == "Exchange"]
            if row.empty and comp_name == "Allocation_Shift":
                row = cls_data[cls_data["Component"] == "Shift"]

            if row.empty:
                continue
            
            gain_val = row["Gain"].sum()
            loss_val = row["Loss"].sum()
            
            # Get color
            color_val = comp_colors.get(
                comp_name,
                "gray"
            )
                
            # Plot Gain Bar (Positive)
            if gain_val > 0:
                ax.bar(
                    i - bar_width/2,
                    gain_val,
                    bar_width,
                    bottom=current_bottom_gain, 
                    color=color_val,
                    label=comp_name if i==0 else ""
                )
                current_bottom_gain += gain_val
            
            # Plot Loss Bar (Negative)
            if loss_val > 0:
                ax.bar(
                    i + bar_width/2,
                    -loss_val,
                    bar_width,
                    bottom=-current_bottom_loss,
                    color=color_val
                )
                current_bottom_loss += loss_val

    ax.set_xticks(
        indices
    )
    ax.set_xticklabels(
        [str(c) for c in classes],
        rotation=45,
        ha='right'
    )
    ax.set_ylabel(
        "Change (Pixel/Area)"
    )
    ax.set_title(
        "Components of Change: Gain and Loss"
    )
    
    # Create deduplicated legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(
        zip(labels, handles)
    )
    # Filter legend to only show what was actually plotted and exists in our order
    ordered_handles = []
    ordered_labels = []
    for comp in component_order:
        if comp in by_label:
            ordered_handles.append(
                by_label[comp]
            )
            ordered_labels.append(
                comp
            )
            
    ax.legend(
        ordered_handles,
        ordered_labels,
        title="Component"
    )
    
    plt.axhline(
        0,
        color='black',
        linewidth=0.8
    )
    plt.tight_layout()
    
    out_file = os.path.join(
        output_path,
        filename
    )
    plt.savefig(
        out_file,
        dpi=300
    )
    plt.close()
    print(
        f"Components plot saved to: {out_file}"
    )


# ------------------------------------------------------------------------------
# 6. Trajectory Analysis
# ------------------------------------------------------------------------------

@nb.jit(nopython=True, parallel=True)
def process_stack_parallel(
    stack: np.ndarray,
    height: int,
    width: int
) -> np.ndarray:
    """
    Calculates the Number of Changes per pixel across the time series.

    As defined in the Trajectory concept, this function counts how many times
    a pixel transitions from one category to another across the stack of
    temporal layers (time points).

    Parameters
    ----------
    stack : np.ndarray
        A 3D numpy array of shape (Time Points, Height, Width) containing the
        categorical data for each time point.
    height : int
        The height (number of rows) of the raster.
    width : int
        The width (number of columns) of the raster.

    Returns
    -------
    np.ndarray
        A 2D numpy array (uint8) of shape (Height, Width). Each pixel value
        represents the total count of changes observed in the time series.
    """
    number_of_changes = np.zeros(
        (height, width),
        dtype=np.uint8
    )
    for i in prange(height):
        for j in range(width):
            changes = 0
            valid_mask = stack[0, i, j] != 0
            
            for t in range(1, stack.shape[0]):
                curr = stack[t, i, j]
                prev = stack[t - 1, i, j]
                
                if curr != 0:
                    valid_mask = True
                    if prev != 0 and curr != prev:
                        changes += 1
            
            if not valid_mask:
                number_of_changes[i, j] = 0
            else:
                number_of_changes[i, j] = changes
    return number_of_changes


def generate_trajectory_map(
    output_path: str,
    image_paths: List[str]
) -> str:
    """
    Generates the Trajectory Map (Number of Changes) for the time series.

    This function orchestrates the loading of the raster stack, the computation
    of the pixel-wise number of changes using parallel processing, and the
    export of the result as a GeoTIFF.

    Parameters
    ----------
    output_path : str
        The root directory containing the 'input_rasters' folder and where the
        output trajectory map will be saved.
    image_paths : List[str]
        A mandatory list of absolute file paths to the raster images, in
        chronological order.

    Returns
    -------
    str
        The file path to the saved trajectory map ('trajectory_map.tif').
    
    Raises
    ------
    FileNotFoundError
        If no raster files are found in the provided list.
    """
    if not image_paths:
        raise FileNotFoundError(
            "No image paths provided for trajectory analysis."
        )

    print(
        f"Found {len(image_paths)} rasters for trajectory analysis."
    )

    # Read metadata from the first raster to establish stack dimensions
    with rasterio.open(image_paths[0]) as src:
        meta = src.meta.copy()
        height = src.height
        width = src.width
        # Capture the data type (e.g., uint8) to minimize RAM usage
        dtype = src.meta['dtype']

    # Pre-allocate memory for the 3D stack
    # Shape: (Time Points, Height, Width)
    num_time_points = len(
        image_paths
    )
    stack = np.zeros(
        (num_time_points, height, width),
        dtype=dtype
    )

    # Load rasters directly into the pre-allocated array slices
    for t, p in enumerate(tqdm(image_paths, desc="Loading stack")):
        with rasterio.open(p) as src:
            stack[t, :, :] = src.read(
                1
            )

    print(
        "Computing trajectories (Number of Changes)..."
    )
    trajectory_map = process_stack_parallel(
        stack,
        height,
        width
    )

    meta.update(
        dtype=np.uint8,
        count=1,
        nodata=0
    )
    out_file = os.path.join(
        output_path,
        "trajectory_map.tif"
    )
    
    with rasterio.open(out_file, "w", **meta) as dst:
        dst.write(
            trajectory_map,
            1
        )
        
    print(
        f"Trajectory map saved to: {out_file}"
    )
    return out_file

