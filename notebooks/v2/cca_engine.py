"""cca_engine.py

Modularized Change Component Analysis (CCA) engine.

This module provides pure, testable functions to load raster stacks,
compute pixel counts, classify trajectories, compute CCA aggregated
components and generate heatmaps. Input validation is provided via
Pydantic models. Heavy numerical routines use Numba for performance.

Do NOT execute module code on import; functions return pandas DataFrames
or file paths so downstream agents can consume results.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numba as nb
import numpy as np
import pandas as pd
import rasterio
from numba import prange
from pydantic import BaseModel, DirectoryPath, FilePath, conint
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# -----------------------------
# Pydantic input models
# -----------------------------


class StackInput(BaseModel):
    """Validation model for raster stack inputs.

    This model enforces type constraints for loading temporal sequences of raster
    data. All raster files must have identical dimensions and geospatial metadata.

    Attributes
    ----------
    image_paths : List[FilePath]
        Paths to raster files. Must be valid file paths. Will be sorted internally
        to ensure correct temporal order during processing.
    nodata : conint(ge=0)
        NoData integer value indicating missing or invalid pixels. Typically 0 or 255.
        Defaults to 255. Pixels with this value are excluded from all analyses.
    """

    image_paths: List[FilePath]
    nodata: conint(ge=0) = 255


class PixelCountInput(BaseModel):
    """Validation model for pixel count and classification accounting.

    Computes per-class pixel statistics (area in pixels) for each temporal snapshot.
    Requires aligned image and year sequences and a class dictionary mapping IDs
    to human-readable names (e.g., 1 → {"name": "Forest", "code": "FOR"}).

    Attributes
    ----------
    image_paths : List[FilePath]
        Ordered paths to classification rasters (one per year).
    years : List[conint(ge=0)]
        Corresponding calendar years. Must have same length as image_paths.
    class_labels_dict : Dict[int, Dict[str, str]]
        Lookup table: class_id → {"name": label, "code": optional}.
        Only classes present in this dict are counted; others are ignored.
    output_path : DirectoryPath
        Directory where output CSV will be written to tables/ subfolder.
    nodata : conint(ge=0)
        NoData indicator value. Defaults to 255.
    """

    image_paths: List[FilePath]
    years: List[conint(ge=0)]
    class_labels_dict: Dict[int, Dict[str, str]]
    output_path: DirectoryPath
    nodata: conint(ge=0) = 255


class TrajectoryInput(BaseModel):
    """Validation model for pixel-level trajectory classification.

    Classifies each pixel's temporal sequence into one of five trajectory types
    representing distinct land-use change dynamics (stable, reversal, etc.).
    Output is a GeoTIFF with values 1–5 (or nodata).

    Attributes
    ----------
    image_paths : List[FilePath]
        Ordered classification rasters covering the time series.
    output_path : DirectoryPath
        Root output directory; trajectory.tif written to rasters/ subfolder.
    years : List[conint(ge=0)]
        Calendar years corresponding to image_paths. Must match length.
    nodata : conint(ge=0)
        NoData value in input rasters. Defaults to 255.
    """
    image_paths: List[FilePath]
    output_path: DirectoryPath
    years: List[conint(ge=0)]
    nodata: conint(ge=0) = 255


class CCAInput(BaseModel):
    """Validation model for Change Component Analysis (CCA) decomposition.

    Orchestrates the full CCA pipeline: builds transition matrices per interval,
    computes per-class gain/loss/exchange/shift components (Pontius Jr. framework),
    and saves CSV outputs. Follows Pontius (2010, 2016) decomposition methodology.

    Attributes
    ----------
    image_paths : List[FilePath]
        Ordered classification rasters for the analysis period.
    years : List[conint(ge=0)]
        Calendar years; must correspond to image_paths.
    class_labels_dict : Dict[int, Dict[str, str]]
        Class metadata: id → {"name": label}. Used in component naming.
    output_path : DirectoryPath
        Output root; CSVs written to tables/ subfolder.
    nodata : conint(ge=0)
        NoData value. Defaults to 255.
    """
    image_paths: List[FilePath]
    years: List[conint(ge=0)]
    class_labels_dict: Dict[int, Dict[str, str]]
    output_path: DirectoryPath
    nodata: conint(ge=0) = 255


class HeatmapInput(BaseModel):
    """Validation model for heatmap visualization generation.

    Loads pre-computed transition matrices (CSV) from the output directory and
    generates publication-ready heatmaps (PNG) with logarithmic coloring and labels.

    Attributes
    ----------
    output_path : DirectoryPath
        Directory containing tables/ subfolder with transition matrix CSVs.
    years : List[conint(ge=0)]
        Temporal range [start_year, ..., end_year]. Used in matrix filename lookup.
    class_labels_dict : Dict[int, Dict[str, str]]
        Class lookup for axis labels in heatmaps.
    """
    output_path: DirectoryPath
    years: List[conint(ge=0)]
    class_labels_dict: Dict[int, Dict[str, str]]


# -----------------------------
# Utility helpers
# -----------------------------


def ensure_output_dirs(output_path: str) -> Dict[str, str]:
    """Create output directory tree and return paths.

    Ensures all required subdirectories exist with 755 permissions. Called
    internally by high-level functions but may be invoked standalone to
    pre-allocate storage.

    Parameters
    ----------
    output_path : str
        Root output directory (created if missing).

    Returns
    -------
    Dict[str, str]
        Mapping: "tables" → path/to/tables, "rasters" → path/to/rasters,
        "charts" → path/to/charts, "maps" → path/to/maps.
    """

    base = Path(output_path)
    out = {
        "tables": str((base / "tables")),
        "rasters": str((base / "rasters")),
        "charts": str((base / "charts")),
        "maps": str((base / "maps")),
    }
    for p in out.values():
        os.makedirs(p, exist_ok=True)
    return out


def load_square_matrix(csv_path: str) -> pd.DataFrame:
    """Load and normalize a transition matrix from CSV.

    Reads a class-indexed transition matrix and ensures rows and columns are
    aligned (square). If row/column labels differ, their union is used with
    NaN values filled as 0.0. Essential for CCA heatmaps and decomposition.

    Parameters
    ----------
    csv_path : str
        Path to CSV file (index_col=0) with class labels in rows and columns
        and transition counts in cells.

    Returns
    -------
    pd.DataFrame
        Square DataFrame (n_classes × n_classes) with string indices and columns.
        All row/column labels are coerced to strings for consistency.

    Raises
    ------
    ValueError
        If matrix remains non-square after alignment.
    """
    df = pd.read_csv(csv_path, index_col=0)

    df.index = df.index.map(str)
    df.columns = df.columns.map(str)

    if list(df.index) != list(df.columns):
        labels = sorted(
            set(df.index).union(df.columns),
            key=lambda x: int(x),
        )
        df = df.reindex(index=labels, columns=labels).fillna(0.0)

    if df.shape[0] != df.shape[1]:
        raise ValueError(
            f"Matrix not square after alignment: {csv_path}",
        )

    return df


# -----------------------------
# Numba-accelerated routines
# -----------------------------


@nb.njit(nogil=True)
def compute_change_frequency_numba(stack: np.ndarray, nodata: int) -> np.ndarray:
    """Numba implementation: counts pixels changing by number of total changes.

    Parameters
    ----------
    stack : np.ndarray
        3D array (time, height, width)
    nodata : int
        NoData value

    Returns
    -------
    np.ndarray
        2D counts array (n_intervals x n_intervals)
    """

    n_times = stack.shape[0]
    height = stack.shape[1]
    width = stack.shape[2]
    n_intervals = n_times - 1
    counts = np.zeros((n_intervals, n_intervals), dtype=np.int64)

    for y in range(height):
        for x in range(width):
            total_changes_for_pixel = 0
            for t in range(n_intervals):
                v_curr = stack[t, y, x]
                v_next = stack[t + 1, y, x]
                if v_curr != nodata and v_next != nodata:
                    if v_curr != v_next:
                        total_changes_for_pixel += 1

            if total_changes_for_pixel > 0:
                for t in range(n_intervals):
                    v_curr = stack[t, y, x]
                    v_next = stack[t + 1, y, x]
                    if v_curr != nodata and v_next != nodata:
                        if v_curr != v_next:
                            counts[t, total_changes_for_pixel - 1] += 1
    return counts


@nb.njit(nogil=True)
def classify_pixel(pixel_series: np.ndarray, nodata_val: int) -> np.uint8:
    """Classify a pixel's temporal trajectory into land-use change category.

    Assigns each pixel one of five trajectory types based on its land-cover
    sequence. Classification follows:

    1. **Stable** (ID=1): Start = End = Middle. No change across time.
       Land use remains constant; represents areas under persistent use.

    2. **Reversal** (ID=2): Start = End ≠ Middle. Pixel changed but returned
       to original class. Indicates cyclical or episodic disturbance (e.g.,
       harvest-regrowth, flooding-recovery).

    3. **Direct Transition** (ID=3): Start → End with single land-cover path.
       Pixel follows unique class sequence without backtracking.
       Represents coherent, linear land-use conversion.

    4. **Complex Transition** (ID=4): Start → End with multiple intermediate
       states (≥2 steps). Represents complex trajectories (e.g.,
       Forest → Agriculture → Pasture → Urban).

    5. **Non-Convergent** (ID=5): End state never reached from Start.
       Indicates complex dynamics without direct end state attainment.

    Parameters
    ----------
    pixel_series : np.ndarray
        1D array of class values over time. Length = n_timesteps.
    nodata_val : int
        NoData indicator. Any pixel with nodata_val → return nodata_val.

    Returns
    -------
    np.uint8
        Trajectory ID (1, 2, 3, 4, 5) or nodata_val if any input is nodata.
    """

    length = len(pixel_series)
    if length == 0:
        return np.uint8(nodata_val)
    for i in range(length):
        if pixel_series[i] == nodata_val:
            return np.uint8(nodata_val)

    start = pixel_series[0]
    end = pixel_series[length - 1]

    if start == end:
        is_stable = True
        for i in range(1, length):
            if pixel_series[i] != start:
                is_stable = False
                break
        if is_stable:
            return np.uint8(1)
        else:
            return np.uint8(2)
    else:
        has_direct = False
        for i in range(length - 1):
            if pixel_series[i] == start and pixel_series[i + 1] == end:
                has_direct = True
                break
        if not has_direct:
            return np.uint8(5)
        else:
            path_changes = 0
            last_val = start
            for i in range(1, length):
                current_val = pixel_series[i]
                if current_val != last_val:
                    path_changes += 1
                    last_val = current_val
            if path_changes == 1:
                return np.uint8(3)
            else:
                return np.uint8(4)


@nb.njit(nogil=True, parallel=True)
def process_stack_parallel(stack: np.ndarray, height: int, width: int, nodata_val: int) -> np.ndarray:
    """Apply pixel trajectory classification to full raster stack in parallel.

    Numba-compiled loop over all pixels with OpenMP parallelization (prange).
    Suitable for large rasters (e.g., 10,000 × 10,000 pixels).

    Parameters
    ----------
    stack : np.ndarray
        3D array (time, height, width) of class values (uint8 or int32).
    height : int
        Raster height in pixels.
    width : int
        Raster width in pixels.
    nodata_val : int
        NoData indicator.

    Returns
    -------
    np.ndarray
        2D uint8 array (height, width) with trajectory IDs (1–5) or nodata_val.
    """

    result = np.zeros((height, width), dtype=np.uint8)
    for y in prange(height):
        for x in range(width):
            pixel_series = stack[:, y, x]
            result[y, x] = classify_pixel(pixel_series, nodata_val)
    return result


@nb.njit(nogil=True, fastmath=True)
def compute_trajectory_changes_numba(trajectory_map: np.ndarray, stack: np.ndarray, nodata: int) -> np.ndarray:
    """Tally transition counts per interval, stratified by trajectory type.

    For each time interval and each trajectory class (ID 2–5), counts the number
    of transitions (class changes) occurring. Trajectory ID 1 (stable) is skipped.
    Used for detailed interval-specific change accounting.

    Parameters
    ----------
    trajectory_map : np.ndarray
        2D uint8 array (height, width) of trajectory IDs from process_stack_parallel.
    stack : np.ndarray
        3D array (time, height, width) of original class values.
    nodata : int
        NoData indicator value.

    Returns
    -------
    np.ndarray
        2D array (n_intervals × 6) where rows are time intervals,
        columns are trajectory IDs [0, 1, 2, 3, 4, 5], and cells are
        transition counts (dtype int64).
    """

    n_times = stack.shape[0]
    height = stack.shape[1]
    width = stack.shape[2]
    n_intervals = n_times - 1
    counts = np.zeros((n_intervals, 6), dtype=np.int64)
    for y in range(height):
        for x in range(width):
            traj_id = trajectory_map[y, x]
            if traj_id < 2 or traj_id > 5:
                continue
            for t in range(n_intervals):
                val_from = stack[t, y, x]
                val_to = stack[t + 1, y, x]
                if (val_from != nodata) and (val_to != nodata) and (val_from != val_to):
                    counts[t, traj_id] += 1
    return counts


# -----------------------------
# High-level modular functions
# -----------------------------


def load_raster_stack(image_paths: List[str]) -> Tuple[np.ndarray, dict]:
    """Load temporal raster sequence into 3D array with geospatial metadata.

    Reads GeoTIFF or other rasterio-compatible formats, stacks into (time, height, width),
    and preserves geotransform/CRS from the first image. Critical for all downstream
    analyses.

    Parameters
    ----------
    image_paths : List[str]
        Paths to raster files (unsorted). Will be sorted internally.
        All rasters must have identical dimensions and geospatial extent.

    Returns
    -------
    Tuple[np.ndarray, dict]
        - stack: 3D array (n_timesteps, height, width) of pixel values.
        - meta: rasterio profile dict from first image (contains CRS, transform, etc.).

    Raises
    ------
    ValueError
        If image_paths is empty.
    rasterio.errors.RasterioIOError
        If any file cannot be read.
    """

    if not image_paths:
        raise ValueError("`image_paths` is empty")
    image_paths = sorted(image_paths)
    stack_list = []
    meta = None
    for i, p in enumerate(image_paths):
        with rasterio.open(p) as src:
            arr = src.read(1)
            stack_list.append(arr)
            if i == 0:
                meta = src.meta.copy()
    stack = np.array(stack_list)
    return stack, meta


def calculate_pixel_counts(params: PixelCountInput) -> pd.DataFrame:
    """Tabulate area (in pixel counts) per class per year.

    Computes per-class coverage statistics for each temporal snapshot,
    accounting for nodata values. Output is a pivot table (Years × ClassName)
    saved as CSV and returned as DataFrame.

    Parameters
    ----------
    params : PixelCountInput
        Validated input with image_paths, years, class_labels_dict, output_path.

    Returns
    -------
    pd.DataFrame
        Pivot table (years × class_names) with pixel counts per class.
        Index is Year; columns are class names (from class_labels_dict).
        Written to {output_path}/tables/pixels_per_class_per_year.csv.

    Raises
    ------
    ValueError
        If len(image_paths) != len(years).
    """

    p = params
    ensure_output_dirs(str(p.output_path))
    if len(p.image_paths) != len(p.years):
        raise ValueError("Length of image_paths must equal length of years")

    records = []
    for year, path in zip(p.years, sorted(map(str, p.image_paths))):
        with rasterio.open(path) as src:
            data = src.read(1)
        values, counts = np.unique(data, return_counts=True)
        for v, c in zip(values, counts):
            v = int(v)
            if v == int(p.nodata):
                continue
            if v not in p.class_labels_dict:
                continue
            records.append({
                "Year": int(year),
                "ClassID": v,
                "ClassName": p.class_labels_dict[v]["name"],
                "Pixels": int(c),
            })

    df_pixels = pd.DataFrame(records)
    pivot_pixels = (
        df_pixels.pivot_table(index="Year", columns="ClassName", values="Pixels", aggfunc="sum")
        .fillna(0)
        .astype(int)
    )

    tables_dir = Path(p.output_path) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tables_dir / "pixels_per_class_per_year.csv"
    pivot_pixels.to_csv(out_csv, index_label="Year")
    return pivot_pixels


def compute_trajectories(params: TrajectoryInput) -> str:
    """Classify all pixels' temporal dynamics and save trajectory map.

    Applies classify_pixel to every pixel in the raster stack, generating
    a single-band GeoTIFF where each pixel is labeled with its trajectory type (1–5)
    or nodata. Output preserves geospatial metadata from input.

    Parameters
    ----------
    params : TrajectoryInput
        Validated input with image_paths, years, output_path, nodata.

    Returns
    -------
    str
        Path to output GeoTIFF: {output_path}/rasters/trajectory.tif

    Notes
    -----
    Output is uint8 (0–255). Missing/nodata pixels retain nodata_val.
    """

    p = params
    dirs = ensure_output_dirs(str(p.output_path))
    image_paths = sorted(map(str, p.image_paths))
    if not image_paths:
        raise ValueError("No image paths provided")

    # Read first image meta
    with rasterio.open(image_paths[0]) as src:
        meta = src.meta.copy()
        height = src.height
        width = src.width

    # Load stack
    stack_list = []
    for path in image_paths:
        with rasterio.open(path) as src:
            stack_list.append(src.read(1))
    stack = np.array(stack_list)

    traj_map = process_stack_parallel(stack, height, width, int(p.nodata))

    # Update meta and write
    meta.update(dtype=rasterio.uint8, count=1, nodata=int(p.nodata), compress="lzw")
    out_path = Path(dirs["rasters"]) / "trajectory.tif"
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(traj_map, 1)
    return str(out_path)


def compute_cca_components(params: CCAInput) -> pd.DataFrame:
    """Decompose land-cover change into Quantity, Exchange, and Shift components.

    Full Change Component Analysis (CCA) pipeline per Pontius Jr. (2010, 2016).
    Builds transition matrices for each interval, computes per-interval matrices
    and aggregated matrices over the full period, then decomposes total gain/loss
    into:

    - **Quantity**: Net gain/loss per class (asymmetry in transitions).
    - **Exchange**: Simultaneous bidirectional transitions (min flux in both directions).
    - **Shift**: Unidirectional change after exchange removal.

    All matrices saved as CSVs; component summary returned as DataFrame.

    Parameters
    ----------
    params : CCAInput
        Validated input with image_paths, years, class_labels_dict, output_path.

    Returns
    -------
    pd.DataFrame
        Per-class components table (columns: Time_Interval, Class, Component,
        Gain, Loss). Saved to {output_path}/tables/change_components.csv.
        Each class has three rows: Allocation_Quantity, Allocation_Exchange,
        Allocation_Shift.

    Notes
    -----
    Intermediate matrices saved as transition_matrix_*.csv files.
    Follows Pontius Jr. decomposition exactly; see class docstring.
    """

    p = params
    dirs = ensure_output_dirs(str(p.output_path))
    image_paths = sorted(map(str, p.image_paths))
    if not image_paths:
        raise ValueError("`image_paths` cannot be empty")

    # Read all rasters and masks
    all_data = []
    all_masks = []
    for path in image_paths:
        with rasterio.open(path) as src:
            data = src.read(1)
            mask = (data == p.nodata) | (data == src.nodata) | np.isnan(data.astype(float))
            all_masks.append(mask)
            all_data.append(data)
    combined_mask = np.any(np.array(all_masks), axis=0)

    flattened = [
        data[~combined_mask].flatten() if np.any(combined_mask) else data.flatten()
        for data in all_data
    ]

    all_classes = np.unique(np.concatenate(flattened)).astype(int)
    n = len(all_classes)
    class_to_idx = {cls: i for i, cls in enumerate(all_classes)}

    # interval matrices
    interval_matrices = {f"{p.years[t]}-{p.years[t+1]}": np.zeros((n, n)) for t in range(len(p.years) - 1)}

    num_pixels = len(flattened[0])
    for r in range(num_pixels):
        traj = [int(flattened[t][r]) for t in range(len(flattened))]
        for t in range(len(traj) - 1):
            s = class_to_idx[traj[t]]
            e = class_to_idx[traj[t + 1]]
            interval_matrices[f"{p.years[t]}-{p.years[t+1]}"][s, e] += 1

    tables_dir = Path(dirs["tables"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    for interval_name, mat in interval_matrices.items():
        fname = f"transition_matrix_{interval_name}.csv"
        out_csv = tables_dir / fname
        pd.DataFrame(mat, index=all_classes, columns=all_classes).to_csv(out_csv)

    # Aggregated matrices
    mat_sum = np.zeros((n, n))
    mat_ext = np.zeros((n, n))
    mat_x = np.zeros((n, n))
    mat_s = np.zeros((n, n))
    mat_u = np.zeros((n, n))

    for r in range(num_pixels):
        traj = [int(flattened[t][r]) for t in range(len(flattened))]
        start_val = traj[0]
        end_val = traj[-1]
        start_idx = class_to_idx[start_val]
        end_idx = class_to_idx[end_val]

        m_r = np.zeros((n, n))
        for t in range(len(traj) - 1):
            s_t = class_to_idx[traj[t]]
            e_t = class_to_idx[traj[t + 1]]
            m_r[s_t, e_t] += 1

        e_r = np.zeros((n, n))
        e_r[start_idx, end_idx] = 1
        diff_m_e = m_r - e_r
        x_r = np.maximum(0, np.minimum(diff_m_e, diff_m_e.T))
        s_r = np.maximum(0, m_r - x_r - e_r)
        u_r = e_r + x_r + s_r - m_r

        mat_sum += m_r
        mat_ext += e_r
        mat_x += x_r
        mat_s += s_r
        mat_u += u_r

    mat_c = np.minimum(mat_ext, mat_ext.T)
    mat_q = mat_ext - mat_c

    aggregated_matrices = {
        "sum": mat_sum,
        "extent": mat_ext,
        "allocation_exchange": mat_c,
        "quantity_allocation_shift": mat_q,
        "alternation_exchange": mat_x,
        "alternation_shift": mat_s,
        "unaccounted_extent": mat_u,
    }
    interval_str = f"{p.years[0]}-{p.years[-1]}"
    for name, mat in aggregated_matrices.items():
        file_name = f"transition_matrix_{name}_{interval_str}.csv"
        output_csv_path = tables_dir / file_name
        pd.DataFrame(mat, index=all_classes, columns=all_classes).to_csv(output_csv_path)

    # Compute components per class using simplified ComponentCalculator logic
    class_components = []
    for idx, class_id in enumerate(all_classes):
        gain_sum = mat_sum[:, idx].sum()
        loss_sum = mat_sum[idx, :].sum()
        q_gain = max(0.0, gain_sum - loss_sum)
        q_loss = max(0.0, loss_sum - gain_sum)
        mutual = np.sum(np.minimum(mat_sum[idx, :], mat_sum[:, idx]))
        exchange = mutual - mat_sum[idx, idx]
        total_trans = loss_sum - mat_sum[idx, idx]
        shift = total_trans - q_loss - exchange
        cls_name = p.class_labels_dict.get(int(class_id), {}).get("name", f"Class {class_id}")
        class_components.append({
            "Time_Interval": interval_str,
            "Class": cls_name,
            "Component": "Allocation_Quantity",
            "Gain": q_gain,
            "Loss": q_loss,
        })
        class_components.append({
            "Time_Interval": interval_str,
            "Class": cls_name,
            "Component": "Allocation_Exchange",
            "Gain": exchange,
            "Loss": exchange,
        })
        class_components.append({
            "Time_Interval": interval_str,
            "Class": cls_name,
            "Component": "Allocation_Shift",
            "Gain": shift,
            "Loss": shift,
        })

    df_components = pd.DataFrame(class_components)
    df_components.to_csv(tables_dir / "change_components.csv", index=False)
    return df_components


def generate_heatmaps(params: HeatmapInput) -> Dict[str, str]:
    """Render transition matrix heatmaps for publication or reporting.

    Loads pre-computed matrices (from compute_cca_components) and generates
    publication-ready PNG heatmaps (300 dpi) with colorbars and labeled axes.
    Suitable for academic papers, reports, or dashboards.

    Parameters
    ----------
    params : HeatmapInput
        Validated input with output_path, years, class_labels_dict.
        Expects transition matrix CSVs in {output_path}/tables/

    Returns
    -------
    Dict[str, str]
        Mapping of matrix type ("sum", "alt_exc", etc.) to PNG path.
        Saved to {output_path}/charts/heatmaps/*.png
        Returns empty dict if no matrices found.
    """

    p = params
    dirs = ensure_output_dirs(str(p.output_path))
    tables_dir = Path(dirs["tables"])

    # metadata for matrices (same keys as notebook)
    MATRIX_META = {
        "sum": ["sum", "Time Intervals", "flow"],
        "alt_exc": ["alternation_exchange", "Alternation Exchange", "flow"],
        "alt_shift": ["alternation_shift", "Alternation Shift", "flow"],
        "ext": ["extent", "Extent", "stock"],
        "all_exc": ["allocation_exchange", "Allocation Exchange", "stock"],
        "qty_shift": ["quantity_allocation_shift", "Quantity & Allocation Shift", "stock"],
        "unacc_ext": ["unaccounted_extent", "Unaccounted Extent", "stock"],
    }

    str_y0 = re.search(r"(\d+)", str(p.years[0])).group(1) if p.years else "0"
    str_y1 = re.search(r"(\d+)", str(p.years[-1])).group(1) if p.years else "0"
    interval_str = f"{str_y0}-{str_y1}"

    saved = {}
    for key, meta in MATRIX_META.items():
        suffix = meta[0]
        csv_path = tables_dir / f"transition_matrix_{suffix}_{interval_str}.csv"
        if not csv_path.exists():
            continue
        df = load_square_matrix(str(csv_path))
        # simple heatmap: use matplotlib (imported at module top)
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(df.values.astype(float), cmap="YlOrRd")
        ax.set_xticks(range(len(df.columns)))
        ax.set_yticks(range(len(df.index)))
        ax.set_xticklabels([str(x) for x in df.columns], rotation=90, fontsize=6)
        ax.set_yticklabels([str(x) for x in df.index], fontsize=6)
        fig.colorbar(im, ax=ax)
        charts_dir = Path(dirs["charts"]) / "heatmaps"
        charts_dir.mkdir(parents=True, exist_ok=True)
        out_fig = charts_dir / f"heatmap_{suffix}_{interval_str}.png"
        fig.savefig(out_fig, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved[key] = str(out_fig)

    return saved


__all__ = [
    # Pydantic input models (contracts for AI agent)
    "StackInput",
    "PixelCountInput",
    "TrajectoryInput",
    "CCAInput",
    "HeatmapInput",
    # High-level public API
    "load_raster_stack",
    "calculate_pixel_counts",
    "compute_trajectories",
    "compute_cca_components",
    "generate_heatmaps",
    # Utilities (for advanced users)
    "ensure_output_dirs",
    "load_square_matrix",
]
