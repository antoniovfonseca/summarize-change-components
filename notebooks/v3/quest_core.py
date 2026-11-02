"""
QUEST core engine: Quantity, Exchange and Shift Through Time.

Shared computation used by the QUEST notebooks. A single pass over the
raster blocks reads each block from every year once and computes the
trajectory, number of changes, interval transition matrices, change
component matrices, class counts and the extent, alternation exchange and
alternation shift rasters. The functions below then build the tables, the
change components and the scale of the maps from those results.

Every output uses the same common valid mask: a location enters the
analysis only if it is valid and belongs to a listed class in every year.
"""
import itertools
import os
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from typing import Dict, List, Optional, Tuple

import numba as nb
import numpy as np
import pandas as pd
import rasterio
from pyproj import Geod, Transformer
from tqdm.auto import tqdm





# ---------------------------------------------------------------------------
# Shared raster processing infrastructure
# ---------------------------------------------------------------------------


# Shared raster processing infrastructure
_thread_local_stack = threading.local()


def _get_thread_datasets(
    image_paths: Tuple[str, ...],
) -> Tuple[rasterio.io.DatasetReader, ...]:
    """
    Return open rasterio datasets for the given paths, cached per thread.

    Rasterio datasets are not safe to share across threads, so each
    worker thread keeps its own open handles and reuses them across
    every window/chunk, instead of opening and closing the file once
    per window.

    Parameters
    ----------
    image_paths : tuple of str
        Paths to the input rasters, in chronological order.

    Returns
    -------
    tuple of rasterio.io.DatasetReader
        One open dataset per input path, ordered as `image_paths`.
    """
    # 1. Check whether this thread already has the right datasets open
    cache = getattr(_thread_local_stack, "datasets", None)

    if cache is None or cache["key"] != image_paths:

        # 2. Close any datasets left over from a previous set of paths
        if cache is not None:
            for src in cache["srcs"]:
                src.close()

        # 3. Open every raster once and store it for reuse
        cache = {
            "key": image_paths,
            "srcs": tuple(
                rasterio.open(p) for p in image_paths
            ),
        }
        _thread_local_stack.datasets = cache

    return cache["srcs"]


def _resolve_windows(
    image_paths: List[str],
) -> List[rasterio.windows.Window]:
    """
    Resolve the spatial chunks used to process the raster time series.

    Uses the native block layout of the first raster when it is tiled.
    When the raster is striped, or its blocks are not multiples of 16,
    uses 256 x 256 windows instead, which match the output tiles written
    by `compute_all_quest_outputs`. Computed once and shared by every
    function that processes the same time series.

    Parameters
    ----------
    image_paths : list of str
        Paths to the input rasters, in chronological order.

    Returns
    -------
    list of rasterio.windows.Window
        Windows covering the full raster extent.
    """
    # 1. Read the first raster to determine block layout
    with rasterio.open(image_paths[0]) as src:
        block_y, block_x = src.block_shapes[0]

        # 2. Use the native blocks when the input is tiled; otherwise use
        # 256 x 256 windows, which match the output tiles
        if block_x == src.width or block_x % 16 or block_y % 16:
            windows = [
                rasterio.windows.Window(
                    col,
                    row,
                    min(256, src.width - col),
                    min(256, src.height - row),
                )
                for row in range(0, src.height, 256)
                for col in range(0, src.width, 256)
            ]
        else:
            windows = [window for ij, window in src.block_windows()]

    return windows




# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------


# Trajectory classification kernels
@nb.njit(
    nogil=True,
    cache=True,
)
def classify_pixel(
    pixel_series: np.ndarray,
    nodata_val: int,
) -> np.uint8:
    """
    Classify a single pixel trajectory into five categories based on MatricesMain11 logic.

    Parameters
    ----------
    pixel_series : np.ndarray
        A 1D numpy array representing the land cover class values
        of a single pixel over time.
    nodata_val : int
        The integer value representing no data.

    Returns
    -------
    np.uint8
        An integer representing the trajectory class:
        1: All stable (Gray)
        2: Extent stable with alternation (Red)
        3: Extent change without alternation (Yellow)
        4: Extent change with alternation - Direct transition exists (Orange)
        5: Extent change - Unaccounted/No direct transition (Dark Blue)
        0: NoData
    """
    # 1. Get the length of the time series
    length = len(
        pixel_series,
    )

    # 2. Check for empty or NoData series
    if length == 0:
        return np.uint8(
            nodata_val,
        )

    for i in range(
        length,
    ):
        if pixel_series[i] == nodata_val:
            return np.uint8(
                nodata_val,
            )

    # 3. Identify Start and End classes
    start = pixel_series[
        0
    ]
    end = pixel_series[
        length - 1
    ]

    # 4. Logic for Stable Extent (Start == End)
    if start == end:
        is_completely_stable = True

        # Check if any intermediate value differs from start
        for i in range(
            1,
            length,
        ):
            if pixel_series[i] != start:
                is_completely_stable = False
                break

        if is_completely_stable:
            # Trajectory 1: All stable
            return np.uint8(
                1,
            )
        else:
            # Trajectory 2: Extent stable with alternation
            return np.uint8(
                2,
            )

    # 5. Logic for Extent Change (Start != End)
    else:
        # Check if the direct transition (Start -> End) exists at any step
        has_direct_transition = False

        for i in range(
            length - 1,
        ):
            if pixel_series[i] == start and pixel_series[i + 1] == end:
                has_direct_transition = True
                break

        if not has_direct_transition:
            # Trajectory 5: Extent change with alternation shift (Unaccounted)
            # The transition Start->End never happens directly.
            return np.uint8(
                5,
            )

        else:
            # Direct transition exists. Now distinguish T3 vs T4.
            # We calculate the "path length" (ignoring consecutive duplicates)

            path_changes = 0
            last_val = start

            for i in range(
                1,
                length,
            ):
                current_val = pixel_series[
                    i
                ]
                if current_val != last_val:
                    path_changes += 1
                    last_val = current_val

            # If there is exactly 1 change (Start -> ... -> End), it is T3.
            if path_changes == 1:
                # Trajectory 3: Extent change without alternation
                return np.uint8(
                    3,
                )
            else:
                # Trajectory 4: Extent change with alternation (Direct exists)
                return np.uint8(
                    4,
                )


@nb.njit(
    nogil=True,
    cache=True,
)
def compute_trajectory_chunk(
    stack: np.ndarray,
    nodata_val: int,
) -> np.ndarray:
    """
    Apply pixel classification to a 3D raster stack.

    Runs on a single thread per block, because the thread pool in
    `compute_all_quest_outputs` already processes blocks in parallel.

    Parameters
    ----------
    stack : np.ndarray
        A 3D numpy array of shape (time, height, width) containing
        the stacked raster data.
    nodata_val : int
        The integer value representing no data.

    Returns
    -------
    np.ndarray
        A 2D numpy array of shape (height, width) containing the
        classified trajectory codes (uint8).
    """
    height = stack.shape[
        1
    ]
    width = stack.shape[
        2
    ]

    # 1. Initialize result array with zeros
    result = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint8,
    )

    # 2. Iterate over rows
    for y in range(
        height,
    ):
        for x in range(
            width,
        ):
            # 3. Extract time series for pixel (y, x)
            pixel_series = stack[
                :,
                y,
                x,
            ]

            # 4. Classify the pixel
            result[
                y,
                x,
            ] = classify_pixel(
                pixel_series,
                nodata_val,
            )

    return result


# Number of changes kernel
@nb.njit(
    nogil=True,
    cache=True,
)
def compute_number_of_changes_chunk(
    stack: np.ndarray,
    nodata: int,
) -> np.ndarray:
    """
    Numba-optimized function to calculate the total number of changes per pixel.

    Parameters
    ----------
    stack : np.ndarray
        3D numpy array (time, height, width) for a specific spatial chunk.
    nodata : int
        Value representing NoData pixels.

    Returns
    -------
    np.ndarray
        2D numpy array (height, width) with the total number of changes.
    """
    time_steps = stack.shape[
        0
    ]
    h = stack.shape[
        1
    ]
    w = stack.shape[
        2
    ]

    out_changes = np.full(
        (
            h,
            w,
        ),
        nodata,
        dtype=np.uint8,
    )

    for y in range(
        h,
    ):
        for x in range(
            w,
        ):
            traj = stack[
                :,
                y,
                x,
            ]

            # Check for NoData in any time step
            has_nodata = False
            for t in range(
                time_steps,
            ):
                if traj[
                    t
                ] == nodata:
                    has_nodata = True
                    break

            if has_nodata:
                continue

            # Calculate total changes
            changes = 0
            for t in range(
                time_steps - 1,
            ):
                if traj[
                    t
                ] != traj[
                    t + 1
                ]:
                    changes += 1

            out_changes[
                y,
                x,
            ] = changes

    return out_changes


# Change counts matrix kernel
@nb.njit(
    nogil=True,
    cache=True,
)
def compute_change_counts_matrix(
    stack: np.ndarray,
    nodata: int,
) -> np.ndarray:
    """
    Counts pixels changing in each interval, categorized by their total
    number of changes across the entire time series.

    Parameters
    ----------
    stack : np.ndarray
        3D array (time, height, width) with pixel values.
    nodata : int
        Value representing NoData. Pixels with NoData in any year are
        skipped.

    Returns
    -------
    np.ndarray
        2D array of shape (n_intervals, n_intervals).
        Rows represent time intervals, columns represent the total changes.
    """
    n_times = stack.shape[0]
    height = stack.shape[1]
    width = stack.shape[2]
    n_intervals = n_times - 1

    # 1. Initialize the counts array
    # Rows: Time Intervals (0 to T-1)
    # Cols: Total life changes (0 = 1 total change, 1 = 2 total changes, etc.)
    counts = np.zeros(
        (
            n_intervals,
            n_intervals,
        ),
        dtype=np.int64,
    )

    # 2. Iterate over every pixel in the raster
    for y in range(
        height,
    ):
        for x in range(
            width,
        ):

            # 3. Skip the pixel if it is NoData in any year, matching the
            # other kernels
            has_nodata = False
            for t in range(
                n_times,
            ):
                if stack[
                    t, y, x,
                ] == nodata:
                    has_nodata = True
                    break

            if has_nodata:
                continue

            # 4. Discover the total number of changes in the pixel's lifetime
            total_changes_for_pixel = 0

            for t in range(
                n_intervals,
            ):
                if stack[
                    t, y, x,
                ] != stack[
                    t + 1, y, x,
                ]:
                    total_changes_for_pixel += 1

            # 5. Register the pixel in the intervals where it changed.
            # Column index is total changes minus 1, so index 0 holds the
            # pixels with 1 total change.
            if total_changes_for_pixel > 0:
                for t in range(
                    n_intervals,
                ):
                    if stack[
                        t, y, x,
                    ] != stack[
                        t + 1, y, x,
                    ]:
                        counts[
                            t,
                            total_changes_for_pixel - 1,
                        ] += 1

    return counts


# Interval transition matrices kernel
@nb.njit(
    nogil=True,
    cache=True,
)
def compute_interval_matrices_chunk(
    stack: np.ndarray,
    nodata: int,
    class_to_idx: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Calculate interval transition matrices for a chunk of raster data.

    Parameters
    ----------
    stack : np.ndarray
        3D array of shape (time, height, width) with pixel values.
    nodata : int
        Value representing NoData.
    class_to_idx : np.ndarray
        1D array mapping class values to matrix indices.
    num_classes : int
        Total number of valid classes.

    Returns
    -------
    np.ndarray
        3D array of shape (n_intervals, num_classes, num_classes)
        containing the accumulated transition counts.
    """
    # 1. Extract dimensions
    n_times = stack.shape[0]
    h = stack.shape[1]
    w = stack.shape[2]
    n_intervals = n_times - 1

    # 2. Initialize output matrices
    out_matrices = np.zeros(
        (
            n_intervals,
            num_classes,
            num_classes,
        ),
        dtype=np.int64,
    )

    # 3. Iterate over spatial dimensions
    for y in range(h):
        for x in range(w):
            # 4. Check for NoData across the pixel's trajectory
            has_nodata = False
            for t in range(n_times):
                if stack[t, y, x] == nodata:
                    has_nodata = True
                    break

            if has_nodata:
                continue

            # 5. Accumulate transitions for valid pixels
            for t in range(n_intervals):
                val_s = stack[t, y, x]
                val_e = stack[t + 1, y, x]

                idx_s = class_to_idx[val_s]
                idx_e = class_to_idx[val_e]

                if idx_s >= 0 and idx_e >= 0:
                    out_matrices[
                        t,
                        idx_s,
                        idx_e,
                    ] += 1

    return out_matrices


# Change component matrices kernel
@nb.njit(
    nogil=True,
    cache=True,
)
def compute_component_matrices_chunk(
    stack: np.ndarray,
    nodata: int,
    class_to_idx: np.ndarray,
    num_classes: int,
    trajectory_chunk: np.ndarray,
) -> tuple:
    """
    Calculate change component matrices for a chunk of raster data.

    The per-pixel transition matrix has at most one nonzero cell per time
    interval, so the kernel visits only those cells and the extent cell.
    Every other cell has zero transitions and zero extent, so its exchange,
    shift and unaccounted values are also zero. A single scratch matrix is
    allocated per chunk and reset after each pixel.

    Parameters
    ----------
    stack : np.ndarray
        3D array of shape (time, height, width) with pixel values.
    nodata : int
        Value representing NoData.
    class_to_idx : np.ndarray
        1D array mapping class values to matrix indices.
    num_classes : int
        Total number of valid classes.
    trajectory_chunk : np.ndarray
        2D array of shape (height, width) with trajectory codes (1-5)
        for each pixel, as classified by `compute_trajectory_chunk`.

    Returns
    -------
    tuple
        A tuple containing 6 np.ndarray matrices:
        - Sum of all transitions
        - Extent matrix
        - Alternation exchange matrix
        - Alternation shift matrix
        - Unaccounted extent matrix
        - Sum of all transitions, split by trajectory, shape
          (5, num_classes, num_classes), where index 0 is Trajectory 1
          and index 4 is Trajectory 5
    """
    # 1. Extract dimensions
    n_times = stack.shape[0]
    h = stack.shape[1]
    w = stack.shape[2]

    # 2. Initialize chunk accumulation matrices
    m_sum = np.zeros(
        (
            num_classes,
            num_classes,
        ),
        dtype=np.int64,
    )
    m_ext = np.zeros_like(m_sum)
    m_x = np.zeros_like(m_sum)
    m_s = np.zeros_like(m_sum)
    m_u = np.zeros_like(m_sum)
    m_by_trajectory = np.zeros(
        (
            5,
            num_classes,
            num_classes,
        ),
        dtype=np.int64,
    )

    # 3. Allocate the per-pixel scratch matrix and the list of its
    # nonzero cells once per chunk
    m_r = np.zeros_like(m_sum)
    cell_rows = np.empty(
        n_times,
        dtype=np.int64,
    )
    cell_cols = np.empty(
        n_times,
        dtype=np.int64,
    )

    # 4. Iterate over spatial dimensions
    for y in range(h):
        for x in range(w):
            # 5. Check for NoData across the pixel's trajectory
            has_nodata = False
            for t in range(n_times):
                if stack[t, y, x] == nodata:
                    has_nodata = True
                    break

            if has_nodata:
                continue

            start_idx = class_to_idx[stack[0, y, x]]
            end_idx = class_to_idx[stack[n_times - 1, y, x]]

            if start_idx < 0 or end_idx < 0:
                continue

            traj = trajectory_chunk[y, x]

            # 6. Fill the transition matrix for pixel r, keeping the list
            # of cells it touches
            n_cells = 0
            for t in range(n_times - 1):
                s_t = class_to_idx[stack[t, y, x]]
                e_t = class_to_idx[stack[t + 1, y, x]]
                if s_t >= 0 and e_t >= 0:
                    if m_r[s_t, e_t] == 0:
                        cell_rows[n_cells] = s_t
                        cell_cols[n_cells] = e_t
                        n_cells += 1
                    m_r[s_t, e_t] += 1
                    m_sum[s_t, e_t] += 1
                    if traj >= 1 and traj <= 5:
                        m_by_trajectory[traj - 1, s_t, e_t] += 1

            # 7. Extent matrix for pixel r has one cell, from start to end
            m_ext[start_idx, end_idx] += 1
            if m_r[start_idx, end_idx] == 0:
                cell_rows[n_cells] = start_idx
                cell_cols[n_cells] = end_idx
                n_cells += 1

            # 8. Exchange, shift and unaccounted extent on the touched cells
            for k in range(n_cells):
                i = cell_rows[k]
                j = cell_cols[k]
                e_ij = 1 if (i == start_idx and j == end_idx) else 0
                e_ji = 1 if (j == start_idx and i == end_idx) else 0
                m_ij = m_r[i, j]

                x_ij = min(
                    m_ij - e_ij,
                    m_r[j, i] - e_ji,
                )
                if x_ij < 0:
                    x_ij = 0

                s_ij = m_ij - x_ij - e_ij
                if s_ij < 0:
                    s_ij = 0

                m_x[i, j] += x_ij
                m_s[i, j] += s_ij
                m_u[i, j] += e_ij + x_ij + s_ij - m_ij

            # 9. Reset the scratch matrix for the next pixel
            for k in range(n_cells):
                m_r[cell_rows[k], cell_cols[k]] = 0

    return m_sum, m_ext, m_x, m_s, m_u, m_by_trajectory


# Extent, exchange and shift components kernel
@nb.njit(
    nogil=True,
    cache=True,
)
def compute_components_chunk(
    stack: np.ndarray,
    nodata: int,
) -> tuple:
    """
    Numba-optimized function to calculate Extent, Exchange, and Shift components.

    The per-pixel transition matrix is indexed by the raw pixel value and
    has at most one nonzero cell per time interval, so the kernel visits
    only those cells. A single 256 x 256 scratch matrix is allocated per
    chunk and reset after each pixel.

    Parameters
    ----------
    stack : np.ndarray
        3D numpy array (time, height, width) for a specific spatial chunk.
    nodata : int
        Value representing NoData pixels.

    Returns
    -------
    tuple
        Three 2D numpy arrays (Extent, Exchange, Shift).
    """
    time_steps = stack.shape[0]
    h = stack.shape[1]
    w = stack.shape[2]

    out_extent = np.full(
        (
            h,
            w,
        ),
        nodata,
        dtype=np.uint8,
    )
    out_alt_exc = np.full(
        (
            h,
            w,
        ),
        nodata,
        dtype=np.uint8,
    )
    out_alt_shift = np.full(
        (
            h,
            w,
        ),
        nodata,
        dtype=np.uint8,
    )

    # Scratch transition matrix and list of its nonzero cells, allocated
    # once per chunk
    m_r = np.zeros(
        (
            256,
            256,
        ),
        dtype=np.int32,
    )
    cell_rows = np.empty(
        time_steps,
        dtype=np.int64,
    )
    cell_cols = np.empty(
        time_steps,
        dtype=np.int64,
    )

    for y in range(h):
        for x in range(w):

            # 1. Check for NoData
            has_nodata = False
            for t in range(time_steps):
                if stack[t, y, x] == nodata:
                    has_nodata = True
                    break

            if has_nodata:
                continue

            # 2. Extent Component
            start_val = stack[0, y, x]
            end_val = stack[time_steps - 1, y, x]
            extent_val = 1 if start_val != end_val else 0

            out_extent[y, x] = extent_val

            # 3. Fill the transition matrix with the changes of pixel r
            n_cells = 0
            total_changes = 0
            for t in range(time_steps - 1):
                s = stack[t, y, x]
                e = stack[t + 1, y, x]
                if s != e:
                    if m_r[s, e] == 0:
                        cell_rows[n_cells] = s
                        cell_cols[n_cells] = e
                        n_cells += 1
                    m_r[s, e] += 1
                    total_changes += 1

            # 4. Alternation Exchange Component
            # The direct transition from the start class to the end class
            # belongs to Extent. It is set aside before pairing opposite
            # transitions, matching compute_component_matrices_chunk.
            # Cells with no transitions contribute zero, so only the
            # touched cells are visited.
            total_exchange = 0
            for k in range(n_cells):
                i = cell_rows[k]
                j = cell_cols[k]
                val1 = m_r[i, j]
                val2 = m_r[j, i]
                if extent_val == 1:
                    if i == start_val and j == end_val:
                        val1 -= 1
                    elif i == end_val and j == start_val:
                        val2 -= 1
                pair_min = val1 if val1 < val2 else val2
                if pair_min > 0:
                    total_exchange += pair_min

            out_alt_exc[y, x] = total_exchange

            # 5. Alternation Shift Component
            total_shift = total_changes - extent_val - total_exchange

            out_alt_shift[y, x] = total_shift if total_shift > 0 else 0

            # 6. Reset the scratch matrix for the next pixel
            for k in range(n_cells):
                m_r[cell_rows[k], cell_cols[k]] = 0

    return out_extent, out_alt_exc, out_alt_shift


# Trajectory interval change counts kernel
@nb.njit(
    nogil=True,
    fastmath=True,
    cache=True,
)
def compute_trajectory_changes(
    trajectory_map: np.ndarray,
    stack: np.ndarray,
    nodata: int,
) -> np.ndarray:
    """
    Count pixels per trajectory type changing in each time interval using Numba.

    Parameters
    ----------
    trajectory_map : np.ndarray
        2D array (height, width) with trajectory IDs (2, 3, 4, 5).
    stack : np.ndarray
        3D array (time, height, width) with pixel values.
    nodata : int
        Value to ignore.

    Returns
    -------
    np.ndarray
        2D array of shape (n_intervals, 6) containing change counts.
        Rows correspond to time intervals.
        Columns correspond to trajectory IDs (0 to 5).
    """
    n_times = stack.shape[0]
    height = stack.shape[1]
    width = stack.shape[2]
    n_intervals = n_times - 1

    # Array to store counts: rows=intervals, cols=trajectory_ids(0-5)
    # We use 6 columns to map the trajectory ID directly as an index
    counts = np.zeros(
        (
            n_intervals,
            6,
        ),
        dtype=np.int64,
    )

    # Iterate over every pixel
    for y in range(
        height,
    ):
        for x in range(
            width,
        ):
            traj_id = trajectory_map[
                y,
                x,
            ]

            # We only care about Alternation(2), Step(3), Complex(4), Unaccounted(5)
            if traj_id < 2 or traj_id > 5:
                continue

            # Check changes across all time intervals
            for t in range(
                n_intervals,
            ):
                val_from = stack[
                    t,
                    y,
                    x,
                ]
                val_to = stack[
                    t + 1,
                    y,
                    x,
                ]

                # Check validity
                if (
                    val_from != nodata
                    and val_to != nodata
                    and val_from != val_to
                ):
                    counts[
                        t,
                        traj_id,
                    ] += 1

    return counts


# Class counts in the common mask kernel
@nb.njit(
    nogil=True,
    cache=True,
)
def compute_class_counts_chunk(
    stack: np.ndarray,
    nodata: int,
    class_to_idx: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Count pixels per class and time point inside the common valid mask.

    The common valid mask keeps only the pixels that are valid and belong
    to a listed class at every time point, so every time point has the
    same total number of pixels.

    Parameters
    ----------
    stack : np.ndarray
        3D array of shape (time, height, width) with pixel values.
    nodata : int
        Value representing NoData.
    class_to_idx : np.ndarray
        1D array mapping class values to matrix indices, with -1 for
        values that are not listed classes.
    num_classes : int
        Total number of valid classes.

    Returns
    -------
    np.ndarray
        2D array of shape (time, num_classes) with pixel counts.
    """
    n_times = stack.shape[0]
    h = stack.shape[1]
    w = stack.shape[2]

    counts = np.zeros(
        (
            n_times,
            num_classes,
        ),
        dtype=np.int64,
    )

    for y in range(h):
        for x in range(w):

            # 1. Keep the pixel only if it is a listed class at every time
            is_valid = True
            for t in range(n_times):
                value = stack[t, y, x]
                if value == nodata or class_to_idx[value] < 0:
                    is_valid = False
                    break

            if not is_valid:
                continue

            # 2. Count the class of the pixel at every time point
            for t in range(n_times):
                counts[t, class_to_idx[stack[t, y, x]]] += 1

    return counts




# ---------------------------------------------------------------------------
# Single-pass worker and orchestrator
# ---------------------------------------------------------------------------


# Worker: read block once, compute all quest outputs
def _read_and_compute_quest_block(
    image_paths: Tuple[str, ...],
    window: rasterio.windows.Window,
    nodata_val: int,
    class_to_idx: np.ndarray,
    num_classes: int,
) -> Dict[str, np.ndarray]:
    """
    Read one spatial block from every input raster ONCE and compute
    every quest output that depends on it.

    Reused downstream for: number-of-changes analysis, trajectory
    analysis, interval transition matrices, component matrices, and
    the extent / alternation raster maps — instead of re-reading the
    same block once per output.

    Every output uses the same common valid mask: a location enters the
    analysis only if it is valid and belongs to a listed class in every
    year. Locations outside the mask are set to NoData in every year
    before any kernel runs, so the rasters, tables, and percentages all
    describe the same set of locations.

    Parameters
    ----------
    image_paths : tuple of str
        Paths to the input rasters, in chronological order.
    window : rasterio.windows.Window
        Spatial block to read from every raster.
    nodata_val : int
        Pixel value representing NoData.
    class_to_idx : np.ndarray
        1D array mapping class values to transition-matrix indices.
    num_classes : int
        Total number of valid classes.

    Returns
    -------
    dict of str to np.ndarray
        Keys: "window", "trajectory_chunk", "changes_chunk",
        "change_counts_matrix", "interval_matrices",
        "extent_chunk", "alt_exchange_chunk", "alt_shift_chunk",
        "component_matrices" (tuple of 6 matrices, the last one being
        the sum of transitions split by trajectory), "trajectory_interval_counts",
        "class_counts", "trajectory_counts".
    """
    # 1. Read the block from every raster, using the cached datasets
    srcs = _get_thread_datasets(image_paths)
    stack = np.array(
        [src.read(1, window=window) for src in srcs],
        dtype=np.uint8,
    )

    # 1.1 Apply the common valid mask: set to NoData, in every year, the
    #     locations that are NoData or an unlisted class in any year
    outside_mask = ~np.all(
        class_to_idx[stack] >= 0,
        axis=0,
    ) | np.any(
        stack == nodata_val,
        axis=0,
    )
    stack[:, outside_mask] = nodata_val

    # 2. Classify pixel trajectories
    trajectory_chunk = compute_trajectory_chunk(
        stack,
        int(nodata_val),
    )

    # 3. Compute total number of changes per pixel
    changes_chunk = compute_number_of_changes_chunk(
        stack,
        nodata_val,
    )

    # 4. Compute the interval x total-changes count matrix
    change_counts_matrix = compute_change_counts_matrix(
        stack=stack,
        nodata=int(nodata_val),
    )

    # 5. Compute interval transition matrices
    interval_matrices = compute_interval_matrices_chunk(
        stack=stack,
        nodata=nodata_val,
        class_to_idx=class_to_idx,
        num_classes=num_classes,
    )

    # 6. Compute overall change component matrices, including the sum
    #    of transitions split by trajectory, using the trajectory
    #    classification from step 2
    component_matrices = compute_component_matrices_chunk(
        stack=stack,
        nodata=nodata_val,
        class_to_idx=class_to_idx,
        num_classes=num_classes,
        trajectory_chunk=trajectory_chunk,
    )

    # 7. Compute extent / alternation exchange / alternation shift
    #    raster components
    extent_chunk, alt_exchange_chunk, alt_shift_chunk = compute_components_chunk(
        stack,
        nodata_val,
    )

    # 8. Compute per-interval change counts by trajectory type, reusing
    #    the trajectory classification and the stack already in memory
    trajectory_interval_counts = compute_trajectory_changes(
        trajectory_map=trajectory_chunk,
        stack=stack,
        nodata=nodata_val,
    )

    # 9. Count pixels per class and time point inside the common valid
    #    mask, for the class size table
    class_counts = compute_class_counts_chunk(
        stack=stack,
        nodata=nodata_val,
        class_to_idx=class_to_idx,
        num_classes=num_classes,
    )

    # 10. Count pixels per trajectory code, for the trajectory tables
    trajectory_counts = np.bincount(
        trajectory_chunk.ravel(),
        minlength=256,
    )

    return {
        "window": window,
        "trajectory_chunk": trajectory_chunk,
        "changes_chunk": changes_chunk,
        "change_counts_matrix": change_counts_matrix,
        "interval_matrices": interval_matrices,
        "component_matrices": component_matrices,
        "extent_chunk": extent_chunk,
        "alt_exchange_chunk": alt_exchange_chunk,
        "alt_shift_chunk": alt_shift_chunk,
        "trajectory_interval_counts": trajectory_interval_counts,
        "class_counts": class_counts,
        "trajectory_counts": trajectory_counts,
    }


# Orchestrator: single-pass computation of all quest outputs
def compute_all_quest_outputs(
    image_paths: List[str],
    output_dir: str,
    years: List[int],
    class_dict: Dict[int, Dict[str, str]],
    nodata_val: int,
    max_workers: Optional[int] = None,
    compress: str = "deflate",
) -> Dict[str, object]:
    """
    Compute every quest output in a single pass over the raster blocks.

    Reads each spatial block from the input time series exactly once
    and derives from it: number-of-changes analysis, trajectory
    classification, interval transition matrices, overall component
    matrices, the sum of transitions split by trajectory, and the
    extent / alternation raster maps, and the pixels per class and year
    inside the common valid mask, and the pixels per trajectory code.
    Replaces ten separate full
    read-throughs of the same input rasters.

    Parameters
    ----------
    image_paths : list of str
        Paths to the input rasters, in chronological order.
    output_dir : str
        Directory where rasters and tables subfolders are created.
    years : list of int
        Years corresponding to each input raster.
    class_dict : dict
        Mapping from class ID to metadata with "name" and "color".
    nodata_val : int
        Pixel value representing NoData.
    max_workers : int, optional
        Number of threads used to read/process blocks in parallel. If
        None, uses the number of CPU cores of the machine.
    compress : str, optional
        Compression algorithm used for the output rasters.

    Returns
    -------
    dict of str to object
        Every downstream artifact: dataframes, matrices, and output
        raster paths, keyed by name.
    """
    # 1. Validate that every year has one raster. The rasters are used
    # in the order given, which must match the order of `years`.
    if len(image_paths) != len(years):
        raise ValueError(
            f"Input mismatch: {len(image_paths)} images "
            f"vs {len(years)} years.",
        )

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    # Prepare class index mapping shared by the transition matrices
    class_ids = sorted(class_dict.keys())
    num_classes = len(class_ids)
    max_id = max(class_ids) if class_ids else 255

    # Cover every uint8 value, so a pixel value above the largest class
    # ID maps to -1 instead of reading outside the array in Numba
    class_to_idx_arr = np.full(
        max(256, max_id + 1),
        -1,
        dtype=np.int64,
    )
    for idx, cid in enumerate(class_ids):
        class_to_idx_arr[cid] = idx

    # 2. Resolve output directories
    rasters_dir = os.path.join(output_dir, "rasters")
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(rasters_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    out_raster_paths = {
        "trajectory": os.path.join(rasters_dir, "trajectory.tif"),
        "number_of_changes": os.path.join(rasters_dir, "number_of_changes.tif"),
        "extent": os.path.join(rasters_dir, "map_extent.tif"),
        "exchange": os.path.join(rasters_dir, "map_alternation_exchange.tif"),
        "shift": os.path.join(rasters_dir, "map_alternation_shift.tif"),
    }

    # 3. Read the output raster profile and resolve processing windows
    image_paths_t = tuple(image_paths)

    with rasterio.open(image_paths_t[0]) as src:
        block_y, block_x = src.block_shapes[0]

        # Match the output tiles to the input blocks, so every window
        # fills whole output tiles. Fall back to 256 x 256 when the
        # input is striped or its blocks are not multiples of 16.
        if block_x == src.width or block_x % 16 or block_y % 16:
            block_x, block_y = 256, 256

        profile = src.profile.copy()
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=nodata_val,
            compress=compress,
            tiled=True,
            blockxsize=block_x,
            blockysize=block_y,
            BIGTIFF="YES",
        )
        if compress == "deflate":
            profile.update(
                zlevel=6,
            )

    windows = _resolve_windows(list(image_paths_t))

    # 4. Initialize accumulators for every CSV-based output
    n_intervals = len(years) - 1

    raw_change_counts = np.zeros((n_intervals, n_intervals), dtype=np.int64)
    raw_trajectory_intervals = np.zeros((n_intervals, 6), dtype=np.int64)
    global_interval_matrices = np.zeros(
        (n_intervals, num_classes, num_classes),
        dtype=np.int64,
    )
    mat_sum = np.zeros((num_classes, num_classes), dtype=np.int64)
    mat_ext = np.zeros((num_classes, num_classes), dtype=np.int64)
    mat_x = np.zeros((num_classes, num_classes), dtype=np.int64)
    mat_s = np.zeros((num_classes, num_classes), dtype=np.int64)
    mat_u = np.zeros((num_classes, num_classes), dtype=np.int64)
    mat_by_trajectory = np.zeros((5, num_classes, num_classes), dtype=np.int64)
    class_counts = np.zeros((len(years), num_classes), dtype=np.int64)
    trajectory_counts = np.zeros(256, dtype=np.int64)

    print(
        f"Computing all quest outputs across {len(windows)} chunks, "
        f"reading each block once...",
    )

    # 5. Open every output raster writer and process all blocks
    with ExitStack() as stack_ctx:
        dst_trajectory = stack_ctx.enter_context(
            rasterio.open(out_raster_paths["trajectory"], "w", **profile)
        )
        dst_changes = stack_ctx.enter_context(
            rasterio.open(out_raster_paths["number_of_changes"], "w", **profile)
        )
        dst_extent = stack_ctx.enter_context(
            rasterio.open(out_raster_paths["extent"], "w", **profile)
        )
        dst_exchange = stack_ctx.enter_context(
            rasterio.open(out_raster_paths["exchange"], "w", **profile)
        )
        dst_shift = stack_ctx.enter_context(
            rasterio.open(out_raster_paths["shift"], "w", **profile)
        )

        # Keep at most `max_in_flight` blocks in memory at any time.
        # Each finished future is dropped right after its result is
        # written, so memory stays flat regardless of the number of
        # blocks in the study area.
        max_in_flight = max_workers * 2
        window_iter = iter(windows)
        pending = set()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            for window in itertools.islice(
                window_iter,
                max_in_flight,
            ):
                pending.add(
                    executor.submit(
                        _read_and_compute_quest_block,
                        image_paths_t,
                        window,
                        nodata_val,
                        class_to_idx_arr,
                        num_classes,
                    )
                )

            progress = tqdm(
                total=len(windows),
                desc="Processing chunks",
            )

            while pending:
                done, pending = wait(
                    pending,
                    return_when=FIRST_COMPLETED,
                )

                for future in done:
                    result = future.result()
                    window = result["window"]

                    # 6. Write every raster output for this block
                    dst_trajectory.write(
                        result["trajectory_chunk"],
                        1,
                        window=window,
                    )
                    dst_changes.write(
                        result["changes_chunk"],
                        1,
                        window=window,
                    )
                    dst_extent.write(
                        result["extent_chunk"],
                        1,
                        window=window,
                    )
                    dst_exchange.write(
                        result["alt_exchange_chunk"],
                        1,
                        window=window,
                    )
                    dst_shift.write(
                        result["alt_shift_chunk"],
                        1,
                        window=window,
                    )

                    # 7. Accumulate every table-based output
                    raw_change_counts += result["change_counts_matrix"]
                    raw_trajectory_intervals += result["trajectory_interval_counts"]
                    global_interval_matrices += result["interval_matrices"]

                    (
                        chunk_sum,
                        chunk_ext,
                        chunk_x,
                        chunk_s,
                        chunk_u,
                        chunk_by_trajectory,
                    ) = result["component_matrices"]
                    mat_sum += chunk_sum
                    mat_ext += chunk_ext
                    mat_x += chunk_x
                    mat_s += chunk_s
                    mat_u += chunk_u
                    mat_by_trajectory += chunk_by_trajectory
                    class_counts += result["class_counts"]
                    trajectory_counts += result["trajectory_counts"]

                    del result
                    progress.update(1)

                    # Refill the queue with the next block, if any
                    next_window = next(
                        window_iter,
                        None,
                    )
                    if next_window is not None:
                        pending.add(
                            executor.submit(
                                _read_and_compute_quest_block,
                                image_paths_t,
                                next_window,
                                nodata_val,
                                class_to_idx_arr,
                                num_classes,
                            )
                        )

            progress.close()

        # 8. Build overviews for every output raster, so downstream
        #    plotting functions can request a decimated read without
        #    GDAL needing to decode the full-resolution file. Without
        #    this, a display-sized read on a large uncompressed raster
        #    can still force a full decode internally and exhaust
        #    available memory.
        overview_factors = [2, 4, 8, 16, 32]
        for dst in (
            dst_trajectory,
            dst_changes,
            dst_extent,
            dst_exchange,
            dst_shift,
        ):
            dst.build_overviews(
                overview_factors,
                rasterio.enums.Resampling.nearest,
            )
            dst.update_tags(ns="rio_overview", resampling="nearest")

    print("\nAll raster maps generated successfully.")

    return {
        "class_ids": class_ids,
        "out_raster_paths": out_raster_paths,
        "raw_change_counts": raw_change_counts,
        "raw_trajectory_intervals": raw_trajectory_intervals,
        "global_interval_matrices": global_interval_matrices,
        "mat_sum": mat_sum,
        "mat_ext": mat_ext,
        "mat_x": mat_x,
        "mat_s": mat_s,
        "mat_u": mat_u,
        "mat_by_trajectory": mat_by_trajectory,
        "class_counts": class_counts,
        "trajectory_counts": trajectory_counts,
    }




# ---------------------------------------------------------------------------
# Map scale
# ---------------------------------------------------------------------------


# Compute display pixel size
def compute_display_pixel_size_km(
    raster_path: str,
    downsample_divisor: int,
) -> float:
    """
    Compute the ground distance, in kilometers, of one displayed pixel.

    The distance is the geodesic length of one pixel along the horizontal
    axis at the center of the map, measured on the WGS84 ellipsoid. This
    works for any coordinate reference system, projected or geographic,
    because the pixel corners are converted to longitude and latitude
    before the distance is measured. The result is multiplied by the
    downsample divisor, since each displayed pixel covers that many
    native pixels.

    In a geographic coordinate reference system, and in some projections,
    the ground size of a pixel changes across the map. In that case the
    scale bar is exact at the center of the map, and the function prints
    the range of the horizontal pixel size between the top and bottom rows
    when that range exceeds 2%.

    Parameters
    ----------
    raster_path : str
        Path to a raster file used to derive the grid and CRS.
    downsample_divisor : int
        Integer factor used to downsample the raster for display.

    Returns
    -------
    float
        Ground distance in kilometers of one displayed pixel at the center
        of the map.
    """
    # 1. Read the grid of the raster
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(
                f"{raster_path} has no coordinate reference system, so the "
                f"scale bar cannot be computed.",
            )

        left, bottom, right, top = src.bounds
        res_x, res_y = src.res
        width = src.width
        crs = src.crs

    to_ll = Transformer.from_crs(
        crs,
        "EPSG:4326",
        always_xy=True,
    )
    geod = Geod(
        ellps="WGS84",
    )

    # 2. Measure a short horizontal span of pixels centered on the map, so
    #    the result is the local pixel size and not the chord of the whole
    #    map width
    span_px = min(
        width,
        100,
    )
    x_center = (left + right) / 2.0
    half_span = span_px * res_x / 2.0

    def _km_per_pixel(
        y: float,
    ) -> float:
        lon_a, lat_a = to_ll.transform(
            x_center - half_span,
            y,
        )
        lon_b, lat_b = to_ll.transform(
            x_center + half_span,
            y,
        )
        _, _, dist_m = geod.inv(
            lon_a,
            lat_a,
            lon_b,
            lat_b,
        )
        return dist_m / span_px / 1_000

    km_center = _km_per_pixel(
        (top + bottom) / 2.0,
    )
    km_top = _km_per_pixel(
        top - res_y / 2.0,
    )
    km_bottom = _km_per_pixel(
        bottom + res_y / 2.0,
    )

    # 3. Report how much the pixel size changes between the top and bottom
    #    rows, when the change exceeds 2%
    km_min = min(
        km_top,
        km_bottom,
        km_center,
    )
    km_max = max(
        km_top,
        km_bottom,
        km_center,
    )
    variation_pct = (km_max - km_min) / km_center * 100

    if variation_pct > 2:
        print(
            f"Scale bar is exact at the center of the map. The horizontal "
            f"pixel size ranges from {km_min * 1_000:.2f} m to "
            f"{km_max * 1_000:.2f} m between the top and bottom rows, a "
            f"{variation_pct:.1f}% range.",
        )

    return km_center * downsample_divisor


def format_scale_bar_label(
    value: float,
    unit: str,
) -> str:
    """
    Format the scale bar label in meters or kilometers.

    The scale bar picks the unit prefix from the bar length, so a long bar
    can come as megameters. This keeps the label in kilometers instead.

    Parameters
    ----------
    value : float
        Bar length in `unit`, as chosen by the scale bar.
    unit : str
        Unit chosen by the scale bar, such as "m", "km" or "Mm".

    Returns
    -------
    str
        Label such as "500 m", "5 km" or "1000 km".
    """
    if unit == "Mm":
        return f"{value * 1_000:g} km"

    return f"{value:g} {unit}"




# ---------------------------------------------------------------------------
# Change component calculator
# ---------------------------------------------------------------------------


# Component calculator class
class ComponentCalculator:
    """
    Compute change components for a matrix.

    Supports pre-decomposed matrices via force_component parameter.
    """

    def __init__(
        self,
        transition_matrix: np.ndarray,
    ) -> None:
        """
        Initialize the calculator with a transition matrix.

        Parameters
        ----------
        transition_matrix : np.ndarray
            Square matrix representing transitions.
        """
        self.matrix = transition_matrix.astype(
            float,
        )
        self.num_classes = transition_matrix.shape[
            0
        ]
        self.class_components: list[
            dict
        ] = []

    def calculate_components(
        self,
        force_component: str = None,
    ) -> "ComponentCalculator":
        """
        Calculate gain, loss, exchange, and shift for all classes.

        Parameters
        ----------
        force_component : str, optional
            If set to "Exchange" or "Shift", forces the interpretation of the
            matrix content to that specific component.

        Returns
        -------
        ComponentCalculator
            Returns self for chaining.
        """
        for class_idx in range(
            self.num_classes,
        ):
            gain_sum = np.sum(
                self.matrix[
                    :,
                    class_idx,
                ],
            )
            loss_sum = np.sum(
                self.matrix[
                    class_idx,
                    :,
                ],
            )

            # Standard net change calculation
            q_gain = max(
                0.0,
                gain_sum - loss_sum,
            )
            q_loss = max(
                0.0,
                loss_sum - gain_sum,
            )

            if force_component == "Exchange":
                # Matrix content is treated purely as exchange
                exchange = loss_sum - self.matrix[
                    class_idx,
                    class_idx,
                ]
                shift = 0.0
                q_gain, q_loss = (
                    gain_sum - loss_sum
                ), (
                    loss_sum - gain_sum
                )
            elif force_component == "Shift":
                # Matrix content is treated purely as shift
                exchange = 0.0
                shift = loss_sum - self.matrix[
                    class_idx,
                    class_idx,
                ]
                q_gain, q_loss = (
                    0.0,
                    0.0,
                )
            else:
                # Standard Pontius decomposition
                mutual = np.sum(
                    np.minimum(
                        self.matrix[
                            class_idx,
                            :,
                        ],
                        self.matrix[
                            :,
                            class_idx,
                        ],
                    ),
                )
                exchange = mutual - self.matrix[
                    class_idx,
                    class_idx,
                ]
                total_trans = loss_sum - self.matrix[
                    class_idx,
                    class_idx,
                ]
                shift = total_trans - q_loss - exchange

            self.class_components.append(
                {
                    "Quantity_Gain": q_gain,
                    "Quantity_Loss": q_loss,
                    "Exchange_Gain": exchange,
                    "Exchange_Loss": exchange,
                    "Shift_Gain": shift,
                    "Shift_Loss": shift,
                },
            )
        return self


# ---------------------------------------------------------------------------
# Tables built from the single-pass results
# ---------------------------------------------------------------------------
def build_quest_tables(
    quest_results: Dict[str, object],
    years: List[int],
) -> Dict[str, object]:
    """
    Build every QUEST table from the results of `compute_all_quest_outputs`.

    Parameters
    ----------
    quest_results : dict
        Output of `compute_all_quest_outputs`.
    years : list of int
        Years of the time series, in the same order as the input rasters.

    Returns
    -------
    dict
        "interval_labels" : list of str, such as "1985-1986".
        "number_change_per_interval" : pd.DataFrame with intervals as rows
            and the number of changes of the pixel as columns.
        "trajectory_contributions_per_interval" : pd.DataFrame with
            intervals as rows and trajectories 2 to 5 as columns.
        "interval_matrices" : dict of interval label to pd.DataFrame.
        "aggregated_matrices" : dict of matrix name to pd.DataFrame, with
            the names time_intervals, extent, allocation_exchange,
            quantity_allocation_shift, alternation_exchange,
            alternation_shift, indirect_extent and indirect_ratio.
        "trajectory_matrices" : dict of trajectory code 1 to 5 to
            pd.DataFrame.
        "interval_str" : str, such as "1985-2021".
    """
    class_ids = quest_results["class_ids"]
    interval_labels = [
        f"{years[i]}-{years[i + 1]}"
        for i in range(len(years) - 1)
    ]

    # 1. Number of changes per interval, with one column per number of
    #    changes of the pixel, up to the largest number observed
    raw_counts = quest_results["raw_change_counts"]
    max_changes_observed = 0
    if np.any(raw_counts > 0):
        col_sums = raw_counts.sum(axis=0)
        if np.any(col_sums > 0):
            max_changes_observed = np.max(np.nonzero(col_sums))

    relevant_counts = raw_counts[:, :max_changes_observed + 1]
    df_change_freq = pd.DataFrame(
        relevant_counts,
        index=interval_labels,
        columns=[
            str(i + 1)
            for i in range(relevant_counts.shape[1])
        ],
    )

    # 2. Trajectory contributions per interval, for trajectories 2 to 5
    df_trajectory_intervals = pd.DataFrame(
        quest_results["raw_trajectory_intervals"][:, 2:6],
        index=interval_labels,
        columns=[2, 3, 4, 5],
    )
    df_trajectory_intervals.index.name = "Interval"
    df_trajectory_intervals.columns.name = "Trajectory_Class"

    # 3. Interval transition matrices
    interval_matrices = {
        interval_name: pd.DataFrame(
            quest_results["global_interval_matrices"][t],
            index=class_ids,
            columns=class_ids,
        )
        for t, interval_name in enumerate(interval_labels)
    }

    # 4. Aggregated matrices. Allocation exchange and quantity allocation
    #    shift come from the extent matrix; the indirect ratio is the
    #    indirect extent as a percentage of the extent.
    mat_ext = quest_results["mat_ext"]
    mat_u = quest_results["mat_u"]
    mat_c = np.minimum(
        mat_ext,
        mat_ext.T,
    )
    mat_q = mat_ext - mat_c

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        mat_ratio = np.where(
            mat_ext != 0,
            (mat_u / mat_ext) * 100.0,
            np.nan,
        )

    aggregated_arrays = {
        "time_intervals": quest_results["mat_sum"],
        "extent": mat_ext,
        "allocation_exchange": mat_c,
        "quantity_allocation_shift": mat_q,
        "alternation_exchange": quest_results["mat_x"],
        "alternation_shift": quest_results["mat_s"],
        "indirect_extent": mat_u,
        "indirect_ratio": mat_ratio,
    }
    aggregated_matrices = {
        name: pd.DataFrame(
            mat,
            index=class_ids,
            columns=class_ids,
        )
        for name, mat in aggregated_arrays.items()
    }

    # 5. Transition matrices split by trajectory, which sum exactly to the
    #    time intervals matrix
    mat_by_trajectory = quest_results["mat_by_trajectory"]
    trajectory_matrices = {
        traj: pd.DataFrame(
            mat_by_trajectory[traj - 1],
            index=class_ids,
            columns=class_ids,
        )
        for traj in range(1, 6)
    }

    if not np.array_equal(
        mat_by_trajectory.sum(axis=0),
        quest_results["mat_sum"],
    ):
        print(
            "WARNING: The 5 trajectory matrices do not sum to the Time "
            "Intervals matrix. Check the trajectory accounting.",
        )

    return {
        "interval_labels": interval_labels,
        "number_change_per_interval": df_change_freq,
        "trajectory_contributions_per_interval": df_trajectory_intervals,
        "interval_matrices": interval_matrices,
        "aggregated_matrices": aggregated_matrices,
        "trajectory_matrices": trajectory_matrices,
        "interval_str": f"{years[0]}-{years[-1]}",
    }


def save_quest_tables(
    tables: Dict[str, object],
    output_dir: str,
) -> str:
    """
    Save the tables from `build_quest_tables` as CSV files.

    Parameters
    ----------
    tables : dict
        Output of `build_quest_tables`.
    output_dir : str
        Directory where the "tables" subfolder is created.

    Returns
    -------
    str
        Path of the tables folder.
    """
    tables_dir = os.path.join(
        output_dir,
        "tables",
    )
    os.makedirs(
        tables_dir,
        exist_ok=True,
    )
    interval_str = tables["interval_str"]

    tables["number_change_per_interval"].to_csv(
        os.path.join(
            tables_dir,
            "number_change_per_interval.csv",
        ),
    )
    tables["trajectory_contributions_per_interval"].to_csv(
        os.path.join(
            tables_dir,
            "trajectory_contributions_per_interval.csv",
        ),
    )

    for interval_name, df_mat in tables["interval_matrices"].items():
        df_mat.to_csv(
            os.path.join(
                tables_dir,
                f"transition_matrix_{interval_name}.csv",
            ),
        )

    for name, df_mat in tables["aggregated_matrices"].items():
        df_mat.to_csv(
            os.path.join(
                tables_dir,
                f"transition_matrix_{name}_{interval_str}.csv",
            ),
        )

    for traj, df_mat in tables["trajectory_matrices"].items():
        df_mat.to_csv(
            os.path.join(
                tables_dir,
                f"transition_matrix_trajectory_{traj}_{interval_str}.csv",
            ),
        )

    return tables_dir


def build_class_area_table(
    class_counts: np.ndarray,
    class_ids: List[int],
    years: List[int],
    class_labels_dict: Dict[int, Dict[str, str]],
) -> pd.DataFrame:
    """
    Build the table of pixels per class and year inside the common valid
    mask.

    The common valid mask keeps only the locations that are valid and
    belong to a listed class in all years, so every year has the same
    total number of pixels.

    Parameters
    ----------
    class_counts : np.ndarray
        2D array of shape (years, classes), as returned in
        `quest_results["class_counts"]`.
    class_ids : list of int
        Class IDs that match the columns of `class_counts`.
    years : list of int
        Years that match the rows of `class_counts`.
    class_labels_dict : dict
        Mapping from class ID to metadata with "name" and "color".

    Returns
    -------
    pd.DataFrame
        Pivot table with years as rows and class names as columns. Classes
        with no pixels in any year are left out.
    """
    # 1. Validate that the counts match the years and classes
    if class_counts.shape != (len(years), len(class_ids)):
        raise ValueError(
            f"Input mismatch: counts with shape {class_counts.shape} "
            f"vs {len(years)} years and {len(class_ids)} classes.",
        )

    # 2. Map the counts to records, skipping classes with no pixels
    records = []
    for year, year_counts in zip(
        years,
        class_counts,
    ):
        for class_id, count in zip(
            class_ids,
            year_counts,
        ):
            if count == 0:
                continue

            records.append(
                {
                    "Year": year,
                    "ClassID": class_id,
                    "ClassName": class_labels_dict[class_id]["name"],
                    "Pixels": int(count),
                },
            )

    df_pixels = pd.DataFrame(
        records,
    )

    if df_pixels.empty:
        return pd.DataFrame()

    # 3. Pivot to years as rows and class names as columns
    return (
        df_pixels.pivot_table(
            index="Year",
            columns="ClassName",
            values="Pixels",
            aggfunc="sum",
        )
        .fillna(
            0,
        )
        .astype(
            int,
        )
    )


# ---------------------------------------------------------------------------
# Change components
# ---------------------------------------------------------------------------
def _matrix_component_rows(
    df_mat: pd.DataFrame,
    matrix_type: str,
    label_time: str,
    class_labels_dict: Dict[int, Dict[str, str]],
    indirect_by_idx: Optional[np.ndarray] = None,
) -> List[dict]:
    """
    Decompose one matrix into the change component rows of each class.

    Parameters
    ----------
    df_mat : pd.DataFrame
        Square matrix with class IDs as index and columns.
    matrix_type : str
        "interval", "extent", "alternation_exchange" or
        "alternation_shift".
    label_time : str
        Value written in the Time_Interval column.
    class_labels_dict : dict
        Mapping from class ID to metadata with "name".
    indirect_by_idx : np.ndarray, optional
        Off-diagonal row totals of the indirect extent matrix, subtracted
        from Shift when `matrix_type` is "alternation_shift".

    Returns
    -------
    list of dict
        Rows with Time_Interval, Class, Component, Gain and Loss.
    """
    force_comp = (
        "Exchange"
        if "exchange" in matrix_type
        else (
            "Shift"
            if "shift" in matrix_type
            else None
        )
    )

    calc = ComponentCalculator(
        df_mat.values,
    ).calculate_components(
        force_component=force_comp,
    )

    if indirect_by_idx is None:
        indirect_by_idx = np.zeros(
            len(df_mat.index),
        )

    rows = []
    for idx, class_id in enumerate(
        [int(c) for c in df_mat.index],
    ):
        cls_name = class_labels_dict.get(
            class_id,
            {},
        ).get(
            "name",
            f"Class {class_id}",
        )
        comp_vals = calc.class_components[idx]

        for comp_name in [
            "Quantity",
            "Exchange",
            "Shift",
        ]:
            label_comp = comp_name
            if matrix_type in [
                "extent",
                "sum",
            ]:
                label_comp = f"Allocation_{comp_name}"
            if "alternation" in matrix_type:
                label_comp = f"Alternation_{comp_name}"

            gain = comp_vals[f"{comp_name}_Gain"]
            loss = comp_vals[f"{comp_name}_Loss"]

            if comp_name == "Shift":
                gain -= indirect_by_idx[idx]
                loss -= indirect_by_idx[idx]

            rows.append(
                {
                    "Time_Interval": label_time,
                    "Class": cls_name,
                    "Component": label_comp,
                    "Gain": gain,
                    "Loss": loss,
                },
            )

    return rows


def compute_change_components_table(
    tables: Dict[str, object],
    class_labels_dict: Dict[int, Dict[str, str]],
) -> pd.DataFrame:
    """
    Compute the change components of every class for every interval and
    for the extent, alternation exchange and alternation shift matrices.

    Alternation Shift excludes the Indirect Extent. For each class, the
    off-diagonal row total of the indirect extent matrix is subtracted.
    After the subtraction, the row and column totals of each class are
    equal, so the gain equals the loss.

    Parameters
    ----------
    tables : dict
        Output of `build_quest_tables`.
    class_labels_dict : dict
        Mapping from class ID to metadata with "name".

    Returns
    -------
    pd.DataFrame
        Columns Time_Interval, Class, Component, Gain and Loss, in the
        same layout as change_components.csv.
    """
    rows = []

    # 1. Interval matrices
    for interval_name, df_mat in tables["interval_matrices"].items():
        rows.extend(
            _matrix_component_rows(
                df_mat=df_mat,
                matrix_type="interval",
                label_time=interval_name,
                class_labels_dict=class_labels_dict,
            ),
        )

    # 2. Aggregated matrices. The indirect extent is subtracted from the
    #    alternation shift of the class where it starts.
    aggregated = tables["aggregated_matrices"]
    indirect_vals = aggregated["indirect_extent"].values.astype(
        float,
    )
    np.fill_diagonal(
        indirect_vals,
        0,
    )
    indirect_by_idx = indirect_vals.sum(
        axis=1,
    )

    for matrix_type in [
        "extent",
        "alternation_exchange",
        "alternation_shift",
    ]:
        rows.extend(
            _matrix_component_rows(
                df_mat=aggregated[matrix_type],
                matrix_type=matrix_type,
                label_time=matrix_type,
                class_labels_dict=class_labels_dict,
                indirect_by_idx=(
                    indirect_by_idx
                    if matrix_type == "alternation_shift"
                    else None
                ),
            ),
        )

    return pd.DataFrame(
        rows,
    )


def summarize_change_components(
    df_components: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sum the change components over all classes.

    Parameters
    ----------
    df_components : pd.DataFrame
        Output of `compute_change_components_table`.

    Returns
    -------
    pd.DataFrame
        Columns Component and Pixels, for Quantity, Allocation_Exchange,
        Allocation_Shift, Alternation_Exchange and Alternation_Shift.
    """
    df = df_components
    component_totals = {
        "Quantity": df[
            (df["Component"] == "Allocation_Quantity")
            & (df["Time_Interval"] == "extent")
        ]["Gain"].sum(),
        "Allocation_Exchange": df[
            (df["Component"] == "Allocation_Exchange")
            & (df["Time_Interval"] == "extent")
        ]["Gain"].sum(),
        "Allocation_Shift": df[
            (df["Component"] == "Allocation_Shift")
            & (df["Time_Interval"] == "extent")
        ]["Gain"].sum(),
        "Alternation_Exchange": df[
            df["Time_Interval"] == "alternation_exchange"
        ]["Gain"].sum(),
        "Alternation_Shift": df[
            df["Time_Interval"] == "alternation_shift"
        ]["Gain"].sum(),
    }

    return pd.DataFrame(
        list(component_totals.items()),
        columns=[
            "Component",
            "Pixels",
        ],
    )
