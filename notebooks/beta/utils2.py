# utils.py

import os
import re
import math
import glob
import pickle
import time
import numpy as np
import pandas as pd
import rasterio
import xlsxwriter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from pathlib import Path
from tqdm import tqdm
from pyproj import Geod, Transformer
from rasterio.enums import Resampling
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_map_utils import north_arrow

# =============================================================================
# FILE AND DIRECTORY MANAGEMENT
# =============================================================================

def setup_directories(output_path):
    """
    Checks if a directory exists and creates it if necessary.

    Args:
        output_path (str): The path of the directory to be created.

    Returns:
        None
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Directory created: {output_path}")
    else:
        print(f"Directory already exists: {output_path}")

def get_ordered_annual_rasters(directory):
    """
    Finds annual .tif files in a directory, validates for duplicates, 
    and returns them sorted by year.

    Args:
        directory (str): Path to the folder containing the .tif files.

    Returns:
        list: Sorted list of file paths based on the year extracted from the filename.
    """
    # Regex to find exactly 4 digits followed by .tif (e.g., 1985.tif)
    pattern = re.compile(r"(\d{4})\.tif$")
    files = glob.glob(os.path.join(directory, "*.tif"))
    
    year_map = {}

    for f in files:
        match = pattern.search(os.path.basename(f))
        if match:
            year = int(match.group(1))
            if year in year_map:
                print(f"Warning: Duplicate year {year} found! Skipping: {f}")
            else:
                year_map[year] = f

    if not year_map:
        print(f"Error: No files matching pattern XXXX.tif found in: {directory}")
        return []

    # Sort years and create the list of paths in the correct sequence
    sorted_years = sorted(year_map.keys())
    ordered_files = [year_map[year] for year in sorted_years]

    print(f"Success: {len(ordered_files)} files found and sorted chronologically.")
    return ordered_files

# =============================================================================
# RASTER DATA PROCESSING
# =============================================================================
def get_raster_profile(file_path):
    """
    Retrieves the profile (metadata) of a GeoTIFF file.

    Args:
        file_path (str): Path to the .tif file.

    Returns:
        dict: The raster profile containing CRS, transform, and dimensions.
    """
    with rasterio.open(file_path) as src:
        return src.profile

# =============================================================================
# RASTER VISUALIZATION SUPPORT
# =============================================================================
def compute_display_pixel_size_km(raster_path, downsample_divisor):
    """
    Computes horizontal resolution in kilometers per displayed pixel.

    Args:
        raster_path (str): Path to the raster file.
        downsample_divisor (int): Factor used to downsample the raster.

    Returns:
        float: Pixel size in kilometers for the display grid.
    """
    with rasterio.open(raster_path) as src:
        left, bottom, right, top = src.bounds
        lat_mid_src = (top + bottom) / 2.0

        # Transform to WGS84 to calculate real-world distance
        to_ll = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        lon_l, lat_mid = to_ll.transform(left, lat_mid_src)
        lon_r, _ = to_ll.transform(right, lat_mid_src)

        geod = Geod(ellps="WGS84")
        _, _, width_m = geod.inv(lon_l, lat_mid, lon_r, lat_mid)

        cols_disp = max(1, src.width // downsample_divisor)

        return (width_m / cols_disp) / 1000

def plot_classified_images(
    class_map,
    years,
    output_path,
    image_paths_arg=None,
    downsample_divisor=1,
    panel_size=(4.0, 6.0),
    dx_km=None,
    resampling_method=Resampling.bilinear,
):
    """
    Plot classified rasters over time with legend, north arrow, and scale bar.

    Args:
        class_map (dict): 
            Mapping from class ID to metadata (name and color).
        years (list): 
            List of years corresponding to each classified raster.
        output_path (str): 
            Directory where the figure will be saved.
        image_paths_arg (list, optional): 
            List of specific raster paths. Defaults to None.
        downsample_divisor (int, optional): 
            Factor to reduce image size for display. Defaults to 1.
        panel_size (tuple, optional): 
            Width and height for each subplot. Defaults to (4.0, 6.0).
        dx_km (float, optional): 
            Pixel size in km. If None, computed from first raster. 
            Defaults to None.
        resampling_method (Resampling, optional): 
            Rasterio resampling method. Defaults to bilinear.

    Returns:
        None: 
            Saves a PNG figure to disk and displays the plot.
    """

    # Resolve input raster paths
    if (
        image_paths_arg and 
        isinstance(image_paths_arg, (list, tuple)) and 
        image_paths_arg
    ):
        image_paths = sorted(list(image_paths_arg))
    else:
        input_dir = os.path.join(output_path, "input_rasters")
        image_paths = sorted(
            glob.glob(os.path.join(input_dir, "*.tif"))
        )

    # Configure subplot grid
    n_images = len(image_paths)
    max_cols = 10
    ncols = min(max_cols, n_images)
    nrows = math.ceil(n_images / max_cols)

    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(
            panel_size[0] * ncols, 
            panel_size[1] * nrows
        ),
        sharey=True,
        constrained_layout=False,
    )

    plt.subplots_adjust(
        left=0.02, 
        right=0.85, 
        top=0.95, 
        bottom=0.05, 
        wspace=0.04, 
        hspace=0.04
    )

    if isinstance(axs, np.ndarray):
        axes = axs.ravel()
    else:
        axes = [axs]

    # Build colormap and normalization for class IDs
    class_ids_sorted = sorted(class_map.keys())
    
    cmap = ListedColormap(
        [class_map[k]["color"] for k in class_ids_sorted]
    )
    
    norm = BoundaryNorm(
        class_ids_sorted + [class_ids_sorted[-1] + 1], 
        cmap.N
    )

    # Derive km per displayed pixel when not provided
    if dx_km is None:
        dx_km = compute_display_pixel_size_km(
            raster_path=image_paths[0],
            downsample_divisor=downsample_divisor,
        )

    # Plot each classified raster
    for i, (path, year) in enumerate(zip(image_paths, years)):
        ax = axes[i]
        with rasterio.open(path) as src:
            h = max(1, src.height // downsample_divisor)
            w = max(1, src.width // downsample_divisor)
            data = src.read(
                1, 
                out_shape=(h, w), 
                resampling=resampling_method
            )

        ax.imshow(data, cmap=cmap, norm=norm)
        ax.set_title(f"{year}", fontweight="bold", fontsize=24)
        ax.axis("off")

    # Disable unused axes
    for j in range(n_images, len(axes)):
        axes[j].axis("off")

    # Create legend
    legend_elements = [
        Rectangle(
            (0, 0), 1, 1, 
            color=class_map[k]["color"], 
            label=class_map[k]["name"]
        )
        for k in sorted(class_map.keys())
    ]

    last_ax = axes[n_images - 1]
    
    last_ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=24,
    )

    # Add scale bar and north arrow
    scalebar = ScaleBar(
        dx=dx_km, 
        units="km", 
        length_fraction=0.35, 
        location="lower right",
        scale_loc="bottom", 
        color="black", 
        box_alpha=0
    )
    
    axes[n_images - 1].add_artist(scalebar)
    
    north_arrow(
        axes[n_images - 1], 
        location="upper left"
    )

    # Save figure
    out_fig = os.path.join(output_path, "map_panel_input_maps.png")
    
    plt.savefig(
        out_fig, 
        format="png", 
        bbox_inches="tight", 
        dpi=300
    )
    plt.show()
    plt.close()

