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
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle

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
    Finds annual .tif files, extracts the year from filenames (even with prefixes),
    validates for duplicates, and returns sorted paths and years.

    Args:
        directory (str): Path to the folder containing the .tif files.

    Returns:
        tuple: (list of sorted file paths, list of sorted years)
    """
    # Regex to find 4 digits anywhere in the filename before .tif
    pattern = re.compile(r"(\d{4})")
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
        print(f"Error: No files with a 4-digit year found in: {directory}")
        return [], []

    # Sort by year
    sorted_years = sorted(year_map.keys())
    ordered_files = [year_map[year] for year in sorted_years]

    print(f"Success: {len(ordered_files)} raster maps found and sorted chronologically.")
    return ordered_files, sorted_years

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
            List of specific raster paths. If None, looks in output_path/input_rasters.
            Defaults to None.
        downsample_divisor (int, optional): 
            Factor to reduce image size for display speed. Defaults to 1.
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

    # 1. Resolve input raster paths
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

    # 2. Configure subplot grid
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

    # 3. Build colormap and normalization
    class_ids_sorted = sorted(class_map.keys())
    
    cmap = ListedColormap(
        [class_map[k]["color"] for k in class_ids_sorted]
    )
    
    norm = BoundaryNorm(
        class_ids_sorted + [class_ids_sorted[-1] + 1], 
        cmap.N
    )

    # 4. Derive scale if not provided
    if dx_km is None:
        try:
            dx_km = compute_display_pixel_size_km(
                raster_path=image_paths[0],
                downsample_divisor=downsample_divisor,
            )
        except:
            dx_km = 0.03  # Fallback value if computation fails

    # 5. Plot each raster
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

    # 6. Add Legend
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

    # 7. Add Scale Bar and North Arrow
    scalebar = ScaleBar(
        dx=dx_km, 
        units="km", 
        length_fraction=0.35, 
        location="lower right",
        scale_loc="bottom", 
        color="black", 
        box_alpha=0
    )
    
    last_ax.add_artist(scalebar)
    
    # Fix: Pass rotation as a dictionary to bypass CRS auto-detection
    north_arrow(
        last_ax, 
        location="upper left",
        rotation={"degrees": 0}
    )

    # 8. Save and show
    out_fig = os.path.join(output_path, "map_panel_input_maps.png")
    
    plt.savefig(
        out_fig, 
        format="png", 
        bbox_inches="tight", 
        dpi=300
    )
    plt.show()
    plt.close()

# =============================================================================
# QUANTITATIVE ANALYSIS
# =============================================================================
def process_and_plot_pixel_counts(
    image_paths,
    years,
    class_labels_dict,
    output_dir,
    noData_value=0,
):
    """
    Processes raster images to count pixels per class, plots a stacked bar chart,
    and exports the results to a CSV file.

    This function iterates over a time series of raster images, calculates the
    area (pixel count) for each land cover class, and generates a stacked bar
    chart to visualize the temporal evolution. It also saves the raw pixel
    counts to a CSV file.

    Args:
        image_paths (list):
            List of file paths to the raster images. Must be sorted chronologically.
        years (list):
            List of years corresponding to the images. Length must match image_paths.
        class_labels_dict (dict):
            Dictionary mapping class IDs (int) to metadata (dict).
            Must contain "name" (str) and "color" (hex str) keys.
        output_dir (str):
            Directory path where the output plot (PNG) and data (CSV) will be saved.
        noData_value (int, optional):
            The pixel value representing "No Data" to be excluded from analysis.
            Defaults to 0.

    Returns:
        pd.DataFrame:
            A pandas DataFrame containing the pivot table of pixel counts,
            indexed by Year and columns by ClassName.
    """
    # 1. Validate inputs
    if len(image_paths) != len(years):
        raise ValueError(
            f"Input mismatch: {len(image_paths)} images vs {len(years)} years."
        )

    records = []

    # 2. Iterate through each year
    for year, path in zip(years, image_paths):
        with rasterio.open(path) as src:
            data = src.read(1)

        values, counts = np.unique(
            data,
            return_counts=True,
        )

        for value, count in zip(values, counts):
            value = int(value)
            
            if value == noData_value:
                continue
            
            if value not in class_labels_dict:
                continue

            records.append(
                {
                    "Year": year,
                    "ClassID": value,
                    "ClassName": class_labels_dict[value]["name"],
                    "Pixels": int(count)
                }
            )

    # 3. Create DataFrame
    df_pixels = pd.DataFrame(records)
    
    pivot_pixels = (
        df_pixels.pivot_table(
            index="Year",
            columns="ClassName",
            values="Pixels",
            aggfunc="sum",
        )
        .fillna(0)
        .astype(int)
    )

    years_array = pivot_pixels.index.values

    # 4. Determine Y-axis scaling
    max_val = pivot_pixels.to_numpy().max()
    
    if max_val >= 1_000_000:
        scale_factor = 1_000_000
        y_label = "Area (million pixels)"
    elif max_val >= 1_000:
        scale_factor = 1_000
        y_label = "Area (thousand pixels)"
    else:
        scale_factor = 1
        y_label = "Area (pixels)"

    pivot_scaled = pivot_pixels / scale_factor

    # 5. Prepare Sorting Logic for Stacked Plot
    first_year = years_array[0]
    last_year = years_array[-1]
    net_change = (
        pivot_scaled.loc[last_year] - pivot_scaled.loc[first_year]
    )
    
    classes_sorted = net_change.sort_values(ascending=False).index.tolist()

    # 6. Generate Stacked Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Map names back to colors
    name_to_color = {
        v["name"]: v["color"] 
        for k, v in class_labels_dict.items()
    }
    
    pivot_scaled[classes_sorted].plot(
        kind="bar", 
        stacked=True, 
        ax=ax, 
        color=[name_to_color[c] for c in classes_sorted],
        width=0.9
    )

    # 7. Configure Axes
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_title("Land Cover Evolution (Pixel Counts)", fontsize=16)
    
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.tight_layout()

    # 8. Save Outputs
    out_img = os.path.join(output_dir, "graph_pixel_counts.png")
    out_csv = os.path.join(output_dir, "table_pixel_counts.csv")
    
    plt.savefig(out_img, dpi=300, bbox_inches="tight")
    pivot_pixels.to_csv(out_csv)
    
    plt.show() 
    plt.close()

    return pivot_pixels
