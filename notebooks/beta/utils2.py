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
    noData_value,
):
    """
    Processes raster images to count pixels per class, plots a stacked bar chart
    (faithful to original layout), and exports the results to a CSV file.

    Args:
        image_paths (list):
            List of file paths to the raster images (must be sorted).
        years (list):
            List of years corresponding to the images.
        class_labels_dict (dict):
            Dictionary mapping class IDs to metadata (must contain "name" and "color").
        output_dir (str):
            Directory path where the output plot and CSV will be saved.
        noData_value (int, optional):
            Pixel value to be treated as NoData (Required).

    Returns:
        pd.DataFrame:
            The pivot table containing pixel counts per year and class.
    """
    # 1. Validate that input lengths match
    if len(image_paths) != len(years):
        raise ValueError(
            f"Input mismatch: {len(image_paths)} images vs {len(years)} years."
        )

    records = []

    # 2. Iterate through each year and corresponding image path
    for year, path in zip(years, image_paths):
        with rasterio.open(path) as src:
            data = src.read(1)

        # Count unique pixel values
        values, counts = np.unique(
            data,
            return_counts=True,
        )

        # Process counts and map to class names
        for value, count in zip(values, counts):
            value = int(value)

            # Filter out NoData values
            if value == noData_value:
                continue

            # Skip classes not defined in the dictionary
            if value not in class_labels_dict:
                continue

            records.append(
                {
                    "Year": year,
                    "ClassID": value,
                    "ClassName": class_labels_dict[value]["name"],
                    "Pixels": int(count),
                }
            )

    # 3. Create DataFrame and Pivot Table
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

    # 4. Determine Y-axis scaling factor and label
    max_val = pivot_pixels.to_numpy().max()

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

    # 5. Prepare color map and sorting logic
    class_ids_plot = sorted(class_labels_dict.keys())

    color_map = {
        class_labels_dict[cid]["name"]: class_labels_dict[cid]["color"]
        for cid in class_ids_plot
    }

    # Calculate Net Change for sorting
    first_year = years_array[0]
    last_year = years_array[-1]
    
    net_change_per_class = (
        pivot_scaled.loc[last_year] - pivot_scaled.loc[first_year]
    )

    # Map names back to IDs for tie-breaking
    name_to_id_map = {
        v["name"]: k
        for k, v in class_labels_dict.items()
    }

    df_sorting = net_change_per_class.to_frame(name="net_change")
    
    df_sorting["class_id"] = df_sorting.index.map(name_to_id_map)

    # Sort: Net Change (Desc) then Class ID (Desc)
    classes_for_stack = list(
        df_sorting.sort_values(
            by=["net_change", "class_id"],
            ascending=[False, False],
        ).index
    )

    # Legend order: Reversed stack order
    classes_for_legend = list(reversed(classes_for_stack))

    # 6. Generate the Stacked Bar Chart (Original Layout)
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(years_array))
    width = 0.9
    base = np.zeros(len(years_array), dtype=float)
    patches_by_class = {}

    for cls in classes_for_stack:
        if cls not in pivot_scaled.columns:
            continue

        values_cls = pivot_scaled[cls].reindex(
            years_array, 
            fill_value=0.0
        ).values

        bars = ax.bar(
            x,
            values_cls,
            bottom=base,
            width=width,
            label=cls,
            color=color_map.get(cls, "gray"),
        )
        patches_by_class[cls] = bars[0]
        base += values_cls

    # 7. Configure Axes
    ax.set_xticks(x)
    ax.set_xticklabels(years_array)

    # Adaptive rotation for X-axis labels
    n_labels = len(years_array)
    if n_labels <= 6:
        rotation = 0
        ha = "center"
    elif n_labels <= 12:
        rotation = 45
        ha = "right"
    else:
        rotation = 90
        ha = "center"

    plt.setp(
        ax.get_xticklabels(),
        rotation=rotation,
        ha=ha,
    )

    ax.tick_params(axis="both", labelsize=14)
    ax.set_ylabel(y_label, fontsize=18)
    ax.set_xlabel("Time points", fontsize=18)
    ax.set_title("Number of pixels per class", fontsize=20)

    # Y-axis limit and formatting
    y_max_scaled = base.max() * 1.1 if base.max() > 0 else 1.0
    ax.set_ylim(0, y_max_scaled)
    
    # Ensure mticker is used (requires 'import matplotlib.ticker as mticker' at top of file)
    import matplotlib.ticker as mticker
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))

    # 8. Add Legend
    handles = [
        patches_by_class[cls]
        for cls in classes_for_legend
        if cls in patches_by_class
    ]
    labels = [
        cls
        for cls in classes_for_legend
        if cls in patches_by_class
    ]

    ax.legend(
        handles,
        labels,
        bbox_to_anchor=(1.05, 1.0),
        loc="upper left",
        frameon=False,
        fontsize=12,
    )

    plt.tight_layout()

    # 9. Save Figure
    out_fig = os.path.join(output_dir, "graph_pixel_per_class.png")
    
    plt.savefig(
        out_fig,
        format="png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()

    # 10. Save CSV
    csv_output_path = os.path.join(
        output_dir,
        "pixels_per_class_per_year.csv",
    )
    pivot_pixels.to_csv(
        csv_output_path,
        index_label="Year",
    )
    
    # Silent return (no print)
    return pivot_pixels
