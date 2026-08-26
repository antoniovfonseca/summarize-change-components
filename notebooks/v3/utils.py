import os
import re
import glob

import ee
import geemap
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import matplotlib.ticker as mticker

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from matplotlib_scalebar.scalebar import ScaleBar
from pyproj import Transformer, Geod
from tqdm import tqdm

###############################################################################
#                                                                             #
#                1. GLOBAL CONFIGURATION & CONSTANTS                          #
#                                                                             #
###############################################################################

# 1. Directory Definitions
DEFAULT_INPUT_DIR = "/content/glance_data/masked"
DEFAULT_OUTPUT_DIR = "/content/glance_data/output"

# 2. Data Definitions
NODATA_VALUE = 255
GLANCE_COLLECTION_ID = "projects/GLANCE/DATASETS/V001"
GLANCE_CLASS_BAND = "LC"

# --- CENTRALIZED GLANCE REGIONS DATABASE ---
GLANCE_REGIONS_REGISTRY = {
    'EU': {
        'geom': [-25.0, 34.0, 45.0, 72.0],
        'ul_xy': (-5505560.00, 3346245.0),
        'center_lon': 20,
        'center_lat': 55
    },
    'AN': {
        'geom': [-180.0, -90.0, 180.0, -60.0],
        'ul_xy': (-3662210.00, 5169375.0),
        'center_lon': 0,
        'center_lat': -90
    },
    'AF': {
        'geom': [-26.0, -35.0, 52.0, 38.0],
        'ul_xy': (-5312270.00, 3707205.0),
        'center_lon': 20,
        'center_lat': 5
    },
    'SA': {
        'geom': [-90.0, -56.0, -30.0, 15.0],
        'ul_xy': (-6918770.00, 4899705.0),
        'center_lon': -60,
        'center_lat': -15
    },
    'AS': {
        'geom': [25.0, 1.0, 180.0, 77.0],
        'ul_xy': (-4805840.00, 5190735.0),
        'center_lon': 100,
        'center_lat': 45
    },
    'NA': {
        'geom': [-170.0, 15.0, -50.0, 85.0],
        'ul_xy': (-7633670.00, 5076465.0),
        'center_lon': -100,
        'center_lat': 50
    },
    'OC': {
        'geom': [110.0, -50.0, 180.0, 0.0],
        'ul_xy': (-6961010.00, 4078425.0),
        'center_lon': 135,
        'center_lat': -15
    }
}

# --- DYNAMIC REGION CONFIGURATION ---
# Initialize placeholders for global variables
ACTIVE_REGION = 'AF'
GLOBAL_GEOM = None
GLANCE_RESOLUTION = [30, 30]
GLANCE_UL_XY = None
GLANCE_CRS_WKT = None

def initialize_active_region(region_code: Union[str, List[str]]) -> None:
    """
    Dynamically configure all global parameters for the active processing region.
    Supports single region strings or a list of region strings for mosaicking.

    Parameters
    ----------
    region_code : str or list of str
        The 2-letter regional code or a list of codes (e.g., ['EU', 'AF', 'SA']).
    """
    global ACTIVE_REGION, GLOBAL_GEOM, GLANCE_UL_XY, GLANCE_CRS_WKT
    
    if isinstance(region_code, list):
        # Handle multiple regions as a unified mosaic
        mosaicker = GlanceMosaicker(region_codes=region_code)
        GLOBAL_GEOM = mosaicker.unified_geometry
        ACTIVE_REGION = "GLOBAL_MOSAIC"
        GLANCE_UL_XY = None # Not applicable for mosaics
        # Use EPSG:6933 (Cylindrical Equal Area) as default global projection
        GLANCE_CRS_WKT = "EPSG:6933"
        print(f"Active region successfully initialized to a global mosaic of: {region_code}")
    else:
        # Handle a single region
        if region_code not in GLANCE_REGIONS_REGISTRY:
            raise ValueError(
                f"Region code '{region_code}' is invalid. "
                f"Choose from: {list(GLANCE_REGIONS_REGISTRY.keys())}"
            )
            
        ACTIVE_REGION = region_code
        config = GLANCE_REGIONS_REGISTRY[region_code]
        
        GLOBAL_GEOM = ee.Geometry.Rectangle(config['geom'], "EPSG:4326", False)
        GLANCE_UL_XY = config['ul_xy']
        
        GLANCE_CRS_WKT = f"""PROJCS["BU MEaSUREs Lambert Azimuthal Equal Area - {ACTIVE_REGION} - V01",
            GEOGCS["GCS_WGS_1984",
                DATUM["D_WGS_1984",
                    SPHEROID["WGS_1984",6378137.0,298.257223563]],
                PRIMEM["Greenwich",0.0],
                UNIT["degree",0.0174532925199433]],
            PROJECTION["Lambert_Azimuthal_Equal_Area"],
            PARAMETER["false_easting",0.0],
            PARAMETER["false_northing",0.0],
            PARAMETER["longitude_of_center",{config['center_lon']}],
            PARAMETER["latitude_of_center",{config['center_lat']}],
            UNIT["meter",1.0]]"""
            
        print(f"Active region successfully initialized to: {ACTIVE_REGION}")

# 3. Class Metadata
GLANCE_METADATA = {
    1: {'name': 'Water',      'color': '#386cb0'},
    2: {'name': 'Ice/Snow',   'color': '#FFFFFF'},
    3: {'name': 'Developed',  'color': '#FF1809'},
    4: {'name': 'Barren',     'color': '#000000'},
    5: {'name': 'Trees',      'color': '#00A212'},
    6: {'name': 'Shrub',      'color': '#9EDF85'},
    7: {'name': 'Herbaceous', 'color': '#FFFF96'}
}

###############################################################################
#                                                                             #
#                  2. HELPER FUNCTIONS (FILE & SYSTEM)                        #
#                                                                             #
###############################################################################

def get_year_from_filename(filepath):
    """
    Extracts the 4-digit year from the filename using a regular expression.

    Parameters
    ----------
    filepath : str
        The full path or filename of the raster image (e.g., '/path/to/glance_2005.tif').

    Returns
    -------
    int
        The 4-digit year extracted from the filename. Returns 0 if no pattern matches.
    """
    filename = os.path.basename(filepath)
    match = re.search(r"(\d{4})\.tif$", filename)
    if match:
        return int(match.group(1))
    return 0

def build_global_valid_mask_and_yearly_images(
    year_list: list,
    collection_id: str,
    band_name: str,
    nodata_val: int,
) -> tuple[ee.Image, list[tuple[int, ee.Image]]]:
    """
    Builds a global valid mask (pixels valid across all years) and returns
    yearly images, all pre-masked with this global mask.
    Ensures strict consistency of valid pixels across all analyses.
    """
    # 1. Build the image stack using the existing helper
    image_stack, band_names = build_glance_stack(
        year_list=year_list,
        collection_id=collection_id,
        band_name=band_name,
        nodata_val=nodata_val,
    )

    # 2. Create the global mask from the stack, ensuring strict intersection
    # A pixel is valid only if it's not NODATA_VALUE in *all* bands of the stack.
    global_mask = image_stack.neq(nodata_val).unmask(0).reduce(ee.Reducer.min())

    # 3. Prepare yearly_images, applying the global_mask to each original image
    yearly_images_masked = []
    for i, year in enumerate(year_list):
        original_image_for_year = image_stack.select(band_names[i])
        masked_image_year = original_image_for_year.updateMask(global_mask).rename(band_name)
        yearly_images_masked.append((year, masked_image_year))

    return global_mask, yearly_images_masked

###############################################################################
#                                                                             #
#                  3. GLanCE VISUALIZATION                                    #
#                                                                             #
###############################################################################

# ---------------------------------------------------------------------------
# 3.1 SELECT GLanCE YEAR
# ---------------------------------------------------------------------------
def get_glance_map(year):
    """
    Generates an interactive map with GLANCE land cover data for a given year.

    This function assumes that the Earth Engine API (ee) has already been
    authenticated and initialized in the main execution environment.

    Parameters
    ----------
    year : int
        The year to visualize (e.g., 2010). 
        GLANCE data is typically available from 2001 to 2019.

    Returns
    -------
    geemap.Map
        An interactive map object centered on the data with the land cover
        layer added. Returns None if data cannot be loaded.
    """

    # 1. Dynamic Visualization Parameters construction
    # Extract IDs to sort correctly
    class_ids = sorted(GLANCE_METADATA.keys())
    
    # Earth Engine expects hex colors without '#', so we strip it
    palette = [GLANCE_METADATA[i]['color'].lstrip('#') for i in class_ids]
    
    vis_params = {
        'min': class_ids[0],
        'max': class_ids[-1],
        'palette': palette
    }

    # 2. Initialize Map
    m = geemap.Map()
    m.setCenter(0, 20, 2)

    # 3. Filter Data
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    try:
        # Use global constant for Collection ID
        collection = ee.ImageCollection(GLANCE_COLLECTION_ID)
        image = collection.filterDate(start_date, end_date).mosaic()
        
        # Use global constant for Band Name
        glance_layer = image.select(GLANCE_CLASS_BAND)

        # 4. Add Layer to Map
        m.addLayer(
            glance_layer,
            vis_params, 
            f"GLANCE LC {year}"
        )
        
        # 5. Dynamic Legend construction
        # Creates a dictionary 'Name': 'Color' for the legend
        legend_dict = {
            meta['name']: meta['color'] 
            for meta in GLANCE_METADATA.values()
        }
        
        m.add_legend(
            title="GLANCE Land Cover",
            legend_dict=legend_dict
        )
        
        return m

    except Exception as e:
        print(f"Error loading GLANCE data for {year}: {e}")
        return None

# ---------------------------------------------------------------------------
# 3.2 PLOT GLanCE
# ---------------------------------------------------------------------------

def view_local_rasters(input_dir=DEFAULT_INPUT_DIR):
    """
    Creates an interactive map displaying all .tif files found in the specified directory.
    This is useful for visually validating the result of the masking process.

    Parameters
    ----------
    input_dir : str, optional
        The directory path containing the masked .tif files.
        Defaults to DEFAULT_INPUT_DIR.

    Returns
    -------
    geemap.Map
        An interactive map object with the raster layers added.
        Returns None if no .tif files are found.
    """
    # 1. Initialize Map
    m = geemap.Map()
    
    # 2. Search for files
    search_pattern = os.path.join(
        input_dir,
        "*.tif"
    )
    files = sorted(glob.glob(search_pattern))
    
    if not files:
        print(f"No .tif files found in {input_dir}")
        return None

    # 3. Prepare Colors (Palette)
    # Extract colors from metadata to match the charts.
    # We pass the list of hex codes (e.g., ['#0000FF', ...])
    class_ids = sorted(GLANCE_METADATA.keys())
    palette = [GLANCE_METADATA[i]['color'] for i in class_ids]
    
    # 4. Add each file as a Layer
    print(f"Loading {len(files)} local raster layers...")
    
    for filepath in files:
        filename = os.path.basename(filepath)
        year_match = re.search(r"(\d{4})", filename)
        layer_name = f"Masked {year_match.group(1)}" if year_match else filename
        
        try:
            m.add_raster(
                filepath, 
                layer_name=layer_name, 
                palette=palette,
                nodata=NODATA_VALUE
            )
        except Exception as e:
            print(f"Could not load {filename}: {e}")

    # 5. Add Legend
    legend_dict = {
        meta['name']: meta['color'] 
        for meta in GLANCE_METADATA.values()
    }
    m.add_legend(
        title="Classes",
        legend_dict=legend_dict
    )

    return m

###############################################################################
#                                                                             #
#                  4. CLASS AREA                                              #
#                                                                             #
###############################################################################

# ---------------------------------------------------------------------------
# 4.1 EXPORT PIXEL COUNTS
# ---------------------------------------------------------------------------
def export_global_pixel_counts_tasks(
    year_list: list,
    drive_folder: str,
    scale: int = 30,
    max_pixels: float = 1e13,
    nodata_val: int = NODATA_VALUE,
    full_year_list: list = None,
) -> list:
    """
    Triggers GEE tasks to calculate pixel counts for global GLANCE images.

    This function avoids client-side timeouts by tiling the global geometry and
    mapping a reduction over the tiles on the server side. The results for each
    tile are exported to a single CSV per year, which can be aggregated later.

    Parameters
    ----------
    year_list : list of int
        A list of 4-digit years to process.
    drive_folder : str
        The Google Drive folder where the CSVs will be saved.
    scale : int, optional
        The scale in meters for the GEE reduction. Default is 30.
    full_year_list : list of int, optional
        The complete timeline to construct the consistent global validity mask.
    max_pixels : float, optional
        The maximum number of pixels to process. Default is 1e13.
    nodata_val : int, optional
        The NoData value used for masking. Default is 255.

    Returns
    -------
    list of ee.batch.Task
        A list containing all the triggered Earth Engine tasks.
    """
    if GLOBAL_GEOM is None:
        raise ValueError(
            "GLOBAL_GEOM is not initialized. Please call "
            "utils.initialize_active_region(region_code) before running tasks."
        )

    glance_collection = ee.ImageCollection(GLANCE_COLLECTION_ID).select(GLANCE_CLASS_BAND)
    tasks_list = []

    if full_year_list is None:
        full_year_list = year_list

    # 1. Build global mask using the FULL timeline to ensure mathematical consistency
    full_stack, _ = build_glance_stack(
        year_list=full_year_list,
        collection_id=GLANCE_COLLECTION_ID,
        band_name=GLANCE_CLASS_BAND,
        nodata_val=nodata_val,
    )
    global_mask = full_stack.neq(nodata_val).unmask(0).reduce(ee.Reducer.min())

    # 2. Build target stack for the specific years we want to export tasks for
    target_stack, target_band_names = build_glance_stack(
        year_list=year_list,
        collection_id=GLANCE_COLLECTION_ID,
        band_name=GLANCE_CLASS_BAND,
        nodata_val=nodata_val,
    )

    for i, year in enumerate(year_list):
        image_year = target_stack.select(target_band_names[i]).rename(GLANCE_CLASS_BAND)
        image_masked = image_year.updateMask(global_mask)

        histogram = image_masked.reduceRegion(
            # force unweighted reduction to guarantee integer pixel counts from GEE server
            reducer=ee.Reducer.frequencyHistogram().unweighted(),
            geometry=GLOBAL_GEOM,
            scale=scale,
            crs="EPSG:4326",
            maxPixels=max_pixels,
            tileScale=16
        )

        counts_dict = ee.Dictionary(histogram.get(GLANCE_CLASS_BAND))
        feature_collection = ee.FeatureCollection([
            ee.Feature(None, counts_dict)
        ])

        export_name = f"Pixel_Counts_LULC_{year}"
        task = ee.batch.Export.table.toDrive(
            collection=feature_collection,
            description=export_name,
            folder=drive_folder,
            fileFormat="CSV"
        )

        task.start()
        tasks_list.append(task)
        print(f"Task submitted for year {year}.")

    return tasks_list

# ---------------------------------------------------------------------------
# 4.1 PLOT PIXEL COUNTS
# ---------------------------------------------------------------------------

def plot_pixel_counts_bar_chart(
    input_dir: str,
    class_labels_dict: dict,
    output_dir: str,
) -> None:
    """
    Reads yearly GEE pixel count CSVs, aggregates them, and generates a 
    stacked bar chart of pixel counts per class over time.

    Parameters
    ----------
    input_dir : str
        Directory path containing the Earth Engine CSVs (e.g., Pixel_Counts_LULC_2001.csv).
    class_labels_dict : dict
        Dictionary mapping class IDs to metadata (must contain "name" and "color").
    output_dir : str
        Directory path where the output plot will be saved.
    """
    tables_dir = os.path.join(output_dir, "tables")
    consolidated_csv_path = os.path.join(tables_dir, "aggregated_pixel_counts.csv")

    # 1. Read and aggregate GEE CSVs or load existing consolidated CSV
    csv_pattern = os.path.join(input_dir, "Pixel_Counts_LULC_*.csv")
    csv_files = glob.glob(csv_pattern)

    pivot_pixels = None

    if not csv_files and os.path.exists(consolidated_csv_path):
        print(f"No raw GEE CSVs found, but detected consolidated CSV. Loading: {consolidated_csv_path}")
        pivot_pixels = pd.read_csv(consolidated_csv_path, index_col="Year")
    elif csv_files:
        yearly_data = {}
        for file in csv_files:
            basename = os.path.basename(file)
            # Strict regex to avoid duplicate files like "(1).csv" or copy files
            match = re.match(r"^Pixel_Counts_LULC_(\d{4})\.csv$", basename)
            if not match:
                continue
            
            year = int(match.group(1))
            df_csv = pd.read_csv(file)
            
            row_dict = {}
            for col in df_csv.columns:
                try:
                    class_id = int(col)
                    if class_id in class_labels_dict:
                        class_name = class_labels_dict[class_id]["name"]
                        row_dict[class_name] = df_csv[col].sum()
                except ValueError:
                    pass
            
            if row_dict:
                if year not in yearly_data:
                    yearly_data[year] = row_dict
                else:
                    for class_name, count in row_dict.items():
                        yearly_data[year][class_name] = yearly_data[year].get(class_name, 0) + count

        if yearly_data:
            pivot_pixels = pd.DataFrame.from_dict(
                yearly_data,
                orient='index'
            ).fillna(0)
            # Round and convert to integer since we are counting discrete pixels
            pivot_pixels = pivot_pixels.round(0).astype(int)
            pivot_pixels.sort_index(inplace=True)
            pivot_pixels.index.name = "Year"
            
            # Save consolidated aggregated data to CSV
            os.makedirs(tables_dir, exist_ok=True)
            pivot_pixels.to_csv(consolidated_csv_path)
            print(f"Consolidated pixel counts saved to: {consolidated_csv_path}")

    if pivot_pixels is None or pivot_pixels.empty:
        print(f"No valid GEE CSV data found in {input_dir} and no consolidated CSV found at {consolidated_csv_path}")
        return

    years_array = pivot_pixels.index.values

    # 2. Determine Y-axis scaling factor and label
    max_val = pivot_pixels.to_numpy().max()

    if max_val >= 1_000_000_000_000:
        scale_factor = 1_000_000_000_000
        y_label = "Class Area (trillion pixels)"
    elif max_val >= 1_000_000_000:
        scale_factor = 1_000_000_000
        y_label = "Class Area (billion pixels)"
    elif max_val >= 1_000_000:
        scale_factor = 1_000_000
        y_label = "Class Area (million pixels)"
    elif max_val >= 1_000:
        scale_factor = 1_000
        y_label = "Class Area (thousand pixels)"
    elif max_val >= 100:
        scale_factor = 100
        y_label = "Class Area (hundred pixels)"
    else:
        scale_factor = 1
        y_label = "Class Area (pixels)"

    pivot_scaled = pivot_pixels / scale_factor

    # 3. Prepare color map and sorting logic
    class_ids_plot = sorted(
        class_labels_dict.keys(),
    )

    color_map = {
        class_labels_dict[class_id]["name"]: class_labels_dict[class_id]["color"]
        for class_id in class_ids_plot
    }

    # Calculate Net Change
    first_year = years_array[0]
    last_year = years_array[-1]
    net_change_per_class = (
        pivot_scaled.loc[last_year]
        - pivot_scaled.loc[first_year]
    )

    # Map names back to IDs for tie-breaking
    name_to_id_map = {
        v["name"]: k
        for k, v in class_labels_dict.items()
    }

    df_sorting = net_change_per_class.to_frame(
        name="net_change",
    )
    df_sorting["class_id"] = df_sorting.index.map(
        name_to_id_map,
    )

    # Sort: Net Change (Desc) then Class ID (Desc)
    classes_for_stack = list(
        df_sorting.sort_values(
            by=[
                "net_change",
                "class_id",
            ],
            ascending=[
                False,
                False,
            ],
        ).index,
    )

    # Legend order: Reversed stack order
    classes_for_legend = list(
        reversed(classes_for_stack),
    )

    # 4. Generate the Stacked Bar Chart
    fig, ax = plt.subplots(
        figsize=(10,6),
    )

    x = np.arange(
        len(years_array),
    )
    width = 0.9
    base = np.zeros(
        len(years_array),
        dtype=float,
    )
    patches_by_class: dict[str, plt.Artist] = {}

    for cls in classes_for_stack:
        if cls not in pivot_scaled.columns:
            continue

        values_cls = pivot_scaled[cls].reindex(
            years_array,
            fill_value=0.0,
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

    # 5. Configure Axes
    ax.set_xticks(
        x,
    )
    ax.set_xticklabels(
        years_array,
    )

    # Adaptive rotation for X-axis labels
    n_labels = len(years_array)

    # 0 degrees if <= 6 labels, else 90 degrees
    if n_labels <= 6:
        rotation = 0
        ha = "center"
    else:
        rotation = 90
        ha = "center"

    plt.setp(
        ax.get_xticklabels(),
        rotation=rotation,
        ha=ha,
    )

    ax.tick_params(
        axis="both",
        labelsize=12,
    )
    ax.set_ylabel(
        y_label,
        fontsize=18,
    )
    ax.set_xlabel(
        "Time points",
        fontsize=18,
    )
    ax.set_title(
        "Class Area at Time Points",
        fontsize=20,
    )

    # Y-axis limit and formatting
    y_max_scaled = base.max() * 1.1 if base.max() > 0 else 1.0
    ax.set_ylim(
        0,
        y_max_scaled,
    )
    ax.yaxis.set_major_locator(
        ticker.MaxNLocator(
            nbins=8,
            integer=True,
        ),
    )

    # 6. Add Legend
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

    leg = ax.legend(
        handles,
        labels,
        title="Class",
        title_fontsize=16,
        bbox_to_anchor=(
            1.02,
            0.5,
        ),
        loc="center left",
        frameon=False,
        fontsize=16,
        alignment="left",
    )

    plt.tight_layout()

    # 7. Save Figure
    charts_dir = os.path.join(
        output_dir,
        "charts",
    )

    os.makedirs(
        charts_dir,
        exist_ok=True,
    )

    out_fig = os.path.join(
        charts_dir,
        "chart_pixel_per_class_net_change.png",
    )

    plt.savefig(
        out_fig,
        format="png",
        bbox_inches="tight",
        dpi=300,
    )
    print(f"Chart successfully saved at: {out_fig}")
    plt.show()

###############################################################################
#                                                                             #
#                  5. NUMBER OF CHANGES FUNCTIONS                             #
#                                                                             #
###############################################################################

# ---------------------------------------------------------------------------
# 5.1 EXPORT NUMBER OF CHANGES PER INTERVAL
# ---------------------------------------------------------------------------
def export_global_change_frequency_tasks(
    year_list: list,
    drive_folder: str,
    scale: int = 300,
    full_year_list: list = None,
) -> list:
    """
    Triggers GEE tasks to calculate the frequency of pixel changes for each
    consecutive interval in a time series and exports them as CSV files.
    """
    if GLOBAL_GEOM is None:
        raise ValueError(
            "GLOBAL_GEOM is not initialized. Please call "
            "utils.initialize_active_region(region_code) before running tasks."
        )

    if full_year_list is None:
        full_year_list = year_list

    # 1. Build global mask using the FULL timeline to ensure mathematical consistency
    full_stack, _ = build_glance_stack(
        year_list=full_year_list,
        collection_id=GLANCE_COLLECTION_ID,
        band_name=GLANCE_CLASS_BAND,
        nodata_val=NODATA_VALUE,
    )
    global_mask = full_stack.neq(NODATA_VALUE).unmask(0).reduce(ee.Reducer.min())

    # 2. Build target stack for the specific years we want to export tasks for
    target_stack, target_band_names = build_glance_stack(
        year_list=year_list,
        collection_id=GLANCE_COLLECTION_ID,
        band_name=GLANCE_CLASS_BAND,
        nodata_val=NODATA_VALUE,
    )

    # 3. Shift the target stack by 1 band to compare t and t+1 in parallel (vectorized)
    stack_t = target_stack.select(target_band_names[:-1])
    stack_t1 = target_stack.select(target_band_names[1:])

    # 4. Calculate total changes across the target years timeline
    total_changes = stack_t.neq(stack_t1).reduce(ee.Reducer.sum()).rename('num_changes')

    total_changes_masked = total_changes.updateMask(global_mask)

    tasks_list = []

    # 5. Iterate through each consecutive interval to mask and export
    for i in range(len(year_list) - 1):
        start_year = year_list[i]
        end_year = year_list[i + 1]

        img_start = target_stack.select(target_band_names[i]).rename(GLANCE_CLASS_BAND).updateMask(global_mask)
        img_end = target_stack.select(target_band_names[i + 1]).rename(GLANCE_CLASS_BAND).updateMask(global_mask)

        # Mask total_changes to only include pixels that changed in THIS interval
        interval_change = img_start.neq(img_end)
        interval_total_changes = total_changes_masked.updateMask(interval_change)

        # 6. Compute the frequency histogram of the total changes for these pixels
        histogram = interval_total_changes.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram().unweighted(),
            geometry=GLOBAL_GEOM,
            scale=scale,
            crs="EPSG:4326",
            maxPixels=1e13,
            tileScale=16,
        ).get('num_changes')

        # Handle possible nulls if no change occurred
        hist_dict = ee.Dictionary(ee.Algorithms.If(histogram, histogram, {}))
        feature = ee.Feature(None, hist_dict)
        fc = ee.FeatureCollection([feature])

        # 7. Configure and start the export task
        export_name = f"Number_Change_{start_year}_{end_year}"
        task = ee.batch.Export.table.toDrive(
            collection=fc,
            description=export_name,
            folder=drive_folder,
            fileNamePrefix=export_name,
            fileFormat="CSV",
        )
        task.start()
        tasks_list.append(task)
        print(f"Task submitted for interval {start_year}-{end_year}")

    return tasks_list

# ---------------------------------------------------------------------------
# 5.1 PLOT NUMBER OF CHANGES DURING TIME INTERVALS
# ---------------------------------------------------------------------------

def plot_global_change_frequency_bar_chart(
    input_dir: str,
    output_dir: str,
) -> None:
    """
    Create a stacked bar chart showing the sequence of changes per interval
    by compiling multiple GEE-exported CSV files, saving a consolidated CSV 
    for performance caching.

    Parameters
    ----------
    input_dir : str
        Directory containing the 'Number_Change_*.csv' files.
    output_dir : str
        Directory to save the resulting figure.
    """
    tables_dir = os.path.join(output_dir, "tables")
    consolidated_csv_path = os.path.join(tables_dir, "aggregated_change_frequency.csv")

    # 1. Read and compile the GEE CSV files or load existing consolidated CSV
    search_pattern = os.path.join(input_dir, "Number_Change_*.csv")
    csv_files = glob.glob(search_pattern)

    df = None

    if not csv_files and os.path.exists(consolidated_csv_path):
        print(f"No raw GEE change frequency CSVs found, but detected consolidated CSV. Loading: {consolidated_csv_path}")
        df = pd.read_csv(consolidated_csv_path, index_col=0)
        df.columns = df.columns.astype(str)
        df = df.round(0).astype(int)
        # Keep only columns that are positive integers (representing 1 or more changes)
        valid_cols = [c for c in df.columns if c.isdigit() and int(c) > 0]
        df = df[valid_cols]
    elif csv_files:
        records = {}
        for file_path in csv_files:
            basename = os.path.basename(file_path)
            # Avoid matching consolidated or other system CSV files
            if "Overall" in basename or "aggregated" in basename:
                continue
                
            interval_str = basename.replace("Number_Change_", "").replace(".csv", "")
            parts = interval_str.split("_")
            if len(parts) == 2:
                label = f"{parts[0]}-{parts[1]}"
            else:
                label = interval_str

            df_temp = pd.read_csv(file_path)
            num_cols = df_temp.select_dtypes(include=['number']).columns
            num_cols = [c for c in num_cols if 'system' not in c]

            if num_cols:
                row_data = df_temp[num_cols].sum()
                records[label] = row_data

        if records:
            df = pd.DataFrame.from_dict(records, orient='index').fillna(0)
            # Round and convert to integer since we are counting discrete pixels
            df = df.round(0).astype(int)
            
            # Fix column names (e.g. float keys "1.0" to integer strings "1")
            new_cols = {}
            for c in df.columns:
                try:
                    int_c = int(float(c))
                    new_cols[c] = str(int_c)
                except ValueError:
                    new_cols[c] = str(c)
            df.rename(columns=new_cols, inplace=True)

            # Keep only columns that are positive integers (representing 1 or more changes)
            valid_cols = [c for c in df.columns if c.isdigit() and int(c) > 0]
            df = df[valid_cols]

            sorted_cols = sorted(df.columns, key=lambda x: int(x))
            df = df[sorted_cols]
            df.sort_index(inplace=True)
            
            # Save consolidated aggregated data to CSV
            os.makedirs(tables_dir, exist_ok=True)
            df.to_csv(consolidated_csv_path)
            print(f"Consolidated change frequency saved to: {consolidated_csv_path}")

    if df is None or df.empty:
        print(f"No valid GEE CSV data found in {input_dir} and no consolidated CSV found at {consolidated_csv_path}")
        return

    # 7. Determine Unit Scaling
    max_val = df.sum(
        axis=1,
    ).max()

    if max_val >= 1_000_000_000_000:
        factor = 1_000_000_000_000.0
        suffix = " (trillion pixels)"
    elif max_val >= 1_000_000_000:
        factor = 1_000_000_000.0
        suffix = " (billion pixels)"
    elif max_val >= 1_000_000:
        factor = 1_000_000.0
        suffix = " (million pixels)"
    elif max_val >= 1_000:
        factor = 1_000.0
        suffix = " (thousand pixels)"
    else:
        factor = 1.0
        suffix = ""

    df_scaled = df / factor

    # 8. Setup Figure and Colors
    fig, ax = plt.subplots(
        figsize=(14,6),
    )

    n_cols = len(
        df.columns,
    )

    cmap = plt.cm.viridis_r

    if n_cols > 1:
        colors = [
            cmap(
                i / (
                    n_cols - 1
                ),
            )
            for i in range(
                n_cols,
            )
        ]
    else:
        colors = [
            cmap(
                0.5,
            ),
        ]

    # 9. Plot Stacked Bars
    bottom = pd.Series(
        0.0,
        index=df_scaled.index,
    )

    for i, col in reversed(
        list(
            enumerate(
                df.columns,
            )
        )
    ):
        vals = df_scaled[
            col
        ]

        if vals.sum() == 0:
            label_txt = "_nolegend_"
        else:
            label_txt = f"{col}"

        ax.bar(
            df_scaled.index,
            vals,
            bottom=bottom,
            label=label_txt,
            color=colors[
                i
            ],
            edgecolor="none",
            linewidth=0.5,
            width=0.9,
        )
        bottom += vals

    # 10. Formatting Axes and Labels
    y_label_text = f"Change{suffix}"

    ax.set_ylabel(
        y_label_text,
        fontsize=18,
    )

    ax.yaxis.set_major_locator(
        ticker.MaxNLocator(
            nbins=5,
            integer=True,
        ),
    )

    ax.set_title(
        "Number of Changes during Time Intervals",
        fontsize=20,
        pad=15,
    )

    # 11. X-Axis labels
    labels = ax.get_xticklabels()
    n_labels = len(
        labels,
    )

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
        labels,
        rotation=rotation,
        ha=ha,
        fontsize=18,
    )

    ax.tick_params(
        axis="y",
        rotation=0,
        labelsize=18,
    )

    # 12. Legend
    handles, labels = ax.get_legend_handles_labels()

    leg = ax.legend(
        handles[::-1],
        labels[::-1],
        title="Changes",
        title_fontsize=16,
        bbox_to_anchor=(
            1.02,
            0.5,
        ),
        loc="center left",
        frameon=False,
        fontsize=16,
    )

    for patch in leg.get_patches():
        patch.set_linewidth(
            0,
        )

    plt.tight_layout()

    # 13. Save Figure
    charts_dir = os.path.join(
        output_dir,
        "charts",
    )
    os.makedirs(
        charts_dir,
        exist_ok=True,
    )

    output_fig = os.path.join(
        charts_dir,
        "chart_number_change_time_interval.png",
    )

    plt.savefig(
        output_fig,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print(
        f"Chart saved to: {output_fig}",
    )

# ---------------------------------------------------------------------------
# 5.2 PLOT NUMBER OF CHANGES OVERALL
# ---------------------------------------------------------------------------

def plot_number_of_changes_distribution(
    input_dir: str,
    output_dir: str,
    nodata_val: int = NODATA_VALUE,
) -> None:
    """
    Generate a single stacked bar chart of the number of changes distribution.

    This function calculates the percentage of unique pixels that underwent
    1, 2, 3, or N total changes relative to the ENTIRE valid study area.
    """
    tables_dir = os.path.join(output_dir, "tables")
    consolidated_csv_path = os.path.join(tables_dir, "aggregated_overall_change_frequency.csv")

    df_counts = None

    # 1. Try to load from consolidated CSV first (cache)
    if os.path.exists(consolidated_csv_path):
        print(f"Loading consolidated overall change frequency from: {consolidated_csv_path}")
        df_counts = pd.read_csv(consolidated_csv_path, index_col="Number_of_Changes")
        df_counts['Count'] = df_counts['Count'].astype(int)
    else:
        # 2. If not in cache, locate the Number of Changes Raster
        raster_patterns = [
            os.path.join(input_dir, "rasters", "Number_of_Changes_Raster_*.tif"),
            os.path.join(input_dir, "Number_of_Changes_Raster_*.tif"),
            os.path.join(output_dir, "rasters", "Number_of_Changes_Raster_*.tif"),
            os.path.join(output_dir, "Number_of_Changes_Raster_*.tif"),
        ]
        
        raster_path = None
        for pattern in raster_patterns:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                raster_path = matches[0]
                break
                
        if not raster_path:
            # Fallback broader search
            matches = glob.glob(os.path.join(input_dir, "**/*Number_of_Changes_Raster*.tif"), recursive=True)
            if matches:
                raster_path = matches[0]

        if not raster_path:
            print(f"Error: Could not find Number_of_Changes_Raster_*.tif in {input_dir} or {output_dir}.")
            return

        print(f"Loading overall change frequency directly from raster: {raster_path}")

        # 3. Read raster and compute unique counts in memory-efficient chunks
        data_counts_dict = {}
        with rasterio.open(raster_path) as src:
            nodata = src.nodata if src.nodata is not None else nodata_val
            for _, window in src.block_windows():
                chunk = src.read(1, window=window)
                valid_pixels = chunk[chunk != nodata]
                if valid_pixels.size > 0:
                    vals, counts = np.unique(valid_pixels, return_counts=True)
                    for v, c in zip(vals, counts):
                        v_int = int(v)
                        data_counts_dict[v_int] = data_counts_dict.get(v_int, 0) + int(c)

        if not data_counts_dict:
            print("Error: No valid data found in the raster to compute overall change frequency.")
            return

        df_counts = pd.DataFrame.from_dict(data_counts_dict, orient='index', columns=['Count'])
        df_counts.index.name = "Number_of_Changes"
        df_counts.sort_index(inplace=True)

        # Save consolidated data to CSV
        os.makedirs(tables_dir, exist_ok=True)
        df_counts.to_csv(consolidated_csv_path)
        print(f"Consolidated overall change frequency saved to: {consolidated_csv_path}")

    # 4. Strict numerical alignment: Ensure index and counts are clean integers
    df_numeric = df_counts.copy()
    df_numeric.index = pd.to_numeric(df_numeric.index, errors='coerce')
    df_numeric = df_numeric.dropna()
    df_numeric['Count'] = pd.to_numeric(df_numeric['Count'], errors='coerce').fillna(0).astype(int)

    # Denominator must be the entire valid study area: stable (0) + all changes (1..N)
    # We strictly exclude NoData (255) from the total area calculations
    df_valid_area = df_numeric[(df_numeric.index >= 0) & (df_numeric.index < nodata_val)]
    total_valid_pixels = df_valid_area['Count'].sum()

    if total_valid_pixels == 0:
        print("Error: No valid study area pixels found for overall change frequency.")
        return

    # Calculate percentages for changes (> 0) relative to the total valid study area
    df_filtered = df_valid_area[df_valid_area.index > 0]
    percentages = {}
    for n_changes, row in df_filtered.iterrows():
        count = row['Count']
        n_changes_int = int(n_changes)
        pct = (count / total_valid_pixels) * 100.0
        percentages[n_changes_int] = pct

    # 4. Setup Colors
    active_changes = [
        k for k, v in percentages.items()
        if v > 0
    ]

    if not active_changes:
        active_changes = list(
            percentages.keys(),
        )

    n_colors = len(
        active_changes,
    )
    cmap = plt.cm.viridis_r

    sorted_changes_desc = sorted(
        active_changes,
        reverse=True,
    )

    colors = {
        n: cmap(
            i / (n_colors - 1)
        )
        if n_colors > 1
        else cmap(0.5)
        for i, n in enumerate(
            sorted(
                active_changes,
            )
        )
    }

    # 5. Create the Figure
    fig, ax = plt.subplots(
        figsize=(
            6,
            6,
        ),
    )

    bottom = 0.0
    for n_change in sorted_changes_desc:
        val = percentages[
            n_change
        ]
        if val > 0:
            ax.bar(
                0,
                val,
                bottom=bottom,
                color=colors[
                    n_change
                ],
                width=0.4,
                edgecolor="none",
            )
            bottom += val

    # 6. Formatting the Axes
    ax.set_ylabel(
        "Change (% of study area)",
        fontsize=16,
    )

    ax.set_title(
        "Number of Changes Overall",
        fontsize=18,
        pad=15,
    )

    for spine in [
        "top",
        "right",
        "bottom",
        "left",
    ]:
        ax.spines[
            spine
        ].set_visible(
            True,
        )
        ax.spines[
            spine
        ].set_color(
            "black",
        )
        ax.spines[
            spine
        ].set_linewidth(
            0.5,
        )

    ax.tick_params(
        axis="y",
        which="major",
        labelsize=18,
    )

    ax.set_xticks(
        [],
    )

    # Note: Keep Y-axis limit at 105 so the scale is always absolute
    max_y = bottom * 1.05 if bottom > 0 else 1.0

    ax.set_ylim(
        0,
        max_y,
    )

    # Define the number of bins
    ax.yaxis.set_major_locator(
        mticker.MaxNLocator(
            integer=True,
            nbins=10,
        ),
    )

    # 7. Legend
    legend_elements = []

    for n in sorted(active_changes):
        legend_elements.append(
            Patch(
                facecolor=colors[n],
                label=str(n),
            )
        )

    ax.legend(
        handles=legend_elements,
        title="Changes",
        title_fontsize=16,
        loc="center left",
        bbox_to_anchor=(
            1.05,
            0.5,
        ),
        fontsize=14,
        frameon=False,
    )

    # Force the main plotting box to always occupy the exact same spatial coordinates in the figure
    fig.subplots_adjust(
        left=0.15,
        right=0.75,
        bottom=0.1,
        top=0.9
    )

    # 8. Save and show the figure
    charts_dir = os.path.join(
        output_dir,
        "charts",
    )
    os.makedirs(
        charts_dir,
        exist_ok=True,
    )

    out_fig_path = os.path.join(
        charts_dir,
        "chart_number_changes_percentage_overall.png",
    )

    plt.savefig(
        out_fig_path,
        dpi=300,
        bbox_inches="tight",
        format="png",
    )
    plt.show()
    print(
        f"Chart saved to: {out_fig_path}",
    )

# ---------------------------------------------------------------------------
# 5.3 EXPORT NUMBER OF CHANGES RASTER
# ---------------------------------------------------------------------------

def export_global_number_of_changes_raster_task(
    year_list: list[int],
    drive_folder: str,
    scale: int = 300,
    full_year_list: list = None,
) -> ee.batch.Task:
    """
    Compute and save a raster representing the total number of class changes per pixel
    using Google Earth Engine, and export it to Google Drive.
    """
    if GLOBAL_GEOM is None:
        raise ValueError(
            "GLOBAL_GEOM is not initialized. Please call "
            "utils.initialize_active_region(region_code) before running tasks."
        )

    if full_year_list is None:
        full_year_list = year_list

    # 1. Combine lists to build a single stack containing all required years safely
    combined_years = sorted(list(set(year_list) | set(full_year_list)))

    # 2. Build the master stack and extract the global mask based on the FULL timeline
    master_stack, master_band_names = build_glance_stack(
        year_list=combined_years,
        collection_id=GLANCE_COLLECTION_ID,
        band_name=GLANCE_CLASS_BAND,
        nodata_val=NODATA_VALUE,
    )
    
    # Create the global validity mask strictly using full_year_list bands
    full_year_bands = [f"y{y}" for y in full_year_list]
    full_stack_subset = master_stack.select(full_year_bands)
    global_mask = full_stack_subset.neq(NODATA_VALUE).unmask(0).reduce(ee.Reducer.min())

    # Build target bands list for changes calculation
    target_band_names = [f"y{y}" for y in year_list]

    # 3. Shift the target stack by 1 band to compare t and t+1 in parallel (vectorized)
    stack_t = master_stack.select(target_band_names[:-1])
    stack_t1 = master_stack.select(target_band_names[1:])

    # 4. Calculate total changes across the timeline
    change_count_img = stack_t.neq(stack_t1).reduce(ee.Reducer.sum()).rename(GLANCE_CLASS_BAND)

    masked_change_count = change_count_img.updateMask(global_mask)

    # Unmask void/nodata pixels to NODATA_VALUE before export to differentiate from 0 (no change)
    final_export_image = masked_change_count.unmask(NODATA_VALUE).toByte()

    start_year = year_list[0]
    end_year = year_list[-1]
    task_name = f"Number_of_Changes_Raster_{start_year}_{end_year}"

    # Create and submit the export task
    task = ee.batch.Export.image.toDrive(
        image=final_export_image,
        description=task_name,
        folder=drive_folder,
        fileNamePrefix=task_name,
        region=GLOBAL_GEOM,
        scale=scale,
        maxPixels=1e13,
        crs="EPSG:4326",
        fileFormat="GeoTIFF"
    )
    task.start()
    
    print(f"Task started: {task_name} (Scale: {scale}m, NoData: {NODATA_VALUE})")
    return task

# ---------------------------------------------------------------------------
# 5.4 PLOT NUMBER OF CHANGES MAP
# ---------------------------------------------------------------------------

# 1. Provide a stub for north_arrow if not defined
def north_arrow(
    ax,
    location="upper right",
    shadow=False,
    rotation=None,
    scale=0.3,
) -> None:
    """
    Placeholder for north_arrow function.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the arrow on.
    location : str, optional
        Location of the arrow.
    shadow : bool, optional
        Whether to draw a shadow.
    rotation : dict, optional
        Rotation parameters.
    scale : float, optional
        Scale of the arrow.
    """
    pass

def compute_display_pixel_size_km(
    raster_path: str,
    downsample_factor: float,
) -> float:
    """
    Compute horizontal resolution in kilometers per displayed pixel.

    Parameters
    ----------
    raster_path : str
        Path to a raster file (or VRT) used to derive spatial extent and CRS.
    downsample_factor : float
        Float factor used to downsample the raster width for display (e.g., 0.05).

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
        lon_l, lat_mid = to_ll.transform(left, lat_mid_src)
        lon_r, _ = to_ll.transform(right, lat_mid_src)
        lon_c, _ = to_ll.transform((left + right) / 2.0, lat_mid_src)

        geod = Geod(ellps="WGS84")
        
        # Handle global rasters by splitting the distance calculation 
        # to avoid the distance between -180 and 180 resulting in 0.
        if abs(lon_r - lon_l) > 350:
            _, _, dist1 = geod.inv(lon_l, lat_mid, lon_c, lat_mid)
            _, _, dist2 = geod.inv(lon_c, lat_mid, lon_r, lat_mid)
            width_m = dist1 + dist2
        else:
            _, _, width_m = geod.inv(lon_l, lat_mid, lon_r, lat_mid)

        cols_disp = max(1, int(src.width * downsample_factor))

        return (width_m / cols_disp) / 1_000


def plot_number_of_changes_map(
    output_dir: str,
    nodata_val: int,
    raster_filename: str = "Number_of_Changes_Raster",
    scale_factor: float = 0.05,
) -> None:
    """
    Plot the Number of Changes raster map with cartographic elements.

    Parameters
    ----------
    output_dir : str
        Directory containing the exported GEE tiles and where the map will be saved.
    nodata_val : int
        Value representing NoData in the raster to be masked out.
    raster_filename : str, optional
        Prefix of the raster tiles to plot.
    scale_factor : float, optional
        Scale factor to downsample the massive global raster to fit into memory.

    Returns
    -------
    None
    """

    # 1. Locate all raster tiles exported by GEE
    raster_files = glob.glob(
        os.path.join(
            output_dir,
            f"{raster_filename}*.tif")
        )
    if not raster_files:
        raise FileNotFoundError(
            f"Raster tiles not found for prefix: {raster_filename}. Make sure the GEE export finished."
        )

    # 2. Create a temporary Virtual Raster (VRT) to merge tiles dynamically
    vrt_path = os.path.join(output_dir, "merged_changes.vrt")
    files_str = " ".join([f'"{f}"' for f in raster_files])
    os.system(f"gdalbuildvrt {vrt_path} {files_str}")

    # 3. Calculate pixel size for scale bar
    pixel_size_km = compute_display_pixel_size_km(
        raster_path=vrt_path,
        downsample_factor=scale_factor
    )

    # 4. Read raster and basic metadata with downsampling
    with rasterio.open(vrt_path) as src:
        out_shape = (
            int(src.height * scale_factor),
            int(src.width * scale_factor),
        )
        data = src.read(
            1,
            out_shape=out_shape,
            resampling=rasterio.enums.Resampling.nearest,
        )

        # Force masking using the provided nodata value
        data = np.ma.masked_equal(data, nodata_val)

        src_crs = src.crs
        # Adjust the affine transform for the new downsampled resolution
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[1]),
            (src.height / data.shape[0]),
        )
        height, width = data.shape

    # 5. Figure
    fig, ax = plt.subplots(
        figsize=(20, 10), 
        dpi=300
    )

    # Data range
    min_val = int(np.ma.min(data))
    max_val = int(np.ma.max(data))

    # 6. Colormap (gray for 0 + jet/viridis for 1..max)
    original_cmap = plt.get_cmap("viridis_r")
    color_list = ["#c0c0c0"] + [
        original_cmap(i) for i in np.linspace(0, 1, max_val)
    ]
    cmap = ListedColormap(color_list)

    # Discrete normalization
    bounds = np.arange(min_val, max_val + 2) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    # 7. Plot raster
    ax.imshow(
        data,
        cmap=cmap,
        interpolation="nearest",
        norm=norm
    )

    # 8. Discrete box legend
    legend_elements = []
    present_values = np.unique(data.compressed())

    for i in range(min_val, max_val + 1):
        if i in present_values:
            legend_elements.append(
                Patch(
                    facecolor=cmap(norm(i)),
                    edgecolor="none",
                    linewidth=0,
                    label=str(i),
                )
            )

    ax.legend(
        handles=legend_elements,
        title="Changes",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=10,
        title_fontsize=12,
        alignment="left",
        handlelength=2.0,
        handleheight=1.5,
    )

    # 9. Cartographic elements
    # If the X-axis is in degrees (max value <= 180), use 1 degree in meters.
    # Otherwise, use the pixel size in meters.
    degree_in_meters = 111320.0
    dx_meters = degree_in_meters if ax.get_xlim()[1] <= 180.5 else (pixel_size_km * 1000)

    # Custom formatter to prevent 'Mm' (Megameters) and force 'km'
    def km_formatter(value, unit):
        if unit == "Mm":
            return f"{int(value * 1000)} km"
        return f"{int(value)} {unit}"

    scalebar = ScaleBar(
        dx=dx_meters,
        units="m",
        length_fraction=0.15,
        location="lower left",
        box_alpha=0.6,
        scale_formatter=km_formatter
    )
    ax.add_artist(scalebar)

    # Note: ensure north_arrow function exists in your utils.py
    try:
        north_arrow(
            ax,
            location="upper right",
            shadow=False,
            rotation={"degrees": 0},
            scale=0.5,
        )
    except NameError:
        print("north_arrow function not found. Skipping north arrow.")

    # 10. Axes styling
    ax.set_title(
        "Number of Changes",
        fontsize=18,
        pad=10
    )
    
    ax.set_aspect("equal")

    to_latlon = Transformer.from_crs(
        src_crs,
        "EPSG:4326",
        always_xy=True
    )

    def format_lon(x, pos):
        x = np.clip(x, 0, width - 1)
        x_proj, y_proj = rasterio.transform.xy(transform, height // 2, x)
        lon, _ = to_latlon.transform(x_proj, y_proj)
        return f"{lon:.1f}°"

    def format_lat(y, pos):
        y = np.clip(y, 0, height - 1)
        x_proj, y_proj = rasterio.transform.xy(transform, y, width // 2)
        _, lat = to_latlon.transform(x_proj, y_proj)
        return f"{lat:.1f}°"

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=10,
        pad=4
    )
    plt.setp(
        ax.get_yticklabels(),
        rotation=90,
        va="center"
    )

    # 11. Save and Show
    maps_dir = os.path.join(
        output_dir,
        "maps"
    )
    os.makedirs(
        maps_dir,
        exist_ok=True
    )
    output_figure_path = os.path.join(
        maps_dir,
        "map_number_of_changes.png"
    )

    plt.savefig(
        output_figure_path,
        dpi=300,
        bbox_inches="tight",
        format="png",
        pad_inches=0.5,
    )
    plt.show()
    print(f"Map figure saved successfully to: {output_figure_path}")


###############################################################################
#                                                                             #
#                  5. TRAJECTORY ANALYSIS                                     #
#                                                                             #
###############################################################################

# ---------------------------------------------------------------------------
# 5.1 EXPORT TRAJECTORY RASTER
# ---------------------------------------------------------------------------
def export_trajectory_task_gee(
    year_list: list,
    drive_folder: str,
    scale: int,
    collection_id: str,
    band_name: str,
    nodata_val: int,
    full_year_list: list = None,
) -> ee.batch.Task:
    """
    Generate the trajectory raster and submit an export task to Google Earth Engine.
    """
    if GLOBAL_GEOM is None:
        raise ValueError(
            "GLOBAL_GEOM is not initialized. Please call "
            "utils.initialize_active_region(region_code) before running tasks."
        )

    if full_year_list is None:
        full_year_list = year_list

    # 1. Combine lists to build a single stack containing all required years safely
    combined_years = sorted(list(set(year_list) | set(full_year_list)))

    # 2. Build the master stack and extract the global mask based on the FULL timeline
    master_stack, master_band_names = build_glance_stack(
        year_list=combined_years,
        collection_id=collection_id,
        band_name=band_name,
        nodata_val=nodata_val,
    )
    
    # Create the global validity mask strictly using full_year_list bands
    full_year_bands = [f"y{y}" for y in full_year_list]
    full_stack_subset = master_stack.select(full_year_bands)
    global_mask = full_stack_subset.neq(nodata_val).unmask(0).reduce(ee.Reducer.min())

    # Build target bands list for trajectory calculation
    target_band_names = [f"y{y}" for y in year_list]

    # 3. Generate the trajectory classification image using the consistent global mask
    trajectory_image = calculate_trajectory_gee(
        master_stack,
        target_band_names,
        global_mask,
        nodata_val,
    )

    # 3. Apply NoData unmasking to match the project's standard
    trajectory_image = trajectory_image.unmask(nodata_val).toUint8()

    # 4. Define the Earth Engine export task
    task_desc = f"Trajectory_Analysis_{year_list[0]}_{year_list[-1]}"
    task = ee.batch.Export.image.toDrive(
        image=trajectory_image,
        description=task_desc,
        folder=drive_folder,
        scale=scale,
        region=GLOBAL_GEOM,
        maxPixels=1e13,
        crs="EPSG:4326",
    )

    # 5. Start the export task
    task.start()
    print(f"Task '{task_desc}' submitted to Google Earth Engine.")

    return task


# ---------------------------------------------------------------------------
# 5.2 EXPORT TRAJECTORY DURING INTERVALS
# ---------------------------------------------------------------------------
def export_trajectory_intervals_csv_gee(
    year_list: list,
    drive_folder: str,
    scale: int = 300,
    full_year_list: list = None,
) -> ee.batch.Task:
    """
    Compute trajectory interval contributions using GEE and export to CSV.
    Returns pixel counts instead of area.
    """
    if GLOBAL_GEOM is None:
        raise ValueError(
            "GLOBAL_GEOM is not initialized. Please call "
            "utils.initialize_active_region(region_code) before running tasks."
        )

    print(f"Preparing Trajectory Contributions GEE Task for {year_list[0]}-{year_list[-1]}...")

    if full_year_list is None:
        full_year_list = year_list

    # 1. Combine lists to build a single stack containing all required years safely
    combined_years = sorted(list(set(year_list) | set(full_year_list)))

    # 2. Build the master stack and extract the global mask based on the FULL timeline
    master_stack, master_band_names = build_glance_stack(
        year_list=combined_years,
        collection_id=GLANCE_COLLECTION_ID,
        band_name=GLANCE_CLASS_BAND,
        nodata_val=NODATA_VALUE,
    )
    
    # Create the global validity mask strictly using full_year_list bands
    full_year_bands = [f"y{y}" for y in full_year_list]
    full_stack_subset = master_stack.select(full_year_bands)
    global_mask = full_stack_subset.neq(NODATA_VALUE).unmask(0).reduce(ee.Reducer.min())

    # Build target bands list for trajectory calculation
    target_band_names = [f"y{y}" for y in year_list]
    
    # 3. Calculate trajectory using the consistent global mask
    trajectory_image = calculate_trajectory_gee(master_stack, target_band_names, global_mask, NODATA_VALUE)

    # 3. Filter valid trajectories (we only care about 2, 3, 4, 5)
    valid_traj_mask = trajectory_image.gte(2).And(trajectory_image.lte(5))
    trajectory_image = trajectory_image.updateMask(valid_traj_mask)

    # 4. Build a single multi-band image where each band represents an interval's masked trajectory
    interval_images = []
    for i in range(len(year_list) - 1):
        y_start = year_list[i]
        y_end = year_list[i + 1]
        band_label = f"i_{y_start}_{y_end}"
        
        img1 = master_stack.select(f"y{y_start}").rename(GLANCE_CLASS_BAND)
        img2 = master_stack.select(f"y{y_end}").rename(GLANCE_CLASS_BAND)
        change_mask = img1.neq(img2)
        
        traj_for_interval = trajectory_image.updateMask(change_mask).rename(band_label)
        interval_images.append(traj_for_interval)
        
    combined_intervals_image = ee.Image(interval_images)

    # 5. Run a SINGLE parallel reduction over all bands (massively faster!)
    histograms = combined_intervals_image.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram().unweighted(),
        geometry=GLOBAL_GEOM,
        scale=scale,
        crs="EPSG:4326",
        maxPixels=1e13,
        tileScale=16,
    )

    # 6. Parse the single dictionary server-side into a FeatureCollection
    length = len(year_list)
    indices = ee.List.sequence(0, length - 2)

    def process_interval(idx):
        idx = ee.Number(idx)
        y_list = ee.List(year_list)
        y_start = ee.Number(y_list.get(idx)).format('%d')
        y_end = ee.Number(y_list.get(idx.add(1))).format('%d')
        interval_label = y_start.cat('-').cat(y_end)
        band_label = ee.String("i_").cat(y_start).cat("_").cat(y_end)

        # Get the histogram dictionary for this specific interval band
        hist_dict = ee.Dictionary(histograms.get(band_label, {}))

        # Return as Feature (row for the CSV)
        return ee.Feature(None, {
            'Interval': interval_label,
            '2': ee.Number(hist_dict.get('2', 0)),
            '3': ee.Number(hist_dict.get('3', 0)),
            '4': ee.Number(hist_dict.get('4', 0)),
            '5': ee.Number(hist_dict.get('5', 0)),
        })

    # 7. Map over the intervals
    features = ee.FeatureCollection(indices.map(process_interval))

    # 6. Prepare the CSV Export task
    y_start = str(year_list[0])
    y_end = str(year_list[-1])
    task_desc = f"Trajectory_Contributions_{y_start}_{y_end}"
    
    task = ee.batch.Export.table.toDrive(
        collection=features,
        description=task_desc,
        folder=drive_folder,
        fileNamePrefix=task_desc,
        fileFormat="CSV"
    )

    # 7. Start the task
    task.start()
    print(f"Task '{task_desc}' submitted to Google Earth Engine.")
    
    return task

# ---------------------------------------------------------------------------
# 5.3 PLOT TRAJECTORY DURING INTERVALS
# ---------------------------------------------------------------------------
def plot_trajectory_contributions(
    input_dir: str,
    output_dir: str,
) -> None:
    """
    Create stacked bar chart for trajectory contributions per interval,
    caching the aggregated data to a consolidated CSV file.
    """
    tables_dir = os.path.join(output_dir, "tables")
    consolidated_csv_path = os.path.join(tables_dir, "aggregated_trajectory_contributions.csv")

    df = None

    # 1. Try to load from consolidated CSV first (cache)
    if os.path.exists(consolidated_csv_path):
        print(f"Loading consolidated trajectory contributions from: {consolidated_csv_path}")
        df = pd.read_csv(consolidated_csv_path, index_col="Interval")
    else:
        # 2. Locate the raw Trajectory Contributions CSV
        search_pattern = os.path.join(input_dir, "Trajectory_Contributions_*.csv")
        csv_files = glob.glob(search_pattern)

        if csv_files:
            raw_csv_path = csv_files[0]
            print(f"Processing raw trajectory contributions from: {raw_csv_path}")
            df_temp = pd.read_csv(raw_csv_path)

            # Set 'Interval' as index if present, else fallback
            interval_col = None
            for col in df_temp.columns:
                if col.lower() == 'interval':
                    interval_col = col
                    break
            
            if interval_col:
                df_temp.set_index(interval_col, inplace=True)
            else:
                non_num_cols = df_temp.select_dtypes(exclude=['number']).columns
                if len(non_num_cols) > 0:
                    df_temp.set_index(non_num_cols[0], inplace=True)
                    df_temp.index.name = "Interval"

            # Keep only columns representing valid trajectories 2, 3, 4, 5
            cols_to_keep = [c for c in df_temp.columns if c.isdigit() and int(c) in [2, 3, 4, 5]]
            df = df_temp[cols_to_keep].copy()
            
            # Round and convert counts to strictly integer
            df = df.round(0).astype(int)

            # Save consolidated copy to tables/
            os.makedirs(tables_dir, exist_ok=True)
            df.to_csv(consolidated_csv_path)
            print(f"Consolidated trajectory contributions saved to: {consolidated_csv_path}")

    if df is None or df.empty:
        print(f"No valid GEE CSV data found in {input_dir} and no consolidated CSV found at {consolidated_csv_path}")
        return

    # Ensure columns are integers to match plotting logic
    df.columns = df.columns.astype(int)

    # Calculate the maximum value to determine scale factor
    max_val = df.sum(axis=1).max()

    if max_val >= 1_000_000_000_000:
        scale_factor = 1_000_000_000_000
        y_label = "Change (trillion pixels)"
    elif max_val >= 1_000_000_000:
        scale_factor = 1_000_000_000
        y_label = "Change (billion pixels)"
    elif max_val >= 1_000_000:
        scale_factor = 1_000_000
        y_label = "Change (million pixels)"
    elif max_val >= 1_000:
        scale_factor = 1_000
        y_label = "Change (thousand pixels)"
    else:
        scale_factor = 1
        y_label = "Change (pixels)"

    # Apply scaling
    df_scaled = df / scale_factor

    # 2. Define colors and stacking order
    colors = {
        2: "#990033",
        3: "#FDE725",
        4: "#ff9900",
        5: "#000066"
    }

    # Stacking order: 5 (bottom), 4, 3, 2 (top)
    stack_order = [5, 4, 3, 2]

    # 3. Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # 4. Plot stacked bars
    bottom = pd.Series(0.0, index=df_scaled.index)

    for traj_id in stack_order:
        if traj_id in df_scaled.columns:
            values = df_scaled[traj_id]
            ax.bar(
                df_scaled.index,
                values,
                label=f"{traj_id}",
                bottom=bottom,
                color=colors[traj_id],
                edgecolor="none",
                width=0.9,
            )
            bottom += values

    # 5. Customize axes and labels
    ax.set_ylabel(
        y_label,
        fontsize=18
    )
    ax.set_title(
        "Trajectories during Time Intervals",
        fontsize=20,
        pad=15
    )
    # X-Axis formatting: Horizontal labels
    ax.tick_params(
        axis="x",
        labelsize=18,
        rotation=90
    )
    # Y-Axis formatting (mticker)
    ax.tick_params(
        axis="y",
        labelsize=18
    )
    ax.yaxis.set_major_locator(
        ticker.MaxNLocator(
            nbins=5,
            integer=True,
        ),
    )
    max_scaled_val = df_scaled.sum(axis=1).max()
    ax.yaxis.set_major_locator(
        ticker.MaxNLocator(
            nbins=5,
            integer=max_scaled_val >= 5,
        ),
    )

    # Spines visible, NO GRID
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)

    # 6. Legend
    handles, labels = ax.get_legend_handles_labels()

    if handles:
        # Reorder handles to match 2, 3, 4, 5
        legend_order_map = {"2": 0, "3": 1, "4": 2, "5": 3}

        # Sort handles based on labels
        sorted_pairs = sorted(
            zip(handles, labels),
            key=lambda x: legend_order_map.get(x[1], 99),
        )
        sorted_handles, sorted_labels = zip(*sorted_pairs)

        ax.legend(
            sorted_handles,
            sorted_labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            title="Trajectory",
            title_fontsize=14,
            alignment="left",
            fontsize=14,
            frameon=False,
        )

    plt.tight_layout()

    # 7. Save figure
    charts_dir = os.path.join(
        output_dir,
        "charts"
    )
    os.makedirs(
        charts_dir,
        exist_ok=True
    )
    output_fig = os.path.join(
        charts_dir,
        "graphic_trajectory_time_interval.png"
    )
    plt.savefig(
        output_fig,
        dpi=300,
        bbox_inches="tight",
        format="png"
    )
    plt.show()

    print(f"Figure saved to: {output_fig}")

# ---------------------------------------------------------------------------
# 5.4 EXPORT TRAJECTORY OVERALL
# ---------------------------------------------------------------------------

def export_trajectory_overall_csv_gee(
    year_list: list,
    drive_folder: str,
    scale: int = 300,
    full_year_list: list = None,
) -> ee.batch.Task:
    """
    Compute overall trajectory contributions using GEE and export to CSV.

    Parameters
    ----------
    year_list : list
        List of integer years to process.
    drive_folder : str
        The destination folder in Google Drive.
    scale : int, optional
        The spatial resolution for the export in meters. Default is 300.
        
    Returns
    -------
    ee.batch.Task
        The submitted Earth Engine task object.
    """
    if GLOBAL_GEOM is None:
        raise ValueError(
            "GLOBAL_GEOM is not initialized. Please call "
            "utils.initialize_active_region(region_code) before running tasks."
        )

    if full_year_list is None:
        full_year_list = year_list

    # 1. Combine lists to build a single stack containing all required years safely
    combined_years = sorted(list(set(year_list) | set(full_year_list)))

    # 2. Build the master stack and extract the global mask based on the FULL timeline
    master_stack, master_band_names = build_glance_stack(
        year_list=combined_years,
        collection_id=GLANCE_COLLECTION_ID,
        band_name=GLANCE_CLASS_BAND,
        nodata_val=NODATA_VALUE,
    )
    
    # Create the global validity mask strictly using full_year_list bands
    full_year_bands = [f"y{y}" for y in full_year_list]
    full_stack_subset = master_stack.select(full_year_bands)
    global_mask = full_stack_subset.neq(NODATA_VALUE).unmask(0).reduce(ee.Reducer.min())

    # Build target bands list for trajectory calculation
    target_band_names = [f"y{y}" for y in year_list]
    
    # 3. Calculate overall trajectory using the consistent global mask
    trajectory_image = calculate_trajectory_gee(master_stack, target_band_names, global_mask, NODATA_VALUE)

    # Ensure the final trajectory image is strictly masked by the same global validity mask before reducing
    trajectory_image_masked = trajectory_image.updateMask(global_mask)

    # 4. Use frequencyHistogram to get counts for ALL categories (including 1) for a consistent denominator
    histograms = trajectory_image_masked.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram().unweighted(),
        geometry=GLOBAL_GEOM,
        scale=scale,
        crs="EPSG:4326",
        maxPixels=1e13,
        tileScale=16,
    )

    y_start = str(year_list[0])
    y_end = str(year_list[-1])
    period_label = f"{y_start}-{y_end}"

    hist_dict = ee.Dictionary(histograms.get('trajectory', ee.Dictionary()))

    # 5. Format into a Feature
    feature = ee.Feature(None, {
        'Period': period_label,
        '1': ee.Number(hist_dict.get('1', 0)),
        '2': ee.Number(hist_dict.get('2', 0)),
        '3': ee.Number(hist_dict.get('3', 0)),
        '4': ee.Number(hist_dict.get('4', 0)),
        '5': ee.Number(hist_dict.get('5', 0)),
    })

    fc = ee.FeatureCollection([feature])

    # 6. Prepare the CSV Export task
    task_desc = f"Trajectory_Overall_{y_start}_{y_end}"
    
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=task_desc,
        folder=drive_folder,
        fileNamePrefix=task_desc,
        fileFormat="CSV"
    )

    # 7. Start the task
    task.start()
    print(f"Task '{task_desc}' submitted to Google Earth Engine.")
    
    return task


# ---------------------------------------------------------------------------
# 5.5 PLOT TRAJECTORY OVERALL
# ---------------------------------------------------------------------------
def plot_trajectory_distribution(
    input_dir: str,
    output_dir: str,
    csv_filename: Optional[str] = None,
    total_pixels: int = 0,
) -> None:
    """
    Generate and save a stacked bar chart of overall trajectory distributions.

    This function processes GEE trajectory count CSV files, aggregates the
    counts, and creates a stacked bar chart showing the percentage of each
    trajectory (2, 3, 4, 5) relative to a given or calculated total study area.
    It automatically looks for Pixel_Counts_LULC_*.csv files in the input
    directory to establish the universal study area denominator.
    It caches the consolidated data in a local subdirectory for optimized
    subsequent calls.

    Parameters
    ----------
    input_dir : str
        Path to the directory containing raw GEE-exported CSV files.
    output_dir : str
        Path to the root output directory where plots and cached tables will be saved.
    csv_filename : str, optional
        Optional specific raw CSV filename to target. If None, the function
        dynamically searches using a predefined pattern. Default is None.
    total_pixels : int, optional
        The total number of valid pixels in the study area. This is used as the
        universal denominator for calculating percentages. If 0 or negative,
        the function attempts to derive the denominator from local Pixel_Counts
        CSVs or local sums. Default is 0.

    Returns
    -------
    None
        Generates a matplotlib plot and saves it as a PNG file.
    """
    tables_dir = os.path.join(
        output_dir,
        "tables",
    )
    consolidated_csv_path = os.path.join(
        tables_dir,
        "aggregated_trajectory_overall.csv",
    )

    df_overall = None

    # 1. Try to load from consolidated cached CSV first
    if os.path.exists(consolidated_csv_path):
        print(f"Loading consolidated overall trajectory from: {consolidated_csv_path}")
        df_overall = pd.read_csv(
            consolidated_csv_path,
            index_col=0,
        )
        df_overall.columns = df_overall.columns.astype(str)
    else:
        # 2. Locate the raw Trajectory Overall CSV
        if csv_filename:
            csv_files = [
                os.path.join(
                    input_dir,
                    csv_filename,
                )
            ]
        else:
            search_pattern = os.path.join(
                input_dir,
                "Trajectory_Overall_*.csv",
            )
            csv_files = glob.glob(search_pattern)

        if csv_files:
            raw_csv_path = csv_files[0]
            print(f"Processing raw overall trajectory from: {raw_csv_path}")
            df_temp = pd.read_csv(raw_csv_path)

            # Keep columns representing all trajectories (1 to 5)
            cols_to_keep = [
                c for c in df_temp.columns
                if c.isdigit() and int(c) in [1, 2, 3, 4, 5]
            ]
            df_overall = df_temp[cols_to_keep].copy()

            # Round and convert counts to strictly integer
            df_overall = df_overall.round(0).astype(int)

            # Save consolidated copy to tables cache
            os.makedirs(
                tables_dir,
                exist_ok=True,
            )
            df_overall.to_csv(consolidated_csv_path)
            print(f"Consolidated overall trajectory saved to: {consolidated_csv_path}")

    if df_overall is None or df_overall.empty:
        print(f"No valid GEE CSV data found in {input_dir}")
        return

    # 3. Parse and extract all valid trajectory columns (1 to 5) to establish the correct denominator
    all_traj_cols = [
        c for c in df_overall.columns
        if c.isdigit() and int(c) in [1, 2, 3, 4, 5]
    ]
    df_numeric_all = df_overall[all_traj_cols].copy().astype(float)
    df_numeric_all.columns = [str(int(float(c))) for c in df_numeric_all.columns]

    # Total valid study area denominator (sum of all trajectories 1, 2, 3, 4, 5)
    total_valid_pixels = float(df_numeric_all.iloc[0].sum())

    if total_valid_pixels == 0:
        print("Error: Total valid pixels for denominator is 0. Cannot plot.")
        return

    # Target columns representing changes (2 to 5) for percentage calculations
    traj_cols = [
        c for c in df_overall.columns
        if c.isdigit() and int(c) in [2, 3, 4, 5]
    ]
    df_numeric_changes = df_overall[traj_cols].copy()
    df_numeric_changes.columns = [str(int(float(c))) for c in df_numeric_changes.columns]

    # Calculate percentages relative to the total valid study area
    percentages = {
        i: float((df_numeric_changes[str(i)].iloc[0] / total_valid_pixels) * 100.0)
        if str(i) in df_numeric_changes.columns else 0.0
        for i in [2, 3, 4, 5]
    }

    ordered_trajs = [5, 4, 3, 2]
    colors = {
        5: "#000066",
        4: "#ff9900",
        3: "#FDE724",
        2: "#990033",
    }

    # 4. Generate the plot
    fig, ax = plt.subplots(
        figsize=(6, 6),
    )

    bottom = 0.0
    for traj in ordered_trajs:
        val = percentages[traj]
        ax.bar(
            0,
            val,
            bottom=bottom,
            color=colors[traj],
            width=0.4,
            edgecolor="none",
        )
        bottom += val

    # 5. Format axes, labels, and boundaries
    ax.set_ylabel(
        "Change (% of study area)",
        fontsize=16,
    )
    ax.set_title(
        "Trajectories Overall",
        fontsize=18,
        pad=15,
    )

    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(0.5)

    ax.tick_params(
        axis="y",
        which="major",
        labelsize=18,
    )
    ax.set_xticks([])

    # Dynamically scale Y-axis to match the total percentage of changes in the study area
    max_y = bottom * 1.05 if bottom > 0 else 1.0
    ax.set_ylim(
        0,
        max_y,
    )

    ax.yaxis.set_major_locator(
        mticker.MaxNLocator(
            integer=True,
            nbins=10,
        )
    )

    # 6. Configure legend and layout alignments
    legend_elements = [
        Patch(facecolor=colors[2], label="2"),
        Patch(facecolor=colors[3], label="3"),
        Patch(facecolor=colors[4], label="4"),
        Patch(facecolor=colors[5], label="5"),
    ]

    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        title="Trajectory",
        title_fontsize=14,
        alignment="left",
        fontsize=14,
        frameon=False,
    )

    fig.subplots_adjust(
        left=0.15,
        right=0.75,
        bottom=0.1,
        top=0.9,
    )

    # 7. Save output figure
    charts_dir = os.path.join(
        output_dir,
        "charts",
    )
    os.makedirs(
        charts_dir,
        exist_ok=True,
    )

    out_fig_path = os.path.join(
        charts_dir,
        "graphic_trajectory_percentage_overall.png",
    )
    plt.savefig(
        out_fig_path,
        dpi=300,
        bbox_inches="tight",
        format="png",
    )
    plt.show()
    print(f"Figure saved to: {out_fig_path}")


# ---------------------------------------------------------------------------
# 5.5 PLOT TRAJECTORY MAP
# ---------------------------------------------------------------------------
def plot_trajectory_map(
    output_dir: str,
    raster_filename: str = "Trajectory_Analysis",
    scale_factor: float = 0.05,
    nodata_val: int = 255,
) -> None:
    """
    Plot the Trajectory raster map with cartographic elements,
    using an optimized downsampling approach.

    Parameters
    ----------
    output_dir : str
        Directory containing the exported GEE tiles and where the map will be saved.
    raster_filename : str, optional
        Prefix of the raster tiles to plot.
    scale_factor : float, optional
        Scale factor to downsample the massive global raster to fit into memory.
    nodata_val : int, optional
        Value representing NoData in the raster to be masked out.
    """

    # 1. Locate all raster tiles exported by GEE
    raster_files = glob.glob(
        os.path.join(
            output_dir,
            f"{raster_filename}*.tif"
        )
    )
    if not raster_files:
        raise FileNotFoundError(
            f"Raster tiles not found for prefix: {raster_filename}. Make sure the GEE export finished."
        )

    # 2. Create a temporary Virtual Raster (VRT) to merge tiles dynamically
    vrt_path = os.path.join(output_dir, "merged_trajectory.vrt")
    files_str = " ".join([f'"{f}"' for f in raster_files])
    os.system(f"gdalbuildvrt {vrt_path} {files_str}")

    # 3. Calculate pixel size for scale bar
    # Make sure compute_display_pixel_size_km is available in utils.py
    pixel_size_km = compute_display_pixel_size_km(
        raster_path=vrt_path,
        downsample_factor=scale_factor
    )

    # 4. Read raster and basic metadata with downsampling
    with rasterio.open(vrt_path) as src:
        out_shape = (
            max(1, int(src.height * scale_factor)),
            max(1, int(src.width * scale_factor)),
        )
        data = src.read(
            1,
            out_shape=out_shape,
            resampling=rasterio.enums.Resampling.nearest,
        )

        # Force masking using the provided nodata value
        data = np.ma.masked_equal(
            data,
            nodata_val
        )

        src_crs = src.crs
        # Adjust the affine transform for the new downsampled resolution
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[1]),
            (src.height / data.shape[0]),
        )
        height, width = data.shape

    # 5. Figure
    fig, ax = plt.subplots(
        figsize=(20, 10),
        dpi=300
    )

    # 6. Colormap
    cmap = ListedColormap(
        [
            "#d9d9d9",  # Trajectory 1
            "#990033",  # Trajectory 2
            "#FDE725",  # Trajectory 3
            "#ff9900",  # Trajectory 4
            "#000066",  # Trajectory 5
        ]
    )
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = BoundaryNorm(bounds, cmap.N)

    # 7. Plot raster
    ax.imshow(
        data,
        cmap=cmap,
        interpolation="nearest",
        norm=norm
    )

    # 8. Legend
    legend_elements = [
        Patch(
            facecolor="#d9d9d9",
            label="1 All stable",
            edgecolor="black",
            linewidth=0,
        ),
        Patch(
            facecolor="#990033",
            label="2 Extent stable with alternation",
            edgecolor="black",
            linewidth=0,
        ),
        Patch(
            facecolor="#FDE725",
            label="3 Extent change without alternation",
            edgecolor="black",
            linewidth=0,
        ),
        Patch(
            facecolor="#ff9900",
            label=(
                "4 Extent change with alternation\n"
                "   where extent transition matches\n"
                "   a time interval transition"
            ),
            edgecolor="black",
            linewidth=0,
        ),
        Patch(
            facecolor="#000066",
            label=(
                "5 Extent change with alternation shift\n"
                "   where extent transition differs\n"
                "   from all time interval transitions"
            ),
            edgecolor="black",
            linewidth=0,
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=14,
        borderpad=1.2,
        title="Trajectory",
        title_fontsize=14,
        alignment="left",
        handletextpad=0.8,
        columnspacing=2,
        labelspacing=0.8,
        handlelength=2.0,
        handleheight=1.5,
    )

    # 9. Cartographic elements
    scalebar = ScaleBar(
        dx=pixel_size_km,
        units="km",
        length_fraction=0.15,
        location="lower left",
        box_alpha=0.6,
        scale_formatter=lambda value, unit: f"{int(value * 1000)} km" if "M" in unit else f"{int(value)} km"
    )
    ax.add_artist(scalebar)

    try:
        # Ensure north_arrow is also in utils
        north_arrow(
            ax,
            location="upper right",
            shadow=False,
            rotation={"degrees": 0},
            scale=0.5,
        )
    except NameError:
        print("north_arrow function not found. Skipping north arrow.")

    # 10. Axes styling
    ax.set_title(
        "Trajectories",
        fontsize=18, 
        pad=10
    )
    ax.set_aspect("equal")

    to_latlon = Transformer.from_crs(
        src_crs,
        "EPSG:4326",
        always_xy=True
    )

    def format_lon(x, pos):
        x = np.clip(x, 0, width - 1)
        x_proj, y_proj = rasterio.transform.xy(transform, height // 2, x)
        lon, _ = to_latlon.transform(x_proj, y_proj)
        return f"{lon:.1f}°"

    def format_lat(y, pos):
        y = np.clip(y, 0, height - 1)
        x_proj, y_proj = rasterio.transform.xy(transform, y, width // 2)
        _, lat = to_latlon.transform(x_proj, y_proj)
        return f"{lat:.1f}°"

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=10,
        pad=4
    )
    plt.setp(
        ax.get_yticklabels(),
        rotation=90,
        va="center"
    )

    # 11. Save and Show
    maps_dir = os.path.join(
        output_dir,
        "maps"
    )
    os.makedirs(
        maps_dir,
        exist_ok=True
    )
    output_figure_path = os.path.join(
        maps_dir,
        "map_trajectories.png"
    )

    plt.savefig(
        output_figure_path,
        dpi=300,
        bbox_inches="tight",
        format="png",
        pad_inches=0.5,
    )
    plt.show()
    print(f"Map figure saved successfully to: {output_figure_path}")


###############################################################################
#                                                                             #
#                  6. CHANGE COMPONENT ANALYSIS                               #
#                                                                             #
###############################################################################

# ---------------------------------------------------------------------------
# 6.1 COMPUTE TRANSITION MATRICES
# ---------------------------------------------------------------------------

def export_global_transition_tasks(
    year_list,
    drive_folder="GLANCE_Transitions",
    scale=30,
    full_year_list: list = None,
):
    """
    Triggers asynchronous GEE tasks to export global transition matrices.

    Each task computes a frequency histogram for a year pair and saves 
    the result as a CSV file in a specific Google Drive folder.

    Parameters
    ----------
    year_list : list of int
        List of years to process (e.g., [2001, 2010, 2019]).
    drive_folder : str, optional
        Name of the folder in Google Drive to save the CSVs. 
        Defaults to "GLANCE_Transitions".
    scale : int, optional
        Spatial resolution in meters. Use 30 for native resolution. 
        Defaults to 30.

    Returns
    -------
    list of ee.batch.Task
        A list of triggered Earth Engine Task objects for monitoring.
    """
    if GLOBAL_GEOM is None:
        raise ValueError(
            "GLOBAL_GEOM is not initialized. Please call "
            "utils.initialize_active_region(region_code) before running tasks."
        )

    # Initialize the GEE ImageCollection
    collection = ee.ImageCollection(GLANCE_COLLECTION_ID)

    # 1. Define transition pairs (consecutive intervals)
    pairs = [
        (year_list[i], year_list[i+1]) 
        for i in range(len(year_list) - 1)
    ]
    
    # 2. Add long-term transition pair (First Year to Last Year)
    if len(year_list) > 2:
        pairs.append(
            (year_list[0], year_list[-1])
        )
    
    triggered_tasks = []

    if full_year_list is None:
        full_year_list = year_list

    # 1. Combine lists to build a single stack containing all required years safely
    combined_years = sorted(list(set(year_list) | set(full_year_list)))
    
    # 2. Build the master stack and extract the global mask based on the FULL timeline
    master_stack, master_band_names = build_glance_stack(
        year_list=combined_years,
        collection_id=GLANCE_COLLECTION_ID,
        band_name=GLANCE_CLASS_BAND,
        nodata_val=NODATA_VALUE,
    )
    
    # Create the global validity mask strictly using full_year_list bands
    full_year_bands = [f"y{y}" for y in full_year_list]
    full_stack_subset = master_stack.select(full_year_bands)
    global_mask = full_stack_subset.neq(NODATA_VALUE).unmask(0).reduce(ee.Reducer.min())

    # 3. Iterate through each pair to define and start export tasks
    for y1, y2 in pairs:
        label = f"transition_{y1}_{y2}"
        
        # Select images directly by year band name from the master stack and apply global mask
        img_start = master_stack.select(f"y{y1}").rename(GLANCE_CLASS_BAND).updateMask(global_mask)
        img_end = master_stack.select(f"y{y2}").rename(GLANCE_CLASS_BAND).updateMask(global_mask)

        # 5. Create transition image: (Start * 100) + End.
        # This now correctly includes stable pixels (e.g., 1*100 + 1 = 101)
        # because we are no longer pre-filtering for inequality.
        transition_image = img_start.multiply(100).add(img_end).rename("transition")

        # 6. Reduce the image to a frequency histogram (Table format)
        # The histogram will now contain codes for both transitions and stability.
        # Using a Feature to wrap the result for CSV export
        transition_stats = transition_image.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram().unweighted(),
            geometry=GLOBAL_GEOM,
            scale=scale,
            crs="EPSG:4326",
            maxPixels=1e13,
            tileScale=16,
        )

        # 7. Create a feature collection with the statistics for export
        feature = ee.Feature(None, transition_stats)
        fc = ee.FeatureCollection([feature])

        # 8. Configure the Batch Export Task to Google Drive
        task = ee.batch.Export.table.toDrive(
            collection=fc,
            description=label,
            folder=drive_folder,
            fileNamePrefix=label,
            fileFormat='CSV'
        )

        # 9. Start the task on the server and store the object
        task.start()
        triggered_tasks.append(task)
        print(f"Task started for {label} (Scale: {scale}m)")

    return triggered_tasks

# ---------------------------------------------------------------------------
# 6.2 COMPUTE AGGREGATION MATRICES
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 6.3 COMPUTE CHANGE COMPONENTS
# ---------------------------------------------------------------------------
def generate_change_components_table(
    year_list: list,
    input_dir: str,
    output_dir: str,
    class_labels_dict: dict,
) -> None:
    """
    Main execution function to process all interval and aggregate matrices and export to CSV.

    Parameters
    ----------
    year_list : list
        List of years to process.
    input_dir : str
        Path to the input directory containing matrices.
    output_dir : str
        Path to the output directory.
    class_labels_dict : dict
        Dictionary mapping class IDs to metadata.
    """
    all_results = []

    # 1. Process Annual Intervals
    for i in tqdm(range(len(year_list) - 1), desc="Processing interval matrices"):
        y_start = _extract_year_str(year_list[i])
        y_end = _extract_year_str(year_list[i + 1])

        all_results.extend(
            process_matrix(
                matrix_type="interval",
                input_dir=input_dir,
                years_list=year_list,
                class_labels_dict=class_labels_dict,
                start_year=y_start,
                end_year=y_end,
            )
        )

    # 2. Process Aggregated Matrices
    aggregate_types = [
        "extent",
        "sum",
        "alternation_exchange",
        "alternation_shift",
    ]
    for mtype in tqdm(
        aggregate_types,
        desc="Processing aggregated matrices"):
        all_results.extend(
            process_matrix(
                matrix_type=mtype,
                input_dir=input_dir,
                years_list=year_list,
                class_labels_dict=class_labels_dict,
            )
        )

    # 3. Save combined results
    if not all_results:
        print("\nNo matrices found to process.")
        return

    df_out = pd.DataFrame(all_results)

    tables_dir = os.path.join(
        output_dir,
        "tables"
    )
    os.makedirs(
        tables_dir,
        exist_ok=True
    )

    output_file = os.path.join(
        tables_dir,
        "change_components.csv"
    )
    df_out.to_csv(
        output_file,
        index=False
    )
    print(f"\nSuccess! Final components saved to: {output_file}")

# ---------------------------------------------------------------------------
# 6.4 HEAT MAPS
# ---------------------------------------------------------------------------

def validate_and_get_interval(
    years: List[int],
    output_path: str,
    class_labels_dict: Dict[int, str]
) -> str:
    """
    Validates global inputs and generates an interval string based on years.

    Parameters
    ----------
    years : list of int
        List of years to process.
    output_path : str
        Directory path for outputs.
    class_labels_dict : dict
        Dictionary mapping class values to labels.

    Returns
    -------
    str
        The formatted interval string (e.g., '2001-2019').

    Raises
    ------
    ValueError
        If any of the inputs are invalid or missing.
    """
    if not (isinstance(years, (list, tuple)) and len(years) >= 2):
        raise ValueError("`years` missing, invalid, or contains fewer than 2 elements.")
    
    if not (isinstance(output_path, str) and output_path):
        raise ValueError("`output_path` missing or invalid.")
        
    if not (isinstance(class_labels_dict, dict) and class_labels_dict):
        raise ValueError("`class_labels_dict` missing or invalid.")

    str_y0 = _extract_year_str(years[0])
    str_y1 = _extract_year_str(years[-1])

    return f"{str_y0}-{str_y1}"

def _extract_year_str(val: Union[str, int]) -> str:
    """
    Extract the first sequence of digits from a year string or integer.

    Parameters
    ----------
    val : str or int
        The value containing the year (e.g., "time_2000" or 2000).

    Returns
    -------
    str
        The extracted year digits.
    """
    match = re.search(r"(\d+)", str(val))
    return match.group(1) if match else str(val)

def load_square_matrix(csv_path: str) -> pd.DataFrame:
    """
    Load a square transition matrix from CSV and align row/column labels.

    Parameters
    ----------
    csv_path : str
        Path to a CSV file where the first column and the header row
        contain class IDs (or labels), and the remaining cells contain
        transition counts.

    Returns
    -------
    pd.DataFrame
        Square DataFrame with string labels on both rows and columns.
        When row and column labels differ, their union is used and
        missing cells are filled with 0.0.
    """
    df = pd.read_csv(csv_path, index_col=0)

    # Detect if this is a raw GEE dictionary CSV (contains a column with '{' and '=')
    is_raw = False
    for col in df.columns:
        if df[col].dropna().empty: # Check if column is empty before accessing iloc[0]
            continue
        val = str(df[col].iloc[0])
        if val.startswith("{") and "=" in val:
            is_raw = True
            break

    if is_raw:
        df = parse_gee_raw_csv(csv_path)

        # Drop GEE metadata columns and rows if they exist
        metadata_fields = {'system:index', '.geo'}
        cols_to_drop = [c for c in df.columns if str(c) in metadata_fields]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        rows_to_drop = [r for r in df.index if str(r) in metadata_fields]
        if rows_to_drop:
            df = df.drop(index=rows_to_drop)

    df.index = df.index.map(str)
    df.columns = df.columns.map(str)

    if list(df.index) != list(df.columns):
        # Safe sorting logic that handles both class IDs and class names
        name_to_id = {v['name']: k for k, v in GLANCE_METADATA.items()}
        for k, v in GLANCE_METADATA.items():
            if "rename" in v:
                name_to_id[v["rename"]] = k

        def safe_sort_key(label):
            label_str = str(label)
            if label_str in name_to_id:
                return (0, name_to_id[label_str])
            try:
                return (0, int(label_str))
            except (ValueError, TypeError):
                return (1, label_str)

        labels = sorted(
            set(df.index).union(df.columns),
            key=safe_sort_key,
        )
        df = df.reindex(
            index=labels,
            columns=labels
        ).fillna(0.0)

    if df.shape[0] != df.shape[1]:
        raise ValueError(f"Matrix not square after alignment: {csv_path}")

    return df

def label_id_to_name(
    labels: Iterable[str],
    class_labels_dict: dict,
) -> list[str]:
    """
    Map class ID strings to human-readable names using class_labels_dict.

    Parameters
    ----------
    labels : Iterable[str]
        Class IDs as strings (e.g. ["1", "2", "3"]).
    class_labels_dict : dict
        Dictionary containing metadata for each class ID.

    Returns
    -------
    list[str]
        List of class names mapped from IDs.
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

    names: list[str] = []
    for lab in labels:
        try:
            cid = int(str(lab))
            names.append(
                id_to_name.get(
                    cid,
                    str(lab),
                ),
            )
        except Exception:
            names.append(str(lab))

    return names

def compute_net_change_from_sum(df_sum: pd.DataFrame) -> pd.Series:
    """
    Compute net change per class from a SUM transition matrix.

    Parameters
    ----------
    df_sum : pd.DataFrame
        Square transition matrix representing total transitions over
        the full time span.

    Returns
    -------
    pd.Series
        Net change for each class (gains - losses).
    """
    m_vals = df_sum.values.astype(float).copy()
    np.fill_diagonal(m_vals, 0.0)

    gains = m_vals.sum(axis=0)
    losses = m_vals.sum(axis=1)
    net_change = gains - losses

    return pd.Series(net_change, index=df_sum.index)

def reorder_all_matrices(matrices_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Reorder rows and columns of all matrices using net change from sum.

    Parameters
    ----------
    matrices_dict : dict[str, pd.DataFrame]
        Dictionary containing all loaded DataFrames.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with all input DataFrames reindexed.
    """
    net_change = compute_net_change_from_sum(df_sum=matrices_dict["sum"])

    order_labels = net_change.sort_values(ascending=True).index.tolist()

    reordered = {}
    for k, df in matrices_dict.items():
        reordered[k] = df.reindex(
            index=order_labels,
            columns=order_labels,
        )

    return reordered


def annotate_heatmap(
    ax: plt.Axes,
    M: np.ndarray,
    fontsize: int = 8,
    show_diagonal: bool = True,
    equalize_diagonal_font: bool = False,
    vmax: Optional[float] = None,
    matrix_key: Optional[str] = None,
    diagonal_vmax: Optional[float] = None, # New parameter for diagonal-specific vmax
) -> None:
    """
    Annotate a heatmap with integer cell values and adaptive text color.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object where the heatmap is plotted.
    M : np.ndarray
        The matrix containing the values to display.
    fontsize : int, optional
        The font size of the annotations (default is 8).
    show_diagonal : bool, optional
        If True, annotates the diagonal in white.
        If False, skips the diagonal.
    equalize_diagonal_font : bool, optional
        If True, uses the same font size for all cells.
    vmax : float, optional
        The maximum scale value for off-diagonal values, used to determine text color threshold.
    matrix_key : str, optional
        The key identifying the matrix type (e.g., "sum", "ext") to control
        diagonal coloring and annotation behavior.
    diagonal_vmax : float, optional
        The maximum value among the diagonal entries, used to determine
        text color on grayscale diagonals.
    """
    if M.size == 0:
        return

    if vmax is None:
        m_off = M.copy()
        np.fill_diagonal(m_off, np.nan)
        finite_vals = m_off[np.isfinite(m_off)]
        vmax = float(np.nanmax(finite_vals)) if finite_vals.size > 0 else 1.0

    thresh_red = 0.6 * vmax  # 60% of vmax is considered red in YlOrRd
    rows, cols = M.shape

    for i in range(rows):
        for j in range(cols):
            val = float(M[i, j])

            if i == j:
                if not show_diagonal:
                    continue
                
                # Define matrices that should have a white diagonal with black text
                DIAGONAL_WHITE_MATRICES = ["qty_shift", "alt_shift"]

                if matrix_key in DIAGONAL_WHITE_MATRICES:
                    color = "black"
                    current_fontsize = fontsize # Use standard font size as requested
                elif diagonal_vmax is not None and diagonal_vmax > 0:
                    # Handle grayscale diagonal: adaptive text color
                    normalized_val = val / diagonal_vmax
                    color = "white" if normalized_val > 0.5 else "black" # Dark background -> white text
                    current_fontsize = max(5, fontsize - 4) # Make font smaller for grayscale
                else:
                    # Default case for solid black diagonal
                    color = "white"
                    current_fontsize = max(5, fontsize - 4) # Make font smaller for black
            else:
                if val >= thresh_red and val > 0:
                    color = "white"
                else:
                    color = "black"
                current_fontsize = fontsize

            txt = f"{int(round(val))}"

            ax.text(
                j, i, txt, ha="center", va="center",
                fontsize=current_fontsize, color=color, clip_on=True
            )

def _unit_label(suffix: str, base_label: str = "Pixels") -> str:
    """
    Build a descriptive label for the colorbar.

    Parameters
    ----------
    suffix : str
        The suffix for the unit (e.g., 'k', 'M').
    base_label : str, optional
        The base label text (default is "Pixels").

    Returns
    -------
    str
        The formatted unit label.
    """
    if suffix == "hundreds":
        suffix = ""

    mapping = {
        "": base_label,
        "k": f"{base_label} (thousands)",
        "M": f"{base_label} (millions)",
        "B": f"{base_label} (billions)",
        "T": f"{base_label} (trillions)",
    }
    return mapping.get(suffix, f"{base_label} ({suffix})")

def _unit_formatter(factor: float, decimals: int = 1, suffix: str = ""):
    """
    Build a tick formatter that scales values by a factor.

    Parameters
    ----------
    factor : float
        The factor to divide the values by (e.g., 1000 or 1000000).
    decimals : int, optional
        The number of decimal places to include (default is 1).
    suffix : str, optional
        The string suffix to append (default is "").

    Returns
    -------
    matplotlib.ticker.FuncFormatter
        A formatter function for the plot ticks.
    """
    fmt = f"{{:.{decimals}f}}{suffix}"
    def _fmt(x: float, pos: int) -> str:
        return fmt.format(x / factor)
    return mticker.FuncFormatter(_fmt)

def plot_heatmap(
    df: pd.DataFrame,
    title: str,
    save_path: str = None,
    figsize: tuple = None,
    cmap: str = "YlOrRd",
    vmin: float = 0.0,
    vmax: float = 50_000_000.0,
    rotate_xticks_deg: int = 90,
    cbar_label: str = "Number of pixels",
    annotate: bool = True,
    cell_size_inch: float = 0.8,
    tick_fontsize: int = None,
    ann_fontsize: int = 12,
    cbar_fraction: float = 0.025,
    cbar_pad: float = 0.02,
    tick_fontsize_x: int = None,
    tick_fontsize_y: int = None,
    axis_label_fontsize: int = None,
    title_fontsize: int = None,
    cbar_tick_labelsize: int = None,
    cbar_label_fontsize: int = 14,
    xlabel: str = "To class",
    ylabel: str = "From class",
    show_diagonal_values: bool = True,
    # This parameter is now controlled internally based on matrix_key
    equalize_diagonal_font: bool = False,
    matrix_key: Optional[str] = None,
) -> None:
    """
    Plot a square matrix as a heatmap with adaptive integer colorbar.

    Parameters
    ----------
    df : pd.DataFrame
        The square dataframe to plot.
    title : str
        The title of the plot.
    save_path : str, optional
        Path to save the figure image.
    figsize : tuple, optional
        Figure size in inches (width, height).
    cmap : str, optional
        Colormap name (default is "YlOrRd").
    vmin : float, optional
        Minimum value for colormap scaling.
    vmax : float, optional
        Maximum value for colormap scaling.
    rotate_xticks_deg : int, optional
        Rotation angle for x-axis ticks.
    cbar_label : str, optional
        Label for the colorbar (base unit, e.g., "Pixels").
    annotate : bool, optional
        Whether to annotate cells with values.
    cell_size_inch : float, optional
        Size of each cell in inches used for auto-figsize.
    tick_fontsize : int, optional
        Deprecated; use tick_fontsize_x/y.
    ann_fontsize : int, optional
        Font size for annotations.
    cbar_fraction : float, optional
        Fraction of original axes to use for colorbar.
    cbar_pad : float, optional
        Padding between axes and colorbar.
    tick_fontsize_x : int, optional
        Font size for x-axis ticks.
    tick_fontsize_y : int, optional
        Font size for y-axis ticks.
    axis_label_fontsize : int, optional
        Font size for axis labels.
    title_fontsize : int, optional
        Font size for the title.
    cbar_tick_labelsize : int, optional
        Font size for colorbar tick labels.
    cbar_label_fontsize : int, optional
        Font size for the colorbar label/title.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    show_diagonal_values : bool, optional
        If True, shows text on the black diagonal.
    equalize_diagonal_font : bool, optional
        If True, equalizes the font size of the diagonal.
    matrix_key : str, optional
        The key identifying the matrix type (e.g., "sum", "ext") to control diagonal coloring.

    Returns
    -------
    None
    """
    # 2. Validate font sizes
    if tick_fontsize_x is None or tick_fontsize_y is None:
        raise ValueError(
            "Set `tick_fontsize_x` and `tick_fontsize_y` explicitly.",
        )

    if axis_label_fontsize is None:
        axis_label_fontsize = 12

    if title_fontsize is None:
        title_fontsize = 14

    # 3. Prepare data and values
    labels = list(
        df.index,
    )
    matrix_values = df.values.astype(
        float,
    )

    # Force diagonal to 0 for Shift matrices for mathematical consistency
    if matrix_key in ["qty_shift", "alt_shift"]:
        np.fill_diagonal(matrix_values, 0.0)

    # 4. Prepare scale ignoring diagonal
    matrix_scale = matrix_values.copy()

    np.fill_diagonal(
        matrix_scale,
        0.0,
    )

    finite_vals = matrix_scale[
        np.isfinite(
            matrix_scale,
        )
    ]

    # 5. Define effective limits for the color scale
    if finite_vals.size == 0:
        has_negative = False
        vmin_eff, vmax_eff = 0.0, 1.0
    else:
        has_negative = float(
            np.nanmin(
                finite_vals,
            ),
        ) < 0.0

        min_val = float(
            np.nanmin(
                finite_vals,
            ),
        )
        max_val = float(
            np.nanmax(
                finite_vals,
            ),
        )

        if has_negative:
            vmin_eff, vmax_eff = min_val, max_val
        else:
            vmin_eff = vmin
            vmax_eff = float(
                max_val,
            ) if vmax is None else float(
                vmax,
            )

        if vmin_eff == vmax_eff:
            vmax_eff += 1.0

    # 6. Determine Figure Size
    nrows, ncols = df.shape

    if figsize is None:
        figsize = (
            cell_size_inch * ncols,
            cell_size_inch * nrows,
        )

    # 7. Initialize Plot
    fig, ax = plt.subplots(
        figsize=figsize,
        constrained_layout=True,
    )

    # 8. Plot layers (Negative in Blues, Positive in YlOrRd)
    cmap_neg = mcolors.LinearSegmentedColormap.from_list(
        "CustomBlues",
        [
            "#08306b",
            "#b3e0ff",
        ],
    )

    if has_negative:
        matrix_pos = np.ma.masked_less_equal(
            matrix_values,
            0.0,
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
            cmap=cmap_neg,
            norm=norm_neg,
        )
    else:
        matrix_pos = np.ma.masked_equal(
            matrix_values,
            0.0,
        )
        norm_pos = mcolors.Normalize(
            vmin=vmin_eff,
            vmax=vmax_eff,
        )
        ax.imshow(
            matrix_pos,
            aspect="equal",
            cmap=plt.cm.YlOrRd,
            norm=norm_pos,
        )

    # 9. Overlay diagonal (now conditional for grayscale or black)
    diag_mask = np.eye(
        nrows,
        dtype=bool,
    )
    
    # Define which matrices should have grayscale diagonals
    DIAGONAL_GRAYSCALE_MATRICES = ["sum", "ext", "all_exc", "alt_exc", "unacc_ext"]
    DIAGONAL_WHITE_MATRICES = ["qty_shift", "alt_shift"]
    
    annotate_heatmap_diagonal_vmax = None # Initialize for annotate_heatmap

    if matrix_key in DIAGONAL_GRAYSCALE_MATRICES:
        diagonal_values = np.diag(matrix_values)
        # Determine vmax for diagonal, ensuring it's at least 1 if all zeros
        vmax_diag_current = np.max(diagonal_values) if diagonal_values.size > 0 and np.max(diagonal_values) > 0 else 1.0
        
        cmap_diag = plt.cm.Greys # Greys colormap (0=white, 1=black)
        norm_diag = mcolors.Normalize(vmin=0, vmax=vmax_diag_current)
        
        # Plot diagonal values with grayscale colormap
        ax.imshow(
            np.ma.masked_where(~diag_mask, matrix_values), # Mask non-diagonal elements
            aspect="equal",
            cmap=cmap_diag,
            norm=norm_diag,
            interpolation="nearest"
        )
        annotate_heatmap_diagonal_vmax = vmax_diag_current
    elif matrix_key in DIAGONAL_WHITE_MATRICES:
        # For white diagonals, do not draw any overlay.
        # The default plot background is white, and the annotation function
        # will handle drawing the "0" values.
        pass
    else:
        # Fallback to a black diagonal for any other unspecified matrix type.
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
            interpolation="nearest"
        )

    # 10. Axis formatting
    ax.set_xticks(
        range(
            len(
                labels,
            ),
        ),
    )
    ax.set_yticks(
        range(
            len(
                labels,
            ),
        ),
    )

    try:
        tick_names = label_id_to_name(
            labels,
            class_labels_dict=GLANCE_METADATA
        )
    except NameError:
        tick_names = labels

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
        xlabel,
        fontsize=axis_label_fontsize,
    )
    ax.set_ylabel(
        ylabel,
        fontsize=axis_label_fontsize,
    )
    ax.set_title(
        title,
        fontsize=title_fontsize,
    )

    # 11. Build continuous colorbar logic
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
                cmap_neg(
                    t,
                ),
            )
        elif v == 0:
            colors_bar.append(
                (
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ),
            )
        else:
            t = max(
                0.0,
                v,
            ) / vmax_eff if vmax_eff > 0.0 else 0.0
            colors_bar.append(
                plt.cm.YlOrRd(
                    t,
                ),
            )

    cmap_bar = mcolors.ListedColormap(
        colors_bar,
    )
    norm_bar = mcolors.Normalize(
        vmin=vmin_eff,
        vmax=vmax_eff,
    )

    scalar_mappable = plt.cm.ScalarMappable(
        cmap=cmap_bar,
        norm=norm_bar,
    )
    scalar_mappable.set_array(
        [],
    )

    # 12. Create Colorbar
    cbar = fig.colorbar(
        scalar_mappable,
        ax=ax,
        fraction=cbar_fraction,
        pad=cbar_pad,
    )

    # 13. Determine Unit Scaling
    max_abs = vmax_eff

    if max_abs >= 1_000_000_000_000:
        factor = 1_000_000_000_000.0
        suffix = "T"
    elif max_abs >= 1_000_000_000:
        factor = 1_000_000_000.0
        suffix = "B"
    elif max_abs >= 1_000_000:
        factor = 1_000_000.0
        suffix = "M"
    elif max_abs >= 1_000:
        factor = 1_000.0
        suffix = "k"
    else:
        factor = 1.0
        suffix = ""

    # 14. Apply Formatters
    cbar.locator = mticker.MaxNLocator(
        nbins=2,
        integer=True,
        steps=[
            1,
            2,
            5,
            10,
        ],
    )

    try:
        cbar.formatter = _unit_formatter(
            factor=factor,
            decimals=0,
            suffix="" # Remove the unit suffix from the colorbar ticks
        )

        final_label = _unit_label(
            suffix, # Restore the descriptive unit in parentheses in the main label
            base_label=cbar_label,
        )
    except NameError:
        final_label = cbar_label

    cbar.set_label(
        final_label,
        rotation=270,
        labelpad=15,
        fontsize=cbar_label_fontsize,
    )

    if cbar_tick_labelsize is not None:
        cbar.ax.tick_params(
            labelsize=cbar_tick_labelsize,
        )

    cbar.update_ticks()

    # 15. Annotate Cells
    if annotate:
        try:
            annotate_heatmap(
                ax=ax,
                M=matrix_values,
                fontsize=ann_fontsize,
                show_diagonal=show_diagonal_values,
                equalize_diagonal_font=equalize_diagonal_font,
                vmax=vmax_eff,
                matrix_key=matrix_key,
                diagonal_vmax=annotate_heatmap_diagonal_vmax, # Pass the new parameter
            )
        except NameError:
            pass

    # 16. Save and Show
    if save_path:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

def generate_all_heatmaps(
    matrices_dict: dict,
    output_path: str,
    interval_str: str,
    years: list,
    style_config: dict,
) -> None:
    """
    Iterate over the matrices dictionary and generate a heatmap for each.

    Parameters
    ----------
    matrices_dict : dict
        Dictionary containing the dataframes to plot.
    output_path : str
        Base directory path to save the generated charts.
    interval_str : str
        String representing the time interval.
    years : list
        List of years processed.
    style_config : dict
        Dictionary containing style configurations for the plot.

    Returns
    -------
    None
    """
    import os

    print(
        "Generating Heatmaps...",
    )

# ---------------------------------------------------------------------------
# 6.5 QUEST CHARTS
# ---------------------------------------------------------------------------

def plot_change_components_time_intervals(
    input_dir: str,
    output_dir: str,
) -> None:
    """
    Plot a stacked bar chart of change components over time intervals.

    Reads the 'change_components.csv' file, aggregates the Gain
    per time interval and component, and generates a scaled stacked
    bar chart.

    Parameters
    ----------
    input_dir : str
        Directory path containing the 'tables' folder with the CSV.
    output_dir : str
        Directory path where the 'charts' folder will be saved.

    Returns
    -------
    None
        Saves the plot to disk and displays it.
    """

    csv_path = os.path.join(
        input_dir,
        "tables",
        "change_components.csv"
    )
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing components file: {csv_path}")

    df = pd.read_csv(csv_path)

    # Filter only time intervals (containing '-')
    time_df = df[df["Time_Interval"].str.contains("-")]

    # Aggregate totals per interval and component (Gain)
    totals = time_df.groupby(["Time_Interval", "Component"])["Gain"].sum().unstack()

    # Choose scale automatically based on max value
    max_val = totals[["Quantity", "Shift", "Exchange"]].to_numpy().max()

    if max_val >= 1_000_000_000_000:
        scale_factor = 1_000_000_000_000
        y_label = "Change (trillion pixels)"
    elif max_val >= 1_000_000_000:
        scale_factor = 1_000_000_000
        y_label = "Change (billion pixels)"
    elif max_val >= 1_000_000:
        scale_factor = 1_000_000
        y_label = "Change (million pixels)"
    elif max_val >= 1_000:
        scale_factor = 1_000
        y_label = "Change (thousand pixels)"
    elif max_val >= 100:
        scale_factor = 100
        y_label = "Change (hundred pixels)"
    else:
        scale_factor = 1
        y_label = "Change (pixels)"

    # Scaled totals per component for plotting
    scaled_totals = totals[["Quantity", "Shift", "Exchange"]] / scale_factor

    # Maximum stacked height
    stacked_max = scaled_totals.sum(axis=1).max()

    fig, ax = plt.subplots(figsize=(14, 6))

    # Colors configuration
    colors = [
        "#1f77b4",
        "#2ca02c",
        "#ffd700"
    ]
    components_color = {
        "Quantity": "#1f77b4",
        "Shift":    "#2ca02c",
        "Exchange": "#ffd700",
    }
    components = [
        "Quantity",
        "Shift",
        "Exchange"
    ]

    # Stacked bars using scaled values
    for idx, comp in enumerate(components):
        bottom_values = scaled_totals.iloc[:, :idx].sum(axis=1) if idx > 0 else 0
        ax.bar(
            totals.index,
            scaled_totals[comp],
            label=comp,
            color=colors[idx],
            edgecolor="none",
            bottom=bottom_values,
            width=0.9,
        )

    # Axes formatting
    ax.set_ylabel(
        y_label,
        fontsize=18
    )
    ax.set_title(
        "Change Components during Time Intervals",
        fontsize=20,
        pad=15
    )
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=18
    )

    # Adaptive rotation for x-axis tick labels
    labels = ax.get_xticklabels()
    n_labels = len(labels)

    if n_labels <= 6:
        rotation, ha = 0, "center"
    elif n_labels <= 12:
        rotation, ha = 45, "right"
    else:
        rotation, ha = 90, "center"

    plt.setp(
        labels,
        rotation=rotation, 
        ha=ha
    )

    # Y-axis limits and ticks
    y_max_scaled = stacked_max * 1.1
    ax.set_ylim(0, y_max_scaled)
    ax.yaxis.set_major_locator(
        ticker.MaxNLocator(
            nbins=6,
            integer=True
            )
    )

    # Legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=components_color["Exchange"], label="Allocation Exchange"),
        plt.Rectangle((0, 0), 1, 1, color=components_color["Shift"], label="Allocation Shift"),
        plt.Rectangle((0, 0), 1, 1, color=components_color["Quantity"], label="Quantity"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        title="Component",
        title_fontsize=16,
        fontsize=16,
        alignment="left",
        frameon=False,
    )

    # Save and show
    plt.tight_layout()

    charts_dir = os.path.join(
        output_dir,
        "charts"
    )
    os.makedirs(
        charts_dir,
        exist_ok=True
    )
    out_fig_path = os.path.join(
        charts_dir,
        "chart_change_components_time_interval.png"
    )
    plt.savefig(
        out_fig_path,
        bbox_inches="tight",
        format="png",
        dpi=300
    )
    plt.show()
    print(f"Chart saved to: {out_fig_path}")

def plot_components_with_alternation(csv_path: str, output_path: str) -> None:
    """
    Plot overall change components as a single stacked bar with auto-scaled y-axis for Real Data.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing change components.
    output_path : str
        Directory where the output figure will be saved.
    """
    # 1. Load data
    df = pd.read_csv(csv_path)

    # 2. Define Colors and component order
    components_color = {
        "Quantity":            "#1f77b4",
        "Allocation_Exchange": "#ffd700",
        "Alternation_Exchange":"#ff8080",
        "Allocation_Shift":    "#2ca02c",
        "Alternation_Shift":   "#990099",
    }

    component_order = [
        "Quantity",
        "Allocation_Shift",
        "Allocation_Exchange",
        "Alternation_Shift",
        "Alternation_Exchange",
    ]

    # 3. Aggregate totals per component
    component_totals = {
        "Quantity": df[(df["Component"] == "Allocation_Quantity") & (df["Time_Interval"] == "extent")]["Gain"].sum(),
        "Allocation_Exchange": df[(df["Component"] == "Allocation_Exchange") & (df["Time_Interval"] == "extent")]["Gain"].sum(),
        "Allocation_Shift": df[(df["Component"] == "Allocation_Shift") & (df["Time_Interval"] == "extent")]["Gain"].sum(),
        "Alternation_Exchange": df[(df["Time_Interval"] == "alternation_exchange")]["Gain"].sum(),
        "Alternation_Shift": df[(df["Time_Interval"] == "alternation_shift")]["Gain"].sum(),
    }

    # 4. Automatic scale based on the sum of all stacked components
    total_change = sum(component_totals.values())

    if total_change >= 1_000_000_000_000:
        scale_factor = 1_000_000_000_000
        y_label = "Change (trillion pixels)"
    elif total_change >= 1_000_000_000:
        scale_factor = 1_000_000_000
        y_label = "Change (billion pixels)"
    elif total_change >= 1_000_000:
        scale_factor = 1_000_000
        y_label = "Change (million pixels)"
    elif total_change >= 1_000:
        scale_factor = 1_000
        y_label = "Change (thousand pixels)"
    elif total_change >= 100:
        scale_factor = 100
        y_label = "Change (hundred pixels)"
    else:
        scale_factor = 1
        y_label = "Change (pixels)"

    # 5. Initialize figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))

    # 6. Plot each component in a stacked bar at a single x-position
    bottom = 0.0
    for component in component_order:
        value = component_totals.get(component, 0.0) / scale_factor
        ax.bar(
            x=0,
            height=value,
            bottom=bottom,
            color=components_color[component],
            edgecolor="none",
            width=0.4,
        )
        bottom += value

    # 7. Axes formatting and labels
    ax.set_ylabel(
        y_label,
        fontsize=16
    )
    ax.set_title(
        "Change Components Overall",
        fontsize=18,
        pad=15
    )
    ax.set_xticks([])
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=18
    )

    # 8. Set y-axis limits and major tick locators
    y_max_scaled = bottom * 1.05 if bottom > 0 else 1.0
    ax.set_ylim(
        0,
        y_max_scaled
    )
    ax.yaxis.set_major_locator(
        mticker.MaxNLocator(
            nbins=10,
            integer=True
            )
    )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))

    # 9. Configure visible spines for the plot frame
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(0.5)

    # 10. Define custom legend elements
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=components_color["Alternation_Exchange"], label="Alternation Exchange"),
        plt.Rectangle((0, 0), 1, 1, color=components_color["Alternation_Shift"], label="Alternation Shift"),
        plt.Rectangle((0, 0), 1, 1, color=components_color["Allocation_Exchange"], label="Allocation Exchange"),
        plt.Rectangle((0, 0), 1, 1, color=components_color["Allocation_Shift"], label="Allocation Shift"),
        plt.Rectangle((0, 0), 1, 1, color=components_color["Quantity"], label="Quantity"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        title="Component",
        title_fontsize=14,
        fontsize=14,
        alignment="left",
        frameon=False,
    )

    # Force the main plotting box to always occupy the exact same spatial coordinates in the figure
    fig.subplots_adjust(
        left=0.15,
        right=0.75,
        bottom=0.1,
        top=0.9
    )

    # 11. Final layout adjustment and export
    charts_dir = os.path.join(
        output_path,
        "charts"
    )
    os.makedirs(
        charts_dir,
        exist_ok=True
    )
    out_fig_path = os.path.join(
        charts_dir,
        "graphic_change_components_overall.png"
    )
    plt.savefig(
        out_fig_path,
        bbox_inches="tight",
        format="png",
        dpi=300
    )
    plt.show()
    print(f"Chart saved to: {out_fig_path}")


def plot_change_components_by_class(
    input_dir: str,
    output_dir: str,
    class_labels_dict: dict,
) -> None:
    """
    Plot per-class gains and losses as stacked bars with auto-scaled y-axis.

    Parameters
    ----------
    input_dir : str
        Directory path containing the 'tables' folder with the CSV.
    output_dir : str
        Directory path where the 'charts' folder will be saved.
    class_labels_dict : dict
        Dictionary mapping class IDs to names/metadata.
    """
    # 1. Load Data
    csv_path = os.path.join(
        input_dir,
        "tables",
        "change_components.csv"
    )
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing components file: {csv_path}")

    df = pd.read_csv(csv_path)

    # 2. Filter and Prepare Data
    target_intervals = ["extent", "alternation_shift", "alternation_exchange"]
    df_sub = df[df["Time_Interval"].isin(target_intervals)].copy()
    # Rename for clarity in the plot
    df_sub["Component"] = df_sub["Component"].str.replace("Allocation_Quantity", "Quantity")

    # 3. Colors and Order
    components_map = {
        "Quantity": "#1f77b4",
        "Allocation_Exchange": "#ffd700",
        "Alternation_Exchange": "#ff8080",
        "Allocation_Shift": "#2ca02c",
        "Alternation_Shift": "#990099",
    }
    # Components derived from 'extent'
    comp_groups = [
        "Quantity",
        "Allocation_Shift",
        "Allocation_Exchange"
    ]

    # 4. Identify Classes and Sort by Net Quantity Change
    classes = sorted(df_sub["Class"].unique())
    class_stats = []

    for cls in classes:
        c_data = df_sub[df_sub["Class"] == cls]
        # Net change is based on Quantity component only
        qty_gain = c_data[c_data["Component"] == "Quantity"]["Gain"].sum()
        qty_loss = c_data[c_data["Component"] == "Quantity"]["Loss"].sum()
        class_stats.append((cls, qty_gain - qty_loss))

    ordered_classes = [x[0] for x in sorted(class_stats, key=lambda x: x[1])]

    # 5. Prepare Plot Data
    plot_data = []
    max_val = 0.0

    for cls in ordered_classes:
        c_data = df_sub[df_sub["Class"] == cls]

        # --- GAINS ---
        gains = {}
        for comp in comp_groups:
            gains[comp] = c_data[c_data["Component"] == comp]["Gain"].sum()

        # Handle Alternation components separately
        raw_alt_exc = c_data[c_data["Component"] == "Alternation_Exchange"]["Gain"].sum()
        raw_alt_shift = c_data[c_data["Component"] == "Alternation_Shift"]["Gain"].sum()
        net_alt = raw_alt_exc + raw_alt_shift

        adj_alt_exc, adj_alt_shift = 0.0, 0.0
        if net_alt > 0.0001:
            if raw_alt_exc > 0:
                adj_alt_exc = raw_alt_exc
                adj_alt_shift = max(0, net_alt - raw_alt_exc)
            else:
                adj_alt_exc = 0
                adj_alt_shift = net_alt

        gains["Alternation_Exchange"] = adj_alt_exc
        gains["Alternation_Shift"] = adj_alt_shift

        # --- LOSSES ---
        losses = {}
        for comp in comp_groups:
            losses[comp] = c_data[c_data["Component"] == comp]["Loss"].sum()

        # Handle Alternation components separately for losses
        raw_alt_exc_l = c_data[c_data["Component"] == "Alternation_Exchange"]["Loss"].sum()
        raw_alt_shift_l = c_data[c_data["Component"] == "Alternation_Shift"]["Loss"].sum()
        net_alt_l = raw_alt_exc_l + raw_alt_shift_l

        adj_alt_exc_l, adj_alt_shift_l = 0.0, 0.0
        if net_alt_l > 0.0001:
            if raw_alt_exc_l > 0:
                adj_alt_exc_l = raw_alt_exc_l
                adj_alt_shift_l = max(0, net_alt_l - raw_alt_exc_l)
            else:
                adj_alt_exc_l = 0
                adj_alt_shift_l = net_alt_l

        losses["Alternation_Exchange"] = adj_alt_exc_l
        losses["Alternation_Shift"] = adj_alt_shift_l

        # Track max height for scaling
        max_val = max(max_val, sum(gains.values()), sum(losses.values()))
        plot_data.append({"class": cls, "gains": gains, "losses": losses})

    # 6. Determine Scale Factor
    # Based on the maximum total gain or loss for any single class
    if max_val >= 1_000_000_000_000:
        scale_factor, y_label = 1_000_000_000_000, "Loss and Gain (trillion pixels)"
    elif max_val >= 1_000_000_000:
        scale_factor, y_label = 1_000_000_000, "Loss and Gain (billion pixels)"
    elif max_val >= 1_000_000:
        scale_factor, y_label = 1_000_000, "Loss and Gain (million pixels)"
    elif max_val >= 1_000:
        scale_factor, y_label = 1_000, "Loss and Gain (thousand pixels)"
    elif max_val >= 100:
        scale_factor, y_label = 100, "Loss and Gain (hundred pixels)"
    else:
        scale_factor, y_label = 1, "Loss and Gain (pixels)"

    # 7. Plotting
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.subplots_adjust(
        left=0.1,
        right=0.75
    )

    # X-axis positions
    x_pos = np.arange(len(ordered_classes))
    width = 0.6

    stack_order = [
        "Quantity",
        "Allocation_Shift",
        "Allocation_Exchange",
        "Alternation_Shift",
        "Alternation_Exchange",
    ]

    for idx, item in enumerate(plot_data):
        # Gains (Upwards)
        bottom_g = 0.0
        for comp in stack_order:
            val = item["gains"][comp] / scale_factor
            if val > 0:
                ax.bar(x_pos[idx], val, width, bottom=bottom_g, color=components_map[comp], edgecolor="none")
                bottom_g += val

        # Losses (Downwards)
        bottom_l = 0.0
        for comp in stack_order:
            val = item["losses"][comp] / scale_factor
            if val > 0:
                ax.bar(x_pos[idx], -val, width, bottom=bottom_l, color=components_map[comp], edgecolor="none")
                bottom_l -= val

    # 8. Formatting
    # Map class IDs to names for tick labels
    class_names = [
        class_labels_dict.get(int(c) if str(c).isdigit() else c, {}).get("name", str(c))
        for c in ordered_classes
    ]

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        class_names,
        rotation=90,
        ha="center",
        fontsize=14
    )
    ax.axhline(
        0,
        color="black",
        linewidth=0.8
    )

    ax.set_ylabel(
        y_label,
        fontsize=20
    )
    ax.set_title(
        "Change Components by Class",
        fontsize=20,
        pad=15
    )
    ax.tick_params(
        axis="x",
        labelsize=16
    )
    ax.tick_params(
        axis="y",
        labelsize=24
    )

    # Set Y-axis limits and format ticks
    limit = max_val / scale_factor * 1.1 if max_val > 0 else 1.0
    ax.set_ylim(-limit, limit)
    ax.yaxis.set_major_locator(
        ticker.MaxNLocator(
            integer=True,
            nbins=10
        )
    )
    ax.yaxis.set_major_formatter(
        ticker.FormatStrFormatter("%d")
    )

    # Create legend handles
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=components_map[c], label=c.replace("_", " "))
        for c in reversed(stack_order)
    ]

    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="Component",
        title_fontsize=16,
        alignment="left",
        fontsize=16,
        frameon=False,
    )

    # Adjust layout to prevent legend from being cut off
    plt.tight_layout()

    charts_dir = os.path.join(
        output_dir,
        "charts"
    )
    os.makedirs(
        charts_dir,
        exist_ok=True
    )

    out_fig_path = os.path.join(
        charts_dir,
        "chart_change_component_change_class.png"
    )
    plt.savefig(
        out_fig_path,
        bbox_inches="tight",
        format="png",
        dpi=300
    )
    plt.show()
    print(f"Chart saved to: {out_fig_path}")


# ---------------------------------------------------------------------------
# 6.6 PLOT MAPS
# ---------------------------------------------------------------------------

def export_quantity_component_task_gee(
    year_list: list,
    drive_folder: str,
    # Default scale to 300m for global analysis
    scale: int = 300,
    nodata_val: int = 255,
    full_year_list: list = None,
) -> ee.batch.Task:
    """
    Compute and export a raster representing the Quantity Component of change using GEE.
    """
    if GLOBAL_GEOM is None:
        raise ValueError(
            "GLOBAL_GEOM is not initialized. Please call "
            "utils.initialize_active_region(region_code) before running tasks."
        )

    # Default to using the main year list for the mask if not specified
    if full_year_list is None:
        full_year_list = year_list

    # 1. Combine lists to build a single stack containing all required years safely
    combined_years = sorted(list(set(year_list) | set(full_year_list)))

    # 2. Build the master stack and extract the global mask based on the FULL timeline
    master_stack, master_band_names = build_glance_stack(
        year_list=combined_years,
        collection_id=GLANCE_COLLECTION_ID,
        band_name=GLANCE_CLASS_BAND,
        nodata_val=nodata_val,
    )
    
    # Create the global validity mask strictly using full_year_list bands
    full_year_bands = [f"y{y}" for y in full_year_list]
    full_stack_subset = master_stack.select(full_year_bands)
    global_mask = full_stack_subset.neq(nodata_val).unmask(0).reduce(ee.Reducer.min())

    # Extract start and end images directly from master stack by name
    start_img = master_stack.select(f"y{year_list[0]}").rename(GLANCE_CLASS_BAND)
    end_img = master_stack.select(f"y{year_list[-1]}").rename(GLANCE_CLASS_BAND)

    # 2. Compute Quantity Component: 1 if start class is not equal to end class, else 0
    quantity_image = start_img.neq(end_img).multiply(1).toByte()

    # 3. Apply the global validity mask (must be valid across ALL years to preserve math integrity)
    quantity_image = quantity_image.updateMask(global_mask)

    # 4. Apply NoData unmasking and set properties
    quantity_image = quantity_image.unmask(nodata_val)
    quantity_image = quantity_image.set('system:no_data_value', nodata_val)

    # 5. Define the Earth Engine export task
    start_year = year_list[0]
    end_year = year_list[-1]
    task_desc = f"Quantity_Component_{start_year}_{end_year}"
    task = ee.batch.Export.image.toDrive(
        image=quantity_image,
        description=task_desc,
        folder=drive_folder,
        scale=scale,
        region=GLOBAL_GEOM,
        maxPixels=1e13,
        crs="EPSG:4326",
    )

    # 6. Start the export task
    task.start()
    print(f"Task '{task_desc}' submitted to Google Earth Engine with NoData: {nodata_val}")

    return task

def plot_quantity_component_map(
    output_dir: str,
    nodata_val: int,
    # Default filename prefix for quantity component raster
    raster_filename: str,
    scale_factor: float = 0.05,
) -> None:
    """
    Plot the Quantity Component (Extent Change) raster map with cartographic elements.

    Parameters
    ----------
    output_dir : str
        Directory containing the exported GEE tiles and where the map will be saved.
    nodata_val : int
        Value representing NoData in the raster to be masked out.
    raster_filename : str
        Prefix of the raster tiles to plot.
    scale_factor : float, optional
        Scale factor to downsample the massive global raster to fit into memory.

    Returns
    -------
    None
    """
    # 1. Locate all raster tiles exported by GEE
    raster_files = glob.glob(os.path.join(
        output_dir,
        f"{raster_filename}*.tif"
        )
    )
    if not raster_files:
        raise FileNotFoundError(
            f"Raster tiles not found for prefix: {raster_filename}. Make sure the GEE export finished."
        )

    # 2. Create a temporary Virtual Raster (VRT) to merge tiles dynamically
    vrt_path = os.path.join(
        output_dir,
        "merged_quantity.vrt"
    )
    files_str = " ".join([f'"{f}"' for f in raster_files])
    os.system(f"gdalbuildvrt {vrt_path} {files_str}")

    # 3. Calculate pixel size for scale bar
    pixel_size_km = compute_display_pixel_size_km(
        raster_path=vrt_path,
        downsample_factor=scale_factor,
    )

    # 4. Read raster and basic metadata with downsampling
    with rasterio.open(vrt_path) as src:
        out_shape = (
            max(1, int(src.height * scale_factor)),
            max(1, int(src.width * scale_factor)),
        )
        data = src.read(
            1,
            out_shape=out_shape,
            resampling=rasterio.enums.Resampling.nearest,
        )

        # Force masking using the provided nodata value
        data = np.ma.masked_equal(
            data,
            nodata_val
        )

        src_crs = src.crs
        # Adjust the affine transform for the new downsampled resolution
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[1]),
            (src.height / data.shape[0]),
        )
        height, width = data.shape

    # 5. Figure
    fig, ax = plt.subplots(
        figsize=(20, 10),
        dpi=300
    )

    # 6. Define a discrete colormap for 0 (no change) and 1 (change)
    cmap = ListedColormap(
        ["#c0c0c0", 
         "#fde725"]
    )
    bounds = [-0.5, 0.5, 1.5]
    norm = BoundaryNorm(bounds, cmap.N)

    # 7. Plot raster
    ax.imshow(
        data,
        cmap=cmap,
        interpolation="nearest",
        norm=norm,
    )

    # 8. Create a manual legend for the two discrete values
    legend_elements = [
        Patch(
            facecolor="#c0c0c0",
            label="0",
            edgecolor="black",
            linewidth=0,
        ),
        Patch(
            facecolor="#fde725",
            label="1",
            edgecolor="black",
            linewidth=0,
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=14,
        borderpad=1.2,
        title="Change",
        title_fontsize=14,
        alignment="left",
        handletextpad=0.8,
        columnspacing=2,
        labelspacing=0.8,
        handlelength=2.0,
        handleheight=1.5,
    )

    # 9. Cartographic elements
    degree_in_meters = 111320.0
    dx_meters = degree_in_meters if ax.get_xlim()[1] <= 180.5 else (pixel_size_km * 1000)

    def km_formatter(value, unit):
        if unit == "Mm":
            return f"{int(value * 1000)} km"
        return f"{int(value)} {unit}"

    scalebar = ScaleBar(
        dx=dx_meters,
        units="m",
        length_fraction=0.15,
        location="lower left",
        box_alpha=0.6,
        scale_formatter=km_formatter,
    )
    ax.add_artist(scalebar)

    try:
        north_arrow(
            ax,
            location="upper right",
            shadow=False,
            rotation={"degrees": 0},
            scale=0.5,
        )
    except NameError:
        print("north_arrow function not found. Skipping north arrow.")

    # 10. Axes styling
    ax.set_title(
        "Extent Change",
        fontsize=18,
        pad=10
    )
    ax.set_aspect("equal")

    to_latlon = Transformer.from_crs(
        src_crs,
        "EPSG:4326",
        always_xy=True
    )

    def format_lon(x, pos):
        x = np.clip(x, 0, width - 1)
        x_proj, y_proj = rasterio.transform.xy(transform, height // 2, x)
        lon, _ = to_latlon.transform(x_proj, y_proj)
        return f"{lon:.1f}°"

    def format_lat(y, pos):
        y = np.clip(y, 0, height - 1)
        x_proj, y_proj = rasterio.transform.xy(transform, y, width // 2)
        _, lat = to_latlon.transform(x_proj, y_proj)
        return f"{lat:.1f}°"

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=10,
        pad=4
    )
    plt.setp(
        ax.get_yticklabels(),
        rotation=90,
        va="center"
    )

    # 11. Save and Show
    maps_dir = os.path.join(output_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)
    output_figure_path = os.path.join(maps_dir, "map_extent_change.png")

    plt.savefig(
        output_figure_path,
        dpi=300,
        bbox_inches="tight",
        format="png",
        pad_inches=0.5,
    )
    plt.show()
    print(f"Map figure saved successfully to: {output_figure_path}")


def export_alternation_components_gee(
    year_list: list,
    drive_folder: str,
    scale: int = 300,
    nodata_val: int = 255,
    full_year_list: list = None,
) -> list:
    """
    Computes and exports all three Alternation components (Exchange, Shift,
    Unaccounted) by performing non-linear math at the pixel-scale before
    spatial aggregation, following the Pontius methodology.

    This function calculates the base Extent (E) and Sum (V) matrices
    per-pixel once, and then derives X, S, and U from them, before launching
    three separate GEE export tasks. A monitoring loop is included to
    process tasks sequentially and avoid quota errors.

    Parameters
    ----------
    year_list : list
        List of integer years for the analysis period (e.g., 2001 to 2019).
    drive_folder : str
        The destination folder in Google Drive for the output CSVs.
    scale : int, optional
        The spatial resolution in meters for the reduction, by default 300.
    nodata_val : int, optional
        The NoData value used for masking, by default 255.
    full_year_list : list, optional
        The complete timeline to construct the consistent global validity mask.
        If None, `year_list` is used.

    Returns
    -------
    list
        A list of the three submitted ee.batch.Task objects.
    """
    import time

    if GLOBAL_GEOM is None:
        raise ValueError(
            "GLOBAL_GEOM is not initialized. Please call "
            "utils.initialize_active_region(region_code) before running tasks."
        )

    # Use the same year list for the mask if a full one isn't provided
    if full_year_list is None:
        full_year_list = year_list

    print("Step 1/4: Building annual image stack and strict global mask...")
    _, yearly_images_masked = build_global_valid_mask_and_yearly_images(
        year_list=year_list,
        collection_id=GLANCE_COLLECTION_ID,
        band_name=GLANCE_CLASS_BAND,
        nodata_val=nodata_val,
    )
    
    # Extract images (already masked) and the global mask itself
    yearly_images = {year: img for year, img in yearly_images_masked}
    global_mask = yearly_images_masked[0][1].mask() # Get mask from first image

    img_start = yearly_images[year_list[0]]
    img_end = yearly_images[year_list[-1]]

    print("Step 2/4: Calculating pixel-wise component bands (X, S, U)...")
    classes = list(GLANCE_METADATA.keys())
    X_bands, S_bands, U_bands = [], [], []

    # Efficiently create image stacks for vectorized calculations
    img_stack_t = ee.Image([yearly_images[y] for y in year_list[:-1]])
    img_stack_t1 = ee.Image([yearly_images[y] for y in year_list[1:]])

    for i in classes:
        for j in classes:
            b_name = f"T_{i}_{j}"

            # --- E_ij and E_ji (Extent) ---
            E_ij = img_start.eq(i).And(img_end.eq(j))
            E_ji = img_start.eq(j).And(img_end.eq(i))

            # --- V_ij and V_ji (Sum of Intervals) - Vectorized calculation ---
            V_ij = img_stack_t.eq(i).And(img_stack_t1.eq(j)).reduce(ee.Reducer.sum())
            V_ji = img_stack_t.eq(j).And(img_stack_t1.eq(i)).reduce(ee.Reducer.sum())

            # --- Pontius Equations (Pixel-wise) ---
            # Unmask with 0 to allow subtraction, then re-mask at the end.
            E_ij, E_ji = E_ij.unmask(0), E_ji.unmask(0)
            V_ij, V_ji = V_ij.unmask(0), V_ji.unmask(0)

            # Eq 7: Alternation Exchange (X_ij)
            diff_ij = V_ij.subtract(E_ij)
            diff_ji = V_ji.subtract(E_ji)
            X_ij = diff_ij.min(diff_ji).max(0)

            # Eq 8: Alternation Shift (S_ij)
            S_ij = V_ij.subtract(X_ij).subtract(E_ij).max(0)

            # Eq 9: Unaccounted Extent (U_ij)
            U_ij = E_ij.add(X_ij).add(S_ij).subtract(V_ij)

            # Append final bands for each component image
            X_bands.append(X_ij.rename(b_name).updateMask(global_mask))
            S_bands.append(S_ij.rename(b_name).updateMask(global_mask))
            U_bands.append(U_ij.rename(b_name).updateMask(global_mask))

    # Assemble the final multi-band images for each component
    img_X = ee.Image(X_bands)
    img_S = ee.Image(S_bands)
    img_U = ee.Image(U_bands)

    component_images = {
        "alternation_exchange": img_X,
        "alternation_shift": img_S,
        "unaccounted_extent": img_U,
    }

    print("Step 3/4: Reducing global area and configuring export tasks...")
    tasks = []
    task_names = []
    # Create a separate export task for each component
    for name, image in component_images.items():
        totals = image.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=GLOBAL_GEOM,
            scale=scale,
            crs="EPSG:4326",
            maxPixels=1e13,
            tileScale=16
        )

        fc = ee.FeatureCollection([ee.Feature(None, totals)])
        task_name = f"transition_matrix_{name}_{year_list[0]}-{year_list[-1]}"

        task = ee.batch.Export.table.toDrive(
            collection=fc,
            description=task_name,
            folder=drive_folder,
            fileNamePrefix=f"raw_{task_name}",
            fileFormat="CSV"
        )
        tasks.append(task)
        task_names.append(task_name)

    print("Step 4/4: Starting tasks sequentially to avoid quota errors...")
    for task, task_name in zip(tasks, task_names):
        # Start one task and wait for it to complete before starting the next
        task.start()
        print(f"🚀 Task started: {task_name}. Monitoring...")
        while task.active():
            minutes_ran = (time.time() - task.status()['start_timestamp_ms']/1000) / 60
            print(
                f"  -> Status: {task.status()['state']} "
                f"({minutes_ran:.1f} minutes elapsed)"
            )
            time.sleep(60)

        final_status = task.status()
        if final_status['state'] != 'COMPLETED':
            error_msg = final_status.get('error_message', 'No error message found.')
            print(f"🚨 Task {task_name} failed: {error_msg}") # noqa
            # Stop processing further tasks if one fails
            raise Exception(f"Task {task_name} failed.")
        else:
            print(f"✅ Task {task_name} completed successfully.")

    return tasks


def plot_alternation_exchange_map(
    output_dir: str,
    nodata_val: int,
    raster_filename: str,
    scale_factor: float = 0.05,
) -> None:
    """
    Plot the Alternation Exchange raster map with cartographic elements.

    Parameters
    ----------
    output_dir : str
        Directory containing the exported GEE tiles and where the map will be saved.
    nodata_val : int
        Value representing NoData in the raster to be masked out.
    raster_filename : str
        Prefix of the raster tiles to plot.
    scale_factor : float, optional
        Scale factor to downsample the global raster to fit into memory.

    Returns
    -------
    None
    """
    # 1. Locate all raster tiles exported by GEE
    raster_files = glob.glob(os.path.join(
        output_dir,
        f"{raster_filename}*.tif")
    )
    if not raster_files:
        raise FileNotFoundError(
            f"Raster tiles not found for prefix: {raster_filename}. Make sure the GEE export finished."
        )

    # 2. Create a temporary Virtual Raster (VRT) to merge tiles
    vrt_path = os.path.join(
        output_dir,
        "merged_exchange.vrt"
    )
    files_str = " ".join([f'"{f}"' for f in raster_files])
    os.system(f"gdalbuildvrt {vrt_path} {files_str}")

    # 3. Calculate pixel size for the scale bar
    pixel_size_km = compute_display_pixel_size_km(
        raster_path=vrt_path,
        downsample_factor=scale_factor,
    )

    # 4. Read raster and basic metadata with downsampling
    with rasterio.open(vrt_path) as src:
        out_shape = (
            max(1, int(src.height * scale_factor)),
            max(1, int(src.width * scale_factor)),
        )
        data = src.read(
            1,
            out_shape=out_shape,
            resampling=rasterio.enums.Resampling.nearest,
        )

        # Mask using the provided nodata value
        data_masked = np.ma.masked_equal(data, nodata_val)

        src_crs = src.crs
        # Adjust the affine transform for the new downsampled resolution
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[1]),
            (src.height / data.shape[0]),
        )
        height, width = data.shape

    # 5. Figure
    fig, ax = plt.subplots(
        figsize=(20, 10),
        dpi=300
    )

    # Determine max value for colormap
    try:
        data_max = int(np.ma.max(data_masked))
    except:
        data_max = 1

    if data_max <= 0:
        data_max = 1

    # 6. Discrete colormap configuration
    original_cmap = plt.get_cmap("viridis_r")
    # Define the color for value 0 (Background/Gray)
    colors_list = ["#c0c0c0"] + [
        original_cmap(i) for i in np.linspace(0, 1, data_max)
    ]
    cmap = ListedColormap(colors_list)
    bounds = np.arange(-0.5, data_max + 1.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    # 7. Plot raster
    ax.imshow(
        data_masked,
        cmap=cmap,
        interpolation="nearest",
        norm=norm,
    )

    # 8. Legend configuration
    legend_elements = []

    # Extract unique values present in the masked raster data
    present_values = np.unique(data_masked.compressed())

    for i in range(0, data_max + 1):
        # Append to legend only if the value is present in the map
        if i in present_values:
            legend_elements.append(
                Patch(
                    facecolor=cmap(norm(i)),
                    edgecolor="none",
                    linewidth=0,
                    label=str(i),
                ),
            )

    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=12,
        borderpad=1.2,
        title="Exchange",
        title_fontsize=14,
        alignment="left",
        handletextpad=0.8,
        columnspacing=2,
        labelspacing=0.8,
        handlelength=2.0,
        handleheight=1.5,
    )

    # 9. Cartographic elements
    degree_in_meters = 111320.0
    # Use pixel size if CRS is projected, otherwise use degree equivalent
    dx_meters = degree_in_meters if ax.get_xlim()[1] <= 180.5 else (pixel_size_km * 1000)

    def km_formatter(value, unit):
        if unit == "Mm":
            return f"{int(value * 1000)} km"
        return f"{int(value)} {unit}"

    scalebar = ScaleBar(
        dx=dx_meters,
        units="m",
        length_fraction=0.15,
        location="lower left",
        box_alpha=0.6,
        scale_formatter=km_formatter,
    )
    ax.add_artist(scalebar)

    try:
        north_arrow(
            ax,
            location="upper right",
            shadow=False,
            rotation={"degrees": 0},
            scale=0.5,
        )
    except NameError:
        print("north_arrow function not found. Skipping north arrow.")

    # 10. Axes styling and title
    ax.set_title(
        "Alternation Exchange",
        fontsize=18,
        pad=10
    )
    ax.set_aspect("equal")

    # Define transformers for lat/lon tick labels
    to_latlon = Transformer.from_crs(
        src_crs,
        "EPSG:4326",
        always_xy=True
    )

    def format_lon(x, pos):
        x = np.clip(x, 0, width - 1)
        x_proj, y_proj = rasterio.transform.xy(transform, height // 2, x)
        lon, _ = to_latlon.transform(x_proj, y_proj)
        return f"{lon:.1f}°"

    def format_lat(y, pos):
        y = np.clip(y, 0, height - 1)
        x_proj, y_proj = rasterio.transform.xy(transform, y, width // 2)
        _, lat = to_latlon.transform(x_proj, y_proj)
        return f"{lat:.1f}°"

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=10,
        pad=4
    )
    plt.setp(
        ax.get_yticklabels(),
        rotation=90,
        va="center"
    )

    # 11. Save and show the figure
    maps_dir = os.path.join(
        output_dir,
        "maps"
    )
    os.makedirs(
        maps_dir,
        exist_ok=True
    )
    output_figure_path = os.path.join(
        maps_dir,
        "map_alternation_exchange.png"
    )

    plt.savefig(
        output_figure_path,
        dpi=300,
        bbox_inches="tight",
        format="png",
        pad_inches=0.5,
    )
    plt.show()
    print(f"Map figure saved successfully to: {output_figure_path}")


def _compute_alternation_component_raster(
    component: str,
    year_list: list,
    scale: int,
    nodata_val: int,
    full_year_list: list = None,
) -> ee.Image:
    """
    Computes a single-band raster with the per-pixel total Alternation
    component ("exchange" or "shift"), summed across all class-transition
    pairs, following the Pontius methodology.
    """
    if component not in ("exchange", "shift"):
        raise ValueError("component must be either 'exchange' or 'shift'.")

    if full_year_list is None:
        full_year_list = year_list

    # Combine lists to build a single stack containing all required years safely
    combined_years = sorted(list(set(year_list) | set(full_year_list)))

    master_stack, master_band_names = build_glance_stack(
        year_list=combined_years,
        collection_id=GLANCE_COLLECTION_ID,
        band_name=GLANCE_CLASS_BAND,
        nodata_val=nodata_val,
    )

    # Global validity mask strictly using full_year_list bands
    full_year_bands = [f"y{y}" for y in full_year_list]
    full_stack_subset = master_stack.select(full_year_bands)
    global_mask = full_stack_subset.neq(nodata_val).unmask(0).reduce(ee.Reducer.min())

    img_start = master_stack.select(f"y{year_list[0]}").rename(GLANCE_CLASS_BAND)
    img_end = master_stack.select(f"y{year_list[-1]}").rename(GLANCE_CLASS_BAND)

    classes = list(GLANCE_METADATA.keys())
    img_stack_t = ee.Image([master_stack.select(f"y{y}").rename(GLANCE_CLASS_BAND) for y in year_list[:-1]])
    img_stack_t1 = ee.Image([master_stack.select(f"y{y}").rename(GLANCE_CLASS_BAND) for y in year_list[1:]])

    component_bands = []
    for i in classes:
        for j in classes:
            if i == j:
                continue

            E_ij = img_start.eq(i).And(img_end.eq(j)).unmask(0)
            E_ji = img_start.eq(j).And(img_end.eq(i)).unmask(0)

            V_ij = img_stack_t.eq(i).And(img_stack_t1.eq(j)).reduce(ee.Reducer.sum()).unmask(0)
            V_ji = img_stack_t.eq(j).And(img_stack_t1.eq(i)).reduce(ee.Reducer.sum()).unmask(0)

            diff_ij = V_ij.subtract(E_ij)
            diff_ji = V_ji.subtract(E_ji)
            X_ij = diff_ij.min(diff_ji).max(0)

            if component == "exchange":
                component_bands.append(X_ij.rename(f"C_{i}_{j}"))
            else:
                S_ij = V_ij.subtract(X_ij).subtract(E_ij).max(0)
                component_bands.append(S_ij.rename(f"C_{i}_{j}"))

    component_image = ee.Image(component_bands).reduce(ee.Reducer.sum()).toInt16()
    component_image = component_image.updateMask(global_mask)
    component_image = component_image.unmask(nodata_val)
    component_image = component_image.set('system:no_data_value', nodata_val)

    return component_image


def export_alternation_exchange_task_gee(
    year_list: list,
    drive_folder: str,
    scale: int = 300,
    nodata_val: int = 255,
    full_year_list: list = None,
) -> ee.batch.Task:
    """
    Compute and export a raster representing the total per-pixel Alternation
    Exchange component, summed across all class-transition pairs.
    """
    if GLOBAL_GEOM is None:
        raise ValueError(
            "GLOBAL_GEOM is not initialized. Please call "
            "utils.initialize_active_region(region_code) before running tasks."
        )

    exchange_image = _compute_alternation_component_raster(
        component="exchange",
        year_list=year_list,
        scale=scale,
        nodata_val=nodata_val,
        full_year_list=full_year_list,
    )

    start_year, end_year = year_list[0], year_list[-1]
    task_desc = f"Alternation_Exchange_{start_year}_{end_year}"
    task = ee.batch.Export.image.toDrive(
        image=exchange_image,
        description=task_desc,
        folder=drive_folder,
        scale=scale,
        region=GLOBAL_GEOM,
        maxPixels=1e13,
        crs="EPSG:4326",
    )

    task.start()
    print(f"Task '{task_desc}' submitted to Google Earth Engine with NoData: {nodata_val}")

    return task


def export_alternation_shift_task_gee(
    year_list: list,
    drive_folder: str,
    scale: int = 300,
    nodata_val: int = 255,
    full_year_list: list = None,
) -> ee.batch.Task:
    """
    Compute and export a raster representing the total per-pixel Alternation
    Shift component, summed across all class-transition pairs.
    """
    if GLOBAL_GEOM is None:
        raise ValueError(
            "GLOBAL_GEOM is not initialized. Please call "
            "utils.initialize_active_region(region_code) before running tasks."
        )

    shift_image = _compute_alternation_component_raster(
        component="shift",
        year_list=year_list,
        scale=scale,
        nodata_val=nodata_val,
        full_year_list=full_year_list,
    )

    start_year, end_year = year_list[0], year_list[-1]
    task_desc = f"Alternation_Shift_{start_year}_{end_year}"
    task = ee.batch.Export.image.toDrive(
        image=shift_image,
        description=task_desc,
        folder=drive_folder,
        scale=scale,
        region=GLOBAL_GEOM,
        maxPixels=1e13,
        crs="EPSG:4326",
    )

    task.start()
    print(f"Task '{task_desc}' submitted to Google Earth Engine with NoData: {nodata_val}")

    return task


def plot_alternation_shift_map(
    output_dir: str,
    nodata_val: int,
    raster_filename: str,
    scale_factor: float = 0.05,
    # This function is now a duplicate of the one at line 3010 and should be removed.
) -> None:
    """
    Plot the Alternation Shift raster map with cartographic elements.

    Parameters
    ----------
    output_dir : str
        Directory containing the exported GEE tiles and where the map will be saved.
    nodata_val : int
        Value representing NoData in the raster to be masked out.
    raster_filename : str
        Prefix of the raster tiles to plot.
    scale_factor : float, optional
        Scale factor to downsample the global raster to fit into memory.

    Returns
    -------
    None
    """
    # 1. Locate all raster tiles exported by GEE
    raster_files = glob.glob(os.path.join(
        output_dir,
        f"{raster_filename}*.tif")
    )
    if not raster_files:
        raise FileNotFoundError(
            f"Raster tiles not found for prefix: {raster_filename}. Make sure the GEE export finished."
        )

    # 2. Create a temporary Virtual Raster (VRT) to merge tiles dynamically
    vrt_path = os.path.join(
        output_dir,
        "merged_shift.vrt"
    )
    files_str = " ".join([f'"{f}"' for f in raster_files])
    os.system(f"gdalbuildvrt {vrt_path} {files_str}")

    # 3. Calculate pixel size for scale bar
    pixel_size_km = compute_display_pixel_size_km(
        raster_path=vrt_path,
        downsample_factor=scale_factor,
    )

    # 4. Read raster and basic metadata with downsampling
    with rasterio.open(vrt_path) as src:
        out_shape = (
            max(1, int(src.height * scale_factor)),
            max(1, int(src.width * scale_factor)),
        )
        data = src.read(
            1,
            out_shape=out_shape,
            resampling=rasterio.enums.Resampling.nearest,
        )

        # Force masking using the provided nodata value
        data_masked = np.ma.masked_equal(data, nodata_val)

        src_crs = src.crs
        # Adjust the affine transform for the new downsampled resolution
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[1]),
            (src.height / data.shape[0]),
        )
        height, width = data.shape

    # 5. Figure
    fig, ax = plt.subplots(
        figsize=(20, 10),
        dpi=300
    )

    # Determine max value for colormap
    try:
        data_max = int(np.ma.max(data_masked))
    except:
        data_max = 1

    if data_max <= 0:
        data_max = 1

    # 6. Discrete Colormap Configuration
    original_cmap = plt.get_cmap("viridis_r")
    # Define the color for value 0 (Background/Gray)
    colors_list = ["#c0c0c0"] + [
        original_cmap(i) for i in np.linspace(0, 1, data_max)
    ]
    cmap = ListedColormap(colors_list)
    bounds = np.arange(-0.5, data_max + 1.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    # 7. Plot raster
    ax.imshow(
        data_masked,
        cmap=cmap,
        interpolation="nearest",
        norm=norm,
    )

    # 8. Legend Configuration
    legend_elements = []

    # Extract unique values actually present in the masked raster data
    present_values = np.unique(data_masked.compressed())

    for i in range(0, data_max + 1):
        # Append to legend ONLY if the value is present in the map
        if i in present_values:
            legend_elements.append(
                Patch(
                    facecolor=cmap(norm(i)),
                    edgecolor="none",
                    linewidth=0,
                    label=str(i),
                ),
            )

    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=12,
        borderpad=1.2,
        title="Shift",
        title_fontsize=14,
        alignment="left",
        handletextpad=0.8,
        columnspacing=2,
        labelspacing=0.8,
        handlelength=2.0,
        handleheight=1.5,
    )

    # 9. Cartographic elements
    degree_in_meters = 111320.0
    # Use pixel size if CRS is projected, otherwise use degree equivalent
    dx_meters = degree_in_meters if ax.get_xlim()[1] <= 180.5 else (pixel_size_km * 1000)

    def km_formatter(value, unit):
        if unit == "Mm":
            return f"{int(value * 1000)} km"
        return f"{int(value)} {unit}"

    scalebar = ScaleBar(
        dx=dx_meters,
        units="m",
        length_fraction=0.15,
        location="lower left",
        box_alpha=0.6,
        scale_formatter=km_formatter,
    )
    ax.add_artist(scalebar)

    try:
        north_arrow(
            ax,
            location="upper right",
            shadow=False,
            rotation={"degrees": 0},
            scale=0.5,
        )
    except NameError:
        print("north_arrow function not found. Skipping north arrow.")

    # 10. Axes styling
    ax.set_title(
        "Alternation Shift",
        fontsize=18,
        pad=10
    )
    ax.set_aspect("equal")

    # Define transformers for lat/lon tick labels
    to_latlon = Transformer.from_crs(
        src_crs,
        "EPSG:4326",
        always_xy=True
    )

    def format_lon(x, pos):
        x = np.clip(x, 0, width - 1)
        x_proj, y_proj = rasterio.transform.xy(transform, height // 2, x)
        lon, _ = to_latlon.transform(x_proj, y_proj)
        return f"{lon:.1f}°"

    def format_lat(y, pos):
        y = np.clip(y, 0, height - 1)
        x_proj, y_proj = rasterio.transform.xy(transform, y, width // 2)
        _, lat = to_latlon.transform(x_proj, y_proj)
        return f"{lat:.1f}°"

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=10,
        pad=4
    )
    plt.setp(
        ax.get_yticklabels(),
        rotation=90,
        va="center"
    )

    # 11. Save and Show
    maps_dir = os.path.join(
        output_dir,
        "maps"
    )
    os.makedirs(
        maps_dir,
        exist_ok=True
    )
    output_figure_path = os.path.join(
        maps_dir,
        "map_alternation_shift.png"
    )

    plt.savefig(
        output_figure_path,
        dpi=300,
        bbox_inches="tight",
        format="png",
        pad_inches=0.5,
    )
    plt.show()
    print(f"Map figure saved successfully to: {output_figure_path}")






def load_and_reorder_matrices(output_path: str, year_list: list) -> Dict[str, Any]:
    """
    Load transition matrices from CSVs and reorder them based on sum net change.

    Parameters
    ----------
    output_path : str
        Base directory path where the matrix CSV files are stored.
    year_list : list
        List of years representing the timeline (e.g., [2001, 2010, 2019]).

    Returns
    -------
    dict
        Dictionary containing the loaded and reordered matrices.
    """
    matrices = {}

    interval_str = f"{year_list[0]}-{year_list[-1]}"

    for key, meta in MATRIX_META.items():
        # Look directly in the output_path instead of the 'tables' subfolder
        csv_path = os.path.join(output_path, f"transition_matrix_{meta[0]}_{interval_str}.csv")
        
        if os.path.exists(csv_path):
            matrices[key] = load_square_matrix(csv_path=csv_path)
        else:
            print(f"Warning: Matrix file not found at {csv_path}")

    if matrices:
        matrices = reorder_all_matrices(matrices_dict=matrices)

    return matrices

# Define matrix metadata dictionary globally for reuse
MATRIX_META = {
    "sum": ["sum", "Time Intervals", "flow"],
    "alt_exc": ["alternation_exchange", "Alternation Exchange", "flow"],
    "alt_shift": ["alternation_shift", "Alternation Shift", "flow"],
    "ext": ["extent", "Extent", "stock"],
    "all_exc": ["allocation_exchange", "Allocation Exchange", "stock"],
    "qty_shift": ["quantity_allocation_shift", "Quantity & Allocation Shift", "stock"],
    "unacc_ext": ["unaccounted_extent", "Indirect", "stock"],
}

def calculate_trajectory_gee(
    image_stack: ee.Image,
    band_names: list,
    global_mask: ee.Image,
    nodata_val: int = NODATA_VALUE,
) -> ee.Image:
    """
    Classify a single pixel trajectory into five categories based on mathematical logic using GEE.

    Parameters
    ----------
    image_stack : ee.Image
        An ee.Image where each band represents a chronological time step.
    band_names : list
        A list of strings representing the ordered band names in the stack.
    nodata_val : int, optional
        NoData value to be masked out at the end, by default NODATA_VALUE (255).

    Returns
    -------
    ee.Image
        An ee.Image containing the classified trajectory codes (1 to 5).
    """
    # 0. Apply the global validity mask to the image stack at the very beginning
    # This ensures all subsequent operations work on a consistent set of valid pixels.
    masked_image_stack = image_stack.updateMask(global_mask)

    # 1. Extract the start and end images from the stack
    start_img = masked_image_stack.select(band_names[0])
    end_img = masked_image_stack.select(band_names[-1])

    # 2. Check if the start class equals the end class
    start_equals_end = start_img.eq(end_img)

    # 3. Shift the stack by 1 band to compare t and t+1 in parallel (vectorized)
    common_names = [f"b_{i}" for i in range(len(band_names) - 1)]
    stack_t = masked_image_stack.select(band_names[:-1]).rename(common_names)
    stack_t1 = masked_image_stack.select(band_names[1:]).rename(common_names)

    # 4. Check for a direct transition and path changes using native multi-band operations
    start_img_stack = ee.Image.cat([start_img] * len(common_names)).rename(common_names)
    end_img_stack = ee.Image.cat([end_img] * len(common_names)).rename(common_names)

    has_direct_transition = stack_t.eq(start_img_stack).And(stack_t1.eq(end_img_stack)).unmask(0).reduce(ee.Reducer.max())
    path_changes = stack_t.neq(stack_t1).reduce(ee.Reducer.sum())

    # 5. Deduce all_match_start mathematically from path_changes (0 changes means completely stable)
    all_match_start = path_changes.eq(0)

    # 6. Assign Trajectory 1 for completely stable pixels
    traj_1 = start_equals_end.And(all_match_start).multiply(1)

    # 7. Assign Trajectory 2 for stable extent with alternation
    traj_2 = start_equals_end.And(all_match_start.Not()).multiply(2)

    # 8. Identify pixels with extent change
    extent_change = start_equals_end.Not()

    # 9. Assign Trajectory 5 for extent change without direct transition
    traj_5 = extent_change.And(has_direct_transition.Not()).multiply(5)

    # 10. Assign Trajectory 3 for extent change without alternation
    traj_3 = extent_change.And(has_direct_transition).And(path_changes.eq(1)).multiply(3)

    # 11. Assign Trajectory 4 for extent change with alternation
    traj_4 = extent_change.And(has_direct_transition).And(path_changes.gt(1)).multiply(4)

    # 12. Combine all trajectory maps into a single output image
    trajectory_image = traj_1.add(traj_2).add(traj_3).add(traj_4).add(traj_5)

    # 13. Apply the global validity mask
    # The global_mask is already applied to the stack, so this final mask is redundant.

    return trajectory_image.rename('trajectory')

def build_glance_stack(
    year_list: list,
    collection_id: str,
    band_name: str,
    nodata_val: int,
) -> tuple:
    """
    Build an Earth Engine image stack from the specified collection,
    without pre-applying complex global masks to keep the computation graph clean.

    Parameters
    ----------
    year_list : list
        List of integer years to process.
    collection_id : str
        The GEE ImageCollection ID.
    band_name : str
        The band name to select.
    nodata_val : int
        The NoData value to mask out.

    Returns
    -------
    tuple
        A tuple containing the ee.Image stack and the list of band names.
    """
    collection = ee.ImageCollection(collection_id).select(band_name)
    images = []
    b_names = []

    for year in year_list:
        b_name = f"y{year}"
        b_names.append(b_name)
        
        img = collection.filter(ee.Filter.calendarRange(year, year, "year")).mosaic()
        images.append(
            img.rename(b_name)
        )

    stack = ee.Image(images)

    return stack, b_names


###############################################################################
#                                                                             #
#                  5.1 CHANGE COMPONENTS                                      #
#                                                                             #
###############################################################################

def export_interval_transition_matrices_gee(
    year_list: list,
    drive_folder: str,
    collection_id: str,
    band_name: str,
    scale: int = 300,
    nodata_val: int = 255,
    full_year_list: list = None,
) -> list:
    """
    Export LULC transition matrices between consecutive years to Google Drive.

    Parameters
    ----------
    year_list : list
        List of years representing the timeline.
    drive_folder : str
        Google Drive folder name for the exported CSV files.
    collection_id : str
        GEE ImageCollection ID containing the LULC rasters.
    band_name : str
        Band name representing the LULC classes.
    scale : int, optional
        Spatial resolution for the export, by default 300.
    full_year_list : list of int, optional
        The complete timeline to construct the consistent global validity mask.
    nodata_val : int, optional
        Value representing NoData/Background to be masked out, by default 255.

    Returns
    -------
    list
        List of submitted ee.batch.Task objects.
    """
    tasks = []

    if full_year_list is None:
        full_year_list = year_list

    # 1. Build global mask using the FULL timeline to ensure mathematical consistency
    full_stack, _ = build_glance_stack(
        year_list=full_year_list,
        collection_id=collection_id,
        band_name=band_name,
        nodata_val=nodata_val,
    )
    global_mask = full_stack.neq(nodata_val).unmask(0).reduce(ee.Reducer.min())

    # 2. Build target stack for the specific years we want to export tasks for
    target_stack, target_band_names = build_glance_stack(
        year_list=year_list,
        collection_id=collection_id,
        band_name=band_name,
        nodata_val=nodata_val,
    )

    yearly_by_year = {
        year: target_stack.select(target_band_names[idx]).rename(band_name)
        for idx, year in enumerate(year_list)
    }

    for i in range(len(year_list) - 1):
        start_year = year_list[i]
        end_year = year_list[i + 1]

        img_start = yearly_by_year[start_year].updateMask(global_mask)
        img_end = yearly_by_year[end_year].updateMask(global_mask)

        transition_img = img_start.multiply(100).add(img_end).rename("transition")

        histogram = transition_img.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram().unweighted(),
            geometry=GLOBAL_GEOM,
            scale=scale,
            crs="EPSG:4326",
            maxPixels=1e13,
            tileScale=16,
        )

        feature = ee.Feature(None, histogram)
        fc = ee.FeatureCollection([feature])

        task_name = f"transition_{start_year}_{end_year}"
        task = ee.batch.Export.table.toDrive(
            collection=fc,
            description=task_name,
            folder=drive_folder,
            fileNamePrefix=f"transition_matrix_{start_year}-{end_year}",
            fileFormat="CSV",
        )
        task.start()
        tasks.append(task)

    return tasks


def format_raw_gee_csv_to_matrix(
    raw_csv_path: str,
    final_csv_path: str,
    classes: list = None
) -> None:
    """
    Reads a raw GEE CSV with 49 transition bands (T_i_j) and formats it
    into a standard 7x7 transition matrix CSV.

    Parameters
    ----------
    raw_csv_path : str
        Path to the raw input CSV file exported from GEE.
    final_csv_path : str
        Path to save the formatted 7x7 matrix CSV file.
    classes : list, optional
        List of class IDs to use for the matrix index and columns.
        If None, defaults to the keys from GLANCE_METADATA.
    """
    if not os.path.exists(raw_csv_path):
        print(f"⚠️ Raw GEE file not found, skipping format: {raw_csv_path}")
        return

    # Default to GLANCE classes if not provided
    if classes is None:
        classes = list(GLANCE_METADATA.keys())

    df_raw = pd.read_csv(raw_csv_path)
    matrix = pd.DataFrame(0.0, index=classes, columns=classes)

    for i in classes:
        for j in classes:
            col_name = f"T_{i}_{j}"
            if col_name in df_raw.columns:
                # Ensure value is numeric, default to 0 if not
                value = pd.to_numeric(df_raw.loc[0, col_name], errors='coerce')
                matrix.at[i, j] = value if pd.notna(value) else 0.0

    # Ensure the output directory exists
    output_dir = os.path.dirname(final_csv_path)
    os.makedirs(output_dir, exist_ok=True)

    matrix.to_csv(final_csv_path)
    print(f"✅ Formatted 7x7 matrix saved to: {final_csv_path}")


def parse_gee_raw_csv(file_path: str) -> pd.DataFrame:
    """
    Parse a raw GEE transition CSV into a square Pandas DataFrame matrix.
    """
    df_raw = pd.read_csv(file_path)
    dict_str = None
    for col in df_raw.columns:
        val = str(df_raw[col].iloc[0])
        if val.startswith("{") and "=" in val:
            dict_str = val
            break

    if not dict_str:
        raise ValueError(f"Raw GEE dictionary string not found in {file_path}.")

    dict_str = dict_str.strip("{}")
    pairs = dict_str.split(", ")

    transitions = {}
    classes = set()

    for pair in pairs:
        if not pair:
            continue
        k, v = pair.split("=")
        if k.strip().lower() == "null":
            continue
        k_int = int(k)
        val_float = float(v)
        # For GLanCE, transition classes are encoded as start * 100 + end
        s_c = k_int // 100
        e_c = k_int % 100
        transitions[(s_c, e_c)] = val_float
        classes.add(s_c)
        classes.add(e_c)

    classes_sorted = sorted(list(classes))
    df_mat = pd.DataFrame(0.0, index=classes_sorted, columns=classes_sorted)

    for (s, e), v in transitions.items():
        df_mat.at[s, e] = v

    return df_mat

class ComponentCalculator:
    """
    Compute change components for a matrix.

    Supports pre-decomposed matrices via force_component parameter.
    """

    def __init__(self, transition_matrix: np.ndarray) -> None:
        """
        Initialize the calculator with a transition matrix.

        Parameters
        ----------
        transition_matrix : np.ndarray
            Square matrix representing transitions.
        """
        self.matrix = transition_matrix.astype(float)
        self.num_classes = transition_matrix.shape[0]
        self.class_components: list[dict] = []

    def calculate_components(self, force_component: str = None) -> "ComponentCalculator":
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
        for class_idx in range(self.num_classes):
            gain_sum = np.sum(self.matrix[:, class_idx])
            loss_sum = np.sum(self.matrix[class_idx, :])

            # Standard net change calculation
            q_gain = max(0.0, gain_sum - loss_sum)
            q_loss = max(0.0, loss_sum - gain_sum)

            if force_component == "Exchange":
                exchange = loss_sum - self.matrix[class_idx, class_idx]
                shift = 0.0
                q_gain, q_loss = gain_sum - loss_sum, loss_sum - gain_sum
            elif force_component == "Shift":
                exchange = 0.0
                shift = loss_sum - self.matrix[class_idx, class_idx]
                q_gain, q_loss = 0.0, 0.0
            else:
                # Standard Pontius decomposition
                mutual = np.sum(np.minimum(self.matrix[class_idx, :], self.matrix[:, class_idx]))
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


def process_matrix(
    matrix_type: str,
    input_dir: str,
    years_list: list,
    class_labels_dict: dict,
    start_year=None,
    end_year=None,
) -> list:
    """
    Search for a transition matrix file and calculate its change components.

    Parameters
    ----------
    matrix_type : str
        Type of matrix ("interval", "extent", "sum", etc.).
    input_dir : str
        Directory where CSV files are stored.
    years_list : list
        List of all years in the timeline.
    class_labels_dict : dict
        Dictionary mapping class IDs to metadata.
    start_year : str or int, optional
        Start year for interval matrices.
    end_year : str or int, optional
        End year for interval matrices.

    Returns
    -------
    list[dict]
        List of dictionaries containing component values per class.
    """
    results = []
    patterns = []

    # 1. Determine naming patterns to search for
    if matrix_type == "interval":
        s_str, e_str = str(start_year), str(end_year)
        patterns.extend([
            f"transition_{s_str}_{e_str}.csv",
            f"transition_matrix_{s_str}-{e_str}.csv",
        ])
        label_time = f"{s_str}-{e_str}"
    else:
        y0_str, yN_str = str(years_list[0]), str(years_list[-1])
        patterns.extend([
            f"transition_matrix_{matrix_type}_{y0_str}-{yN_str}.csv",
        ])
        label_time = matrix_type

    # 2. Find the existing file in the main directory
    full_path = None
    for p in patterns:
        path = os.path.join(input_dir, p)
        if os.path.exists(path):
            full_path = path
            break

    if not full_path:
        return []

    # 3. Process components
    force_comp = (
        "Exchange" if "exchange" in matrix_type else ("Shift" if "shift" in matrix_type else None)
    )

    # Load matrix safely handling standard square CSVs or raw GEE dict formats
    try:
        df_mat = pd.read_csv(full_path, index_col=0)
        if df_mat.shape[1] > 0 and isinstance(df_mat.iloc[0, 0], str) and df_mat.iloc[0, 0].startswith("{"):
            raise ValueError("Raw GEE format detected")
    except (ValueError, TypeError):
        # Assumes parse_gee_raw_csv is defined in utils.py from the previous step
        df_mat = parse_gee_raw_csv(full_path)

    calc = ComponentCalculator(df_mat.values).calculate_components(force_component=force_comp)

    for idx, class_id in enumerate([int(c) for c in df_mat.index]):
        cls_name = class_labels_dict.get(class_id, {}).get("name", f"Class {class_id}")
        comp_vals = calc.class_components[idx]

        for comp_name in ["Quantity", "Exchange", "Shift"]:
            label_comp = comp_name
            if matrix_type in ["extent", "sum"]:
                label_comp = f"Allocation_{comp_name}"
            if "alternation" in matrix_type:
                label_comp = f"Alternation_{comp_name}"

            results.append({
                "Time_Interval": label_time,
                "Class": cls_name,
                "Component": label_comp,
                "Gain": comp_vals[f"{comp_name}_Gain"],
                "Loss": comp_vals[f"{comp_name}_Loss"],
            })
    return results

def generate_all_heatmaps(
    matrices_dict: dict,
    output_path: str,
    interval_str: str,
    years: list,
    style_config: dict,
) -> None:
    """
    Iterate over the matrices dictionary and generate a heatmap for each.

    Parameters
    ----------
    matrices_dict : dict
        Dictionary containing the dataframes to plot.
    output_path : str
        Base directory path to save the generated charts.
    interval_str : str
        String representing the time interval.
    years : list
        List of years processed.
    style_config : dict
        Dictionary containing style configurations for the plot.

    Returns
    -------
    None
    """
    import os

    print(
        "Generating Heatmaps...",
    )

    # Enforce unified color scale and maximum value of 50 million pixels globally
    style_config = style_config.copy()
    style_config["vmax"] = 50_000_000.0
    style_config["cmap"] = "YlOrRd"

    charts_dir = os.path.join(
        output_path,
        "charts",
    )

    os.makedirs(
        charts_dir,
        exist_ok=True,
    )

    title_map = {
        "ext": "Extent", # This was the old title_map
        "sum": "Time Intervals",
        "all_exc": "Allocation Exchange",
        "alloc_shift": "Allocation Shift",
        "qty_shift": "Quantity & Allocation Shift",
        "alt_exc": "Alternation Exchange",
        "alt_shift": "Alternation Shift",
        "unacc_ext": "Indirect",
    } # I'm keeping this logic but it will now be consistent

    for key, df in matrices_dict.items():
        if df is None or df.empty:
            continue

        base_name = title_map.get(
            key,
            key.capitalize(),
        )

        if key == "sum" or "alt" in key:
            formatted_interval = interval_str.replace("-", "...")
            full_title = f"{base_name} {formatted_interval}"
        else:
            full_title = f"{base_name} {interval_str}"

        out_file = os.path.join(
            charts_dir,
            f"heatmap_{key}_{interval_str}.png",
        )

        plot_heatmap(
            df=df,
            title=full_title,
            save_path=out_file,
            **style_config,
            matrix_key=key
        )

        print(
            f"-> Saved heatmap: {out_file}",
        )







###############################################################################
#                                                                             #
#                  6. Convert CSV to Transition Matrix                        #
#                                                                             #
###############################################################################

def load_global_transition_matrices(
    drive_path
):
    """
    Loads exported GEE CSVs from Drive and converts them to transition matrices.

    Parameters
    ----------
    drive_path : str
        The full path to the Google Drive folder containing the CSV files.

    Returns
    -------
    dict of pd.DataFrame
        A dictionary where keys are 'YYYY_YYYY' and values are pivot matrices.
    """

    # 1. Identify all CSV files in the specified directory
    search_pattern = os.path.join(drive_path, "*.csv")
    file_list = glob.glob(search_pattern)
    
    # 2. Create a mapping of class IDs to names from metadata
    class_names = {
        k: v['name'] 
        for k, v in GLANCE_METADATA.items()
    }
    
    all_matrices = {}

    # 3. Iterate through each file to reconstruct the matrix
    for filepath in file_list:
        df_raw = pd.read_csv(filepath)
        
        # 4. Extract the transition label from the filename
        filename = os.path.basename(filepath)
        label = filename.replace(".csv", "").replace("transition_", "")
        
        hist_data = {}
        numeric_cols = [c for c in df_raw.columns if str(c).isdigit()]
        
        # 5. Parse the histogram
        if numeric_cols:
            for col in numeric_cols:
                hist_data[col] = df_raw[col].sum()
        else:
            target_col = 'LC' if 'LC' in df_raw.columns else 'transition'
            if target_col not in df_raw.columns:
                continue
            hist_str = str(df_raw[target_col].iloc[0])
            clean_str = hist_str.strip('{}')
            pairs = clean_str.split(', ')
            for pair in pairs:
                if '=' in pair:
                    k, v = pair.split('=')
                    hist_data[k] = float(v)

        records = []
        
        # 6. Decode transition codes into 'From' and 'To' classes
        for code, count in hist_data.items():
            code_int = int(float(code))
            id_from = code_int // 100
            id_to = code_int % 100

            # 7. Filter records using valid metadata classes
            if id_from in class_names and id_to in class_names:
                records.append({
                    "From": class_names[id_from],
                    "To": class_names[id_to],
                    "Pixels": int(count)
                })

        # 8. Pivot the records into a formal transition matrix
        if records:
            df_temp = pd.DataFrame(records)
            matrix = df_temp.pivot(
                index="From", 
                columns="To", 
                values="Pixels"
            ).fillna(0)

            # 9. Store the resulting DataFrame in the output dictionary
            all_matrices[label] = matrix
            
    return all_matrices

def convert_matrices_to_area(
    matrices_dict,
    pixel_size=30
):
    """
    Converts pixel counts in transition matrices to area in square kilometers.

    Parameters
    ----------
    matrices_dict : dict of pd.DataFrame
        A dictionary where values are transition matrices in pixel counts.
    pixel_size : int, optional
        The edge length of a single pixel in meters. Defaults to 30.

    Returns
    -------
    dict of pd.DataFrame
        A dictionary of transition matrices with values in km^2.
    """
    # 1. Calculate the conversion factor from pixels to km^2
    # Area of one pixel in m^2 = pixel_size * pixel_size
    # Conversion to km^2 = m^2 / 1,000,000
    conversion_factor = (pixel_size ** 2) / 1000000
    
    area_matrices = {}

    # 2. Iterate through the dictionary of matrices
    for label, matrix in matrices_dict.items():
        # 3. Multiply the entire DataFrame by the conversion factor
        # Pandas handles the element-wise multiplication automatically
        area_matrix = matrix * conversion_factor
        
        # 4. Round the results to two decimal places for readability
        area_matrices[label] = area_matrix.round(2)
        
    return area_matrices

def save_area_matrices_to_csv(
    area_matrices,
    output_dir
):
    """
    Saves a dictionary of area matrices to individual CSV files.

    Parameters
    ----------
    area_matrices : dict of pd.DataFrame
        Dictionary containing the transition matrices in km^2.
    output_dir : str
        The directory path where the CSV files will be saved.

    Returns
    -------
    list of str
        A list of file paths to the saved CSV files.
    """
    # 1. Check if the output directory exists and create it if necessary
    if not os.path.exists(output_dir):
        os.makedirs(
            output_dir, 
            exist_ok=True
        )
    
    saved_files = []

    # 2. Iterate through the dictionary to process each transition matrix
    for label, matrix in area_matrices.items():
        # 3. Construct the specific filename for the km2 results
        filename = f"transition_matrix_km2_{label}.csv"
        filepath = os.path.join(
            output_dir, 
            filename
        )
        
        # 4. Save the DataFrame to CSV including the class names in the index
        matrix.to_csv(
            filepath,
            index=True
        )
        
        saved_files.append(filepath)
        print(f"Successfully saved: {filename}")
        
    return saved_files

###############################################################################
#                                                                             #
#                  7. Compute Sum matrix                                      #
#                                                                             #
###############################################################################

def compute_sum_matrix(
    input_dir: str,
    output_path: str,
    file_prefix: str = "transition_",
) -> pd.DataFrame:
    """
    Compute the SUM transition matrix by aggregating all annual intervals.

    Parameters
    ----------
    input_dir : str
        Path to the directory containing annual transition CSV files.
    output_path : str
        Full path (including filename) to save the resulting SUM matrix.
    file_prefix : str, optional
        Prefix of the annual transition files to look for, by default "transition_".

    Returns
    -------
    pd.DataFrame
        The aggregated SUM transition matrix.
    """
    # 1. List all annual transition files (e.g., transition_2001_2002.csv, transition_2002_2003.csv...)
    pattern_underscore = os.path.join(input_dir, f"{file_prefix}????_????.csv")
    pattern_hyphen = os.path.join(input_dir, f"{file_prefix}????-????.csv")
    
    all_files = glob.glob(pattern_underscore) + glob.glob(pattern_hyphen)

    if not all_files:
        raise FileNotFoundError(
            f"No annual transition matrices found in {input_dir} with prefix '{file_prefix}'",
        )

    # 2. Sort files to ensure chronological order (optional, but good practice)
    all_files.sort()

    df_sum = None

    # 3. Iterate and aggregate
    for file_path in all_files:
        # Load current annual matrix (using load_square_matrix to handle raw/square formats)
        df_annual = load_square_matrix(file_path)

        if df_sum is None:
            # Initialize with the first matrix
            df_sum = df_annual.copy()
        else:
            # Sum values cell by cell
            df_sum = df_sum.add(
                df_annual,
                fill_value=0.0,
            )

    # 4. Save the consolidated SUM matrix
    if df_sum is not None:
        df_sum.to_csv(output_path)
        print(f"SUM matrix successfully saved to: {output_path}")

    return df_sum

###############################################################################
#                                                                             #
#                  8. Compute Exchange and Shift                              #
#                                                                             #
###############################################################################
def compute_and_save_components(
    df_sum: pd.DataFrame,
    df_ext: pd.DataFrame,
    output_dir: str,
    period_label: str = "2001-2019",
) -> None:
    """
    Decompose Sum and Extent matrices into change components.

    Logic:
    1. Allocation: Derived from Extent matrix (Aggregate level).
    2. Alternation: Derived from (Sum - Extent) (Trajectory level).
    This implementation follows the logic from Pontius et al. where the
    diagonal (stability) is correctly handled for each component.

    Parameters
    ----------
    df_sum : pd.DataFrame
        Aggregated transition matrix (Sum of annual intervals).
    df_ext : pd.DataFrame
        Direct transition matrix (Start year vs End year).
    output_dir : str
        Directory path to save the resulting CSV files.
    period_label : str, optional
        Year range label for filename, by default "2001-2019".

    Returns
    -------
    None
    """
    # 1. Align and Sort Matrices based on GLANCE_METADATA order
    name_to_id = {v['name']: k for k, v in GLANCE_METADATA.items()}

    def _sort_key(label):
        if label in name_to_id:
            return (0, name_to_id[label])
        try:
            return (0, int(label))
        except (ValueError, TypeError):
            return (1, str(label))

    all_labels = sorted(
        list(set(df_sum.index).union(df_sum.columns)),
        key=_sort_key
    )

    df_s = df_sum.reindex(index=all_labels, columns=all_labels).fillna(0.0)
    df_e = df_ext.reindex(index=all_labels, columns=all_labels).fillna(0.0)

    # Convert to numpy arrays for calculation
    mat_sum = df_s.values
    mat_ext = df_e.values

    # 2. Calculate Allocation Components (from Extent matrix)
    # Allocation Exchange (C) = min(E_ij, E_ji). This preserves the diagonal.
    mat_c = np.minimum(mat_ext, mat_ext.T)

    # Quantity & Allocation Shift (Q) = E - C. Diagonal becomes zero.
    mat_q = mat_ext - mat_c

    # 3. Calculate Alternation Components (from Sum - Extent)
    alternation_raw = mat_sum - mat_ext

    # Alternation Exchange (X) = max(0, min(A_ij, A_ji)), where A = Sum - Extent.
    # This also preserves the diagonal (S_ii - E_ii).
    mat_x = np.maximum(0, np.minimum(alternation_raw, alternation_raw.T))

    # Alternation Shift (S) is the positive part of the remainder after exchange.
    mat_s = np.maximum(0, alternation_raw - mat_x)

    # Indirect (Unaccounted) Component (U) is calculated from the identity:
    # U = E + X + S - V (where V is Sum)
    mat_u = mat_ext + mat_x + mat_s - mat_sum

    # 4. Consolidate and Export all components to CSV
    components = {
        "sum": mat_sum,
        "extent": mat_ext,
        "allocation_exchange": mat_c,
        "quantity_allocation_shift": mat_q,
        "alternation_exchange": mat_x,
        "alternation_shift": mat_s,
        "unaccounted_extent": mat_u,  # Use key 'unaccounted_extent' for consistency with heatmap generator
    }

    for name, data in components.items():
        df_out = pd.DataFrame(
            data,
            index=all_labels,
            columns=all_labels
        )

        fname = f"transition_matrix_{name}_{period_label}.csv"
        path = os.path.join(output_dir, fname)
        df_out.to_csv(path)

        print(f"Component saved: {fname}")

###############################################################################
#                                                                             #
#                  9. Reorder Matrices by net change                          #
#                                                                             #
###############################################################################
def reorder_matrices_by_net_change(
    df_sum: pd.DataFrame,
    df_ext: pd.DataFrame,
    df_ext_exc: pd.DataFrame,
    df_ext_shift: pd.DataFrame,
    df_alt_exc: pd.DataFrame,
    df_alt_shift: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Reorder matrices from largest losers to largest gainers using net change.

    Parameters
    ----------
    df_sum : pd.DataFrame
        The aggregated SUM matrix used to calculate the sorting order.
    df_ext, df_ext_exc, df_ext_shift, df_alt_exc, df_alt_shift : pd.DataFrame
        The other component matrices to be reordered.

    Returns
    -------
    tuple[pd.DataFrame, ...]
        All input dataframes reindexed with the same optimized order.
    """
    # 1. Calculate Net Change (Gains - Losses)
    # Diagonal is ignored to focus only on transitions
    m_values = df_sum.values.copy()
    np.fill_diagonal(m_values, 0.0)
    
    gains = m_values.sum(axis=0)
    losses = m_values.sum(axis=1)
    net_change = gains - losses
    
    # 2. Define the sorting order (ascending: losers first)
    net_series = pd.Series(net_change, index=df_sum.index)
    order_labels = net_series.sort_values(ascending=True).index.tolist()

    # 3. Helper to apply the same order to any dataframe
    def _apply_order(df: pd.DataFrame) -> pd.DataFrame:
        return df.reindex(index=order_labels, columns=order_labels).fillna(0.0)

    # 4. Return all matrices reordered
    return (
        _apply_order(df_sum),
        _apply_order(df_ext),
        _apply_order(df_ext_exc),
        _apply_order(df_ext_shift),
        _apply_order(df_alt_exc),
        _apply_order(df_alt_shift),
    )


def export_global_overall_change_frequency_csv_gee(
    year_list: list,
    drive_folder: str,
    scale: int = 300,
    full_year_list: list = None,
) -> ee.batch.Task:
    """
    Compute and export a single CSV representing the overall frequency of changes
    (how many pixels changed 0, 1, 2, ... N times) across the entire timeline.
    """
    if GLOBAL_GEOM is None:
        raise ValueError(
            "GLOBAL_GEOM is not initialized. Please call "
            "utils.initialize_active_region(region_code) before running tasks."
        )

    print(f"Preparing Overall Change Frequency GEE Task for {year_list[0]}-{year_list[-1]}...")

    if full_year_list is None:
        full_year_list = year_list

    # 1. Build global mask using the FULL timeline to ensure mathematical consistency
    full_stack, _ = build_glance_stack(
        year_list=full_year_list,
        collection_id=GLANCE_COLLECTION_ID,
        band_name=GLANCE_CLASS_BAND,
        nodata_val=NODATA_VALUE,
    )
    global_mask = full_stack.neq(NODATA_VALUE).unmask(0).reduce(ee.Reducer.min())

    # 2. Build target stack for the specific years we want to export tasks for
    target_stack, target_band_names = build_glance_stack(
        year_list=year_list,
        collection_id=GLANCE_COLLECTION_ID,
        band_name=GLANCE_CLASS_BAND,
        nodata_val=NODATA_VALUE,
    )

    # 3. Shift the target stack by 1 band to compare t and t+1 in parallel (vectorized)
    stack_t = target_stack.select(target_band_names[:-1])
    stack_t1 = target_stack.select(target_band_names[1:])

    # 4. Calculate total changes across the entire timeline
    total_changes = stack_t.neq(stack_t1).reduce(ee.Reducer.sum()).rename('num_changes')

    total_changes_masked = total_changes.updateMask(global_mask)

    # 5. Compute the frequency histogram of the total changes for these pixels
    histogram = total_changes_masked.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram().unweighted(),
        geometry=GLOBAL_GEOM,
        scale=scale,
        crs="EPSG:4326",
        maxPixels=1e13,
        tileScale=16,
    ).get('num_changes')

    # Handle possible nulls if no change occurred
    hist_dict = ee.Dictionary(ee.Algorithms.If(histogram, histogram, {}))
    feature = ee.Feature(None, hist_dict)
    fc = ee.FeatureCollection([feature])

    # 6. Configure and start the export task
    start_year = year_list[0]
    end_year = year_list[-1]
    export_name = f"Number_Change_Overall_{start_year}_{end_year}"
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=export_name,
        folder=drive_folder,
        fileNamePrefix=export_name,
        fileFormat="CSV",
    )
    task.start()
    print(f"Task '{export_name}' submitted to Google Earth Engine.")
    return task


###############################################################################
#                                                                             #
#                  10. GLOBAL MOSAIC CLASS & FUNCTIONS                        #
#                                                                             #
###############################################################################

class GlanceMosaicker:
    """
    Manages continental GLanCE regions to build a consistent global mosaic
    and unified geometry operations directly inside Google Earth Engine.
    """

    def __init__(self, region_codes: List[str]) -> None:
        """
        Initialize the mosaicker with a customizable set of continental regions.
        
        Parameters
        ----------
        region_codes : list of str
            List of regional codes to compose the mosaic (e.g., ['EU', 'AF', 'SA']).
        """
        if not region_codes:
            raise ValueError("You must provide at least one region code to construct the mosaic.")
            
        self.region_codes = region_codes
        self._validate_regions()
        self.unified_geometry = self._build_unified_geometry()

    def _validate_regions(self) -> None:
        """Validate if all provided region codes exist in the registry."""
        for code in self.region_codes:
            if code not in GLANCE_REGIONS_REGISTRY:
                raise ValueError(
                    f"Region code '{code}' is invalid. "
                    f"Choose from: {list(GLANCE_REGIONS_REGISTRY.keys())}"
                )

    def _build_unified_geometry(self) -> ee.Geometry:
        """
        Combine the individual rectangular bounding boxes into a single 
        ee.Geometry.MultiPolygon nativelly in GEE.
        """
        geometries = []
        for code in self.region_codes:
            geom_coords = GLANCE_REGIONS_REGISTRY[code]['geom']
            # Create individual non-planar rectangular polygons
            rect = ee.Geometry.Rectangle(geom_coords, "EPSG:4326", False)
            geometries.append(rect)
        
        # Perform a spatial union to create a single clean multipolygon geometry
        return ee.Geometry.MultiPolygon(geometries)

    def get_global_crs_and_transform(self, scale: int = 300) -> Tuple[str, List[float]]:
        """
        Get the global Cylindrical Equal Area projection parameters (EPSG:6933).
        This CRS preserves area properties globally, making pixel counts and area calculations
        extremely accurate across different latitudes.
        
        Parameters
        ----------
        scale : int
            The spatial resolution in meters (e.g., 300 or 30).
            
        Returns
        -------
        tuple
            A tuple containing the CRS string ("EPSG:6933") and the affine transform list.
        """
        crs = "EPSG:6933"
        # Setup a standard global transform aligned to the coordinate origin
        transform = [scale, 0, 0, 0, -scale, 0]
        return crs, transform

    def get_mosaicked_image_collection(self, year: int) -> ee.Image:
        """
        Retrieve and mosaic the GLanCE land cover images for the specified year
        restricted to the unified continental boundary.
        
        Parameters
        ----------
        year : int
            The target year to mosaic (e.g., 2019).
            
        Returns
        -------
        ee.Image
            The unified, masked land cover image for the selected year.
        """
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        # Filter GLanCE collection by date and band
        collection = (
            ee.ImageCollection(GLANCE_COLLECTION_ID)
            .filterDate(start_date, end_date)
            .select(GLANCE_CLASS_BAND)
        )

        # Mosaic overlapping zones (resolving potential duplicates via first-order placement)
        # and clip strictly to our continental limits
        mosaicked_image = collection.mosaic().clip(self.unified_geometry)
        
        # Enforce uint8 casting to preserve data type limits
        return mosaicked_image.toByte()

def plot_alternation_shift_map(
    output_dir: str,
    nodata_val: int,
    raster_filename: str,
    scale_factor: float = 0.05,
) -> None:
    """
    Plot the Alternation Shift raster map with cartographic elements.

    Parameters
    ----------
    output_dir : str
        Directory containing the exported GEE tiles and where the map will be saved.
    nodata_val : int
        Value representing NoData in the raster to be masked out.
    raster_filename : str
        Prefix of the raster tiles to plot.
    scale_factor : float, optional
        Scale factor to downsample the global raster to fit into memory.

    Returns
    -------
    None
    """
    # 1. Locate all raster tiles exported by GEE
    raster_files = glob.glob(os.path.join(
        output_dir,
        f"{raster_filename}*.tif")
    )
    if not raster_files:
        raise FileNotFoundError(
            f"Raster tiles not found for prefix: {raster_filename}. Make sure the GEE export finished."
        )

    # 2. Create a temporary Virtual Raster (VRT) to merge tiles
    vrt_path = os.path.join(
        output_dir,
        "merged_shift.vrt"
    )
    files_str = " ".join([f'"{f}"' for f in raster_files])
    os.system(f"gdalbuildvrt {vrt_path} {files_str}")

    # 3. Calculate pixel size for the scale bar
    pixel_size_km = compute_display_pixel_size_km(
        raster_path=vrt_path,
        downsample_factor=scale_factor,
    )

    # 4. Read raster and basic metadata with downsampling
    with rasterio.open(vrt_path) as src:
        out_shape = (
            max(1, int(src.height * scale_factor)),
            max(1, int(src.width * scale_factor)),
        )
        data = src.read(
            1,
            out_shape=out_shape,
            resampling=rasterio.enums.Resampling.nearest,
        )

        # Mask using the provided nodata value
        data_masked = np.ma.masked_equal(data, nodata_val)

        src_crs = src.crs
        # Adjust the affine transform for the new downsampled resolution
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[1]),
            (src.height / data.shape[0]),
        )
        height, width = data.shape

    # 5. Figure
    fig, ax = plt.subplots(
        figsize=(20, 10),
        dpi=300
    )

    # Determine max value for colormap
    try:
        data_max = int(np.ma.max(data_masked))
    except:
        data_max = 1

    if data_max <= 0:
        data_max = 1

    # 6. Discrete colormap configuration
    original_cmap = plt.get_cmap("viridis_r")
    # Define the color for value 0 (Background/Gray)
    colors_list = ["#c0c0c0"] + [
        original_cmap(i) for i in np.linspace(0, 1, data_max)
    ]
    cmap = ListedColormap(colors_list)
    bounds = np.arange(-0.5, data_max + 1.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    # 7. Plot raster
    ax.imshow(
        data_masked,
        cmap=cmap,
        interpolation="nearest",
        norm=norm,
    )

    # 8. Legend configuration
    legend_elements = []

    # Extract unique values present in the masked raster data
    present_values = np.unique(data_masked.compressed())

    for i in range(0, data_max + 1):
        # Append to legend only if the value is present in the map
        if i in present_values:
            legend_elements.append(
                Patch(
                    facecolor=cmap(norm(i)),
                    edgecolor="none",
                    linewidth=0,
                    label=str(i),
                ),
            )

    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=12,
        borderpad=1.2,
        title="Shift",
        title_fontsize=14,
        alignment="left",
        handletextpad=0.8,
        columnspacing=2,
        labelspacing=0.8,
        handlelength=2.0,
        handleheight=1.5,
    )

    # 9. Cartographic elements
    degree_in_meters = 111320.0
    # Use pixel size if CRS is projected, otherwise use degree equivalent
    dx_meters = degree_in_meters if ax.get_xlim()[1] <= 180.5 else (pixel_size_km * 1000)

    def km_formatter(value, unit):
        if unit == "Mm":
            return f"{int(value * 1000)} km"
        return f"{int(value)} {unit}"

    scalebar = ScaleBar(
        dx=dx_meters,
        units="m",
        length_fraction=0.15,
        location="lower left",
        box_alpha=0.6,
        scale_formatter=km_formatter,
    )
    ax.add_artist(scalebar)

    try:
        north_arrow(
            ax,
            location="upper right",
            shadow=False,
            rotation={"degrees": 0},
            scale=0.5,
        )
    except NameError:
        print("north_arrow function not found. Skipping north arrow.")

    # 10. Axes styling and title
    ax.set_title(
        "Alternation Shift",
        fontsize=18,
        pad=10
    )
    ax.set_aspect("equal")

    # Define transformers for lat/lon tick labels
    to_latlon = Transformer.from_crs(
        src_crs,
        "EPSG:4326",
        always_xy=True
    )

    def format_lon(x, pos):
        x = np.clip(x, 0, width - 1)
        x_proj, y_proj = rasterio.transform.xy(transform, height // 2, x)
        lon, _ = to_latlon.transform(x_proj, y_proj)
        return f"{lon:.1f}°"

    def format_lat(y, pos):
        y = np.clip(y, 0, height - 1)
        x_proj, y_proj = rasterio.transform.xy(transform, y, width // 2)
        _, lat = to_latlon.transform(x_proj, y_proj)
        return f"{lat:.1f}°"

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=10,
        pad=4
    )
    plt.setp(
        ax.get_yticklabels(),
        rotation=90,
        va="center"
    )

    # 11. Save and show the figure
    maps_dir = os.path.join(
        output_dir,
        "maps"
    )
    os.makedirs(
        maps_dir,
        exist_ok=True
    )
    output_figure_path = os.path.join(
        maps_dir,
        "map_alternation_shift.png"
    )

    plt.savefig(
        output_figure_path,
        dpi=300,
        bbox_inches="tight",
        format="png",
        pad_inches=0.5,
    )
    plt.show()
    print(f"Map figure saved successfully to: {output_figure_path}")






def load_and_reorder_matrices(output_path: str, year_list: list) -> Dict[str, Any]:
    """
    Load transition matrices from CSVs and reorder them based on sum net change.

    Parameters
    ----------
    output_path : str
        Base directory path where the matrix CSV files are stored.
    year_list : list
        List of years representing the timeline (e.g., [2001, 2010, 2019]).

    Returns
    -------
    dict
        Dictionary containing the loaded and reordered matrices.
    """
    matrices = {}

    interval_str = f"{year_list[0]}-{year_list[-1]}"

    for key, meta in MATRIX_META.items():
        # Look directly in the output_path
        csv_path = os.path.join(output_path, f"transition_matrix_{meta[0]}_{interval_str}.csv")
        
        if os.path.exists(csv_path):
            matrices[key] = load_square_matrix(csv_path=csv_path)
        else:
            print(f"Warning: Matrix file not found at {csv_path}")

    # Reorder all loaded matrices based on the net change from the 'sum' matrix
    if matrices:
        matrices = reorder_all_matrices(matrices_dict=matrices)

    return matrices

# Define matrix metadata dictionary globally for reuse
MATRIX_META = {
    "sum": ["sum", "Time Intervals", "flow"],
    "alt_exc": ["alternation_exchange", "Alternation Exchange", "flow"],
    "alt_shift": ["alternation_shift", "Alternation Shift", "flow"],
    "ext": ["extent", "Extent", "stock"],
    "all_exc": ["allocation_exchange", "Allocation Exchange", "stock"],
    "qty_shift": ["quantity_allocation_shift", "Quantity & Allocation Shift", "stock"],
    "unacc_ext": ["unaccounted_extent", "Indirect", "stock"],
}


def calculate_trajectory_gee(
    image_stack: ee.Image,
    band_names: list,
    global_mask: ee.Image,
    nodata_val: int = NODATA_VALUE,
) -> ee.Image:
    """
    Classify a single pixel trajectory into five categories based on mathematical logic using GEE.

    Parameters
    ----------
    image_stack : ee.Image
        An ee.Image where each band represents a chronological time step.
    band_names : list
        A list of strings representing the ordered band names in the stack.
    nodata_val : int, optional
        NoData value to be masked out at the end, by default NODATA_VALUE (255).

    Returns
    -------
    ee.Image
        An ee.Image containing the classified trajectory codes (1 to 5).
    """
    # 0. Apply the global validity mask to the image stack at the very beginning.
    masked_image_stack = image_stack.updateMask(global_mask)

    # 1. Extract the start and end images from the stack
    start_img = masked_image_stack.select(band_names[0])
    end_img = masked_image_stack.select(band_names[-1])

    # 2. Check if the start class equals the end class
    start_equals_end = start_img.eq(end_img)

    # 3. Shift the stack by 1 band to compare t and t+1 in parallel (vectorized)
    common_names = [f"b_{i}" for i in range(len(band_names) - 1)]
    stack_t = masked_image_stack.select(band_names[:-1]).rename(common_names)
    stack_t1 = masked_image_stack.select(band_names[1:]).rename(common_names)

    # 4. Check for a direct transition and path changes using native multi-band operations
    start_img_stack = ee.Image.cat([start_img] * len(common_names)).rename(common_names)
    end_img_stack = ee.Image.cat([end_img] * len(common_names)).rename(common_names)

    has_direct_transition = stack_t.eq(start_img_stack).And(stack_t1.eq(end_img_stack)).unmask(0).reduce(ee.Reducer.max())
    path_changes = stack_t.neq(stack_t1).reduce(ee.Reducer.sum())

    # 5. Deduce all_match_start mathematically from path_changes (0 changes means completely stable)
    all_match_start = path_changes.eq(0)

    # 6. Assign Trajectory 1 for completely stable pixels
    traj_1 = start_equals_end.And(all_match_start).multiply(1)

    # 7. Assign Trajectory 2 for stable extent with alternation
    traj_2 = start_equals_end.And(all_match_start.Not()).multiply(2)

    # 8. Identify pixels with extent change
    extent_change = start_equals_end.Not()

    # 9. Assign Trajectory 5 for extent change without direct transition
    traj_5 = extent_change.And(has_direct_transition.Not()).multiply(5)

    # 10. Assign Trajectory 3 for extent change without alternation
    traj_3 = extent_change.And(has_direct_transition).And(path_changes.eq(1)).multiply(3)

    # 11. Assign Trajectory 4 for extent change with alternation
    traj_4 = extent_change.And(has_direct_transition).And(path_changes.gt(1)).multiply(4)

    # 12. Combine all trajectory maps into a single output image
    trajectory_image = traj_1.add(traj_2).add(traj_3).add(traj_4).add(traj_5)

    # 13. The global_mask is already applied to the stack, so a final mask is redundant.

    return trajectory_image.rename('trajectory')


def build_glance_stack(
    year_list: list,
    collection_id: str,
    band_name: str,
    nodata_val: int,
) -> tuple:
    """
    Build an Earth Engine image stack from the specified collection,
    without pre-applying complex global masks to keep the computation graph clean.

    Parameters
    ----------
    year_list : list
        List of integer years to process.
    collection_id : str
        The GEE ImageCollection ID.
    band_name : str
        The band name to select.
    nodata_val : int
        The NoData value to mask out.

    Returns
    -------
    tuple
        A tuple containing the ee.Image stack and the list of band names.
    """
    collection = ee.ImageCollection(collection_id).select(band_name)
    images = []
    b_names = []

    for year in year_list:
        b_name = f"y{year}"
        b_names.append(b_name)
        
        img = collection.filter(ee.Filter.calendarRange(year, year, "year")).mosaic()
        images.append(
            img.rename(b_name)
        )

    stack = ee.Image(images)

    return stack, b_names


###############################################################################
#                                                                             #
#                  5.1 CHANGE COMPONENTS                                      #
#                                                                             #
###############################################################################

def export_interval_transition_matrices_gee(
    year_list: list,
    drive_folder: str,
    collection_id: str,
    band_name: str,
    scale: int = 300,
    nodata_val: int = 255,
    full_year_list: list = None,
) -> list:
    """
    Export LULC transition matrices between consecutive years to Google Drive.

    Parameters
    ----------
    year_list : list
        List of years representing the timeline.
    drive_folder : str
        Google Drive folder name for the exported CSV files.
    collection_id : str
        GEE ImageCollection ID containing the LULC rasters.
    band_name : str
        Band name representing the LULC classes.
    scale : int, optional
        Spatial resolution for the export, by default 300.
    full_year_list : list of int, optional
        The complete timeline to construct the consistent global validity mask.
    nodata_val : int, optional
        Value representing NoData/Background to be masked out, by default 255.

    Returns
    -------
    list
        List of submitted ee.batch.Task objects.
    """
    tasks = []

    # Default to using the main year list for the mask if not specified
    if full_year_list is None:
        full_year_list = year_list

    # 1. Build global mask using the FULL timeline to ensure mathematical consistency
    full_stack, _ = build_glance_stack(
        year_list=full_year_list,
        collection_id=collection_id,
        band_name=band_name,
        nodata_val=nodata_val,
    )
    global_mask = full_stack.neq(nodata_val).unmask(0).reduce(ee.Reducer.min())

    # 2. Build target stack for the specific years we want to export tasks for
    target_stack, target_band_names = build_glance_stack(
        year_list=year_list,
        collection_id=collection_id,
        band_name=band_name,
        nodata_val=nodata_val,
    )

    yearly_by_year = {
        year: target_stack.select(target_band_names[idx]).rename(band_name)
        for idx, year in enumerate(year_list)
    }

    for i in range(len(year_list) - 1):
        start_year = year_list[i]
        end_year = year_list[i + 1]

        img_start = yearly_by_year[start_year].updateMask(global_mask)
        img_end = yearly_by_year[end_year].updateMask(global_mask)

        # Encode transitions: (Start * 100) + End
        transition_img = img_start.multiply(100).add(img_end).rename("transition")

        histogram = transition_img.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram().unweighted(),
            geometry=GLOBAL_GEOM,
            scale=scale,
            crs="EPSG:4326",
            maxPixels=1e13,
            tileScale=16,
        )

        feature = ee.Feature(None, histogram)
        fc = ee.FeatureCollection([feature])

        task_name = f"transition_{start_year}_{end_year}"
        task = ee.batch.Export.table.toDrive(
            collection=fc,
            description=task_name,
            folder=drive_folder,
            fileNamePrefix=f"transition_matrix_{start_year}-{end_year}",
            fileFormat="CSV",
        )
        task.start()
        tasks.append(task)

    return tasks

def parse_gee_raw_csv(file_path: str) -> pd.DataFrame:
    """
    Parse a raw GEE transition CSV into a square Pandas DataFrame matrix.
    """
    df_raw = pd.read_csv(file_path)
    dict_str = None
    for col in df_raw.columns:
        val = str(df_raw[col].iloc[0])
        if val.startswith("{") and "=" in val:
            dict_str = val
            break

    if not dict_str:
        raise ValueError(f"Raw GEE dictionary string not found in {file_path}.")

    dict_str = dict_str.strip("{}")
    pairs = dict_str.split(", ")

    transitions = {}
    classes = set()

    for pair in pairs:
        if not pair:
            continue
        k, v = pair.split("=")
        if k.strip().lower() == "null":
            continue
        k_int = int(k)
        val_float = float(v)
        # For GLanCE, transition classes are encoded as start * 100 + end
        s_c = k_int // 100
        e_c = k_int % 100
        transitions[(s_c, e_c)] = val_float
        classes.add(s_c)
        classes.add(e_c)

    classes_sorted = sorted(list(classes))
    df_mat = pd.DataFrame(0.0, index=classes_sorted, columns=classes_sorted)

    for (s, e), v in transitions.items():
        df_mat.at[s, e] = v

    return df_mat


class ComponentCalculator:
    """
    Compute change components for a matrix.

    Supports pre-decomposed matrices via force_component parameter.
    """

    def __init__(self, transition_matrix: np.ndarray) -> None:
        """
        Initialize the calculator with a transition matrix.

        Parameters
        ----------
        transition_matrix : np.ndarray
            Square matrix representing transitions.
        """
        self.matrix = transition_matrix.astype(float)
        self.num_classes = transition_matrix.shape[0]
        self.class_components: list[dict] = []

    def calculate_components(self, force_component: str = None) -> "ComponentCalculator":
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
        for class_idx in range(self.num_classes):
            gain_sum = np.sum(self.matrix[:, class_idx])
            loss_sum = np.sum(self.matrix[class_idx, :])

            # Standard net change calculation
            q_gain = max(0.0, gain_sum - loss_sum)
            q_loss = max(0.0, loss_sum - gain_sum)

            if force_component == "Exchange":
                exchange = loss_sum - self.matrix[class_idx, class_idx]
                shift = 0.0
                q_gain, q_loss = gain_sum - loss_sum, loss_sum - gain_sum
            elif force_component == "Shift":
                exchange = 0.0
                shift = loss_sum - self.matrix[class_idx, class_idx]
                q_gain, q_loss = 0.0, 0.0
            else:
                # Standard Pontius decomposition
                mutual = np.sum(np.minimum(self.matrix[class_idx, :], self.matrix[:, class_idx]))
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



def process_matrix(
    matrix_type: str,
    input_dir: str,
    years_list: list,
    class_labels_dict: dict,
    start_year=None,
    end_year=None,
) -> list:
    """
    Search for a transition matrix file and calculate its change components.

    Parameters
    ----------
    matrix_type : str
        Type of matrix ("interval", "extent", "sum", etc.).
    input_dir : str
        Directory where CSV files are stored.
    years_list : list
        List of all years in the timeline.
    class_labels_dict : dict
        Dictionary mapping class IDs to metadata.
    start_year : str or int, optional
        Start year for interval matrices.
    end_year : str or int, optional
        End year for interval matrices.

    Returns
    -------
    list[dict]
        List of dictionaries containing component values per class.
    """
    results = []
    patterns = []

    # 1. Determine naming patterns to search for
    if matrix_type == "interval":
        s_str, e_str = str(start_year), str(end_year)
        patterns.extend([
            f"transition_{s_str}_{e_str}.csv",
            f"transition_matrix_{s_str}-{e_str}.csv",
        ])
        label_time = f"{s_str}-{e_str}"
    else:
        y0_str, yN_str = str(years_list[0]), str(years_list[-1])
        patterns.extend([
            f"transition_matrix_{matrix_type}_{y0_str}-{yN_str}.csv",
        ])
        label_time = matrix_type

    # 2. Find the existing file in the main directory
    full_path = None
    for p in patterns:
        path = os.path.join(input_dir, p)
        if os.path.exists(path):
            full_path = path
            break

    if not full_path:
        return []

    # 3. Process components
    force_comp = (
        "Exchange" if "exchange" in matrix_type else ("Shift" if "shift" in matrix_type else None)
    )

    # Load matrix safely handling standard square CSVs or raw GEE dict formats
    try:
        df_mat = pd.read_csv(full_path, index_col=0)
        if df_mat.shape[1] > 0 and isinstance(df_mat.iloc[0, 0], str) and df_mat.iloc[0, 0].startswith("{"):
            raise ValueError("Raw GEE format detected")
    except (ValueError, TypeError):
        # Assumes parse_gee_raw_csv is defined in utils.py from the previous step
        df_mat = parse_gee_raw_csv(full_path)

    calc = ComponentCalculator(df_mat.values).calculate_components(force_component=force_comp)

    for idx, class_id in enumerate([int(c) for c in df_mat.index]):
        cls_name = class_labels_dict.get(class_id, {}).get("name", f"Class {class_id}")
        comp_vals = calc.class_components[idx]

        for comp_name in ["Quantity", "Exchange", "Shift"]:
            label_comp = comp_name
            if matrix_type in ["extent", "sum"]:
                label_comp = f"Allocation_{comp_name}"
            if "alternation" in matrix_type:
                label_comp = f"Alternation_{comp_name}"

            results.append({
                "Time_Interval": label_time,
                "Class": cls_name,
                "Component": label_comp,
                "Gain": comp_vals[f"{comp_name}_Gain"],
                "Loss": comp_vals[f"{comp_name}_Loss"],
            })
    return results


def generate_all_heatmaps(
    matrices_dict: dict,
    output_path: str,
    interval_str: str,
    years: list,
    style_config: dict,
) -> None:
    """
    Iterate over the matrices dictionary and generate a heatmap for each.

    Parameters
    ----------
    matrices_dict : dict
        Dictionary containing the dataframes to plot.
    output_path : str
        Base directory path to save the generated charts.
    interval_str : str
        String representing the time interval.
    years : list
        List of years processed.
    style_config : dict
        Dictionary containing style configurations for the plot.

    Returns
    -------
    None
    """
    import os

    print(
        "Generating Heatmaps...",
    )

    # Enforce a unified color scale and a maximum value for consistency
    style_config = style_config.copy()
    style_config["vmax"] = 50_000_000.0
    style_config["cmap"] = "YlOrRd"

    charts_dir = os.path.join(
        output_path,
        "charts",
    )

    os.makedirs(
        charts_dir,
        exist_ok=True,
    )

    title_map = {
        "ext": "Extent",
        "sum": "Time Intervals",
        "all_exc": "Allocation Exchange",
        "alloc_shift": "Allocation Shift",
        "qty_shift": "Quantity & Allocation Shift",
        "alt_exc": "Alternation Exchange",
        "alt_shift": "Alternation Shift",
        "unacc_ext": "Indirect",
    }

    for key, df in matrices_dict.items():
        if df is None or df.empty:
            continue

        base_name = title_map.get(
            key,
            key.capitalize(),
        )

        if key == "sum" or "alt" in key:
            formatted_interval = interval_str.replace("-", "...")
            full_title = f"{base_name} {formatted_interval}"
        else:
            full_title = f"{base_name} {interval_str}"

        out_file = os.path.join(
            charts_dir,
            f"heatmap_{key}_{interval_str}.png",
        )

        plot_heatmap(
            df=df,
            title=full_title,
            save_path=out_file,
            **style_config,
            matrix_key=key
        )

        print(
            f"-> Saved heatmap: {out_file}",
        )







###############################################################################
#                                                                             #
#                  6. Convert CSV to Transition Matrix                        #
#                                                                             #
###############################################################################

def load_global_transition_matrices(
    drive_path
):
    """
    Loads exported GEE CSVs from Drive and converts them to transition matrices.

    Parameters
    ----------
    drive_path : str
        The full path to the Google Drive folder containing the CSV files.

    Returns
    -------
    dict of pd.DataFrame
        A dictionary where keys are 'YYYY_YYYY' and values are pivot matrices.
    """

    # 1. Identify all CSV files in the specified directory
    search_pattern = os.path.join(drive_path, "*.csv")
    file_list = glob.glob(search_pattern)
    
    # 2. Create a mapping of class IDs to names from metadata
    class_names = {
        k: v['name'] 
        for k, v in GLANCE_METADATA.items()
    }
    
    all_matrices = {}

    # 3. Iterate through each file to reconstruct the matrix
    for filepath in file_list:
        df_raw = pd.read_csv(filepath)
        
        # 4. Extract the transition label from the filename
        filename = os.path.basename(filepath)
        label = filename.replace(".csv", "").replace("transition_", "")
        
        hist_data = {}
        numeric_cols = [c for c in df_raw.columns if str(c).isdigit()]
        
        # 5. Parse the histogram
        if numeric_cols:
            for col in numeric_cols:
                hist_data[col] = df_raw[col].sum()
        else:
            target_col = 'LC' if 'LC' in df_raw.columns else 'transition'
            if target_col not in df_raw.columns:
                continue
            hist_str = str(df_raw[target_col].iloc[0])
            clean_str = hist_str.strip('{}')
            pairs = clean_str.split(', ')
            for pair in pairs:
                if '=' in pair:
                    k, v = pair.split('=')
                    hist_data[k] = float(v)

        records = []
        
        # 6. Decode transition codes into 'From' and 'To' classes
        for code, count in hist_data.items():
            code_int = int(float(code))
            id_from = code_int // 100
            id_to = code_int % 100

            # 7. Filter records using valid metadata classes
            if id_from in class_names and id_to in class_names:
                records.append({
                    "From": class_names[id_from],
                    "To": class_names[id_to],
                    "Pixels": int(count)
                })

        # 8. Pivot the records into a formal transition matrix
        if records:
            df_temp = pd.DataFrame(records)
            matrix = df_temp.pivot(
                index="From", 
                columns="To", 
                values="Pixels"
            ).fillna(0)

            # 9. Store the resulting DataFrame in the output dictionary
            all_matrices[label] = matrix
            
    return all_matrices

def convert_matrices_to_area(
    matrices_dict,
    pixel_size=30
):
    """
    Converts pixel counts in transition matrices to area in square kilometers.

    Parameters
    ----------
    matrices_dict : dict of pd.DataFrame
        A dictionary where values are transition matrices in pixel counts.
    pixel_size : int, optional
        The edge length of a single pixel in meters. Defaults to 30.

    Returns
    -------
    dict of pd.DataFrame
        A dictionary of transition matrices with values in km^2.
    """
    # 1. Calculate the conversion factor from pixels to km^2
    # Area of one pixel in m^2 = pixel_size * pixel_size
    # Conversion to km^2 = m^2 / 1,000,000
    conversion_factor = (pixel_size ** 2) / 1000000
    
    area_matrices = {}

    # 2. Iterate through the dictionary of matrices
    for label, matrix in matrices_dict.items():
        # 3. Multiply the entire DataFrame by the conversion factor
        # Pandas handles the element-wise multiplication automatically
        area_matrix = matrix * conversion_factor
        
        # 4. Round the results to two decimal places for readability
        area_matrices[label] = area_matrix.round(2)
        
    return area_matrices

def save_area_matrices_to_csv(
    area_matrices,
    output_dir
):
    """
    Saves a dictionary of area matrices to individual CSV files.

    Parameters
    ----------
    area_matrices : dict of pd.DataFrame
        Dictionary containing the transition matrices in km^2.
    output_dir : str
        The directory path where the CSV files will be saved.

    Returns
    -------
    list of str
        A list of file paths to the saved CSV files.
    """
    # 1. Check if the output directory exists and create it if necessary
    if not os.path.exists(output_dir):
        os.makedirs(
            output_dir, 
            exist_ok=True
        )
    
    saved_files = []

    # 2. Iterate through the dictionary to process each transition matrix
    for label, matrix in area_matrices.items():
        # 3. Construct the specific filename for the km2 results
        filename = f"transition_matrix_km2_{label}.csv"
        filepath = os.path.join(
            output_dir, 
            filename
        )
        
        # 4. Save the DataFrame to CSV including the class names in the index
        matrix.to_csv(
            filepath,
            index=True
        )
        
        saved_files.append(filepath)
        print(f"Successfully saved: {filename}")
        
    return saved_files


###############################################################################
#                                                                             #
#                  7. Compute Sum matrix                                      #
#                                                                             #
###############################################################################

def compute_sum_matrix(
    input_dir: str,
    output_path: str,
    file_prefix: str = "transition_",
) -> pd.DataFrame:
    """
    Compute the SUM transition matrix by aggregating all annual intervals.

    Parameters
    ----------
    input_dir : str
        Path to the directory containing annual transition CSV files.
    output_path : str
        Full path (including filename) to save the resulting SUM matrix.
    file_prefix : str, optional
        Prefix of the annual transition files to look for, by default "transition_".

    Returns
    -------
    pd.DataFrame
        The aggregated SUM transition matrix.
    """
    # 1. List all annual transition files (e.g., transition_2001_2002.csv, transition_2002_2003.csv...)
    pattern_underscore = os.path.join(input_dir, f"{file_prefix}????_????.csv")
    pattern_hyphen = os.path.join(input_dir, f"{file_prefix}????-????.csv")
    
    all_files = glob.glob(pattern_underscore) + glob.glob(pattern_hyphen)

    if not all_files:
        raise FileNotFoundError(
            f"No annual transition matrices found in {input_dir} with prefix '{file_prefix}'",
        )

    # 2. Sort files to ensure chronological order (optional, but good practice)
    all_files.sort()

    df_sum = None

    # 3. Iterate and aggregate
    for file_path in all_files:
        # Load current annual matrix (using load_square_matrix to handle raw/square formats)
        df_annual = load_square_matrix(file_path)

        if df_sum is None:
            # Initialize with the first matrix
            df_sum = df_annual.copy()
        else:
            # Sum values cell by cell
            df_sum = df_sum.add(
                df_annual,
                fill_value=0.0,
            )

    # 4. Save the consolidated SUM matrix
    if df_sum is not None:
        df_sum.to_csv(output_path)
        print(f"SUM matrix successfully saved to: {output_path}")

    return df_sum


###############################################################################
#                                                                             #
#                  8. Compute Exchange and Shift                              #
#                                                                             #
###############################################################################
def compute_and_save_components(
    df_sum: pd.DataFrame,
    df_ext: pd.DataFrame,
    output_dir: str,
    period_label: str = "2001-2019",
) -> None:
    """
    Decompose Sum and Extent matrices into change components.

    Logic:
    1. Allocation: Derived from Extent matrix (Aggregate level).
    2. Alternation: Derived from (Sum - Extent) (Trajectory level).
    This implementation follows the logic from Pontius et al. where the
    diagonal (stability) is correctly handled for each component.

    Parameters
    ----------
    df_sum : pd.DataFrame
        Aggregated transition matrix (Sum of annual intervals).
    df_ext : pd.DataFrame
        Direct transition matrix (Start year vs End year).
    output_dir : str
        Directory path to save the resulting CSV files.
    period_label : str, optional
        Year range label for filename, by default "2001-2019".

    Returns
    -------
    None
    """
    # 1. Align and Sort Matrices based on GLANCE_METADATA order
    name_to_id = {v['name']: k for k, v in GLANCE_METADATA.items()}

    def _sort_key(label):
        if label in name_to_id:
            return (0, name_to_id[label])
        try:
            return (0, int(label))
        except (ValueError, TypeError):
            return (1, str(label))

    all_labels = sorted(
        list(set(df_sum.index).union(df_sum.columns)),
        key=_sort_key
    )

    df_s = df_sum.reindex(index=all_labels, columns=all_labels).fillna(0.0)
    df_e = df_ext.reindex(index=all_labels, columns=all_labels).fillna(0.0)

    # Convert to numpy arrays for calculation
    mat_sum = df_s.values
    mat_ext = df_e.values

    # 2. Calculate Allocation Components (from Extent matrix)
    # Allocation Exchange (C) = min(E_ij, E_ji). This preserves the diagonal.
    mat_c = np.minimum(mat_ext, mat_ext.T)

    # Quantity & Allocation Shift (Q) = E - C. Diagonal becomes zero.
    mat_q = mat_ext - mat_c

    # 3. Calculate Alternation Components (from Sum - Extent)
    alternation_raw = mat_sum - mat_ext

    # Alternation Exchange (X) = max(0, min(A_ij, A_ji)), where A = Sum - Extent.
    # This also preserves the diagonal (S_ii - E_ii).
    mat_x = np.maximum(0, np.minimum(alternation_raw, alternation_raw.T))

    # Alternation Shift (S) is the positive part of the remainder after exchange.
    mat_s = np.maximum(0, alternation_raw - mat_x)

    # Indirect (Unaccounted) Component (U) is calculated from the identity:
    # U = E + X + S - V (where V is Sum)
    mat_u = mat_ext + mat_x + mat_s - mat_sum

    # 4. Consolidate and Export all components to CSV
    components = {
        "sum": mat_sum,
        "extent": mat_ext,
        "allocation_exchange": mat_c,
        "quantity_allocation_shift": mat_q,
        "alternation_exchange": mat_x,
        "alternation_shift": mat_s,
        "unaccounted_extent": mat_u,  # Use key 'unaccounted_extent' for consistency with heatmap generator
    }

    for name, data in components.items():
        df_out = pd.DataFrame(
            data,
            index=all_labels,
            columns=all_labels
        )

        fname = f"transition_matrix_{name}_{period_label}.csv"
        path = os.path.join(output_dir, fname)
        df_out.to_csv(path)

        print(f"Component saved: {fname}")


###############################################################################
#                                                                             #
#                  9. Reorder Matrices by net change                          #
#                                                                             #
###############################################################################
def reorder_matrices_by_net_change(
    df_sum: pd.DataFrame,
    df_ext: pd.DataFrame,
    df_ext_exc: pd.DataFrame,
    df_ext_shift: pd.DataFrame,
    df_alt_exc: pd.DataFrame,
    df_alt_shift: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Reorder matrices from largest losers to largest gainers using net change.

    Parameters
    ----------
    df_sum : pd.DataFrame
        The aggregated SUM matrix used to calculate the sorting order.
    df_ext, df_ext_exc, df_ext_shift, df_alt_exc, df_alt_shift : pd.DataFrame
        The other component matrices to be reordered.

    Returns
    -------
    tuple[pd.DataFrame, ...]
        All input dataframes reindexed with the same optimized order.
    """
    # 1. Calculate Net Change (Gains - Losses)
    # Diagonal is ignored to focus only on transitions
    m_values = df_sum.values.copy()
    np.fill_diagonal(m_values, 0.0)
    
    gains = m_values.sum(axis=0)
    losses = m_values.sum(axis=1)
    net_change = gains - losses
    
    # 2. Define the sorting order (ascending: losers first)
    net_series = pd.Series(net_change, index=df_sum.index)
    order_labels = net_series.sort_values(ascending=True).index.tolist()

    # 3. Helper to apply the same order to any dataframe
    def _apply_order(df: pd.DataFrame) -> pd.DataFrame:
        return df.reindex(index=order_labels, columns=order_labels).fillna(0.0)

    # 4. Return all matrices reordered
    return (
        _apply_order(df_sum),
        _apply_order(df_ext),
        _apply_order(df_ext_exc),
        _apply_order(df_ext_shift),
        _apply_order(df_alt_exc),
        _apply_order(df_alt_shift),
    )



def export_global_overall_change_frequency_csv_gee(
    year_list: list,
    drive_folder: str,
    scale: int = 300,
    full_year_list: list = None,
) -> ee.batch.Task:
    """
    Compute and export a single CSV representing the overall frequency of changes
    (how many pixels changed 0, 1, 2, ... N times) across the entire timeline.
    """
    if GLOBAL_GEOM is None:
        raise ValueError(
            "GLOBAL_GEOM is not initialized. Please call "
            "utils.initialize_active_region(region_code) before running tasks."
        )

    print(f"Preparing Overall Change Frequency GEE Task for {year_list[0]}-{year_list[-1]}...")

    if full_year_list is None:
        full_year_list = year_list

    # 1. Build global mask using the FULL timeline to ensure mathematical consistency
    full_stack, _ = build_glance_stack(
        year_list=full_year_list,
        collection_id=GLANCE_COLLECTION_ID,
        band_name=GLANCE_CLASS_BAND,
        nodata_val=NODATA_VALUE,
    )
    global_mask = full_stack.neq(NODATA_VALUE).unmask(0).reduce(ee.Reducer.min())

    # 2. Build target stack for the specific years we want to export tasks for
    target_stack, target_band_names = build_glance_stack(
        year_list=year_list,
        collection_id=GLANCE_COLLECTION_ID,
        band_name=GLANCE_CLASS_BAND,
        nodata_val=NODATA_VALUE,
    )

    # 3. Shift the target stack by 1 band to compare t and t+1 in parallel (vectorized)
    stack_t = target_stack.select(target_band_names[:-1])
    stack_t1 = target_stack.select(target_band_names[1:])

    # 4. Calculate total changes across the entire timeline
    total_changes = stack_t.neq(stack_t1).reduce(ee.Reducer.sum()).rename('num_changes')

    total_changes_masked = total_changes.updateMask(global_mask)

    # 5. Compute the frequency histogram of the total changes for these pixels
    histogram = total_changes_masked.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram().unweighted(),
        geometry=GLOBAL_GEOM,
        scale=scale,
        crs="EPSG:4326",
        maxPixels=1e13,
        tileScale=16,
    ).get('num_changes')

    # Handle possible nulls if no change occurred
    hist_dict = ee.Dictionary(ee.Algorithms.If(histogram, histogram, {}))
    feature = ee.Feature(None, hist_dict)
    fc = ee.FeatureCollection([feature])

    # 6. Configure and start the export task
    start_year = year_list[0]
    end_year = year_list[-1]
    export_name = f"Number_Change_Overall_{start_year}_{end_year}"
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=export_name,
        folder=drive_folder,
        fileNamePrefix=export_name,
        fileFormat="CSV",
    )
    task.start()
    print(f"Task '{export_name}' submitted to Google Earth Engine.")
    return task


###############################################################################
#                                                                             #
#                  10. GLOBAL MOSAIC CLASS & FUNCTIONS                        #
#                                                                             #
###############################################################################

class GlanceMosaicker:
    """
    Manages continental GLanCE regions to build a consistent global mosaic
    and unified geometry operations directly inside Google Earth Engine.
    """

    def __init__(self, region_codes: List[str]) -> None:
        """
        Initialize the mosaicker with a customizable set of continental regions.
        
        Parameters
        ----------
        region_codes : list of str
            List of regional codes to compose the mosaic (e.g., ['EU', 'AF', 'SA']).
        """
        if not region_codes:
            raise ValueError("You must provide at least one region code to construct the mosaic.")
            
        self.region_codes = region_codes
        self._validate_regions()
        self.unified_geometry = self._build_unified_geometry()

    def _validate_regions(self) -> None:
        """Validate if all provided region codes exist in the registry."""
        for code in self.region_codes:
            if code not in GLANCE_REGIONS_REGISTRY:
                raise ValueError(
                    f"Region code '{code}' is invalid. "
                    f"Choose from: {list(GLANCE_REGIONS_REGISTRY.keys())}"
                )

    def _build_unified_geometry(self) -> ee.Geometry:
        """
        Combine the individual rectangular bounding boxes into a single 
        ee.Geometry.MultiPolygon nativelly in GEE.
        """
        geometries = []
        for code in self.region_codes:
            geom_coords = GLANCE_REGIONS_REGISTRY[code]['geom']
            # Create individual non-planar rectangular polygons
            rect = ee.Geometry.Rectangle(geom_coords, "EPSG:4326", False)
            geometries.append(rect)
        
        # Perform a spatial union to create a single clean multipolygon geometry
        return ee.Geometry.MultiPolygon(geometries)

    def get_global_crs_and_transform(self, scale: int = 300) -> Tuple[str, List[float]]:
        """
        Get the global Cylindrical Equal Area projection parameters (EPSG:6933).
        This CRS preserves area properties globally, making pixel counts and area calculations
        extremely accurate across different latitudes.
        
        Parameters
        ----------
        scale : int
            The spatial resolution in meters (e.g., 300 or 30).
            
        Returns
        -------
        tuple
            A tuple containing the CRS string ("EPSG:6933") and the affine transform list.
        """
        crs = "EPSG:6933"
        # Setup a standard global transform aligned to the coordinate origin
        transform = [scale, 0, 0, 0, -scale, 0]
        return crs, transform

    def get_mosaicked_image_collection(self, year: int) -> ee.Image:
        """
        Retrieve and mosaic the GLanCE land cover images for the specified year
        restricted to the unified continental boundary.
        
        Parameters
        ----------
        year : int
            The target year to mosaic (e.g., 2019).
            
        Returns
        -------
        ee.Image
            The unified, masked land cover image for the selected year.
        """
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        # Filter GLanCE collection by date and band
        collection = (
            ee.ImageCollection(GLANCE_COLLECTION_ID)
            .filterDate(start_date, end_date)
            .select(GLANCE_CLASS_BAND)
        )

        # Mosaic overlapping zones (resolving potential duplicates via first-order placement)
        # and clip strictly to our continental limits
        mosaicked_image = collection.mosaic().clip(self.unified_geometry)
        
        # Enforce uint8 casting to preserve data type limits
        return mosaicked_image.toByte()
