<img src="https://raw.githubusercontent.com/antoniovfonseca/summarize-change-components/refs/heads/main/logos/app_cover.png" width="1000" height="180"> 

# **Summarize Components of Change in a Cross-Tab Matrix**

---


**Authors:** [Antonio Fonseca](https://orcid.org/my-orcid?orcid=0000-0001-6309-6204), [Robert Gilmore Pontius Jr](https://wordpress.clarku.edu/rpontius/)

**Institution:** [Clark University](https://www.clarku.edu/)

**Purpose:**

This notebook summarizes the overall change and the class-level changes during time intervals by computing five componentes: Quantity, Allocation Exchange, Allocation Shift, Alternation Exchange, and Alternation Shift.

**Data Description:**
- **Source:** [MapBiomas Project](https://brasil.mapbiomas.org/en/)
- **Coverage:** Western Bahia, Brazil, 5-year time intervals from 1990 to 2020.
- **Resolution and Format:** TIF files at 30m resolution.

**Notebook Outline:**
1. **Environment Setup:** Install required packages, import libraries, mount Google Drive, and set input/output paths.
2. **Data Preparation:** Define the years, class dictionary, and load and mask raster files.
3. **Confusion Matrix Generation:** Interval matrices, Extent matrix, Sum matrix, and Alternation matrix.
4. **Change Component Calculation:** Calculate the components of change Quantity, Allocation Exchange, Allocation Shift, Alternation Exchange, and Alternation Shift.
6. **Trajectory Analysis:** Temporal pattern of  change observed in pixel data over a time series, categorized into four distinct types on the map.

**Acknowledgements:**

The United States National Aeronautical and Space Administration supported this work through the Land-Cover and Land-Use Change Mission Directorate via the grant 80NSSC23K0508 entitled ["Irrigation as climate-change adaptation in the Cerrado biome of Brazil evaluated with new quantitative methods, socio-economic analysis, and scenario models."](https://lcluc.umd.edu/projects/irrigation-climate-change-adaptation-cerrado-biome-brazil-evaluated-new-quantitative)


<img src="https://raw.githubusercontent.com/antoniovfonseca/summarize-change-components/refs/heads/main/logos/nasa_lulc_dark.png" width="150" height="100"> <img src="https://raw.githubusercontent.com/antoniovfonseca/summarize-change-components/refs/heads/main/logos/mapbiomas_geral.png" width="120" height="70">
          <img src="https://raw.githubusercontent.com/antoniovfonseca/summarize-change-components/refs/heads/main/logos/clark_logo_horizontal.png" width="150" height="70">
