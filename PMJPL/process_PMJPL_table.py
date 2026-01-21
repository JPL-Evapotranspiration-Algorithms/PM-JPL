"""
Module: process_PMJPL_table.py

This module provides a function to process input data for the PM-JPL (Penman-Monteith Jet Propulsion Laboratory) model.
It prepares the required variables from a pandas DataFrame, handles missing or alternative column names, computes derived variables as needed, and runs the PM-JPL model to generate output variables, which are appended to the input DataFrame.
"""
import logging

import numpy as np
import pandas as pd
import rasters as rt
from rasters import MultiPoint, WGS84

from dateutil import parser
from pandas import DataFrame

from .constants import *
from .PMJPL import PMJPL

logger = logging.getLogger(__name__)

# FIXME include additional inputs required by PM-JPL that were not required by PT-JPL

def process_PMJPL_table(
        input_df: DataFrame,
        upscale_to_daylight: bool = False,
        regenerate_net_radiation: bool = False,
        offline_mode: bool = False) -> DataFrame:
    """
    Processes an input DataFrame to prepare all required variables for the PM-JPL model,
    runs the model, and returns a DataFrame with the model outputs appended as new columns.

    This function is designed to work with tabular (pandas DataFrame) data, such as point or site-level measurements or extracted pixel values from gridded products. It is compatible with DataFrames produced by ECOSTRESS Cal-Val or similar sources, and is suitable for both single-site and batch sensitivity analysis workflows.

    The function is commonly used as a forward process in sensitivity or perturbation analysis, and can be chained with net radiation calculations prior to running the PM-JPL model.

    Expected Input DataFrame Columns:
        - 'NDVI': Normalized Difference Vegetation Index (required)
        - 'ST_C': Surface temperature in Celsius (required)
        - 'albedo': Surface albedo (required)
        - 'Ta_C' or 'Ta': Air temperature in Celsius (required)
        - 'RH': Relative humidity (0-1, required)
        - 'Rn_Wm2': Net radiation (W/m^2, required)
        - 'time_UTC': Time in UTC (required)
        - 'geometry': Geometry object (optional, will be constructed from 'lat' and 'lon' if missing)
        - 'lat', 'lon': Latitude and longitude (optional, used to construct geometry if needed)

    The function will attempt to load or compute any missing optional variables using spatial context if possible.

    Returns:
        DataFrame: The input DataFrame with PM-JPL model outputs added as columns. Output columns include:
            - 'LE': Total instantaneous evapotranspiration
            - 'LE_canopy': Canopy transpiration
            - 'LE_soil': Soil evaporation
            - 'LE_interception': Interception evaporation
            - 'PET': Potential evapotranspiration

    Example:
        Suppose you have a CSV file with columns: NDVI, ST_C, albedo, Ta_C, RH, Rn, time_UTC, lat, lon

        ```python
        import pandas as pd
        from PMJPL.process_PMJPL_table import process_PMJPL_table

        # Load your data
        df = pd.read_csv('my_input_data.csv')

        # Process the table and run the PM-JPL model
        output_df = process_PMJPL_table(df)

        # The output DataFrame will have new columns: 'LE', 'LE_canopy', 'LE_soil', 'LE_interception', 'PET'
        print(output_df.head())
        ```

    Notes:
        - If any required columns are missing, a KeyError will be raised.
        - If geometry is not provided, latitude and longitude columns are required to construct spatial context.
        - All input columns should be numeric and of compatible shape.
        - This function is suitable for batch-processing site-level or point data tables for ET partitioning and for use in sensitivity analysis workflows.
    """
    logger.info("starting PM-JPL table processing")

    # Extract and typecast surface temperature (ST_C) and NDVI
    ST_C = np.array(input_df.ST_C).astype(np.float64)
    emissivity = np.array(input_df.emissivity).astype(np.float64)
    NDVI = np.array(input_df.NDVI).astype(np.float64)

    # Mask NDVI values below threshold (0.06) as NaN
    NDVI = np.where(NDVI > 0.06, NDVI, np.nan).astype(np.float64)

    # Extract and typecast albedo
    albedo = np.array(input_df.albedo).astype(np.float64)

    # Handle air temperature column name differences (Ta_C or Ta)
    if "Ta_C" in input_df:
        Ta_C = np.array(input_df.Ta_C).astype(np.float64)
    elif "Ta" in input_df:
        Ta_C = np.array(input_df.Ta).astype(np.float64)
    else:
        raise KeyError("Input DataFrame must contain either 'Ta_C' or 'Ta' column.")

    # Extract and typecast relative humidity and net radiation
    RH = np.array(input_df.RH).astype(np.float64)

    if "SWin_Wm2" in input_df:
        SWin_Wm2 = np.array(input_df.SWin_Wm2).astype(np.float64)
    else:
        SWin_Wm2 = None

    if "Rn_Wm2" in input_df:
        Rn_Wm2 = np.array(input_df.Rn_Wm2).astype(np.float64)
    else:
        Rn_Wm2 = None

    if "Rn_daily_Wm2" in input_df:
        Rn_daylight_Wm2 = np.array(input_df.Rn_daily_Wm2).astype(np.float64)
    else:
        Rn_daylight_Wm2 = None

    if "Tmin_C" in input_df:
        Tmin_C = np.array(input_df.Tmin_C).astype(np.float64)
    else:
        Tmin_C = None

    if "elevation_m" in input_df:
        elevation_m = np.array(input_df.elevation_m).astype(np.float64)
    if "elevation_km" in input_df:
        elevation_m = np.array(input_df.elevation_km).astype(np.float64) * 1000.0
    else:
        elevation_m = None

    if "IGBP" in input_df:
        IGBP = np.array(input_df.IGBP).astype(np.int8)
    else:
        IGBP = None

    # --- Handle geometry and time columns ---
    import pandas as pd
    from rasters import MultiPoint, WGS84
    from shapely.geometry import Point

    def ensure_geometry(df):
        if "geometry" in df:
            if isinstance(df.geometry.iloc[0], str):
                def parse_geom(s):
                    s = s.strip()
                    if s.startswith("POINT"):
                        coords = s.replace("POINT", "").replace("(", "").replace(")", "").strip().split()
                        return Point(float(coords[0]), float(coords[1]))
                    elif "," in s:
                        coords = [float(c) for c in s.split(",")]
                        return Point(coords[0], coords[1])
                    else:
                        coords = [float(c) for c in s.split()]
                        return Point(coords[0], coords[1])
                df = df.copy()
                df['geometry'] = df['geometry'].apply(parse_geom)
        return df

    input_df = ensure_geometry(input_df)

    logger.info("started extracting geometry from PM-JPL input table")

    if "geometry" in input_df:
        # Convert Point objects to coordinate tuples for MultiPoint
        if hasattr(input_df.geometry.iloc[0], "x") and hasattr(input_df.geometry.iloc[0], "y"):
            coords = [(pt.x, pt.y) for pt in input_df.geometry]
            geometry = MultiPoint(coords, crs=WGS84)
        else:
            geometry = MultiPoint(input_df.geometry, crs=WGS84)
    elif "lat" in input_df and "lon" in input_df:
        lat = np.array(input_df.lat).astype(np.float64)
        lon = np.array(input_df.lon).astype(np.float64)
        geometry = MultiPoint(x=lon, y=lat, crs=WGS84)
    else:
        raise KeyError("Input DataFrame must contain either 'geometry' or both 'lat' and 'lon' columns.")

    logger.info("completed extracting geometry from PM-JPL input table")

    logger.info("started extracting time from PM-JPL input table")
    time_UTC = pd.to_datetime(input_df.time_UTC).tolist()
    logger.info("completed extracting time from PM-JPL input table")

    # --- Pass time and geometry to the model ---
    results = PMJPL(
        geometry=geometry,
        NDVI=NDVI,
        Ta_C=Ta_C,
        ST_C=ST_C,
        emissivity=emissivity,
        RH=RH,
        Rn_Wm2=Rn_Wm2,
        Rn_daylight_Wm2=Rn_daylight_Wm2,
        SWin_Wm2=SWin_Wm2,
        albedo=albedo,
        Tmin_C=Tmin_C,
        elevation_m=elevation_m,
        time_UTC=time_UTC,
        upscale_to_daylight=upscale_to_daylight,
        regenerate_net_radiation=regenerate_net_radiation,
        offline_mode=offline_mode
    )

    output_df = input_df.copy()
    for key, value in results.items():
        output_df[key] = value

    logger.info("PM-JPL table processing complete")

    return output_df
