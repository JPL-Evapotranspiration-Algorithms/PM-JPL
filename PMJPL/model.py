"""
MOD16 model of evapotranspiration

This implementation follows the MOD16 Version 1.5 Collection 6 algorithm described in the MOD16 user's guide.
https://landweb.nascom.nasa.gov/QA_WWW/forPage/user_guide/MOD16UsersGuide2016.pdf

Developed by Gregory Halverson in the Jet Propulsion Laboratory Year-Round Internship Program (Columbus Technologies and Services), in coordination with the ECOSTRESS mission and master's thesis studies at California State University, Northridge.
"""

# All imports moved to the top
import logging
from typing import Dict, Union
from datetime import datetime

import numpy as np
import rasters as rt
from rasters import Raster, RasterGrid, RasterGeometry, VectorGeometry

from check_distribution import check_distribution
from GEOS5FP import GEOS5FP
from NASADEM import NASADEM
from verma_net_radiation import verma_net_radiation
from SEBAL_soil_heat_flux import calculate_SEBAL_soil_heat_flux
from MCD12C1_2019_v006 import load_MCD12C1_IGBP

from carlson_leaf_area_index import carlson_leaf_area_index
from carlson_fractional_vegetation_cover import carlson_fractional_vegetation_cover
from carlson_leaf_area_index import carlson_leaf_area_index

from daylight_evapotranspiration import lambda_Jkg_from_Ta_C, daylight_ET_from_instantaneous_LE

from meteorology_conversion import SVP_Pa_from_Ta_C
from meteorology_conversion import calculate_air_density
from meteorology_conversion import calculate_specific_heat
from meteorology_conversion import calculate_specific_humidity
from meteorology_conversion import calculate_surface_pressure
from meteorology_conversion import celcius_to_kelvin

from priestley_taylor import delta_Pa_from_Ta_C
from PTJPL import calculate_relative_surface_wetness
from PTJPL import RH_THRESHOLD, MIN_FWET

from .constants import *
from .exceptions import *
from .PMJPL_parameter_from_IGBP import PMJPL_parameter_from_IGBP
from .calculate_gamma import calculate_gamma
from .soil_moisture_constraint import calculate_fSM
from .minimum_temperature_factor import Tmin_factor
from .correctance_factor import calculate_correctance_factor
from .VPD_factor import calculate_VPD_factor
from .canopy_conductance import calculate_canopy_conductance
from .wet_canopy_resistance import calculate_wet_canopy_resistance
from .canopy_aerodynamic_resistance import calculate_canopy_aerodynamic_resistance
from .wet_soil_evaporation import calculate_wet_soil_evaporation
from .potential_soil_evaporation import calculate_potential_soil_evaporation
from .interception import calculate_interception
from .transpiration import calculate_transpiration
from .leaf_conductance_to_sensible_heat import leaf_conductance_to_sensible_heat
from .potential_stomatal_conductance import potential_stomatal_conductance
from .open_minimum_temperature import open_minimum_temperature
from .closed_minimum_temperature import closed_minimum_temperature
from .open_vapor_pressure_deficit import open_vapor_pressure_deficit
from .closed_vapor_pressure_deficit import closed_vapor_pressure_deficit
from .leaf_conductance_to_evaporated_water import leaf_conductance_to_evaporated_water
from .maximum_boundary_layer_resistance import maximum_boundary_layer_resistance
from .minimum_boundary_layer_resistance import minimum_boundary_layer_resistance

__author__ = 'Qiaozhen Mu, Maosheng Zhao, Steven W. Running, Gregory Halverson'

logger = logging.getLogger(__name__)

def PMJPL(
    NDVI: Union[Raster, np.ndarray],
    ST_C: Union[Raster, np.ndarray] = None,
    emissivity: Union[Raster, np.ndarray] = None,
    albedo: Union[Raster, np.ndarray] = None,
    Rn_Wm2: Union[Raster, np.ndarray] = None,
    G_Wm2: Union[Raster, np.ndarray] = None,
    SWin_Wm2: Union[Raster, np.ndarray] = None,
    Ta_C: Union[Raster, np.ndarray] = None,
    Tmin_C: Union[Raster, np.ndarray] = None,
    RH: Union[Raster, np.ndarray] = None,
    IGBP: Union[Raster, np.ndarray] = None,
    FVC: Union[Raster, np.ndarray] = None,
    geometry: RasterGeometry = None,
    time_UTC: datetime = None,
    GEOS5FP_connection: GEOS5FP = None,
    resampling: str = "nearest",
    Ps_Pa: Union[Raster, np.ndarray] = None,
    elevation_m: Union[Raster, np.ndarray] = None,
    delta_Pa: Union[Raster, np.ndarray] = None,
    lambda_Jkg: Union[Raster, np.ndarray] = None,
    gamma_Jkg: Union[Raster, np.ndarray, float] = None,
    gl_sh: Union[Raster, np.ndarray] = None,
    CL: Union[Raster, np.ndarray] = None,
    Tmin_open: Union[Raster, np.ndarray] = None,
    Tmin_closed: Union[Raster, np.ndarray] = None,
    VPD_open: Union[Raster, np.ndarray] = None,
    VPD_closed: Union[Raster, np.ndarray] = None,
    gl_e_wv: Union[Raster, np.ndarray] = None,
    RBL_max: Union[Raster, np.ndarray] = None,
    RBL_min: Union[Raster, np.ndarray] = None,
    RH_threshold: float = RH_THRESHOLD,
    min_fwet: float = MIN_FWET,
    IGBP_upsampling_resolution_meters: float = IGBP_UPSAMPLING_RESOLUTION_METERS,
    upscale_to_daylight: bool = False,
    Rn_daylight_Wm2: Union[Raster, np.ndarray] = None,
    day_of_year: np.ndarray = None,
    regenerate_net_radiation: bool = False,
    offline_mode: bool = False) -> Dict[str, Raster]:
    """
    MOD16 Penman-Monteith Evapotranspiration Model (Version 1.5, Collection 6)

    Implements the MOD16 algorithm for partitioning daylight latent heat flux (LE) into canopy transpiration, soil evaporation, and wet canopy evaporation, following Mu et al. (2011) and the MOD16 User Guide (2016).

    Scientific Overview:
    -------------------
    This function estimates daylight evapotranspiration (ET) and its components using satellite-derived vegetation indices, land cover, and meteorological data. It applies the Penman-Monteith equation, partitioning energy and resistances according to biophysical and meteorological constraints. The model is designed for gridded raster inputs but supports numpy arrays for flexibility.

    Key Steps:
    - Retrieves or computes all necessary meteorological and surface variables (temperature, humidity, pressure, radiation, NDVI, LAI, FVC, etc.).
    - Calculates net radiation (Verma et al., 1989), soil heat flux (SEBAL; Bastiaanssen et al., 1998), and meteorological properties (Allen et al., 1998).
    - Computes resistances and conductances for canopy and soil evaporation, including biome-specific and environmental constraints (Mu et al., 2011; Monteith, 1965).
    - Partitions LE into wet canopy evaporation, transpiration, and soil evaporation, applying soil moisture and surface wetness constraints.

    Parameters
    ----------
    NDVI : Raster or np.ndarray
        Normalized Difference Vegetation Index.
    ST_C : Raster or np.ndarray, optional
        Surface temperature (Celsius).
    emissivity : Raster or np.ndarray, optional
        Surface emissivity.
    albedo : Raster or np.ndarray, optional
        Surface albedo.
    Rn_Wm2 : Raster or np.ndarray, optional
        Net radiation (W/m^2).
    G_Wm2 : Raster or np.ndarray, optional
        Soil heat flux (W/m^2).
    SWin_Wm2 : Raster or np.ndarray, optional
        Incoming shortwave radiation (W/m^2).
    Ta_C : Raster or np.ndarray, optional
        Air temperature (Celsius).
    Tmin_C : Raster or np.ndarray, optional
        Minimum air temperature (Celsius).
    RH : Raster or np.ndarray, optional
        Relative humidity (fraction).
    IGBP : Raster or np.ndarray, optional
        Land cover classification (IGBP).
    FVC : Raster or np.ndarray, optional
        Fractional vegetation cover.
    geometry : RasterGeometry, optional
        Spatial geometry for raster data.
    time_UTC : datetime, optional
        Timestamp for meteorological data.
    GEOS5FP_connection : GEOS5FP, optional
        Meteorological data source.
    resampling : str, optional
        Resampling method for raster data.
    Ps_Pa : Raster or np.ndarray, optional
        Surface pressure (Pa).
    elevation_m : Raster or np.ndarray, optional
        Elevation (meters).
    delta_Pa : Raster or np.ndarray, optional
        Slope of saturation vapor pressure curve (Pa/°C).
    lambda_Jkg : Raster or np.ndarray, optional
        Latent heat of vaporization (J/kg).
    gamma_Jkg : Raster, np.ndarray, or float, optional
        Psychrometric constant (J/kg).
    gl_sh : Raster or np.ndarray, optional
        Leaf conductance to sensible heat.
    CL : Raster or np.ndarray, optional
        Potential stomatal conductance.
    Tmin_open, Tmin_closed : Raster or np.ndarray, optional
        Open/closed minimum temperature by land cover.
    VPD_open, VPD_closed : Raster or np.ndarray, optional
        Open/closed vapor pressure deficit by land cover.
    gl_e_wv : Raster or np.ndarray, optional
        Leaf conductance to evaporated water vapor.
    RBL_max, RBL_min : Raster or np.ndarray, optional
        Maximum/minimum boundary layer resistance.
    RH_threshold : float, optional
        RH threshold for surface wetness.
    min_fwet : float, optional
        Minimum surface wetness.
    IGBP_upsampling_resolution_meters : float, optional
        Resolution for upsampling IGBP data.
    upscale_to_daylight : bool, optional
        Whether to upscale instantaneous values to daylight estimates.
    regenerate_net_radiation : bool, optional
        Whether to regenerate net radiation from surface components even if Rn_Wm2 is provided.

    Returns
    -------
    Dict[str, Raster]
        Dictionary of output rasters, including:
        - 'LEi_Wm2': Wet canopy evaporation (W/m^2)
        - 'LEc_Wm2': Canopy transpiration (W/m^2)
        - 'LEs': Soil evaporation (W/m^2)
        - 'LE_Wm2': Total latent heat flux (W/m^2)
        - Additional intermediate variables (e.g., resistances, conductances, meteorological properties)

    References
    ----------
    Mu, Q., Zhao, M., & Running, S. W. (2011). Improvements to a MODIS global terrestrial evapotranspiration algorithm. Remote Sensing of Environment, 115(8), 1781-1800. https://doi.org/10.1016/j.rse.2011.02.019
    MOD16 User Guide (2016): https://landweb.nascom.nasa.gov/QA_WWW/forPage/user_guide/MOD16UsersGuide2016.pdf
    Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). Crop evapotranspiration—Guidelines for computing crop water requirements—FAO Irrigation and drainage paper 56.
    Monteith, J. L. (1965). Evaporation and environment. Symposia of the Society for Experimental Biology, 19, 205-234.
    Carlson, T. N., & Ripley, D. A. (1997). On the relation between NDVI, fractional vegetation cover, and leaf area index. Remote Sensing of Environment, 62(3), 241-252.
    Bastiaanssen, W. G. M., et al. (1998). A remote sensing surface energy balance algorithm for land (SEBAL). Journal of Hydrology, 212-213, 198-212.
    Verma, S. B., Rosenberg, N. J., & Blad, B. L. (1989). Microclimate, evapotranspiration, and water status of maize under shelterbelt and non-shelterbelt conditions. Agricultural and Forest Meteorology, 46(1), 21-34.
    """
    results = {}

    # Check for missing variables in offline mode before any GEOS-5 FP retrievals
    if offline_mode:
        missing_vars = []
        
        if Ta_C is None:
            missing_vars.append("Ta_C")
        if Tmin_C is None:
            missing_vars.append("Tmin_C")
        if RH is None:
            missing_vars.append("RH")
        if elevation_m is None:
            missing_vars.append("elevation_m")
        if IGBP is None:
            missing_vars.append("IGBP")

        if missing_vars:
            raise MissingOfflineParameter(
                f"The following PM-JPL inputs are missing in offline mode: {', '.join(missing_vars)}"
            )

    if geometry is None and isinstance(NDVI, Raster):
        geometry = NDVI.geometry

    if GEOS5FP_connection is None:
        GEOS5FP_connection = GEOS5FP()

    if Ta_C is None and geometry is not None and time_UTC is not None:
        Ta_C = GEOS5FP_connection.Ta_C(
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling
        )

    if Ta_C is None:
        raise ValueError("air temperature (Ta_C) not given")

    check_distribution(Ta_C, "Ta_C")

    if Tmin_C is None and geometry is not None and time_UTC is not None:
        Tmin_K = GEOS5FP_connection.Tmin_K(
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling
        )

        Tmin_C = Tmin_K - 273.15

    if Tmin_C is None:
        raise ValueError("minimum temperature (Tmin_C) not given")

    check_distribution(Tmin_C, "Tmin_C")

    if RH is None and geometry is not None and time_UTC is not None:
        RH = GEOS5FP_connection.RH(
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling
        )

    if RH is None:
        raise ValueError("relative humidity (RH) not given")
        
    if elevation_m is None and geometry is not None:
        elevation_m = NASADEM.elevation_m(geometry=geometry)

    if IGBP is None and geometry is not None:
        if isinstance(geometry, VectorGeometry):
            IGBP_geometry = geometry           
        elif isinstance(geometry, RasterGeometry):
            IGBP_geometry = geometry.UTM(IGBP_upsampling_resolution_meters)
        else:
            raise ValueError(f"invalid geometry type for IGBP retrieval: {type(geometry)}")

        IGBP = load_MCD12C1_IGBP(geometry=IGBP_geometry)
    
    check_distribution(np.float32(IGBP), "IGBP")

    if regenerate_net_radiation or (Rn_Wm2 is None and albedo is not None and ST_C is not None and emissivity is not None):
        if SWin_Wm2 is None and geometry is not None and time_UTC is not None:
            logger.info("retrieving shortwave radiation (SWin_Wm2) from GEOS-5 FP")
            SWin_Wm2 = GEOS5FP_connection.SWin(
                time_UTC=time_UTC,
                geometry=geometry,
                resampling=resampling
            )
        elif SWin_Wm2 is not None:
            logger.info("using given shortwave radiation (SWin_Wm2)")

        if upscale_to_daylight:
            logger.info("running Verma net radiation with daylight upscaling")
        else:
            logger.info("running instantaneous Verma net radiation")

        Rn_results = verma_net_radiation(
            SWin_Wm2=SWin_Wm2,
            albedo=albedo,
            ST_C=ST_C,
            emissivity=emissivity,
            Ta_C=Ta_C,
            RH=RH,
            upscale_to_daylight=upscale_to_daylight,
        )

        Rn_Wm2 = Rn_results["Rn_Wm2"]

        if "Rn_daylight_Wm2" in Rn_results:
            Rn_daylight_Wm2 = Rn_results["Rn_daylight_Wm2"]

    elif Rn_Wm2 is not None:
        logger.info("using given net radiation (Rn_Wm2) for PM-JPL processing")

    if Rn_Wm2 is None:
        missing_vars = []
        if albedo is None:
            missing_vars.append('albedo')
        if ST_C is None:
            missing_vars.append('ST_C')
        if emissivity is None:
            missing_vars.append('emissivity')
        if missing_vars:
            raise ValueError(f"net radiation (Rn_Wm2) not given, and missing required variables to calculate: {', '.join(missing_vars)}")
        else:
            raise ValueError("net radiation (Rn_Wm2) not given and cannot be calculated")

    check_distribution(Rn_Wm2, "Rn_Wm2")

    if G_Wm2 is None and Rn_Wm2 is not None and ST_C is not None and NDVI is not None and albedo is not None:
        G_Wm2 = calculate_SEBAL_soil_heat_flux(
            Rn=Rn_Wm2,
            ST_C=ST_C,
            NDVI=NDVI,
            albedo=albedo
        )

    if G_Wm2 is None:
        raise ValueError("soil heat flux (G) not given")
    
    check_distribution(G_Wm2, "G_Wm2")
    results["G_Wm2"] = G_Wm2

    LAI = carlson_leaf_area_index(NDVI)

    # calculate fraction of vegetation cover if it's not given
    if FVC is None:
        # calculate fraction of vegetation cover from NDVI
        FVC = carlson_fractional_vegetation_cover(NDVI)

    # calculate surface air pressure if it's not given
    if Ps_Pa is None:
        # calculate surface air pressure is Pascal
        Ps_Pa = calculate_surface_pressure(elevation_m=elevation_m, Ta_C=Ta_C)

    # calculate Penman-Monteith/Priestley-Taylor delta term if it's not given
    if delta_Pa is None:
        # calculate Penman-Monteith/Priestley-Taylor delta term in Pascal per degree Celsius
        delta_Pa = delta_Pa_from_Ta_C(Ta_C)

    # calculate latent heat of vaporization if it's not given
    if lambda_Jkg is None:
        # calculate latent heat of vaporization in Joules per kilogram
        lambda_Jkg = lambda_Jkg_from_Ta_C(Ta_C)

    logger.info("calculating PM-MOD meteorology")

    # calculate air temperature in Kelvin
    Ta_K = celcius_to_kelvin(Ta_C)

    # calculate saturation vapor pressure in Pascal from air temperature in Celsius
    SVP_Pa = SVP_Pa_from_Ta_C(Ta_C)

    # calculate vapor pressure in Pascal from releative humidity and saturation vapor pressure
    Ea_Pa = RH * SVP_Pa

    # specific humidity of air
    # as a ratio of kilograms of water to kilograms of air and water
    # from surface pressure and actual water vapor pressure
    specific_humidity = calculate_specific_humidity(Ea_Pa, Ps_Pa)
    check_distribution(specific_humidity, "specific_humidity")
    results['specific_humidity'] = specific_humidity

    # calculate air density (rho) in kilograms per cubic meter
    rho_kgm3 = calculate_air_density(Ps_Pa, Ta_K, specific_humidity)
    check_distribution(rho_kgm3, "rho_kgm3")
    results["rho_kgm3"] = rho_kgm3

    # calculate specific heat capacity of the air (Cp)
    # in joules per kilogram per kelvin
    # from specific heat of water vapor (CPW)
    # and specific heat of dry air (CPD)
    Cp_Jkg = calculate_specific_heat(specific_humidity)
    check_distribution(Cp_Jkg, "Cp_Jkg")
    results["Cp"] = Cp_Jkg

    # calculate delta term if it's not given
    if delta_Pa is None:
        # slope of saturation vapor pressure curve in Pascal per degree
        delta_Pa = delta_Pa_from_Ta_C(Ta_C)

    check_distribution(delta_Pa, "delta_Pa")
    results["delta_Pa"] = delta_Pa

    # calculate gamma term if it's not given
    if gamma_Jkg is None:
        # calculate psychrometric gamma in Joules per kilogram
        gamma_Jkg = calculate_gamma(
            Ta_C=Ta_C,
            Ps_Pa=Ps_Pa,
            Cp_Jkg=Cp_Jkg
        )

    # vapor pressure deficit in Pascal
    VPD_Pa = rt.clip(SVP_Pa - Ea_Pa, 0.0, None)

    # calculate relative surface wetness (fwet)
    # from relative humidity
    fwet = calculate_relative_surface_wetness(
        RH=RH,
        RH_threshold=RH_threshold,
        min_fwet=min_fwet
    )
    
    check_distribution(fwet, "fwet")
    results['fwet'] = fwet

    logger.info("calculating PM-MOD resistances")

    # query leaf conductance to sensible heat (gl_sh) in seconds per meter
    if gl_sh is None:
        gl_sh = leaf_conductance_to_sensible_heat(IGBP, geometry)
        
    check_distribution(gl_sh, "gl_sh")
    results['gl_sh'] = gl_sh

    # calculate wet canopy resistance to sensible heat (rhc) in seconds per meter
    rhc = calculate_wet_canopy_resistance(gl_sh, LAI, fwet)
    check_distribution(rhc, "rhc")
    results['rhc'] = rhc

    # calculate resistance to radiative heat transfer through air (rrc)
    rrc = np.float32(rho_kgm3 * Cp_Jkg / (4.0 * SIGMA * Ta_K ** 3.0))
    check_distribution(rrc, "rrc")
    results['rrc'] = rrc

    # calculate aerodynamic resistance (rhrc)
    rhrc = np.float32((rhc * rrc) / (rhc + rrc))
    check_distribution(rhrc, "rhrc")
    results['rhrc'] = rhrc

    # calculate leaf conductance to evaporated water vapor (gl_e_wv)
    if gl_e_wv is None:
        gl_e_wv = leaf_conductance_to_evaporated_water(IGBP, geometry, IGBP_upsampling_resolution_meters)

    check_distribution(gl_e_wv, "gl_e_wv")
    results['gl_e_wv'] = gl_e_wv

    rvc = calculate_wet_canopy_resistance(gl_e_wv, LAI, fwet)
    check_distribution(rvc, "rvc")
    results['rvc'] = rvc

    # caluclate available radiation to the canopy (Ac)
    Ac = Rn_Wm2 * FVC
    check_distribution(Ac, "Ac")
    results['Ac'] = Ac

    # calculate wet latent heat flux (LEi)
    LE_interception_Wm2 = calculate_interception(
        delta_Pa=delta_Pa,
        Ac=Ac,
        rho=rho_kgm3,
        Cp=Cp_Jkg,
        VPD_Pa=VPD_Pa,
        FVC=FVC,
        rhrc=rhrc,
        fwet=fwet,
        rvc=rvc,
    )
    
    check_distribution(LE_interception_Wm2, "LE_interception_Wm2")
    results['LE_interception_Wm2'] = LE_interception_Wm2

    # calculate correctance factor (rcorr)
    rcorr = calculate_correctance_factor(Ps_Pa, Ta_K)
    check_distribution(rcorr, "rcorr")
    results['rcorr'] = rcorr

    # biome-specific mean potential stomatal conductance per unit leaf area
    if CL is None:
        CL = potential_stomatal_conductance(IGBP, geometry, IGBP_upsampling_resolution_meters)

    check_distribution(CL, "CL")
    results['CL'] = CL

    # open minimum temperature by land-cover
    if Tmin_open is None:
        Tmin_open = open_minimum_temperature(IGBP, geometry, IGBP_upsampling_resolution_meters)

    check_distribution(Tmin_open, "Tmin_open")
    results['Tmin_open'] = Tmin_open

    # closed minimum temperature by land-cover
    if Tmin_closed is None:
        Tmin_closed = closed_minimum_temperature(IGBP, geometry, IGBP_upsampling_resolution_meters)

    check_distribution(Tmin_closed, "Tmin_closed")
    results['Tmin_closed'] = Tmin_closed

    check_distribution(Tmin_C, "Tmin_C")

    # minimum temperature factor for stomatal conductance
    mTmin = Tmin_factor(Tmin_C, Tmin_open, Tmin_closed)
    check_distribution(mTmin, "mTmin")
    results['mTmin'] = mTmin

    # open vapor pressure deficit by land-cover
    if VPD_open is None:
        VPD_open = open_vapor_pressure_deficit(IGBP, geometry, IGBP_upsampling_resolution_meters)

    check_distribution(VPD_open, "VPD_open")
    results['VPD_open'] = VPD_open

    # closed vapor pressure deficit by land-cover
    if VPD_closed is None:
        VPD_closed = closed_vapor_pressure_deficit(IGBP, geometry, IGBP_upsampling_resolution_meters)

    check_distribution(VPD_closed, "VPD_closed")
    results['VPD_closed'] = VPD_closed

    # vapor pressure deficit factor for stomatal conductance
    mVPD = calculate_VPD_factor(VPD_open, VPD_closed, VPD_Pa)
    check_distribution(mVPD, "mVPD")
    results['mVPD'] = mVPD

    # stomatal conductance (gs1)
    gs1 = CL * mTmin * mVPD * rcorr
    check_distribution(gs1, "gs1")
    results['gs1'] = gs1

    # correct cuticular conductance constant to leaf cuticular conductance (Gcu) using correction factor (rcorr)
    Gcu = CUTICULAR_CONDUCTANCE * rcorr
    check_distribution(Gcu, "Gcu")
    results['Gcu'] = Gcu

    # canopy conductance
    Cc = calculate_canopy_conductance(LAI, fwet, gl_sh, gs1, Gcu)
    check_distribution(Cc, "Cc")
    results['Cc'] = Cc

    # surface resistance to evapotranspiration as inverse of canopy conductance (Cc)
    rs = rt.clip(1.0 / Cc, 0.0, MAX_RESISTANCE)
    check_distribution(rs, "rs")
    results['rs'] = rs

    # convective heat transfer as inverse of leaf conductance to sensible heat (gl_sh)
    rh = 1.0 / gl_sh
    check_distribution(rs, "rh")
    results['rh'] = rs

    # radiative heat transfer (rr)
    rr = rho_kgm3 * Cp_Jkg / (4.0 * SIGMA * Ta_K ** 3)
    check_distribution(rr, "rr")
    results['rr'] = rr

    # parallel resistance (ra)
    ra = (rh * rr) / (rh + rr)
    check_distribution(ra, "ra")
    results["ra"] = ra

    # transpiration
    LE_canopy_Wm2 = calculate_transpiration(
        delta_Pa=delta_Pa,
        Ac=Ac,
        rho_kgm3=rho_kgm3,
        Cp_Jkg=Cp_Jkg,
        VPD_Pa=VPD_Pa,
        FVC=FVC,
        ra=ra,
        fwet=fwet,
        rs=rs,
    )

    check_distribution(LE_canopy_Wm2, "LE_canopy_Wm2")
    results['LE_canopy_Wm2'] = LE_canopy_Wm2

    # soil evaporation
    # aerodynamic resistant constraints from land-cover
    if RBL_max is None:
        RBL_max = maximum_boundary_layer_resistance(IGBP, geometry, IGBP_upsampling_resolution_meters)

    check_distribution(RBL_max, "RBL_max")
    results['RBL_max'] = RBL_max

    if RBL_min is None:
        RBL_min = minimum_boundary_layer_resistance(IGBP, geometry, IGBP_upsampling_resolution_meters)

    check_distribution(RBL_min, "RBL_min")
    results['RBL_min'] = RBL_min

    # canopy aerodynamic resistance in seconds per meter
    rtotc = calculate_canopy_aerodynamic_resistance(VPD_Pa, VPD_open, VPD_closed, RBL_max, RBL_min)
    check_distribution(rtotc, "rtotc")
    results['rtotc'] = rtotc

    # total aerodynamic resistance
    rtot = rcorr * rtotc
    check_distribution(rtot, "rtot")
    results['rtot'] = rtot

    # resistance to radiative heat transfer through air
    rrs = np.float32(rho_kgm3 * Cp_Jkg / (4.0 * SIGMA * Ta_K ** 3))
    check_distribution(rrs, "rrs")
    results['rrs'] = rrs

    # aerodynamic resistance at the soil surface
    ras = (rtot * rrs) / (rtot + rrs)
    check_distribution(ras, "ras")
    results['ras'] = ras

    # available radiation at the soil
    Asoil = rt.clip((1.0 - FVC) * Rn_Wm2 - G_Wm2, 0.0, None)
    check_distribution(Asoil, "Asoil")
    results['Asoil'] = Asoil

    # separate wet soil evaporation and potential soil evaporation
    # wet soil evaporation
    wet_soil_evaporation_Wm2 = calculate_wet_soil_evaporation(
        delta_Pa=delta_Pa,
        Asoil=Asoil,
        rho_kgm3=rho_kgm3,
        Cp_Jkg=Cp_Jkg,
        FVC=FVC,
        VPD_Pa=VPD_Pa,
        ras=ras,
        fwet=fwet,
        rtot=rtot,
    )

    check_distribution(wet_soil_evaporation_Wm2, "wet_soil_evaporation_Wm2")
    results['wet_soil_evaporation_Wm2'] = wet_soil_evaporation_Wm2

    # potential soil evaporation
    potential_soil_evaporation_Wm2 = calculate_potential_soil_evaporation(
        delta_Pa=delta_Pa,
        Asoil=Asoil,
        rho=rho_kgm3,
        Cp_Jkg=Cp_Jkg,
        FVC=FVC,
        VPD_Pa=VPD_Pa,
        ras=ras,
        fwet=fwet,
        rtot=rtot,
    )

    check_distribution(potential_soil_evaporation_Wm2, "potential_soil_evaporation_Wm2")
    results['potential_soil_evaporation_Wm2'] = potential_soil_evaporation_Wm2

    # soil moisture constraint
    fSM = calculate_fSM(RH, VPD_Pa)
    check_distribution(fSM, "fSM")
    results['fSM'] = fSM

    # soil evaporation
    LE_soil_Wm2 = rt.clip(wet_soil_evaporation_Wm2 + potential_soil_evaporation_Wm2 * fSM, 0.0, None)
    LE_soil_Wm2 = rt.where(np.isnan(LE_soil_Wm2), 0.0, LE_soil_Wm2)
    check_distribution(LE_soil_Wm2, "LE_soil_Wm2")
    results['LE_soil_Wm2'] = LE_soil_Wm2

    # sum partitions into total latent heat flux
    LE_Wm2 = rt.clip(LE_interception_Wm2 + LE_canopy_Wm2 + LE_soil_Wm2, 0.0, Rn_Wm2)
    check_distribution(LE_Wm2, "LE_Wm2")
    results['LE_Wm2'] = LE_Wm2

    # --- daylight Upscaling Option ---
    if upscale_to_daylight and time_UTC is not None:
        logger.info("started daylight ET upscaling (PMJPL)")

        # Use new upscaling function from daylight_evapotranspiration
        daylight_results = daylight_ET_from_instantaneous_LE(
            LE_instantaneous_Wm2=LE_Wm2,
            Rn_instantaneous_Wm2=Rn_Wm2,
            G_instantaneous_Wm2=G_Wm2,
            day_of_year=day_of_year,
            time_UTC=time_UTC,
            geometry=geometry
        )
        # Add all returned daylight results to output
        results.update(daylight_results)

        logger.info("completed daylight ET upscaling (PMJPL)")

    return results
