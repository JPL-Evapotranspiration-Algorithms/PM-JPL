
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

from daylight_evapotranspiration import lambda_Jkg_from_Ta_C

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
from .model import PMJPL
from .process_PMJPL_table import process_PMJPL_table
from .ECOv002_static_tower_PMJPL_inputs import load_ECOv002_static_tower_PMJPL_inputs
from .ECOv002_calval_PMJPL_inputs import load_ECOv002_calval_PMJPL_inputs
from .generate_PMJPL_GEOS5FP_inputs import generate_PMJPL_GEOS5FP_inputs
from .verify import verify
from .exceptions import *

__author__ = 'Qiaozhen Mu, Maosheng Zhao, Steven W. Running, Gregory Halverson'
