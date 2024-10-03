import unittest
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.constants as scc

from em_methods.fuels.fuel_derivatives import fuel_properties, H2_KG_CITY

# Override logger to always use debug
logger = logging.getLogger("sim")
logger.setLevel(logging.DEBUG)
# BASETESTPATH: str = os.path.join("test", "pv")

CARBON_TAX: float = 54.58e-6  # $/g CO2 emissions, April 2024 EU

class TestFUEL(unittest.TestCase):
    def test_fuel_eu(self):
        """
        Test run the single diode equation
        """
        city = "Sines"
        h2_kg_value = H2_KG_CITY[city]["constant"]
        # Based on prices from August 2024
        h2_data_eu = fuel_properties(
            119.96, 2.01568, 8.23890e-5, h2_kg_value, 0, 1, 5.23, CARBON_TAX
        )  # prices of August 2024
        NH3_data = fuel_properties(
        18.646, 17.03022, 6.960942e-4, h2_kg_value, 0, 0.6666667, 0.46, CARBON_TAX
        )
        CH4_data = fuel_properties(
        50.00, 16.04236, 6.557164e-4, h2_kg_value, 1, 0.5, 1.49, CARBON_TAX
        )
        CH3OH_data = fuel_properties(
        19.930, 32.04, 0.7866, h2_kg_value, 1, 0.5, 0.38, CARBON_TAX
        )
        jet_data = fuel_properties(
        43.00, 170, 0.8201, h2_kg_value, 13.5, 0.0689655172, 1.21, CARBON_TAX
        )
