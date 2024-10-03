import datetime
from datetime import datetime
import logging
import unittest

import matplotlib.pyplot as plt
import numpy as np
from pytz import timezone

from em_methods.pv import solar_angle, solar_power
from em_methods.pv import eot_main, solar_declination

# Override logger to always use debug
logger = logging.getLogger("sim")
logger.setLevel(logging.DEBUG)
# BASETESTPATH: str = os.path.join("test", "pv")

PLOTS=True

class TestIrr(unittest.TestCase):
    def test_run(self):
        """
        Test different cases to check if everything runs ok
        """
        test_dt1 = datetime.today()
        solar_angle(-9.142685, 38.7, test_dt1)
        test_dt2 = datetime(2024, 5, 20, 10, 0, 0, tzinfo=timezone("CET"))
        solar_angle(-9.142685, 38.7, test_dt2)
        test_dt3 = datetime(2024, 5, 20, 10, 0, 0, tzinfo=timezone("America/New_York"))
        solar_angle(-9.142685, 38.7, test_dt3)

    def test_solar_power(self):
        """
        Determine the solar angles (based on Cristina results)
        """
        date = datetime(2024, 11, 8, 15, 0, 0)
        zenith, azimuth, *_ = solar_angle(-9.142685, 38.736946, date)
        self.assertAlmostEqual(zenith, 67.6, 1)
        self.assertAlmostEqual(azimuth, 221.2, 1)

    def test_solar_power_angle(self):
        """
        Determine solar power irrandiance in location (based on Cristina results)
        """
        date = datetime(2024, 6, 21, 13, 0, 0)
        ghi, gti, gain = solar_power(-9.142685, 38.736946, 0.002, 35, date)
        self.assertAlmostEqual(ghi, 0.992, 3)
        self.assertAlmostEqual(gti, 0.968, 3)
        self.assertAlmostEqual(gain, -2.5, 1)

    @unittest.skipIf(PLOTS==False, "Test defined to not show plots")
    def test_internal_functions(self):
        """
        Test various internal functions
        """
        days = np.linspace(0, 365, 365, dtype=np.integer)
        eots = eot_main(days)
        declinations = solar_declination(days)
        plt.plot(days, eots, label="EoT")
        plt.plot(days, declinations, label="Solar Declination")
        plt.xlabel("Days")
        plt.legend()
        plt.show()
