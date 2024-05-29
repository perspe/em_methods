import unittest
import datetime
from datetime import datetime
from pytz import timezone
import logging

from em_methods.pv import solar_angle, solar_power

# Override logger to always use debug
logger = logging.getLogger("sim")
logger.setLevel(logging.DEBUG)
# BASETESTPATH: str = os.path.join("test", "pv")

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
        pin, pdirect, pmodule = solar_power(-9.142685, 38.736946, 0.002, 35, date)
        # self.assertAlmostEqual(zenith, 67.6, 1)
        # self.assertAlmostEqual(azimuth, 221.2, 1)

