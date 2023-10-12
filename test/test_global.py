
from em_methods import Units
import unittest

class TestGlobalFunctions(unittest.TestCase):
    def test_units(self):
        """Test units code"""
        self.assertAlmostEqual(Units.convertUnits(Units.NM, Units.UM), 1e-3, 4)
        self.assertAlmostEqual(Units.NM.convertTo(Units.UM), 1e-3, 4)
        self.assertAlmostEqual(Units.UM.convertTo(Units.NM), 1e3, 4)
        self.assertAlmostEqual(Units.DM.convertTo(Units.DM), 1, 4)
        self.assertAlmostEqual(Units.NM.convertTo(Units.M), 1e-9, 4)
        self.assertAlmostEqual(Units.NM.convertTo(Units.NM), 1, 4)