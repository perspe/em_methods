import unittest
import os
import logging
from em_methods.lumerical.fdtd import fdtd_run, fdtd_run_analysis, fdtd_add_material

# Override logger to always use debug
logger = logging.getLogger('sim')
logger.setLevel(logging.WARNING)

BASETESTPATH: str = os.path.join("test", "fdtd")

class TestFDTD(unittest.TestCase):
    def test_single_run(self):
        """ Make a single run of a test file with everything default """
        fdtd_file: str = os.path.join(BASETESTPATH, "test_planar.fsp")
        properties = {
            "::model":
		        {"RT_mode": 1},
            "Planar_layers":
		        {"Perovskite": 250e-9,
		            "Spiro": 100e-9}
        }
        results = {
            "data":
                {"solar_generation": "Jsc"},
            "results":
                {"T": "T",
                 "R": "T"}
        }
        fdtd_run(fdtd_file, properties, results, delete=True)

    def test_fdtd_run_internals(self):
        """ Test internal components of fdtd_run function """
        fdtd_file: str = os.path.join(BASETESTPATH, "test_planar.fsp")
        properties = {
            "::model":
		        {"RT_mode": 1},
            "Planar_layers":
		        {"Perovskite": 250e-9,
		            "Spiro": 100e-9}
        }
        results = {
            "data":
                {"solar_generation": "Jsc"},
            "results":
                {"T": "T",
                 "R": "T"}
        }
        fdtd_run(fdtd_file, properties, results, override_prefix="test_override")
        override_file: str = os.path.join(BASETESTPATH, "test_override_test_planar.fsp")
        self.assertTrue(os.path.isfile(override_file))
        os.remove(override_file)

    def test_fdtd_log(self):
        """ Check if data obtained from log is correct """
        fdtd_file: str = os.path.join("test", "fdtd", "test_planar.fsp")
        properties = {
            "::model":
  		        {"RT_mode": 1},
            "Planar_layers":
                {"Perovskite": 250e-9,
                            "Spiro": 100e-9}
        }
        results = {
            "results":
                {"T": "T",
                "solar_generation": "Jsc",
                 "R": "T"}
        }
        res, _, _, autoshutoff = fdtd_run(fdtd_file, properties, results, override_prefix="log")
        self.assertAlmostEqual(res["results.solar_generation.Jsc"], 183.453, 3)
        self.assertEqual(autoshutoff[-1][0], 100)
        self.assertEqual(autoshutoff[-1][1], 8.42115e-06)

    def test_fdtd_run_analysis(self):
        """ Test fdtd_run_analysis function """
        fdtd_file: str = os.path.join("test", "fdtd", "log_test_planar.fsp")
        results = {
            "results":
                {"T": "T",
                "solar_generation": "Jsc",
                 "R": "T"}
        }
        res = fdtd_run_analysis(fdtd_file, results)
        self.assertAlmostEqual(res["results.solar_generation.Jsc"], 183.453, 3)

    def test_fdtd_graphical(self):
        """ Test if FDTD interface opens with the proper arguments """
        fdtd_file: str = os.path.join(BASETESTPATH, "test_planar.fsp")
        properties = {
            "::model":
		        {"RT_mode": 1},
            "Planar_layers":
		        {"Perovskite": 250e-9,
		            "Spiro": 100e-9}
        }
        results = {
            "data":
                {"solar_generation": "Jsc"},
            "results":
                {"T": "T",
                 "R": "T"}
        }
        fdtd_kw = {"hide": False}
        fdtd_run(fdtd_file, properties, results, delete=True, fdtd_kw=fdtd_kw)