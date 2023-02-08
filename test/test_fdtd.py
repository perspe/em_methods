import unittest
import os
import logging
from em_methods.fdtd.fdtd import fdtd_run, fdtd_add_material

# Override logger to always use debug
logger = logging.getLogger('simulation')
logger.setLevel(logging.DEBUG)

class TestFDTD(unittest.TestCase):
    def test_single_run(self):
        """ Make a single run of a test file with everything default """
        fdtd_file: str = os.path.join("test", "fdtd", "test_planar.fsp")
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
        fdtd_run(fdtd_file, properties, results)

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
            "data":
                {"solar_generation": "Jsc"},
            "results":
                {"T": "T",
                 "R": "T"}
        }
        res, fdtd_runtime, sim_run, autoshutoff = fdtd_run(fdtd_file, properties, results, override_prefix="log_test")
        self.assertEqual(autoshutoff[-1][0], 100)
        self.assertEqual(autoshutoff[-1][1], 4.77719e-06)