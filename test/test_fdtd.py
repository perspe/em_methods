import unittest
import os
import logging
from em_methods.lumerical.fdtd import fdtd_run, fdtd_run_analysis, fdtd_add_material, rcwa_run, absorption_per_angle_results, angular_results
import lumapi

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

    def test_structures(self):
        """ Test for extracting higher amounts of data"""
        fdtd_file: str = os.path.join(BASETESTPATH, "test_periodic.fsp")
        properties = {
        }
        results = {
            "data":
                {"Ttot": "Ex"},
            "results":
                {"Ttot": "T"}
        }
        fdtd_kw = {"hide": True}
        fdtd_run(fdtd_file, properties, results, delete=True, fdtd_kw=fdtd_kw)

    def test_rcwa(self):
        """ Test running a RCWA solver and extracting the results """
        rcwa_file: str = os.path.join(BASETESTPATH, "test_rcwa.fsp")
        properties = {}
        results = {
            "results":
                {"RCWA": "total_energy"}
        }
        results = rcwa_run(rcwa_file, properties, results)

    def test_solvers(self):
        """ Test running different solvers in the same file """
        rcwa_file: str = os.path.join(BASETESTPATH, "test_rcwa.fsp")
        properties = {}
        results = {
            "results":
                {"RCWA": "total_energy"}
        }
        results = rcwa_run(rcwa_file, properties, results)
        results = {
            "results":
                {"R": "T"}
        }
        results = fdtd_run(rcwa_file, properties, results)

    def test_ang_abs(self):
        """ Test absorption per angle function """
        fdtd_file: str = os.path.join(BASETESTPATH, "best_4t_planar_tandem.fsp")
        properties = {}
        monitor_name = "solar_generation_Si"
        results = {"results":{
                        f"{monitor_name}::field": ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Px", "Py", "Pz","x", "y", "z", "f", ],
                        f"{monitor_name}::index": ["index_x", "index_y", "index_z"]},
                    "source": "source"}
        run_res = fdtd_run(
                    fdtd_file,
                    properties = properties,
                    get_results = results,
                    override_prefix=f"test",
                    delete = True
                    )
        total_absorption_per_angle, *_ = absorption_per_angle_results(monitor_name, run_res)
        self.assertAlmostEqual(sum(total_absorption_per_angle), 52.6439, 3)

    def test_angular_results(self):
        """ Test results per angle function """
        fdtd_file: str = os.path.join(BASETESTPATH, "4t_void_tandem_angle.fsp")
        properties = {}
        monitor_name = "Ttot"
        results = {
                    "results": {f"{monitor_name}": "T"},
                    "data": {f"{monitor_name}": ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "x", "y"]},
                    "source": "source",
                }
        run_res = fdtd_run(
                    fdtd_file,
                    properties = properties,
                    get_results = results,
                    override_prefix=f"test",
                    delete = True
                    )
        angular_avg, *_ = angular_results(monitor_name, run_res)
        print(angular_avg)
        self.assertAlmostEqual(sum(angular_avg), 0.067, 3)
        
        
        