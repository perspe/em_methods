from typing import Dict
import unittest
import os
import logging
from em_methods.lumerical.fdtd import (
    fdtd_run,
    fdtd_run_analysis,
    fdtd_add_material,
    rcwa_run,
    filtered_pabs,
    fdtd_batch,
    rcwa_batch,
)
from em_methods.lumerical.lum_helper import overrideSlurm
import lumapi

# Override logger to always use debug
logger = logging.getLogger("sim_file")
logger.setLevel(logging.DEBUG)
logger = logging.getLogger("sim")
logger.setLevel(logging.DEBUG)

BASETESTPATH: str = os.path.join("test", "fdtd")


class TestFDTD(unittest.TestCase):
    def test_single_run(self):
        """Make a single run of a test file with everything default"""
        fdtd_file: str = os.path.join(BASETESTPATH, "test_planar.fsp")
        properties = {
            "::model": {"RT_mode": 1},
            "Planar_layers": {"Perovskite": 250e-9, "Spiro": 100e-9},
        }
        results = {"data": {"solar_generation": "Jsc"}, "results": {"T": "T", "R": "T"}}
        fdtd_run(fdtd_file, properties, results, delete=True, fdtd_kw={"hide": True})

    def test_fdtd_run_internals(self):
        """Test internal components of fdtd_run function"""
        fdtd_file: str = os.path.join(BASETESTPATH, "test_planar.fsp")
        properties = {
            "::model": {"RT_mode": 1},
            "Planar_layers": {"Perovskite": 250e-9, "Spiro": 100e-9},
        }
        results = {"data": {"solar_generation": "Jsc"}, "results": {"T": "T", "R": "T"}}
        fdtd_run(fdtd_file, properties, results, override_prefix="test_override")
        override_file: str = os.path.join(BASETESTPATH, "test_override_test_planar.fsp")
        self.assertTrue(os.path.isfile(override_file))
        os.remove(override_file)

    def test_fdtd_log(self):
        """Check if data obtained from log is correct"""
        fdtd_file: str = os.path.join("test", "fdtd", "test_planar.fsp")
        properties = {
            "::model": {"RT_mode": 1},
            "Planar_layers": {"Perovskite": 250e-9, "Spiro": 100e-9},
        }
        results = {"results": {"T": "T", "solar_generation": "Jsc", "R": "T"}}
        res, _, _, autoshutoff = fdtd_run(
            fdtd_file, properties, results, override_prefix="log"
        )
        self.assertAlmostEqual(res["results.solar_generation.Jsc"], 183.453, 3)
        self.assertEqual(autoshutoff[-1][0], 100)
        self.assertEqual(autoshutoff[-1][1], 8.42115e-06)

    def test_fdtd_run_analysis(self):
        """Test fdtd_run_analysis function"""
        fdtd_file: str = os.path.join("test", "fdtd", "log_test_planar.fsp")
        results = {"results": {"T": "T", "solar_generation": "Jsc", "R": "T"}}
        res = fdtd_run_analysis(fdtd_file, results)
        self.assertAlmostEqual(res["results.solar_generation.Jsc"], 183.453, 3)

    def test_fdtd_graphical(self):
        """Test if FDTD interface opens with the proper arguments"""
        fdtd_file: str = os.path.join(BASETESTPATH, "test_planar.fsp")
        properties = {
            "::model": {"RT_mode": 1},
            "Planar_layers": {"Perovskite": 250e-9, "Spiro": 100e-9},
        }
        results = {"data": {"solar_generation": "Jsc"}, "results": {"T": "T", "R": "T"}}
        fdtd_kw = {"hide": False}
        fdtd_run(fdtd_file, properties, results, delete=True, fdtd_kw=fdtd_kw)

    def test_structures(self):
        """Test for extracting higher amounts of data"""
        fdtd_file: str = os.path.join(BASETESTPATH, "test_periodic.fsp")
        properties = {}
        results = {"data": {"Ttot": "Ex"}, "results": {"Ttot": "T"}}
        fdtd_kw = {"hide": True}
        fdtd_run(fdtd_file, properties, results, delete=True, fdtd_kw=fdtd_kw)

    def test_rcwa(self):
        """Test running a RCWA solver and extracting the results"""
        rcwa_file: str = os.path.join(BASETESTPATH, "test_rcwa.fsp")
        properties = {}
        results = {"results": {"RCWA": "total_energy"}}
        results = rcwa_run(rcwa_file, properties, results)

    def test_rcwa_batch(self):
        """Test running a RCWA solver and extracting the results"""
        rcwa_file: str = os.path.join(BASETESTPATH, "test_rcwa.fsp")
        results = {"results": {"RCWA": "total_energy"}}
        properties = [{} for i in range(2)]
        results = rcwa_batch(rcwa_file, properties, results)
        print(results)

    def test_solvers(self):
        """Test running different solvers in the same file"""
        rcwa_file: str = os.path.join(BASETESTPATH, "test_rcwa.fsp")
        properties = {}
        results = {"results": {"RCWA": "total_energy"}}
        results = rcwa_run(rcwa_file, properties, results)
        results = {"results": {"R": "T"}}
        results = fdtd_run(rcwa_file, properties, results)

    def test_filter(self):
        """Test if filter works properly"""
        fdtd_file: str = os.path.join(BASETESTPATH, "test_2t_tandem_planar_model.fsp")
        properties = {}
        results = {
            "results": {
                "Pabs_all_materials": ["Pabs", "Pabs_total"],
                "Pabs_all_materials::index": ["x", "y", "z", "f", "index_z"],
                "SG_WBG": ["Pabs", "Jsc", "Pabs_total"],
                "SG_LBG": ["Pabs", "Jsc", "Pabs_total"],
            }
        }
        sim = fdtd_run(fdtd_file, properties, results, delete=False)
        sim = sim[0]
        wbg_pabs = filtered_pabs(
            "Pabs_all_materials",
            sim,
            (fdtd_file, "Perovskite WBG"),
            "am1.5",
            tol=1 * 10**-15,
        )
        jsc_res_wbg = round(abs(wbg_pabs["jsc"]), 2)
        solar_gen_wbg = round(0.1 * sim["results.SG_WBG.Jsc"], 2)
        self.assertAlmostEqual(jsc_res_wbg, solar_gen_wbg, 0)

    def test_fdtd_batch(self):
        fdtd_file: str = os.path.join(BASETESTPATH, "test_planar.fsp")
        results = {
            "data": {"solar_generation": "Jsc"},
        }
        properties_list = []
        for tpvk in [100, 200]:
            properties = {
                "::model": {"RT_mode": 1},
                "Planar_layers": {"Perovskite": tpvk * 1e-9, "Spiro": 100e-9},
            }
            properties_list.append(properties)
        fdtd_batch(fdtd_file, properties_list, results, fdtd_kw={"hide": True})


class TestFDTDServer(unittest.TestCase):
    def setUp(self) -> None:
        self.base_test_file: str = os.path.join(BASETESTPATH, "test_planar.fsp")
        self.default_results = {"data": {"solar_generation": "Jsc"}}
        self.base_properties = {
            "::model": {"RT_mode": 1},
            "Planar_layers": {"Perovskite": 250e-9},
        }

    def test_fdtd_batch(self):
        """Run different configurations to run on slurm server"""
        # Run in 2x8 config
        properties_list = [self.base_properties for _ in range(2)]
        overrideSlurm_args = {"ntasks": 8}
        res = fdtd_batch(
            self.base_test_file,
            properties_list,
            self.default_results,
            func=overrideSlurm,
            fdtd_kw={"hide": True},
            **overrideSlurm_args,
        )
        for _, runtime, _, _ in res:
            logger.info(f"Config (2x8) Runtime: {runtime}")

        # Run in 4x4 config
        properties_list = [self.base_properties for _ in range(4)]
        overrideSlurm_args = {"ntasks": 4}
        res = fdtd_batch(
            self.base_test_file,
            properties_list,
            self.default_results,
            func=overrideSlurm,
            fdtd_kw={"hide": True},
            **overrideSlurm_args,
        )
        for _, runtime, _, _ in res:
            logger.info(f"Config (4x4) Runtime: {runtime}")
