import time
import unittest
import os
import logging
from em_methods.lumerical import (
    charge_run,
    LumericalError,
    SimInfo,
)
from em_methods.lumerical.charge import (
    __get_gen,
    run_fdtd_and_charge,
    run_fdtd_and_charge_legacy,
)
import pandas as pd

# Override logger to always use debug
logger = logging.getLogger("sim")
logger.setLevel(logging.DEBUG)

BASETESTPATH_CHARGE: str = os.path.join("test", "charge")
BASETESTPATH_FDTD_CHARGE: str = os.path.join("test", "fdtd_and_charge")


def test_function(charge):
    logger.debug(f"Function runned in {charge}")
    return


class TestCHARGE(unittest.TestCase):
    def test_single_run(self):
        """Make a single run of a test file with everything default"""
        charge_file: str = os.path.join(BASETESTPATH_CHARGE, "teste_planar_2d.ldev")
        properties = {"::model": {"tITO": 0.1e-6, "tSnO2": 0.04e-6, "tSpiro": 0.1e-6}}
        get_results = {"CHARGE": "AZO"}
        try:
            charge_run(
                charge_file,
                properties,
                get_results,
                delete=False,
                device_kw={"hide": True},
            )
        except LumericalError:
            logger.critical("Error running file")

    def test_multiple_run(self):
        """Make a single run of a test file with everything default"""
        charge_file: str = os.path.join(BASETESTPATH_CHARGE, "teste_planar_2d.ldev")
        properties = {"::model": {"tITO": 0.1e-6, "tSnO2": 0.04e-6, "tSpiro": 0.1e-6}}
        get_results = {"CHARGE": "AZO"}
        for i in range(100):
            try:
                charge_run(
                    charge_file,
                    properties,
                    get_results,
                    delete=True,
                    device_kw={"hide": True},
                )
            except LumericalError:
                logger.critical(f"Error running file {i}")

    def test_run_function(self):
        """Test run with internal function to charge"""
        charge_file: str = os.path.join(BASETESTPATH_CHARGE, "teste_planar_2d.ldev")
        properties = {"::model": {"tITO": 0.1e-6, "tSnO2": 0.04e-6, "tSpiro": 0.1e-6}}
        get_results = {"CHARGE": "AZO"}
        charge_run(
            charge_file,
            properties,
            get_results,
            delete=True,
            func=test_function,
            device_kw={"hide": True},
        )
        # Check if CheckRunState finishes properly
        time.sleep(2)

    @unittest.expectedFailure
    def test_run_error(self):
        """Test run file that gives error"""
        charge_file: str = os.path.join(BASETESTPATH_CHARGE, "teste_run_error.ldev")
        properties = {"::model": {"tITO": 0.1e-6, "tSnO2": 0.04e-6, "tSpiro": 0.1e-6}}
        get_results = {"CHARGE": "AZO"}
        charge_run(
            charge_file,
            properties,
            get_results,
            delete=True,
            func=test_function,
            device_kw={"hide": True},
        )

    def test_version_error(self):
        """Test run file that gives error"""
        charge_file: str = os.path.join(BASETESTPATH_CHARGE, "teste_version_error.ldev")
        properties = {"::model": {"tITO": 0.1e-6, "tSnO2": 0.04e-6, "tSpiro": 0.1e-6}}
        try:
            charge_run(
                charge_file,
                properties,
                [0, 1, 2, "AZO"],
                delete=True,
                func=test_function,
                device_kw={"hide": True},
            )
        except LumericalError:
            pass
        # Check if CheckRunState finishes properly
        time.sleep(10)

    def test_get_info(self):
        """Make a single run of a test file with everything default"""
        charge_file: str = os.path.join(BASETESTPATH_CHARGE, "teste_planar_2d.ldev")
        properties = {"::model": {"tITO": 0.1e-6, "tSnO2": 0.04e-6, "tSpiro": 0.1e-6}}
        info = {"simulation region si": "x span"}
        _, _, _, info = charge_run(
            charge_file,
            properties,
            [0, 1, 2, "AZO"],
            get_info=info,
            delete=True,
            device_kw={"hide": True},
        )
        logger.info(info)

    def test_multiple_full(self):
        logger.info(
            """
        ------------------------ NEW RUN --------------------------
        """
        )
        Perovskite = SimInfo(
            "solar_generation_PVK", "G_PVK.mat", "Perovskite", "ITO_top", "ITO"
        )
        path = os.path.join("test", "charge")
        active_region_list = [Perovskite]
        charge_file = "cell_for_test_psk.ldev"
        fdtd_file = "cell_for_test.fsp"
        x = [x * (10**-6) / 1000.0 for x in range(40, 180, 10)]
        properties = {"::model": {"tITO": 0.1e-6, "tSnO2": 0.1e-6, "tSpiro": 0.4e-6}}
        PCE = []
        for n in x:
            properties["::model"]["tSnO2"] = n
            pce, *_ = run_fdtd_and_charge(
                active_region_list, properties, charge_file, path, fdtd_file, "2d"
            )
            PCE.append(pce)
        print(PCE)

    def test_get_gen(self):
        """
        Test function to extract generation profiles
        """
        test_file_fdtd = os.path.join(
            BASETESTPATH_FDTD_CHARGE, "test_planar_tandem_4t.fsp"
        )
        pvk_siminfo = SimInfo(
            "solar_generation_PVK", "G_PVK.mat", "Perovskite", "ITO_top", "ITO"
        )
        si_siminfo = SimInfo(
            "solar_generation_Si", "G_Si.mat", "Si", "AZO", "ITO_bottom"
        )
        active_regions = [si_siminfo, pvk_siminfo]
        jsc = __get_gen(test_file_fdtd, {}, active_regions, avg_mode=True)

    def test_run_fdtd_and_charge_4t(self):
        """Run single instance of run_fdtd_and_charge function"""
        test_file_fdtd = os.path.join(
            BASETESTPATH_FDTD_CHARGE, "test_planar_tandem_4t.fsp"
        )
        test_file_charge = os.path.join(
            BASETESTPATH_FDTD_CHARGE, "test_planar_tandem_4t.ldev"
        )
        pvk_siminfo = SimInfo(
            "solar_generation_PVK", "G_PVK.mat", "Perovskite", "ITO_top", "ITO"
        )
        si_siminfo = SimInfo(
            "solar_generation_Si", "G_Si.mat", "Si", "AZO", "ITO_bottom"
        )
        active_regions = [si_siminfo, pvk_siminfo]
        run_fdtd_and_charge(active_regions, {}, test_file_charge, test_file_fdtd)

    def test_run_fdtd_and_charge_4t_non_default(self):
        """
        Run single instance of run_fdtd_and_charge using non-default values
        for some variables
        """
        test_file_fdtd = os.path.join(
            BASETESTPATH_FDTD_CHARGE, "test_planar_tandem_4t.fsp"
        )
        test_file_charge = os.path.join(
            BASETESTPATH_FDTD_CHARGE, "test_planar_tandem_4t.ldev"
        )
        pvk_siminfo = SimInfo(
            "solar_generation_PVK", "G_PVK.mat", "Perovskite", "ITO_top", "ITO", True
        )
        si_siminfo = SimInfo(
            "solar_generation_Si", "G_Si.mat", "Si", "AZO", "ITO_bottom", 4.73e-15
        )
        active_regions = [si_siminfo, pvk_siminfo]
        properties = {
            "::model": {
                "tPerovskite": 163.43e-9,
                "tlayer": 20e-9,
                "tITO": 50e-9,
                "tTCO": 50e-9,
                "n": 2.80,
            }
        }
        si_bias_regime = {
            "method_solver": "GUMMEL",
            "voltage": 0.8,
            "voltage_points": 31,
        }
        pvk_bias_regime = {
            "method_solver": "GUMMEL",
            "voltage": 1.8,
            "voltage_points": 61,
        }
        run_fdtd_and_charge(
            active_regions,
            properties,
            test_file_charge,
            test_file_fdtd,
            def_sim_region="2d",
            override_bias_regime_args=[si_bias_regime, pvk_bias_regime],
            min_edge=[0.005e-6, 0.001e-6],
            override_get_gen_args={"avg_mode": True},
        )

    def test_run_fdtd_and_charge_2t(self):
        test_file_fdtd = os.path.join(
            BASETESTPATH_FDTD_CHARGE, "test_2t_void_tandem.fsp"
        )
        test_file_charge = os.path.join(
            BASETESTPATH_FDTD_CHARGE, "test_2t_void_tandem.ldev"
        )
        pvk_siminfo = SimInfo(
            "solar_generation_PVK", "G_PVK.mat", "Perovskite", "interlayer", "ITO", True
        )
        si_siminfo = SimInfo(
            "solar_generation_Si", "G_Si.mat", "Si", "AZO", "interlayer", 4.73e-15
        )
        active_regions = [si_siminfo, pvk_siminfo]
        # properties = {
        #     "::model": {
        #         "tPerovskite": 163.43e-9,
        #         "tlayer": 20e-9,
        #         "tITO": 50e-9,
        #         "tTCO": 50e-9,
        #         "n": 2.80,
        #     }
        # }
        si_bias_regime = {
            "method_solver": "GUMMEL",
            "voltage": 0.8,
            "voltage_points": 31,
        }
        pvk_bias_regime = {
            "method_solver": "GUMMEL",
            "voltage": 1.8,
            "voltage_points": 61,
        }
        run_fdtd_and_charge(
            active_regions,
            {},
            test_file_charge,
            test_file_fdtd,
            def_sim_region="2d",
            override_bias_regime_args=[si_bias_regime, pvk_bias_regime],
            min_edge=[0.005e-6, 0.001e-6],
            override_get_gen_args={"avg_mode": True},
        )

    def test_run_fdtd_and_charge_legacy(self):
        """Test compatibility function for run_fdtd_and_charge"""
        # 4T
        fdtd_name = "test_planar_tandem_4t.fsp"
        charge_name = "test_planar_tandem_4t.ldev"
        Perovskite = SimInfo(
            "solar_generation_PVK", "G_PVK.mat", "Perovskite", "ITO_top", "ITO"
        )
        Si = SimInfo("solar_generation_Si", "G_Si.mat", "Si", "AZO", "ITO_bottom")
        region = [Si, Perovskite]
        B_list = [4.73e-15, True]
        v_max = [0.8, 1.6]
        min_edge = [0.005 * 10**-6, 0.001 * 10**-6]
        range_num_points = [31, 61]
        properties = {
            "::model": {
                "tPerovskite": 163.43e-9,
                "tlayer": 20e-9,
                "tITO": 50e-9,
                "tTCO": 50e-9,
                "n": 2.80,
            }
        }
        run_fdtd_and_charge_legacy(
            region,
            properties,
            charge_name,
            BASETESTPATH_FDTD_CHARGE,
            fdtd_name,
            v_max=v_max,
            def_sim_region="2d",
            run_FDTD=True,
            B=B_list,
            avg_mode=True,
            method_solver="GUMMEL",
            min_edge=min_edge,
            range_num_points=range_num_points,
            # save_csv=True,
        )
        pass
