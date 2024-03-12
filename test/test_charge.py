import time
import unittest
import os
import logging
from em_methods.lumerical import charge_run, LumericalError
import pandas as pd

# Override logger to always use debug
logger = logging.getLogger("dev")

BASETESTPATH: str = os.path.join("test", "charge")

def test_function(charge):
    logger.debug(f"Function runned in {charge}")
    return

class TestCHARGE(unittest.TestCase):
    def test_single_run(self):
        """Make a single run of a test file with everything default"""
        charge_file: str = os.path.join(BASETESTPATH, "teste_planar_2d.ldev")
        properties = {"::model": {"tITO": 0.1e-6, "tSnO2": 0.04e-6, "tSpiro": 0.1e-6}}
        charge_run(
            charge_file,
            properties,
            [0, 1, 2, "AZO"],
            delete=True,
            device_kw={"hide": True},
        )

    def test_run_function(self):
        """ Test run with internal function to charge """
        charge_file: str = os.path.join(BASETESTPATH, "teste_planar_2d.ldev")
        properties = {"::model": {"tITO": 0.1e-6, "tSnO2": 0.04e-6, "tSpiro": 0.1e-6}}
        charge_run(
            charge_file,
            properties,
            [0, 1, 2, "AZO"],
            delete=True,
            func=test_function,
            device_kw={"hide": True},
        )
        # Check if CheckRunState finishes properly
        time.sleep(2)

    @unittest.expectedFailure
    def test_run_error(self):
        """ Test run file that gives error """
        charge_file: str = os.path.join(BASETESTPATH, "teste_run_error.ldev")
        properties = {"::model": {"tITO": 0.1e-6, "tSnO2": 0.04e-6, "tSpiro": 0.1e-6}}
        charge_run(
            charge_file,
            properties,
            [0, 1, 2, "AZO"],
            delete=True,
            func=test_function,
            device_kw={"hide": True},
        )

    def test_version_error(self):
        """ Test run file that gives error """
        charge_file: str = os.path.join(BASETESTPATH, "teste_version_error.ldev")
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
