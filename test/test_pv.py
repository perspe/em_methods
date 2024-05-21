import unittest
import os
import logging
import numpy as np

from em_methods.pv import single_diode_rp, luqing_liu_diode

# Override logger to always use debug
logger = logging.getLogger('sim')
logger.setLevel(logging.INFO)
# BASETESTPATH: str = os.path.join("test", "fdtd")

class TestPV(unittest.TestCase):
    def test_single_diode_rp(self):
        """
        Test run the single diode equation
        """
        eta, rs, rsh, j0, jl = 1.5, 2, 3000, 1.5e-11, 21.7
        temp = 298
        voltage = np.linspace(0, 1.2, 100)
        j = single_diode_rp(voltage, jl, j0, rs, rsh, eta, temp)
        self.assertEqual(round(j[0], 4), 21.6855)

    def test_luqing_liu(self):
        """
        Test the luqing liu equation
        """
        eta, rs, rsh, temp =  1.5, 2, 3000, 298
        voc, jsc, jmpp = 1.09, 21.66, 20.34
        voltage = np.linspace(0, 1.2)
        current = luqing_liu_diode(voltage, jsc, jmpp, voc, rs, rsh, eta, temp)
        self.assertEqual(round(current[0], 4), 21.6596)



