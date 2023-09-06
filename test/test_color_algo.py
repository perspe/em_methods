import unittest
import os
import logging
import numpy as np
from em_methods.color_algo import cs_srgb, cs_hdtv, cs_smpte

# Override logger to always use debug
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

BASETESTPATH: str = os.path.join("test", "color_algo")

# Base transformations for comparison
T_hdtv = np.array([[6.20584986, -1.71746142, -1.04788582],
                   [-2.71554014, 5.51336837, 0.09687197],
                   [0.19384968, -0.39357359, 2.9841102]])
T_smpte = np.array([[10.6604194, -5.29004035, -1.65427378],
                    [-3.24746719, 6.00793853, 0.10684093],
                    [0.17120955, -0.5989372, 3.19255446]])
T_srgb = np.array([[9.85408376, -4.67437307, -1.51601289],
                   [-2.94438758, 5.69885136, 0.12623678],
                   [0.16915322, -0.62022847, 3.21391116]])

class testColor(unittest.TestCase):
    def test_tmatrix(self):
        """ Validate the tmatrix for the 3 main color_algo components """
        self.assertTrue(np.allclose(cs_hdtv.T, T_hdtv, rtol=0.01), "HDTV Transformation matrix")
        self.assertTrue(np.allclose(cs_smpte.T, T_smpte, rtol=0.01), "SMPTE Transformation matrix")
        self.assertTrue(np.allclose(cs_srgb.T, T_srgb, rtol=0.01), "SRGB Transformation matrix")

    def test_R_to_RGB(self):
        """ Determine RGB from reflectance data """
        test_file = os.path.join(BASETESTPATH, "R_CrossGratings_green.txt")
        data = np.loadtxt(test_file, delimiter=",", skiprows=3)
        data[:, 0] *= 1e9
        xyz = cs_srgb.spec_to_xyz(data[:, 0], data[:, 1]+1, units="nm")
        rgb = cs_srgb.spec_to_rgb(data[:, 0], data[:, 1]+1, units="nm")
        html = cs_srgb.spec_to_hex(data[:, 0], data[:, 1]+1, units="nm")
        self.assertTrue(np.allclose(np.array(xyz), (0.28941489, 0.53960382, 0.1709813), rtol=0.01), f"Wrong XYZ value {xyz}")
        self.assertTrue(rgb==(7, 255, 29))
        self.assertTrue(html=="#07ff1d")