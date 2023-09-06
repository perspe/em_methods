import numpy as np
import os
import time
from pathlib import Path
from scipy.interpolate import interp1d
import logging
from typing import Tuple

logger = logging.getLogger()

# Get module paths
file_path = Path(os.path.abspath(__file__))
parent_path = file_path.parent.parent
data_path = os.path.join(parent_path, "data")
# File with CIE color matching function
CIE_CMF = os.path.join(data_path, "cie-cmf.txt")


def __xyz_from_xy(x, y):
    """Return the vector (x, y, 1-x-y)."""
    return np.array((x, y, 1 - x - y))

def rgb_to_hex(r: int, g: int, b: int) -> str:
    """
    Convert RGB values into Hex color format
    """
    return "#%02x%02x%02x" % (r, g, b)

def hex_to_rgb(hex_string: str) -> Tuple[int, int, int]: 
    """"
    Convert Hex code to RGB
    """
    hex_string = hex_string.upper()
    r_hex = hex_string[1:3]
    g_hex = hex_string[3:5]
    b_hex = hex_string[5:7]
    return int(r_hex, 16), int(g_hex, 16), int(b_hex, 16)

# The CIE colour matching function for 380 - 780 nm in 5 nm intervals
# with the respective interpolation functions
cmf = np.loadtxt(CIE_CMF)
_cmf_interp_red = interp1d(cmf[:, 0], cmf[:, 1])
_cmf_interp_green = interp1d(cmf[:, 0], cmf[:, 2])
_cmf_interp_blue = interp1d(cmf[:, 0], cmf[:, 3])


class ColourSystem:
    """
    Representation of the {CS} color system.
    The color system is defined by the CIE x, y, and z=1-x-y coordinates
    of its three primary illuminants and its "white point"
    Properties:
        - T (transformation matrix to convert to XYZ coordinates)
    Methods:
        - spec2xyz, spec2rgb, spec2html: convert spectrum to XYZ or RGB coordinates
        - xyz2rgb, xyz2hex: Convert XYZ values to RGB and HEX
    """
    # TODO: Implement gamma correction
    def __init__(self, red, green, blue, white) -> None:
        """
        Initialise the ColourSystem object.
        Pass vectors (ie NumPy arrays of shape (3,)) for each of the
        red, green, blue  chromaticities and the white illuminant
        defining the colour system.
        """
        # Chromaticities
        self._red, self._green, self._blue = red, green, blue
        self._white = white
        self._determine_T_matrix(self._red, self._blue, self._green, self._white)
        self._cmf_interp_list = [
            _cmf_interp_red,
            _cmf_interp_green,
            _cmf_interp_blue,
        ]

    def _frac_rgb_to_hex(self, r: float, g: float, b: float) -> str:
        """
        Convert from fractional rgb values to HTML-style hex string.
        This is needed for the XYZ to RGB convertion
        """
        rgb = np.array([r, g, b])
        hex_rgb = (255 * rgb).astype(int)
        return "#{:02x}{:02x}{:02x}".format(*hex_rgb)

    def _determine_T_matrix(self, red, green, blue, white) -> None:
        """
        Internal function to determine the transformation matrix
        """
        # The chromaticity matrix (rgb -> xyz) and its inverse 
        M = np.c_[red, blue, green] # similar to np.vstack((red, green, blue)).T
        MI = np.linalg.inv(M)
        # White scaling array
        wscale = MI@white
        # xyz -> rgb transformation matrix
        self._T = MI / wscale[:, np.newaxis]

    @property
    def T(self):
        """ Transformation matrix to convert xyz to rgb """
        return self._T

    def xyz_to_hex(self, x: float, y: float, z: float) -> str:
        """
        Transform XYZ to HTML/HEX representation of colour.
        """
        xyz = [x, y, z]
        rgb = self.T@xyz
        if np.any(rgb < 0):
            # We're not in the RGB gamut: approximate by desaturating
            w = -np.min(rgb)
            rgb += w
        if not np.all(rgb == 0):
            # Normalize the rgb vector
            rgb /= np.max(rgb)
        # The calculated rgb components are normalized on their maximum
        # value. If xyz is out the rgb gamut, it is desaturated until it
        # comes into gamut.
        return self._frac_rgb_to_hex(*rgb)

    def xyz_to_rgb(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """ Convert XYZ values to RGB values """
        hex_string = self.xyz_to_hex(x, y, z)
        return hex_to_rgb(hex_string)

    def spec_to_xyz(self, wavelength, spec, units="nm") -> Tuple[float, float, float]:
        """
        Convert a spectrum to an xyz point.
        This only considers if the results are between 380-780 nm
        Args:
            - wavelength: array of wavelengths from the spectrum
            - spec: array with the spectrum
            - units (default: nm): units for the wavelength
        Return:
            X, Y, Z
        """
        # Check all the arguments
        if units == "nm":
            pass
        elif units == "um":
            logger.debug("Updated wavelength calculation to nm from um")
            wavelength *= 1e4
        elif units == "m":
            logger.debug("Wavelength updated to nm from m")
            wavelength *= 1e9
        else:
            logger.error("Unknown units")
            raise Exception("Unknown units (avilable: nm, um)")
        if np.min(wavelength) < 380 or np.max(wavelength) > 780:
            logger.warning(f"Determined data limits: {np.min(wavelength)}::{np.max(wavelength)}")
            raise Exception("Wavelength range outside allowed bounds (380, 780)")
        XYZ = np.sum([spec * cmf_color(wavelength) for cmf_color in self._cmf_interp_list], axis=1)
        logging.debug(f"XYZ values: {XYZ}")
        den = np.sum(XYZ)        
        logging.debug(f"den: {den}")
        if den == 0.0:
            return XYZ[0], XYZ[0], XYZ[0]
        XYZ /= den
        return XYZ[0], XYZ[1], XYZ[2]

    def spec_to_rgb(self, wavelength, spec, units="nm") -> Tuple[int, int, int]:
        """Convert a spectrum to an rgb value."""
        x, y, z = self.spec_to_xyz(wavelength, spec, units=units)
        return self.xyz_to_rgb(x, y, z)

    def spec_to_hex(self, wavelength, spec, units="nm") -> str:
        """Convert a spectrum intro an html/hex code"""
        x, y, z = self.spec_to_xyz(wavelength, spec, units=units)
        return self.xyz_to_hex(x, y, z)


illuminant_D65 = __xyz_from_xy(0.3127, 0.3291)
cs_hdtv = ColourSystem(
    red=__xyz_from_xy(0.67, 0.33),
    green=__xyz_from_xy(0.21, 0.71),
    blue=__xyz_from_xy(0.15, 0.06),
    white=illuminant_D65,
)

cs_smpte = ColourSystem(
    red=__xyz_from_xy(0.63, 0.34),
    green=__xyz_from_xy(0.31, 0.595),
    blue=__xyz_from_xy(0.155, 0.070),
    white=illuminant_D65,
)

cs_srgb = ColourSystem(
    red=__xyz_from_xy(0.64, 0.33),
    green=__xyz_from_xy(0.30, 0.60),
    blue=__xyz_from_xy(0.15, 0.06),
    white=illuminant_D65,
)

# cs = cs_hdtv

# Update the docstring for the different colorspace functions
def __update_cs_docstring(cs_function, replacement: str):
    """ Replaces the {CS} in the doctring with the updated value """
    cs_function.__doc__ = cs_function.__doc__.replace("{CS}", replacement)

__update_cs_docstring(cs_srgb, "SRGB")
__update_cs_docstring(cs_hdtv, "HDTV")
__update_cs_docstring(cs_smpte, "SMTPE")


""" Other functions"""

def timeelapsed(start, end):  # TAKES IN SECONDS
    timeelapsed = end - start
    hours = int(round(timeelapsed / 3600, 0))
    minutes = int(round(timeelapsed % 3600 / 60, 0))
    seconds = int(round(timeelapsed % 60, 0))
    ctime = time.ctime(end)[4:-5]
    print(
        f"\n \n \t Simulations Finished on {ctime} \n \n"
        + "\t Duration: "
        + f"{hours} hours, {minutes} minutes and {seconds} seconds."
    )

def satcolorgen(T):  # TAKES IN A LIST FOR X=[380:781:5]
    cs = cs_hdtv
    spec = np.array(T)
    satcolor = cs.spec_to_rgb(spec, out_fmt="html")
    satcolor = hex2rgb(satcolor)
    return satcolor

def colordist(current, desired):  # TAKES IN CURRENT AS RGB, DESIRED AS HEX
    desired_rgb = hex2rgb(desired)
    R, G, B = [*desired_rgb]
    r, g, b = [*current]
    colordistance = ((R - r) ** 2 + (G - g) ** 2 + (B - b) ** 2) ** (1 / 2)
    return colordistance

if __name__ == "__main__":
    satcolorgen(list(range(380, 781,5)))
