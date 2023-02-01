
import numpy as np

import os

# File with CIE color matching function
CIE_CMF = "cie-cmf.txt"

def xyz_from_xy(x, y):
    """Return the vector (x, y, 1-x-y)."""
    return np.array((x, y, 1-x-y))

class ColourSystem:
    """A class representing a colour system.

    A colour system defined by the CIE x, y and z=1-x-y coordinates of
    its three primary illuminants and its "white point".

    TODO: Implement gamma correction

    """

    # The CIE colour matching function for 380 - 780 nm in 5 nm intervals
    cmf = np.loadtxt(CIE_CMF, usecols=(1,2,3))

    def __init__(self, red, green, blue, white):
        """Initialise the ColourSystem object.

        Pass vectors (ie NumPy arrays of shape (3,)) for each of the
        red, green, blue  chromaticities and the white illuminant
        defining the colour system.

        """

        # Chromaticities
        self.red, self.green, self.blue = red, green, blue
        self.white = white
        # The chromaticity matrix (rgb -> xyz) and its inverse
        self.M = np.vstack((self.red, self.green, self.blue)).T 
        self.MI = np.linalg.inv(self.M)
        # White scaling array
        self.wscale = self.MI.dot(self.white)
        # xyz -> rgb transformation matrix
        self.T = self.MI / self.wscale[:, np.newaxis]

    def xyz_to_rgb(self, xyz, out_fmt=None):
        """Transform from xyz to rgb representation of colour.

        The output rgb components are normalized on their maximum
        value. If xyz is out the rgb gamut, it is desaturated until it
        comes into gamut.

        By default, fractional rgb components are returned; if
        out_fmt='html', the HTML hex string '#rrggbb' is returned.

        """

        rgb = self.T.dot(xyz)
        if np.any(rgb < 0):
            # We're not in the RGB gamut: approximate by desaturating
            w = - np.min(rgb)
            rgb += w
        if not np.all(rgb==0):
            # Normalize the rgb vector
            rgb /= np.max(rgb)

        if out_fmt == 'html':
            return self.rgb_to_hex(rgb)
        return rgb

    def rgb_to_hex(self, rgb):
        """Convert from fractional rgb values to HTML-style hex string."""

        hex_rgb = (255 * rgb).astype(int)
        return '#{:02x}{:02x}{:02x}'.format(*hex_rgb)

    def spec_to_xyz(self, spec):
        """Convert a spectrum to an xyz point.

        The spectrum must be on the same grid of points as the colour-matching
        function, self.cmf: 380-780 nm in 5 nm steps.

        """

        XYZ = np.sum(spec[:, np.newaxis] * self.cmf, axis=0)
        den = np.sum(XYZ)
        if den == 0.:
            return XYZ
        return XYZ / den

    def spec_to_rgb(self, spec, out_fmt=None):
        """Convert a spectrum to an rgb value."""

        xyz = self.spec_to_xyz(spec)
        return self.xyz_to_rgb(xyz, out_fmt)

illuminant_D65 = xyz_from_xy(0.3127, 0.3291)
cs_hdtv = ColourSystem(red=xyz_from_xy(0.67, 0.33),
                       green=xyz_from_xy(0.21, 0.71),
                       blue=xyz_from_xy(0.15, 0.06),
                       white=illuminant_D65)

cs_smpte = ColourSystem(red=xyz_from_xy(0.63, 0.34),
                        green=xyz_from_xy(0.31, 0.595),
                        blue=xyz_from_xy(0.155, 0.070),
                        white=illuminant_D65)

cs_srgb = ColourSystem(red=xyz_from_xy(0.64, 0.33),
                       green=xyz_from_xy(0.30, 0.60),
                       blue=xyz_from_xy(0.15, 0.06),
                       white=illuminant_D65)

cs = cs_hdtv

def rgb2hex(r,g,b):
    """Return color as #rrggbb for the given color values."""
    return '#%02x%02x%02x' % (r,g,b)
import time

def hex2rgb(hex_string):  # FUNÇÃO DE CONVERSÃO DO HEXCODE
    if type(hex_string) == str:
        hex_string = hex_string.upper()
        r_hex = hex_string[1:3]
        g_hex = hex_string[3:5]
        b_hex = hex_string[5:7]
        return int(r_hex, 16), int(g_hex, 16), int(b_hex, 16)
    else:
        return hex_string

def timeelapsed(start, end): #TAKES IN SECONDS
    timeelapsed = end - start
    hours = int(round(timeelapsed/3600, 0))
    minutes = int(round(timeelapsed % 3600 / 60, 0))
    seconds = int(round(timeelapsed % 60, 0))
    ctime = time.ctime(end)[4:-5]
    print(f'\n \n \t Simulations Finished on {ctime} \n \n' +
          '\t Duration: ' +
          f'{hours} hours, {minutes} minutes and {seconds} seconds.')

def satcolorgen(T): #TAKES IN A LIST FOR X=[380:781:5]
    cs = cs_hdtv
    spec = np.array(T)
    satcolor = cs.spec_to_rgb(spec, out_fmt='html')
    satcolor = hex2rgb(satcolor)
    return satcolor


def colordist(current, desired): #TAKES IN CURRENT AS RGB, DESIRED AS HEX

    desired_rgb = hex2rgb(desired)
    R, G, B = [*desired_rgb]
    r, g, b = [*current]
    colordistance = ((R-r)**2+(G-g)**2+(B-b)**2)**(1/2)
    return colordistance
