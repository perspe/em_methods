""" Main file containing the structures for the Grids necessary for the 
PWEM and RCWA """

import logging
from logging.config import fileConfig
import os
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.fft import fft2
from scipy.linalg import toeplitz

# Get module logger
base_path = os.path.dirname(os.path.abspath(__file__))
fileConfig(os.path.join(base_path, 'logging.ini'))
logger = logging.getLogger('dev')


class GridHasNoObjectError(Exception):
    """ Exception to raise when the grid does not have any object """
    pass


class UniformGrid():
    """ Main Class to define non-structured layers (avoids fft) """
    __slots__ = ("_xlims", "_ylims", "_thickness", "_e",
                 "_u")

    def __init__(self,
                 thickness: float,
                 e_default: complex = 1,
                 u_default: complex = 1,
                 xlims: List[float] = [-0.5, 0.5],
                 ylims: List[float] = [-0.5, 0.5]):
        self._xlims = xlims
        self._ylims = ylims
        self._thickness = thickness
        self._e = e_default
        self._u = u_default

    """ Define Accessible properties """

    @property
    def er(self) -> complex:
        return self._e

    @property
    def u0(self) -> complex:
        return self._u

    @property
    def limits(self) -> npt.NDArray:
        return np.array(
            [self._xlims[0], self._xlims[1], self._ylims[0], self._ylims[1]])

    @property
    def grid_size(self) -> Tuple[float, float]:
        xsize: float = max(self._xlims) - min(self._xlims)
        ysize: float = max(self._ylims) - min(self._ylims)
        return xsize, ysize

    @property
    def thickness(self) -> float:
        return self._thickness

    def convolution_matrices(
        self, p: int, q: int
    ) -> Tuple[npt.NDArray[np.complexfloating],
               npt.NDArray[np.complexfloating]]:
        """ Return the convolution matrix """
        base_matrix = np.eye((2 * p + 1) * (2 * q + 1))
        return base_matrix * self._e, base_matrix * self._u


class Grid2D():
    """Main class to create the grid
    This class is initialized with the x/y lengths and x/y max and min values
    Methods:
        Add primitives to the grid:
            - square(xcenter, ycenter, size, rotation=0)
            - rectange(xcenter, ycenter, xsize, ysize, rotation=0)
            - circle(xcenter, ycenter, radius)
            - ellipse(xcenter, ycenter, xradius, yradius, rotation=0)
            - triangle() TODO
        convolution_matrix() (Return convolution matrix for e and u)
    """
    __slots__ = ("_xlen", "_ylen", "_xlims", "_ylims", "_thickness", "_xrange",
                 "_yrange", "_XX", "_YY", "_e", "_u", "_has_object")

    def __init__(self,
                 xlen: int,
                 ylen: int,
                 thickness: float,
                 e_default: complex = 1,
                 u_default: complex = 1,
                 xlims: List[float] = [-0.5, 0.5],
                 ylims: List[float] = [-0.5, 0.5]):
        self._xlen = xlen
        self._ylen = ylen
        self._xlims = xlims
        self._ylims = ylims
        self._thickness = thickness
        self._xrange = np.linspace(min(xlims), max(xlims), xlen)
        self._yrange = np.linspace(max(ylims), min(ylims), ylen)
        self._XX, self._YY = np.meshgrid(self._xrange, self._yrange)
        self._e = np.ones_like(self._XX, dtype=np.complex64) * e_default
        self._u = np.ones_like(self._XX, dtype=np.complex64) * u_default
        self._has_object = False

    """ Define the accessible properties """

    @property
    def XYgrid(
            self) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        return self._XX, self._YY

    @property
    def er(self) -> npt.NDArray[np.complexfloating]:
        return self._e

    @property
    def u0(self) -> npt.NDArray[np.complexfloating]:
        return self._u

    @property
    def limits(self) -> npt.NDArray:
        return np.array(
            [self._xlims[0], self._xlims[1], self._ylims[0], self._ylims[1]])

    @property
    def grid_size(self) -> Tuple[float, float]:
        xsize: float = np.max(self._xrange) - np.min(self._xrange)
        ysize: float = np.max(self._yrange) - np.min(self._yrange)
        return xsize, ysize

    @property
    def thickness(self) -> float:
        return self._thickness

    """ Help functions """

    def has_object(self) -> bool:
        """ Check if any object has been added to the grid """
        return self._has_object

    def reinit(self) -> None:
        """ Remove all objects from the grid """
        self._has_object = False
        self._e = np.ones_like(self._e)
        self._u = np.ones_like(self._u)

    """ Functions to add the primitives """

    def add_rectangle(self,
                      er: complex,
                      u0: complex,
                      size: Tuple[float, float],
                      *,
                      center: Union[float, Tuple[float, float]] = 0,
                      rotation: float = 0) -> None:
        angle = np.radians(rotation)
        xcenter = center[0] if isinstance(center, tuple) else center
        ycenter = center[1] if isinstance(center, tuple) else center
        XX_new = (self._XX - xcenter) * np.cos(angle) - (
            self._YY - ycenter) * np.sin(angle)
        YY_new = (self._XX - xcenter) * np.sin(angle) + (
            self._YY - ycenter) * np.cos(angle)
        mask_x = (XX_new <= size[0] / 2) & (XX_new >= -size[0] / 2)
        mask_y = (YY_new <= size[1] / 2) & (YY_new >= -size[1] / 2)
        mask = mask_x & mask_y
        self._u[mask] = u0
        self._e[mask] = er
        self._has_object = True

    def add_square(self,
                   er: complex,
                   u0: complex,
                   size: float,
                   *,
                   center: Union[float, Tuple[float, float]] = 0,
                   rotation: float = 0) -> None:
        """ Add a square to the grid """
        self.add_rectangle(er,
                           u0, (size, size),
                           center=center,
                           rotation=rotation)

    def add_ellipse(self,
                    er: complex,
                    u0: complex,
                    radius: Tuple[float, float],
                    *,
                    center: Union[float, Tuple[float, float]] = 0,
                    rotation: float = 0) -> None:
        angle = np.radians(rotation)
        xcenter = center[0] if isinstance(center, tuple) else center
        ycenter = center[1] if isinstance(center, tuple) else center
        XX_new = (self._XX - xcenter) * np.cos(angle) - (
            self._YY - ycenter) * np.sin(angle)
        YY_new = (self._XX - xcenter) * np.sin(angle) + (
            self._YY - ycenter) * np.cos(angle)
        a, b = radius
        mask = ((XX_new / a)**2 + (YY_new / b)**2 <= 1)
        self._u[mask] = u0
        self._e[mask] = er
        self._has_object = True

    def add_cirle(self,
                  er: complex,
                  u0: complex,
                  radius: float,
                  center: Union[float, Tuple[float, float]] = 0) -> None:
        """
        Add a circle to the grid
        Args:
            er/u0: Complex er and u0 values to be defined in the grid
            radius: radius of the circle
            center: center to place the sphere
        """
        self.add_ellipse(er, u0, (radius, radius), center=center)

    def add_triangle(self, er: complex, u0: complex, a: Tuple[float, float],
                     b: Tuple[float, float], c: Tuple[float, float]):
        # Sort the values according to their x value
        sorted_list = [a, b, c]
        sorted_list.sort(key=lambda x: x[0])
        a, b, c = tuple(sorted_list)
        logger.debug(f"Sorted Values:{a=}::{b=}::{c=}")
        ax, ay = a
        bx, by = b
        cx, cy = c
        # Build the linear regressions that connect each point combination
        m_ab = (by - ay) / (bx - ax) if bx != ax else 1e20
        m_ac = (cy - ay) / (cx - ax) if cx != bx else 1e20
        m_bc = (cy - by) / (cx - bx) if cx != bx else 1e20
        logger.debug(f"{m_ab=}::{m_ac=}::{m_bc=}")
        b_ab = ay - m_ab * ax
        b_ac = ay - m_ac * ax
        b_bc = by - m_bc * bx
        # Determine the limits for the triangle
        lim_ab = self._XX > (self._YY - b_ab) / m_ab
        lim_bc = self._XX < (self._YY - b_bc) / m_bc
        # The ambiguity is all the the AC line
        if m_ac < 0:
            lim_ac = self._YY > m_ac * self._XX + b_ac
        elif m_ac == 0 and by >= ay:
            lim_ac = self._YY > m_ac * self._XX + b_ac
        elif m_ac == 0 and by <= ay:
            lim_ac = self._YY > m_ac * self._XX + b_ac
        else:
            lim_ac = self._YY < m_ac * self._XX + b_ac

        plt.figure()
        plt.imshow(lim_ab)
        plt.figure()
        plt.imshow(lim_bc)
        plt.figure()
        plt.imshow(lim_ac)
        plt.show()
        mask = lim_ab & lim_ac & lim_bc
        # logger.debug(f"{mask=}")
        self._u[mask] = u0
        self._e[mask] = er

    def convolution_matrices(
        self, p: int, q: int
    ) -> Tuple[npt.NDArray[np.complexfloating],
               npt.NDArray[np.complexfloating]]:
        """
        Return the convolution matrices for e and u
        Args:
            p/q (int): Number of harmonics (x and y, respectively)
        """
        # Obtain the p*q area of the 2D fft
        fft_e0 = fft2(self._e, norm="forward", workers=-2)
        fft_u0 = fft2(self._u, norm="forward", workers=-2)
        fft_e0_zoom = fft_e0[:2 * p + 1, :2 * q + 1]
        fft_u0_zoom = fft_u0[:2 * p + 1, :2 * q + 1]
        conv_e0 = toeplitz(fft_e0_zoom.flatten())
        conv_u0 = toeplitz(fft_u0_zoom.flatten())
        logger.debug(f"{conv_e0=}\n{conv_u0=}")
        return conv_e0, conv_u0

    """ Function for testing purposes """

    def _test_triangle(self):
        from math import floor
        # Initial variables
        Nx = 512
        Lx = 0.0175
        Ly = 0.015
        Ny = round(Nx * Ly / Lx)
        self._xlen = Nx
        self._ylen = Ny
        w = 0.8 * Ly
        # Base values
        self._e = 6 * np.ones((Ny, Nx))
        self._u = np.ones((Ny, Nx))
        self._thickness = 0.005
        # SEtup grid
        dx = Lx / Nx
        dy = Ly / Ny
        logger.info(f"{dx=}::{dy=}")
        xa = np.arange(Nx - 1) * dx
        xa = xa - np.mean(xa)
        ya = np.arange(Ny - 1) * dy
        ya = ya - np.mean(ya)
        # Build Device
        h = 0.5 * np.sqrt(3) * w
        ny = round(h / dy)
        ny1 = round((Ny - ny) / 2)
        ny2 = ny1 + ny - 1
        logger.info(f"{ny=}::{h=}::{ny1=}::{ny2=}")
        for ny in range(ny1, ny2):
            f = (ny - ny1) / (ny2 - ny1)
            nx = round(f * w / Lx * Nx)
            nx1 = 1 + floor((Nx - nx) / 2)
            nx2 = nx1 + nx
            self._e[ny, nx1:nx2] = 2
