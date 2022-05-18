""" Main file containing the structures for the Grids necessary for the 
PWEM and RCWA """

import logging
from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.fft import fft2, fftshift
from scipy.linalg import convolution_matrix


class GridHasNoObjectError(Exception):
    """ Exception to raise when the grid does not have any object """
    pass


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
    def __init__(self,
                 xlen: int,
                 ylen: int,
                 xlims: List[float] = [-0.5, 0.5],
                 ylims: List[float] = [-0.5, 0.5]):
        self._xlen = xlen
        self._ylen = ylen
        self._xlims = xlims
        self._ylims = ylims
        self._xrange = np.linspace(min(xlims), max(xlims), xlen)
        self._yrange = np.linspace(max(ylims), min(ylims), ylen)
        self._XX, self._YY = np.meshgrid(self._xrange, self._yrange)
        self._e = np.ones_like(self._XX, dtype=np.complex64)
        self._u = np.ones_like(self._XX, dtype=np.complex64)
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
    def grid_size(self) -> Tuple[float, float]:
        xsize: float = np.max(self._xrange) - np.min(self._xrange)
        ysize: float = np.max(self._yrange) - np.min(self._yrange)
        return xsize, ysize

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

    # def add_triangle(self, er: complex, u0: complex, a: Tuple[float, float],
    #                  b: Tuple[float, float], c: Tuple[float, float]):
    #     # TODO
    #     # Sort the values according to their x value
    #     sorted_list = [a, b, c]
    #     sorted_list.sort(key=lambda x: x[0])
    #     a, b, c = tuple(sorted_list)
    #     logging.debug(f"Sorted Values:{a=}::{b=}::{c=}")
    #     ax, ay = a
    #     bx, by = b
    #     cx, cy = c
    #     # Build the linear regressions that connect each point combination
    #     m_ab = (by - ay) / (bx - ax)
    #     m_ac = (cy - ay) / (cx - ax)
    #     m_bc = (cy - by) / (cx - bx)
    #     logging.debug(f"{m_ab=}::{m_ac=}::{m_bc=}")
    #     b_ab = ay - m_ab * ax
    #     b_ac = ay - m_ac * ax
    #     b_bc = by - m_bc * bx

    #     sign_m_ab = np.sign(m_ab * self._XX - self._YY)
    #     sign_m_ac = np.sign(m_ac * self._XX - self._YY)
    #     sign_m_bc = np.sign(m_bc * self._XX - self._YY)
    #     eq1 = (sign_m_ab + 1) * (sign_m_ac - 1) * (m_ab * self._XX + b_ab -
    #                                                self._YY)
    #     eq2 = (sign_m_ac + 1) * (sign_m_bc + 1) * (m_ac * self._XX + b_ac -
    #                                                self._YY)
    #     eq3 = (sign_m_bc - 1) * (sign_m_ab - 1) * (m_bc * self._XX + b_bc -
    #                                                self._YY)
    #     mask = (eq1 + eq2 + eq3 < 0)
    #     logging.debug(f"{mask=}")
    #     self._u[mask] = u0
    #     self._e[mask] = er

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
        fft_e0_shift = fftshift(fft_e0)
        fft_u0_shift = fftshift(fft_u0)
        center_x = int(self._xlen / 2)
        center_y = int(self._ylen / 2)
        logging.debug(f"{center_x=}::{center_y}")
        fft_e0_zoom = fft_e0_shift[center_x - (p - 1):center_x + p,
                                   center_y - (q - 1):center_y + q]
        fft_u0_zoom = fft_u0_shift[center_x - (p - 1):center_x + p,
                                   center_y - (q - 1):center_y + q]
        conv_e0 = convolution_matrix(fft_e0_zoom.flatten(),
                                     (2 * p + 1) * (2 * q + 1),
                                     mode="same")
        conv_u0 = convolution_matrix(fft_u0_zoom.flatten(),
                                     (2 * p + 1) * (2 * q + 1),
                                     mode="same")
        return conv_e0, conv_u0
