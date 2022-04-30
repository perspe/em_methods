from typing import List, Tuple, Union
import numpy as np
import numpy.typing as npt
from scipy.fft import fft2, fftshift
from scipy.linalg import convolution_matrix, eig, inv
import matplotlib.pyplot as plt
import logging
from logging.config import fileConfig

fileConfig('logging.ini')
logger = logging.getLogger('dev')
""" Main Grid Classes """


class PWEMGrid():
    """Main class to create the PWEM grid
    This class is initialized with the x/y lengths and x/y max and min values
    Methods:
        Add primitives to the grid:
            - square(xcenter, ycenter, size, rotation=0)
            - rectange(xcenter, ycenter, xsize, ysize, rotation=0)
            - circle(xcenter, ycenter, radius)
            - ellipse(xcenter, ycenter, xradius, yradius, rotation=0)
            - triangle()
        show_grid() (Plot the current grid)
        convolution() (Return convolution matrix for e and u)
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
        self._xrange = np.linspace(xlims[0], xlims[1], xlen)
        self._yrange = np.linspace(ylims[1], ylims[0], ylen)
        self._XX, self._YY = np.meshgrid(self._xrange, self._yrange)
        self._e = np.ones_like(self._XX, dtype=np.complex64)
        self._u = np.ones_like(self._XX, dtype=np.complex64)
        self._has_object = False

    """ Define the accessible properties """

    @property
    def XYgrid(self):
        return self._XX, self._YY

    @property
    def er(self):
        return self._e

    @property
    def u0(self):
        return self._u

    @property
    def grid_size(self):
        xsize: float = np.max(self._xrange) - np.min(self._xrange)
        ysize: float = np.max(self._yrange) - np.min(self._yrange)
        return xsize, ysize


    """ Help functions """

    def has_object(self):
        """ Check if any object has been added to the grid """
        return self._has_object

    """ Functions to add the primitives """

    def add_rectangle(self,
                      er: complex,
                      u0: complex,
                      size: Tuple[float, float],
                      *,
                      center: Union[float, Tuple[float, float]] = 0,
                      rotation: float = 0):
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
                   rotation: float = 0):
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
                    rotation: float = 0):
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
                  center: Union[float, Tuple[float, float]] = 0):
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
        # TODO
        # Sort the values according to their x value
        sorted_list = [a, b, c]
        sorted_list.sort(key=lambda x: x[0])
        a, b, c = tuple(sorted_list)
        logging.debug(f"Sorted Values:{a=}::{b=}::{c=}")
        ax, ay = a
        bx, by = b
        cx, cy = c
        # Build the linear regressions that connect each point combination
        m_ab = (by - ay) / (bx - ax)
        m_ac = (cy - ay) / (cx - ax)
        m_bc = (cy - by) / (cx - bx)
        logging.debug(f"{m_ab=}::{m_ac=}::{m_bc=}")
        b_ab = ay - m_ab * ax
        b_ac = ay - m_ac * ax
        b_bc = by - m_bc * bx

        sign_m_ab = np.sign(m_ab * self._XX - self._YY)
        sign_m_ac = np.sign(m_ac * self._XX - self._YY)
        sign_m_bc = np.sign(m_bc * self._XX - self._YY)
        eq1 = (sign_m_ab + 1) * (sign_m_ac - 1) * (m_ab * self._XX + b_ab -
                                                   self._YY)
        eq2 = (sign_m_ac + 1) * (sign_m_bc + 1) * (m_ac * self._XX + b_ac -
                                                   self._YY)
        eq3 = (sign_m_bc - 1) * (sign_m_ab - 1) * (m_bc * self._XX + b_bc -
                                                   self._YY)
        plt.imshow(eq1)
        plt.show()
        plt.imshow(eq2)
        plt.show()
        plt.imshow(eq3)
        plt.show()
        plt.imshow(eq1 + eq2 + eq3)
        plt.show()
        mask = (eq1 + eq2 + eq3 < 0)
        logging.debug(f"{mask=}")
        self._u[mask] = u0
        self._e[mask] = er

    def convolution_matrices(self, p: int, q: int):
        """
        Return the convolution matrices for e and u
        Args:
            p/q (int): Number of harmonics (x and y, respectively)
        """
        # Obtain the p*q area of the 2D fft
        fft_e0 = fft2(self._e, norm="forward")
        fft_u0 = fft2(self._u, norm="forward")
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


""" Main PWEM Solver """


class GridHasNoObjectError(Exception):
    pass


def PWEMSolve(grid: PWEMGrid,
              block_vector: npt.NDArray[np.float64],
              p: int,
              q: int,
              n_eig: int = 5):
    """
    Main function to solve the PWEM problem defined in the provided Grid
    Args:
        grid: Grid with the structure constructed
        block_vector: Block vector with x and y components
        p/q: Number of harmonics in x and y
        n_eig: Number of eigenvalues to determine
    Returns:
    """
    # Check if all variables are valid
    if not grid.has_object():
        raise GridHasNoObjectError
    if not isinstance(p, int) and not isinstance(q, int):
        raise ValueError("p and q should be integer numbers")
    if block_vector.shape[0] != 2:
        raise IndexError("block vector should have exactly 2 lines (x and y)")
    if n_eig > (2*p-1) * (2*q-1):
        raise ValueError("n_eig should be smaller than (2*p-1)(2*q-1)")
    # Start calculations
    conv_e0, conv_u0 = grid.convolution_matrices(p, q)
    i_conv_u0 = inv(conv_u0)
    kx = 2 * np.pi * np.arange(-(p + 1), p)
    ky = 2 * np.pi * np.arange(-(q + 1), q)
    logging.debug(f"Checking shapes: {kx.shape=}\t{conv_e0.shape=}")
    results = np.zeros((n_eig, block_vector.shape[1]))
    logging.debug(f"{results.shape=}")
    for index, (block_x, block_y) in enumerate(zip(block_vector[0, :], block_vector[1, :])):
        # Build the Kx and Ky matrices
        kx_b = block_x - kx
        ky_b = block_y - ky
        Kx_mesh, Ky_mesh = np.meshgrid(kx_b, ky_b)
        Kx = np.diag(Kx_mesh.flatten())
        Ky = np.diag(Ky_mesh.flatten())
        # Solve the eigenvalue problem
        A = Kx @ i_conv_u0 @ Kx + Ky @ i_conv_u0 @ Ky
        eigs, *_ = eig(A, conv_e0)
        # Handle the eigenvalues ordering
        eigs = np.sort(np.real(eigs))
        logging.debug(f"Five lowest eigs: {eigs[:5]=}")
        # TODO: Consider the grid size... <29-04-22, yourname> #
        norm_eigs = (1 / (2 * np.pi)) * np.sqrt(eigs)
        # Add results to the final structure
        for i in range(n_eig):
            results[i, index] = norm_eigs[i]
    return results


""" Make Tests to run """


def grid_constructor():
    """ Test several elements of the grid """
    # Test adding objects to the grid
    grid = PWEMGrid(525, 525)  # Base grid with a=1
    logging.debug(f"Before adding objects: {grid.has_object()=}")
    grid.add_square(2.5, 2, 0.5)
    logging.debug(f"After adding objects: {grid.has_object()=}")
    grid.add_rectangle(3, 2.5, (0.15, 0.1))
    grid.add_rectangle(3.5, 3, (0.15, 0.1), rotation=45)
    grid.add_cirle(4, 3.5, 0.15, 0.25)
    grid.add_ellipse(4, 3.5, (0.15, 0.1), center=-0.25)
    grid.add_ellipse(4, 3.5, (0.15, 0.1), center=(-0.25, 0.25))
    plt.imshow(np.real(grid.er))
    plt.show()


def pwem_square():
    """ Calculate a simple Band Diagram """
    grid = PWEMGrid(525, 525)  # Base grid with a=1
    grid.add_square(9, 2, 0.5)
    plt.imshow(np.real(grid.er))
    plt.show()
    num_points = 30
    beta_x = np.r_[np.linspace(0, np.pi, num_points),
                   np.ones((num_points)) * np.pi,
                   np.linspace(np.pi, 0, num_points)]
    beta_y = np.r_[np.zeros(num_points),
                   np.linspace(0, np.pi, num_points),
                   np.linspace(np.pi, 0, num_points)]
    logging.debug(f"{beta_x.shape=}::{beta_y.shape=}")
    logging.debug(f"{beta_x=}\n{beta_y=}")
    beta = np.stack([beta_x, beta_y])
    logging.debug(f"{beta=}")
    logging.debug(f"{beta.shape=}")
    eigs = PWEMSolve(grid, beta, 5, 5, 10)
    for eig_i in eigs:
        plt.plot(eig_i)
    plt.show()

def pwem_circle():
    grid = PWEMGrid(525, 525)  # Base grid with a=1
    grid.add_cirle(9.5, 1, 0.25)
    plt.imshow(np.real(grid.er))
    plt.show()
    num_points = 30
    beta_x = np.r_[np.linspace(0, np.pi, num_points),
                   np.ones((num_points)) * np.pi,
                   np.linspace(np.pi, 0, num_points)]
    beta_y = np.r_[np.zeros(num_points),
                   np.linspace(0, np.pi, num_points),
                   np.linspace(np.pi, 0, num_points)]
    beta = np.stack([beta_x, beta_y])
    eigs = PWEMSolve(grid, beta, 6, 6, 10)
    for eig_i in eigs:
        plt.plot(eig_i)
    plt.show()

def pwem_compare():
    grid1 = PWEMGrid(525, 525)  # Base grid with a=1
    grid1.add_ellipse(9.5, 1, (0.1, 0.2))
    grid2 = PWEMGrid(525, 525)  # Base grid with a=1
    grid2.add_cirle(9.5, 1, 0.25)
    num_points = 30
    beta_x = np.r_[np.linspace(0, np.pi, num_points),
                   np.ones((num_points)) * np.pi,
                   np.linspace(np.pi, 0, num_points)]
    beta_y = np.r_[np.zeros(num_points),
                   np.linspace(0, np.pi, num_points),
                   np.linspace(np.pi, 0, num_points)]
    beta = np.stack([beta_x, beta_y])
    eigs1 = PWEMSolve(grid1, beta, 6, 6, 2)
    eigs2 = PWEMSolve(grid2, beta, 6, 6, 2)
    for eig1_i, eig2_i in zip(eigs1, eigs2):
        plt.plot(eig1_i, 'b--')
        plt.plot(eig2_i, 'r--')
    plt.show()

def main():
    """Run tests
    """
    # grid_constructor()
    # pwem_square()
    # pwem_circle()
    pwem_compare()


if __name__ == "__main__":
    main()
