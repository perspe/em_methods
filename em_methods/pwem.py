""" Module with the information for the PWEM method
This file contains
PWEMGrid: Create the grid to perform the PWEM method
PWEMSolve: Run the PWEM method
Block: Create the list of block wavevectors
"""
import logging
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.linalg import eig, inv

from grid import Grid2D, GridHasNoObjectError

# Get module logger
logger = logging.getLogger()
""" Class to obtain the Block Vectors for the structure """


class Block():
    """ Main class that defines all the block vectors for a specific structure """
    def __init__(self,
                 size: float,
                 n_points: int,
                 sym_fraction: str = "1/8") -> None:
        # Initial variables
        zeros = np.zeros(n_points)
        diag_points: int = int(np.sqrt(2) * n_points)
        change_xy = np.linspace(0, np.pi / size, n_points)
        # inv_change_xy = change_xy.copy()[:-1]
        self.label_pos: List[int] = []
        self.label_name: List[str] = []
        if sym_fraction == "1/8":
            self.Kx = np.r_[change_xy, zeros + np.pi / size,
                            np.linspace(np.pi / size, 0, diag_points)]
            self.Ky = np.r_[zeros, change_xy,
                            np.linspace(np.pi / size, 0, diag_points)]
            self.label_pos = [
                0, n_points, 2 * n_points, 2 * n_points + diag_points
            ]
            self.label_name = ["G", "X", "M", "G"]
        else:  # Assume "full"
            kx = np.linspace(-size, size, 2 * n_points)
            ky = np.linspace(-size, size, 2 * n_points)
            _Kx, _Ky = np.meshgrid(kx, ky)
            self.Kx: npt.NDArray[np.floating] = _Kx.flatten()
            self.Ky: npt.NDArray[np.floating] = _Ky.flatten()
        logging.debug(f"{self.Kx=}\n{self.Ky=}")

    @property
    def kx(self) -> npt.NDArray[np.floating]:
        return self.Kx

    @property
    def ky(self) -> npt.NDArray[np.floating]:
        return self.Ky

    @property
    def label(self) -> Tuple[List[str], List[int]]:
        return self.label_name, self.label_pos


""" Main PWEM Solver """

def PWEMSolve2D(grid: Grid2D, block: Block, p: int, q: int, n_eig: int = 5):
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
    if n_eig > (2 * p - 1) * (2 * q - 1):
        raise ValueError("n_eig should be smaller than (2*p-1)(2*q-1)")
    # Start calculations
    conv_e0, conv_u0 = grid.convolution_matrices(p, q)
    i_conv_u0 = inv(conv_u0)
    kx = 2 * np.pi * np.arange(-(p + 1), p)
    ky = 2 * np.pi * np.arange(-(q + 1), q)
    logging.debug(f"Checking shapes: {kx.shape=}\t{conv_e0.shape=}")
    results = np.zeros((n_eig, block.Kx.shape[0]))
    logging.debug(f"{results.shape=}")
    for index, (block_x, block_y) in enumerate(zip(block.Kx, block.Ky)):
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
        # Normalized eigs using the grid size
        norm_eigs = (grid.grid_size[0] / (2 * np.pi)) * np.sqrt(eigs)
        # Add results to the final structure
        for i in range(n_eig):
            results[i, index] = norm_eigs[i]
    return results


""" Make Tests to run """


def grid_constructor():
    """ Test several elements of the grid """
    # Test adding objects to the grid
    grid = Grid2D(525, 525, 10)  # Base grid with a=1
    logging.debug(f"Before adding objects: {grid.has_object()=}")
    grid.add_square(2.5, 2, 0.5)
    logging.debug(f"After adding objects: {grid.has_object()=}")
    grid.add_rectangle(3, 2.5, (0.15, 0.1))
    grid.add_rectangle(3.5, 3, (0.15, 0.1), rotation=45)
    grid.add_circle(4, 3.5, 0.15, 0.25)
    grid.add_ellipse(4, 3.5, (0.15, 0.1), center=-0.25)
    grid.add_ellipse(4, 3.5, (0.15, 0.1), center=(-0.25, 0.25))
    plt.imshow(np.real(grid.er))
    plt.show()


def pwem_square():
    """ Calculate a simple Band Diagram """
    grid = Grid2D(525, 525, 10)  # Base grid with a=1
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
    eigs = PWEMSolve2D(grid, beta, 5, 5, 10)
    for eig_i in eigs:
        plt.plot(eig_i)
    plt.show()


def pwem_full():
    grid = Grid2D(525, 525)  # Base grid with a=1
    grid.add_cirle(9.5, 1, 0.25)
    block = Block(grid.grid_size[0], 25)
    eigs = PWEMSolve2D(grid, block, 8, 8, 5)
    block_label, block_place = block.label
    plt.xticks(block_place, block_label)
    for eig_i in eigs:
        plt.plot(eig_i, 'b--')
    plt.show()

def pwem_particle():
    grid = Grid2D(1050, 1050, 100)
    grid.add_circle(10, 10, 0.1)
    e0, u0 = grid.convolution_matrices(2, 2)
    plt.imshow(np.real(e0))
    plt.show()
    grid.reinit()
    grid.add_circle(10, 10, 0.25)
    e0, u0 = grid.convolution_matrices(2, 2)
    plt.imshow(np.real(e0))
    plt.show()


def pwem_compare():
    grid = Grid2D(525, 525, 10)
    block = Block(grid.grid_size[0], 25)
    # _, ax = plt.subplots()
    # ax.set_xticks(block.label[1], block.label[0])
    print(block.label[0], block.label[1])

    for size_i in np.linspace(0.2, 0.7, 5):
        grid.add_circle(3, 1, size_i)
        eigs = PWEMSolve2D(grid, block, 10, 10, 3)
        np.savetxt(f"Eig_{size_i}.txt", eigs.T)
        # for index, eig_i in enumerate(eigs):
            # ax.plot(eig_i, 'b')
        grid.reinit()
    # plt.show()


def conv_compare():
    grid = Grid2D(1025, 1025, 10)
    grid.add_square(3 + 0.5j, 1, 0.15, center=(-0.15, 0))
    for _ in range(10):
        grid.convolution_matrices(10, 10)


def main():
    """Run tests
    """
    # pwem_compare()
    pwem_particle()


if __name__ == "__main__":
    main()
