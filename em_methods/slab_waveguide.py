"""
Implementation of slab waveguide analysis from EMPossible
Youtube lessons
"""
import logging
import math
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.linalg import eig


def slab_waveguide1D(n1: float,
                     n2: float,
                     fiber_size: float,
                     bound_size: float,
                     lam0: float = 1.0,
                     res: int = 30,
                     export: Union[None, str] = None):
    """
    Solve the eigenvalue condition for the 1D slab waveguide system
    Args:
        n1 - Fiber refractive index
        n2 - Cladding refractive index
        fiber_size - size of the fiber in function of lam0 (incident wav)
        bound_size - size of the cladding in function of the
                     incident wavelength
        res - grid resolution (default: 30)
        lam0 - incident wavelength (default: 1)
        export - Export name to export files or None
    Return:
        neff - Effective refractive index of the medium (eigenvalues)
        eigenvec - The matrix with the eigenvectors
        y_grid - Linearly spaced grid representing the fiber
    """
    # Normalize fiber and cladding to lam0
    fiber_size *= lam0
    bound_size *= lam0
    # Determine grid parameters
    dx = lam0 / res
    Sx = fiber_size + 2 * bound_size
    Nx = math.ceil(Sx / dx)
    Sx = Nx * dx
    # Create the refractive index array
    n_fiber = round(fiber_size / dx)
    nx_fiber = round((Nx - n_fiber) / 2)
    N = np.concatenate(([n2] * nx_fiber, [n1] * n_fiber, [n2] * nx_fiber))
    if N.shape[0] != Nx:
        logging.error(Nx, N.shape)
        raise Exception("N matrix != Nx: N matrix different size from grid")

    # Perform finite difference analysis
    k0 = 2 * np.pi / lam0
    # Build DX2 and N matrix
    Ddiag = -2 * np.ones(Nx)
    Ddiag_off = np.ones(Nx)
    Dx = sparse.diags([Ddiag_off, Ddiag, Ddiag_off], [-1, 0, 1],
                      shape=(Nx, Nx),
                      format="dia")
    Dx2 = Dx / (k0 * dx)**2
    N = sparse.diags(N, 0, format="dia")
    A = Dx2 + N**2
    # Obtain the complete eigenvalue solutions and not the sparse solution
    # Should be changed for bigger matrices
    A = A.todense()
    eigval, eigvec = eig(A)
    neff = np.sqrt(eigval)
    # Sort the eigenvalues and eigenvalues properly
    neff_sort_args = np.argsort(np.real(neff))
    inv_sort_order = neff_sort_args[::-1]
    neff = neff[inv_sort_order]
    # Exclude unwanted values
    eigvec = eigvec[:, inv_sort_order]
    mask_neff = neff >= min(n1, n2)
    logging.debug(mask_neff)
    neff = neff[mask_neff]
    eigvec = np.abs(eigvec[:, mask_neff])
    logging.debug(f"Neff: {neff}")
    logging.debug(f"{eigvec.shape=}")
    y_grid = np.linspace(-bound_size - fiber_size / 2,
                         bound_size + fiber_size / 2, Nx)
    if export is not None:
        vec_array = np.concatenate((y_grid[np.newaxis, :].T, eigvec), axis=1)
        eig_array = np.concatenate(([0], neff.T))
        final_array = np.concatenate((eig_array[np.newaxis, :], vec_array),
                                     axis=0)
        np.savetxt(export, np.real(final_array))
    return neff, eigvec, y_grid


def _plot_waveguide(neff, modes, fiber_size, y):
    """
    Automation script to plot the first n modes of a slab waveguide
    Args:
        neff: array with the nreff values ordered
        modes: matrix with the eigenvectors ordered according to the neff data
        y_grid: simulated grid
        n_modes: number of modes to plot
        fiber_size: size of the fiber in lam0
    """
    n_modes = modes.shape[1]
    xmax = 30
    x_box_coord = [-1, -1, xmax, xmax]
    plt.title(f"Fiber Size: {fiber_size}")
    plt.fill(x_box_coord, [fiber_size / 2, y[-1], y[-1], fiber_size / 2],
             'lightgrey')
    plt.fill(x_box_coord, [-fiber_size / 2, y[0], y[0], -fiber_size / 2],
             'lightgrey')
    for i in range(n_modes):
        norm_modes = modes[:, i] / np.max(modes[:, i])
        plt.plot(norm_modes + 4 * i, y, color='blue')
        plt.annotate(f"neff={np.real(neff[i]):.2f}", (4 * i, 4 * y[0] / 5),
                     rotation=-90)
    plt.ylim(y[0], y[-1])
    plt.xlim(-1, xmax)
    plt.tick_params(bottom=False, labelbottom=False)
    plt.ylabel("x", fontsize=7)
    plt.yticks(fontsize=7)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.INFO)
    fiber_space = np.round(np.arange(0.1, 2, 0.2), 2)
    for f_i in fiber_space:
        eg, ev, y = slab_waveguide1D(2, 1, f_i, 4, res=40)
        plt.figure()
        _plot_waveguide(eg, ev, f_i, y)
        plt.savefig(f"fiber_{f_i}.png", dpi=150, facecolor="white")
