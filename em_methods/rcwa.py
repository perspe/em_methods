""" Main RCWA Functions - utilizing the core structure from rcwa_core """

import logging
from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt

from em_methods.grid import Grid2D, UniformGrid
from em_methods.rcwa_core.rcwa_core import SMatrix, SPMatrix, SRef, STrn
from em_methods.rcwa_core.rcwa_core import initialize_components, r_t_fields

# Get module logger
logger = logging.getLogger("simulation")

Layer = Union[UniformGrid, Grid2D]


class LayerNotProperlyDefined(Exception):
    pass


def rcwa(
    layer_stack: List[Layer],
    theta: float,
    phi: float,
    lmb: float,
    pol: Tuple[complex, complex],
    p: int,
    q: int,
    inc_med: Tuple[complex, complex],
    trn_med: Tuple[complex, complex],
) -> Tuple[float, float]:
    """Calculate the rcwa for a single wavelength"""
    # Check if all layers are in the same spot
    lims = layer_stack[0].limits
    dx, dy = layer_stack[0].grid_size
    for layer in layer_stack:
        lim_i = layer.limits
        if not np.array_equal(lims, lim_i):
            logger.error("All layers must have the same limits")
            raise LayerNotProperlyDefined("All layers must have the same limits")

    # Initialize all the necessary elements for the calculation
    components = initialize_components(
        theta, phi, lmb, pol, p, q, dx, dy, inc_med, trn_med
    )
    k0, sfree, Kx, Ky, Kz_ref, Kz_trn, kz_inc, p_vector = components
    # Update loop for the scattering matrices
    sref = SRef(Kx, Ky, Kz_ref, inc_med[0], inc_med[1], sfree)
    sglobal = sref
    for layer in layer_stack:
        logger.debug("------------------ New Layer ----------------------")
        erc, urc = layer.convolution_matrices(p, q)
        er_size = erc.shape[0] * erc.shape[1]
        if (
            np.count_nonzero(erc) / er_size < 0.5
            and np.count_nonzero(urc) / er_size < 0.5
        ):
            logger.info("Building sparce SMatrix")
            smatrix = SPMatrix(Kx, Ky, erc, urc, sfree, k0, layer.thickness)
        else:
            logger.info("Building full SMatrix")
            smatrix = SMatrix(Kx, Ky, erc, urc, sfree, k0, layer.thickness)
        sglobal = sglobal * smatrix
    strn = STrn(Kx, Ky, Kz_trn, trn_med[0], trn_med[1], sfree)
    sglobal = sglobal * strn
    # Determine the R, T
    delta: npt.NDArray = np.zeros((2 * p + 1) * (2 * q + 1))
    delta[int(delta.size / 2)] = 1
    e_src = np.r_[p_vector[0] * delta, p_vector[1] * delta]
    logger.debug(f"{e_src=}")
    r, t = r_t_fields(sglobal, sref, strn, e_src, Kx, Ky, Kz_ref, Kz_trn)
    R = np.sum(np.real(-Kz_ref / kz_inc) @ r)
    T = np.sum(np.real(inc_med[1] * Kz_trn / (trn_med[1] * kz_inc)) @ t)
    return R, T
