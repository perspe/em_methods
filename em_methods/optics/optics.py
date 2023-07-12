"""
This file contains severall functions used for optics purposes
"""

import scipy.optimize as sco
import numpy as np

def _bruggerman(n_eff, n_qd: complex, n_pvk: complex, p_qd: float):
    """ Bruggerman equation:
    n_eff is similar to x in a function
    The point where the returned y=0 is the n_eff value
    """
    y = p_qd * (n_qd - n_eff) / (n_qd + 2 * n_eff) + (1 - p_qd) * (
        n_pvk - n_eff) / (n_pvk + 2 * n_eff)
    return y


def bruggerman_dispersion(n1, n2, p):
    """
    Calculate the effective medium dispersion between two materials 1 and 2
    Args:
        n1, n2 (complex arrays): refractive indices for each material
        p (float): fraction
    Returns:
        n_eff (complex array): effective medium
    """
    n_eff = np.ones_like(n1, dtype=np.complex128)
    for (index, n1_i), n2_i in zip(enumerate(n1), n2):
        n_eff[index] = sco.newton(_bruggerman, (n1_i + n2_i) / 2,
                                  args=(n1_i, n2_i, p))
    return n_eff
