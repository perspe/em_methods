import logging
import numpy as np
import numpy.typing as npt
from scipy.linalg import inv, eig
from typing import Tuple
from em_methods.grid import Grid2D
import os

from logging.config import fileConfig

# Get module logger
base_path = os.path.dirname(os.path.abspath(__file__))
fileConfig(os.path.join(base_path, '..', 'logging.ini'))
logger = logging.getLogger('dev')
""" Matrices for the Scattering Matrix Connection """


class SMBase:
    """
    Base Class of the Scattering Matrix.
    Use of slots for fast access of the basic structures in the class (S, V, W)
    Defines the standard SMAtrix __mul__ (redhaffer product)
    Defines the __repr__ for printing SMatrices
    """
    __slots__ = ('_S11', '_S12', '_S21', '_S22', '_V', '_W')

    def __init__(self, m: int, n: int) -> None:
        self._S11 = np.zeros((m, n))
        self._S12 = np.eye(m, n)
        self._S21 = np.eye(m, n)
        self._S22 = np.zeros((m, n))

    def __mul__(self, SB):
        eye = np.eye(self._S11.shape[0], self._S11.shape[1])
        redhaf_matrix = SMBase(self._S11.shape[0], self._S11.shape[1])
        inv_1 = self._S12 @ inv(eye - SB._S11 @ self._S22, check_finite=False)
        inv_2 = SB._S21 @ inv(eye - self._S22 @ SB._S11, check_finite=False)
        redhaf_matrix._S11 = self._S11 + inv_1 @ SB._S11 @ self._S21
        redhaf_matrix._S12 = inv_1 @ SB._S12
        redhaf_matrix._S21 = inv_2 @ self._S21
        redhaf_matrix._S22 = SB._S22 + inv_2 @ self._S22 @ SB._S12
        logger.debug(f"__mul__ \n{redhaf_matrix}")
        return redhaf_matrix

    def __repr__(self) -> str:
        return f"--------\n{str(self._S11)}\n{str(self._S12)}\n{str(self._S21)}\n{str(self._S22)}\n------"


class SFree(SMBase):
    """ Scattering Matrix for the Free Space Region """
    def __init__(self, Kx: npt.NDArray, Ky: npt.NDArray) -> None:
        eye = np.eye(Kx.shape[0], Kx.shape[1])
        P = np.block([[Kx @ Ky, eye - Kx @ Kx], [Ky @ Ky - eye, -Kx @ Ky]])
        eigval, self._W = eig(P @ P, check_finite=False)
        eig_matrix = np.diag(1 / np.sqrt(eigval))
        self._V = P @ self._W @ eig_matrix
        logger.debug(f"SFree V0:\n{self._V}")


class SRef(SMBase):
    """ Reflection Region Scattering Matrix """
    def __init__(self, Kx: npt.NDArray, Ky: npt.NDArray, e_ref: complex,
                 u_ref: complex, sfree: SFree) -> None:
        eye = np.eye(Kx.shape[0], Kx.shape[1])
        matrix = np.block([[Kx @ Ky, e_ref * u_ref * eye - Kx @ Kx],
                           [Ky @ Ky - e_ref * u_ref * eye, -Kx @ Ky]])
        P, Q = matrix / e_ref, matrix / u_ref
        self._W = P @ Q
        eigs, self._W = eig(self._W, check_finite=False)
        eig_matrix = np.diag(1 / np.sqrt(eigs))
        self._V = Q @ self._W @ eig_matrix
        logger.debug(f"\nEig_Matrix:\n{eigs}\n Eiv Matrix:\n{self._W}")
        logger.debug(f"\nVref:\n{self._V}")
        # Calculate Scattering Matrix
        inv_W0 = inv(sfree._W, check_finite=False)
        inv_V0 = inv(sfree._V, check_finite=False)
        A = inv_W0 @ self._W + inv_V0 @ self._V
        B = inv_W0 @ self._W - inv_V0 @ self._V
        inv_A = inv(A, check_finite=False)
        self._S11 = -inv_A @ B
        self._S12 = 2 * inv_A
        self._S21 = 0.5 * (A - B @ inv_A @ B)
        self._S22 = B @ inv_A
        logger.debug("SMatrix")
        logger.debug(self)


class STrn(SMBase):
    """ Transmission Region Scattering Matrix """
    def __init__(self, Kx: npt.NDArray, Ky: npt.NDArray, e_trn: complex,
                 u_trn: complex, sfree: SFree) -> None:
        eye = np.eye(Kx.shape[0], Kx.shape[1])
        matrix = np.block([[Kx @ Ky, e_trn * u_trn * eye - Kx @ Kx],
                           [Ky @ Ky - e_trn * u_trn * eye, -Kx @ Ky]])
        P, Q = matrix / e_trn, matrix / u_trn
        self._W = P @ Q
        eigs, self._W = eig(self._W, check_finite=False)
        eig_matrix = np.diag(1 / np.sqrt(eigs))
        self._V = Q @ self._W @ eig_matrix
        logger.debug(f"\nEig_Matrix:\n{eigs}\n Eiv Matrix:\n{self._W}")
        logger.debug(f"\nVref:\n{self._V}")
        # Calculate Scattering Matrix
        inv_W0 = inv(sfree._W, check_finite=False)
        inv_V0 = inv(sfree._V, check_finite=False)
        A = inv_W0 @ self._W + inv_V0 @ self._V
        B = inv_W0 @ self._W - inv_V0 @ self._V
        inv_A = inv(A, check_finite=False)
        self._S11 = B @ inv_A
        self._S12 = 0.5 * (A - B @ inv_A @ B)
        self._S21 = 2 * inv_A
        self._S22 = -inv_A @ B
        logger.debug("SMatrix")
        logger.debug(self)


class SMatrix(SMBase):
    """ General Scattering Matrix for the Layered Stack """
    def __init__(self, Kx: npt.NDArray, Ky: npt.NDArray, er: npt.NDArray,
                 u0: npt.NDArray, sfree: SFree, k0: float, L: float) -> None:
        inv_er = inv(er, check_finite=False)
        inv_u0 = inv(u0, check_finite=False)
        logger.info(f"{Kx.shape}::{Ky.shape}::{inv_er.shape}::{inv_u0.shape}")
        P = np.block([[Kx @ inv_er @ Ky, u0 - Kx @ inv_er @ Kx],
                      [Ky @ inv_er @ Ky - u0, -Ky @ inv_er @ Kx]])
        Q = np.block([[Kx @ inv_u0 @ Ky, er - Kx @ inv_u0 @ Kx],
                      [Ky @ inv_u0 @ Ky - er, -Ky @ inv_u0 @ Kx]])
        self._W = np.array(P @ Q, dtype=np.complex128)
        eigs, self._W = eig(self._W, check_finite=False)
        logger.debug(f"self._W =\n{self._W}")
        inv_W = inv(self._W, check_finite=False)
        eig_matrix = np.diag(1 / np.sqrt(eigs))
        self._V = Q @ self._W @ (eig_matrix)
        inv_V = inv(self._V, check_finite=False)
        logger.debug(f"Eig_Matrix:\n{eig_matrix}\nEiv_Matrix:\n{self._W}")
        # Calculate the SMM elements
        A = inv_W @ sfree._W + inv_V @ sfree._V
        B = inv_W @ sfree._W - inv_V @ sfree._V
        logger.debug(f"A\n{A}\nB\n{B}")
        X = np.diag(np.exp(-np.sqrt(eigs) * k0 * L))
        logger.debug(f"X\n{X}")
        inv_A = inv(A)
        XB = X @ B
        XA = X @ A
        self._S11 = inv(A - XB @ inv_A @ XB,
                        check_finite=False) @ (XB @ inv_A @ XA - B)
        self._S12 = inv(A - XB @ inv_A @ XB,
                        check_finite=False) @ X @ (A - B @ inv_A @ B)
        self._S21 = self._S12
        self._S22 = self._S11
        logger.debug("SMatrix")
        logger.debug(self)


""" Intermediary functions for calculations """


def r_t_fields(s_global: SMBase, s_ref: SMBase, s_trn: SMBase,
               e_src: npt.NDArray, Kx: npt.NDArray, Ky: npt.NDArray,
               Kz_ref: npt.NDArray,
               Kz_trn: npt.NDArray) -> Tuple[complex, complex]:
    """ Calculate the Reflected and Transmited fields """
    # Calculate reflected and transmited fields
    c_src = inv(s_ref._W) @ e_src.T
    e_ref = s_ref._W @ s_global._S11 @ c_src
    e_trn = s_trn._W @ s_global._S21 @ c_src
    logger.debug(f"e_ref: {e_ref}")
    logger.debug(f"e_trn: {e_trn}")
    # Compute the longitudinal components
    e_ref_x, e_ref_y = e_ref[:int(e_ref.size / 2)], e_ref[int(e_ref.size / 2):]
    e_trn_x, e_trn_y = e_trn[:int(e_trn.size / 2)], e_trn[int(e_trn.size / 2):]
    rz = -inv(Kz_ref) @ (Kx @ e_ref_x + Ky @ e_ref_y)
    tz = -inv(Kz_trn) @ (Kx @ e_trn_x + Ky @ e_trn_y)
    logger.debug(f"[{e_ref_x} {e_ref_y} {rz}]::[{e_trn_x} {e_trn_y} {tz}]")
    r2 = np.sum(np.abs([e_ref_x, e_ref_y, rz])**2)
    t2 = np.sum(np.abs([e_trn_x, e_trn_y, tz])**2)
    return r2, t2


def initialize_components(theta: float, phi: float, lmb: float,
                          pol: Tuple[complex, complex], p: int, q: int,
                          dx: float, dy: float, inc_med: Tuple[complex,
                                                               complex],
                          trn_med: Tuple[complex, complex]):
    """ Initialize all the components necessary for the RCWA Algorithm """
    logger.debug("Initialization values for SMatrix")
    k0: float = 2 * np.pi / lmb
    logger.debug(f"{k0=}")
    e_ref, u_ref = inc_med
    e_trn, u_trn = trn_med
    # Determine the incidence k vector
    kx_inc: float = np.sqrt(e_ref * u_ref) * np.sin(theta) * np.cos(phi) / k0
    ky_inc: float = np.sqrt(e_ref * u_ref) * np.sin(theta) * np.sin(phi) / k0
    kz_inc: float = np.sqrt(u_ref * e_ref - k0 * (kx_inc**2 + ky_inc**2))
    logger.debug(f"k_inc: [{kx_inc} {ky_inc} {kz_inc}]")
    # Determine the wavevector matrices from the number of harmonics
    kx_p = kx_inc - np.arange(-p, p + 1) * 2 * np.pi / (dx * k0)
    ky_q = ky_inc - np.arange(-q, q + 1) * 2 * np.pi / (dy * k0)
    kx_mesh, ky_mesh = np.meshgrid(kx_p, ky_q)
    kz_ref_int = np.array(u_ref.conjugate() * e_ref.conjugate() - kx_mesh**2 -
                          ky_mesh**2,
                          dtype=np.complex64)
    kz_trn_int = np.array(u_trn.conjugate() * e_trn.conjugate() - kx_mesh**2 -
                          ky_mesh**2,
                          dtype=np.complex64)
    kz_ref = -np.conjugate(np.sqrt(kz_ref_int))
    kz_trn = np.conjugate(np.sqrt(kz_trn_int))
    Kx = np.diag(kx_mesh.flatten())
    Ky = np.diag(ky_mesh.flatten())
    Kz_ref = np.diag(kz_ref.flatten())
    Kz_trn = np.diag(kz_trn.flatten())
    logger.debug(f"K_matrices:\n{kx_p=}\n{ky_q=}")
    logger.debug(f"K_matrices:\n{Kx=}\n{Ky=}\n{Kz_ref=}\n{Kz_trn=}")
    # Determine the Free Space Scattering Matrix
    sfree = SFree(Kx, Ky)
    # Reduced polarization vector
    if theta == 0:
        ate = np.array([0, 1, 0])
        atm = np.array([1, 0, 0])
    else:
        ate = np.array([-np.sin(phi), np.cos(phi), 0])
        atm = np.array([
            np.cos(phi) * np.cos(theta),
            np.cos(theta) * np.sin(phi), -np.sin(theta)
        ])
    # Create the composite polariztion vector
    p_vector = np.add(pol[1] * ate, pol[0] * atm)
    p_vector: npt.NDArray = p_vector[[0, 1]]
    return k0, sfree, Kx, Ky, Kz_ref, Kz_trn, kz_inc, p_vector
