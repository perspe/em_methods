import logging
from logging.config import fileConfig
import os
from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy.linalg import eig, inv
from scipy.sparse import diags
import scipy.sparse as scs
import scipy.sparse.linalg as scsl
from scipy.sparse.base import spmatrix

# Get module logger
base_path = os.path.dirname(os.path.abspath(__file__))
fileConfig(os.path.join(base_path, '..', 'logging.ini'))
logger = logging.getLogger('dev_file')
""" Matrices for the Scattering Matrix Connection """

# Other useful variables
spm_format = {"dtype": np.complex64, "format": "csc"}
np.set_printoptions(precision=3, linewidth=225)


class SMBase:
    """
    Base Class of the Scattering Matrix.
    Use of slots for fast access of the basic structures in the class (S, V, W)
    Defines the standard SMAtrix __mul__ (redhaffer product)
    Defines the __repr__ for printing SMatrices
    Everything is defined as sparse matrices
    """
    __slots__ = ('_S11', '_S12', '_S21', '_S22', '_V', '_W')

    def __init__(self, m: int, n: int) -> None:
        logger.info("Initializing SMBase")
        self._S11: npt.NDArray = np.zeros((m, n), dtype=np.complex64)
        self._S12: npt.NDArray = np.eye(m, n, dtype=np.complex64)
        self._S21: npt.NDArray = np.eye(m, n, dtype=np.complex64)
        self._S22: npt.NDArray = np.zeros((m, n), dtype=np.complex64)

    def __mul__(self, SB):
        eye = scs.eye(self._S11.shape[0], self._S11.shape[1], **spm_format)
        redhaf_matrix = SMBase(self._S11.shape[0], self._S11.shape[1])
        term_1: npt.NDArray = eye.toarray() - SB._S11 @ self._S22
        term_2: npt.NDArray = eye.toarray() - self._S22 @ SB._S11
        inv_1: npt.NDArray = self._S12 @ inv(np.asfortranarray(term_1))
        inv_2: npt.NDArray = SB._S21 @ inv(np.asfortranarray(term_2))
        redhaf_matrix._S11 = self._S11 + inv_1 @ SB._S11 @ self._S21
        redhaf_matrix._S12 = inv_1 @ SB._S12
        redhaf_matrix._S21 = inv_2 @ self._S21
        redhaf_matrix._S22 = SB._S22 + inv_2 @ self._S22 @ SB._S12
        logger.debug(f"__mul__ \n{redhaf_matrix}")
        return redhaf_matrix

    def __repr__(self) -> str:
        return f"""
----------------- SMatrix ---------------------
S11 ({self._S11.shape}):
{self._S11}
S12 ({self._S12.shape}):
{self._S12}
S21 ({self._S21.shape}):
{self._S21}
S22 ({self._S22.shape}):
{self._S22}
-----------------------------------------------
"""


class SFree(SMBase):
    """ Scattering Matrix for the Free Space Region """
    def __init__(self, Kx: spmatrix, Ky: spmatrix) -> None:
        logger.info("Starting to build SFree...")
        eye = scs.eye(Kx.shape[0], Kx.shape[1], **spm_format)
        KxKy = Kx @ Ky
        Kx2 = Kx @ Kx
        Ky2 = Ky @ Ky
        kz: spmatrix = eye - Kx2 - Ky2
        Kz: spmatrix = kz.sqrt().conjugate()
        logger.debug(f"Kz({Kz.shape})\n{Kz}")
        Q: spmatrix = scs.bmat([[KxKy, eye - Kx2], [Ky2 - eye, -KxKy]],
                               **spm_format)
        self._W: spmatrix = scs.block_diag((eye, eye), **spm_format)
        eig_matrix: spmatrix = scs.block_diag((1j * Kz, 1j * Kz), **spm_format)
        logger.debug(f"Eig_Matrix({eig_matrix.shape}):\n{eig_matrix}")
        logger.info("Calculating SFree V...")
        self._V: spmatrix = Q @ (scsl.inv(eig_matrix))
        logger.debug(f"SFree V0({self._V.shape}):\n{self._V}")
        logger.info("Finished SFree...")


class SRef(SMBase):
    """ Reflection Region Scattering Matrix """
    def __init__(self, Kx: spmatrix, Ky: spmatrix, Kz_ref: spmatrix,
                 e_ref: complex, u_ref: complex, sfree: SFree) -> None:
        logger.info("Initializing SRef...")
        eye: spmatrix = scs.eye(Kx.shape[0], Kx.shape[1], **spm_format)
        KxKy = Kx @ Ky
        Kx2 = Kx @ Kx
        Ky2 = Ky @ Ky
        matrix: spmatrix = scs.bmat([[KxKy, e_ref * u_ref * eye - Kx2],
                                     [Ky2 - e_ref * u_ref * eye, -KxKy]],
                                    **spm_format)
        Q = matrix / u_ref
        logger.debug(f"\nQ:\n{Q}")
        self._W: spmatrix = scs.block_diag((eye, eye), **spm_format)
        eig_matrix: spmatrix = scs.block_diag((-1j * Kz_ref, -1j * Kz_ref),
                                              **spm_format)
        logger.info("Calculating SRef V")
        self._V: spmatrix = Q @ scsl.inv(eig_matrix)
        logger.debug(f"""
Eig_Matrix ({eig_matrix.shape}):
{eig_matrix}

Eiv Matrix ({self._W.shape}):
{self._W}

V ({self._V.shape}):
{self._V}
""")
        # Calculate Scattering Matrix
        inv_W0: spmatrix = scsl.inv(sfree._W)
        inv_V0: spmatrix = scsl.inv(sfree._V)
        A: spmatrix = inv_W0 @ self._W + inv_V0 @ self._V
        B: spmatrix = inv_W0 @ self._W - inv_V0 @ self._V
        inv_A: spmatrix = scsl.inv(A)
        logger.info("Calculating SRef SMatrix Elements")
        self._S11 = (-inv_A @ B).toarray()
        self._S12 = (2 * inv_A).toarray()
        self._S21 = (0.5 * (A - B @ inv_A @ B)).toarray()
        self._S22 = (B @ inv_A).toarray()
        logger.debug(self)
        logger.info("Finished initializing SRef...")


class STrn(SMBase):
    """ Transmission Region Scattering Matrix """
    def __init__(self, Kx: spmatrix, Ky: spmatrix, Kz_trn: spmatrix,
                 e_trn: complex, u_trn: complex, sfree: SFree) -> None:
        logger.info("Initializing STrn...")
        eye: spmatrix = scs.eye(Kx.shape[0], Kx.shape[1], **spm_format)
        KxKy = Kx @ Ky
        Kx2 = Kx @ Kx
        Ky2 = Ky @ Ky
        matrix = scs.bmat([[KxKy, e_trn * u_trn * eye - Kx2],
                           [Ky2 - e_trn * u_trn * eye, -KxKy]], **spm_format)
        Q: spmatrix = matrix / u_trn
        self._W: spmatrix = scs.block_diag((eye, eye), **spm_format)
        eig_matrix: spmatrix = scs.block_diag((1j * Kz_trn, 1j * Kz_trn),
                                              **spm_format)
        logger.info("Calculating STrn V")
        self._V: spmatrix = Q @ scsl.inv(eig_matrix)
        logger.debug(f"""
Eig_Matrix ({eig_matrix.shape}):
{eig_matrix}

Eiv Matrix ({self._W.shape}):
{self._W}

V ({self._V.shape}):
{self._V}
""")
        # Calculate Scattering Matrix
        inv_W0 = scsl.inv(sfree._W)
        inv_V0 = scsl.inv(sfree._V)
        A = inv_W0 @ self._W + inv_V0 @ self._V
        B = inv_W0 @ self._W - inv_V0 @ self._V
        inv_A = scsl.inv(A)
        logger.info("Calculating STrn SMatrix Elements")
        self._S11 = (B @ inv_A).toarray()
        self._S12 = (0.5 * (A - B @ inv_A @ B)).toarray()
        self._S21 = (2 * inv_A).toarray()
        self._S22 = (-inv_A @ B).toarray()
        logger.debug(self)
        logger.info("Finished initializing STrn...")


class SMatrix(SMBase):
    """ General Scattering Matrix for the Layered Stack """
    def __init__(self, Kx: spmatrix, Ky: spmatrix, er: npt.NDArray,
                 u0: npt.NDArray, sfree: SFree, k0: float, L: float) -> None:
        logger.info(f"Initializing SMatrix...")
        inv_er = scs.csc_matrix(inv(er, check_finite=False),
                                dtype=np.complex64)
        inv_u0 = scs.csc_matrix(inv(u0, check_finite=False),
                                dtype=np.complex64)
        logger.debug(f"""
inv_er ({inv_er.shape}):
{inv_er}
inv_u0 ({inv_u0.shape})
{inv_u0}
""")
        logger.info(f"Initializing P/Q Matrices...")
        inv_erKx = inv_er @ Kx
        inv_erKy = inv_er @ Ky
        inv_u0Kx = inv_u0 @ Kx
        inv_u0Ky = inv_u0 @ Ky
        P: spmatrix = scs.bmat([[Kx @ inv_erKy, u0 - Kx @ inv_erKx],
                                [Ky @ inv_erKy - u0, -Ky @ inv_erKx]],
                               **spm_format)
        Q: spmatrix = scs.bmat([[Kx @ inv_u0Ky, er - Kx @ inv_u0Kx],
                                [Ky @ inv_u0Ky - er, -Ky @ inv_u0Kx]],
                               **spm_format)
        # Remove big matrices from memory
        del inv_erKx
        del inv_erKy
        del inv_u0Kx
        del inv_u0Ky
        logger.debug(f"""
P({P.shape}::{P.count_nonzero()})
{P}
Q ({Q.shape}::{Q.count_nonzero()})
{Q}
""")
        self._W = P @ Q
        logger.debug(
            f"Omega2({self._W.shape}::{self._W.count_nonzero()})\n{self._W}")
        logger.info("Solve for eigenvalues")
        eigs, self._W = eig(np.asfortranarray(self._W.toarray()),
                            check_finite=False,
                            overwrite_a=True)
        # Round results to remove unwanted side effects
        self._W: npt.NDArray = np.asfortranarray(np.round(self._W, 10) + 0,
                                                 dtype=np.complex64)
        eigs = np.round(eigs, 9) + 0
        eigs = -np.sqrt(eigs)
        eig_matrix: spmatrix = scs.diags(eigs, **spm_format)
        logger.info("Determine SMatrix V")
        self._V: npt.NDArray = Q @ self._W @ scsl.inv(eig_matrix).toarray()
        self._V: npt.NDArray = np.asfortranarray(self._V, dtype=np.complex64)
        logger.debug(f"""
Eig_Matrix ({eig_matrix.shape}):
{eig_matrix}
Eiv_Matrix ({self._W.shape}):
{self._W}
V ({self._V.shape}):
{self._V}
""")
        logger.info("Determine SMatrix Pre-Elements")
        # Calculate the SMM elements
        inv_W: npt.NDArray = inv(self._W, check_finite=False)
        inv_V: npt.NDArray = inv(self._V, check_finite=False)
        A = inv_W @ sfree._W.toarray() + inv_V @ sfree._V.toarray()
        B = inv_W @ sfree._W.toarray() - inv_V @ sfree._V.toarray()
        del inv_V
        del inv_W
        logger.debug(f"""
A({A.shape})
{A}
B({B.shape})
{B}
""")
        X = scsl.expm(eig_matrix * k0 * L)
        logger.debug(f"X({X.shape})\n{X}")
        inv_A = inv(np.asfortranarray(A), check_finite=False)
        XB = X.toarray() @ B
        XA = X.toarray() @ A
        D = inv(np.asfortranarray(A - XB @ inv_A @ XB), check_finite=False)
        logger.info("Determine SMatrix Elements")
        self._S11 = D @ (XB @ inv_A @ XA - B)
        self._S12 = D @ X.toarray() @ (A - B @ inv_A @ B)
        self._S21 = self._S12
        self._S22 = self._S11
        logger.debug(self)
        logger.info(f"Finished Initializing SMatrix...")


""" Intermediary functions for calculations """


def r_t_fields(s_global: SMBase, s_ref: SMBase, s_trn: SMBase,
               e_src: npt.NDArray, Kx: spmatrix, Ky: spmatrix,
               Kz_ref: spmatrix, Kz_trn: spmatrix) -> Tuple[complex, complex]:
    """ Calculate the Reflected and Transmited fields """
    # Calculate reflected and transmited fields
    c_src = scsl.inv(s_ref._W).toarray() @ e_src
    logger.debug(f"c_src: {c_src}")
    e_ref = s_ref._W.toarray() @ s_global._S11 @ c_src
    e_trn = s_trn._W.toarray() @ s_global._S21 @ c_src
    e_ref = e_ref.flatten()
    logger.debug(f"e_ref{e_ref.shape}: {e_ref}")
    logger.debug(f"e_trn{e_trn.shape}: {e_trn}")
    # Compute the longitudinal components
    e_ref_x, e_ref_y = e_ref[:int(e_ref.size / 2)], e_ref[int(e_ref.size / 2):]
    e_trn_x, e_trn_y = e_trn[:int(e_trn.size / 2)], e_trn[int(e_trn.size / 2):]
    rz = -scsl.inv(Kz_ref) @ (Kx.toarray() @ e_ref_x + Ky.toarray() @ e_ref_y)
    tz = -scsl.inv(Kz_trn) @ (Kx.toarray() @ e_trn_x + Ky.toarray() @ e_trn_y)
    logger.debug(
        f"E_ref:[{e_ref_x} {e_ref_y} {rz}]\n E_Trn:[{e_trn_x} {e_trn_y} {tz}]")
    r2 = np.abs(e_ref_x)**2 + np.abs(e_ref_y)**2 + np.abs(rz)**2
    t2 = np.abs(e_trn_x)**2 + np.abs(e_trn_y)**2 + np.abs(tz)**2
    logger.debug(f"{r2=}\n{t2=}")
    return r2, t2


def initialize_components(theta: float, phi: float, lmb: float,
                          pol: Tuple[complex, complex], p: int, q: int,
                          dx: float, dy: float, inc_med: Tuple[complex,
                                                               complex],
                          trn_med: Tuple[complex, complex]):
    """ Initialize all the components necessary for the RCWA Algorithm """
    logger.info("Initialization values for SMatrix")
    k0: float = 2 * np.pi / lmb
    logger.debug(f"{k0=}")
    e_ref, u_ref = inc_med
    e_trn, u_trn = trn_med
    theta = np.radians(theta)
    phi = np.radians(phi)
    # Determine the incidence k vector
    kx_inc: float = np.sqrt(e_ref * u_ref) * np.sin(theta) * np.cos(phi)
    ky_inc: float = np.sqrt(e_ref * u_ref) * np.sin(theta) * np.sin(phi)
    kz_inc: float = np.sqrt(u_ref * e_ref) * np.cos(theta)
    logger.debug(f"k_inc: [{kx_inc} {ky_inc} {kz_inc}]")
    # Determine the wavevector matrices from the number of harmonics
    kx_p = kx_inc - (2 * np.pi * np.arange(-p, p + 1)) / (dx * k0)
    ky_q = ky_inc - (2 * np.pi * np.arange(-q, q + 1)) / (dy * k0)
    kx_mesh, ky_mesh = np.meshgrid(kx_p, ky_q)
    kz_ref_int = np.array(u_ref.conjugate() * e_ref.conjugate() - kx_mesh**2 -
                          ky_mesh**2,
                          dtype=np.complex64)
    kz_trn_int = np.array(u_trn.conjugate() * e_trn.conjugate() - kx_mesh**2 -
                          ky_mesh**2,
                          dtype=np.complex64,
                          order="F")
    kz_ref = -np.conjugate(np.sqrt(kz_ref_int))
    kz_trn = np.conjugate(np.sqrt(kz_trn_int))
    Kx: spmatrix = diags(kx_mesh.flatten(), **spm_format)
    Ky: spmatrix = diags(ky_mesh.flatten(), **spm_format)
    Kz_ref: spmatrix = diags(kz_ref.flatten(), **spm_format)
    Kz_trn: spmatrix = diags(kz_trn.flatten(), **spm_format)
    logger.debug(f"K_matrices:\n{kx_p=}\n{ky_q=}")
    logger.info(
        f"K_matrices: {Kx.shape}::{Ky.shape}::{Kz_ref.shape}::{Kz_trn.shape}")
    logger.debug(f"K_matrices:\nKx\n{Kx}\nKy\n{Ky}\nKz_ref\n{Kz_ref}\nKz_trn\n{Kz_trn}")
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
    logger.debug(f"{p_vector=}")
    return k0, sfree, Kx, Ky, Kz_ref, Kz_trn, kz_inc, p_vector
