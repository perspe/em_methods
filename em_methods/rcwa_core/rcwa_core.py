import logging
import numpy as np
import numpy.typing as npt
from scipy.linalg import inv, eig

from logging.config import fileConfig

fileConfig('../logging.ini')
logger = logging.getLogger('dev')


class SMBase:
    """
    Base Class of the Scattering Matrix.
    Use of slots for fast access of the basic structures in the class (S, V, W)
    Defines the standard SMAtrix __mul__ (redhaffer product)
    Defines the __repr__ for printing SMatrices
    """
    __slots__ = ('_S11', '_S12', '_S21', '_S22', '_V', '_W')

    def __init__(self, m: int, n: int) -> None:
        self._S11 = np.zeros((m, n), order="F")
        self._S12 = np.eye(m, n, order="F")
        self._S21 = np.eye(m, n, order="F")
        self._S22 = np.zeros((m, n), order="F")

    def __mul__(self, SB):
        eye = np.eye(self._S11.shape[0], self._S11.shape[1], order="F")
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
        eye = np.eye(Kx.shape[0], Kx.shape[1], order="F")
        P = np.block([[Kx @ Ky, eye - Kx @ Kx], [Ky @ Ky - eye, -Kx @ Ky]])
        eigval, self._W = eig(P @ P, check_finite=False)
        eig_matrix = np.diag(1 / np.sqrt(eigval))
        self._V = P @ self._W @ eig_matrix
        logger.debug(f"SFree V0:\n{self._V}")


class SRef(SMBase):
    """ Reflection Region Scattering Matrix """
    def __init__(self, Kx: npt.NDArray, Ky: npt.NDArray, e_ref: complex,
                 u_ref: complex, sfree: SFree) -> None:
        eye = np.eye(Kx.shape[0], Kx.shape[1], order="F")
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
        eye = np.eye(Kx.shape[0], Kx.shape[1], order="F")
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
        P = np.block([[Kx @ inv_er @ Ky, u0 - Kx @ inv_er @ Kx],
                      [Ky @ inv_er @ Ky - u0, -Ky @ inv_er @ Kx]])
        Q = np.block([[Kx @ inv_u0 @ Ky, er - Kx @ inv_u0 @ Kx],
                      [Ky @ inv_u0 @ Ky - er, -Ky @ inv_u0 @ Kx]])
        self._W = np.array(P @ Q, dtype=np.complex128, order="F")
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


""" Test Section """


def test_redhaff():
    size = 1
    kx = np.diag(np.linspace(0, 1, size))
    ky = np.diag(np.linspace(0, 1, size))
    kx = np.array(kx, order="F")
    ky = np.array(ky, order="F")
    e = np.eye(size, size, dtype=np.complex128, order="F")
    logger.debug(f"Input Arrays:\n{kx}")
    sfree = SFree(kx, ky)
    logger.debug("---------------- Layer 1 -------------------------")
    SM = SMatrix(kx, ky, 5.0449 * e, e, sfree, 314.1593, 0.005)
    logger.debug("---------------- Layer 2 -------------------------")
    SM2 = SMatrix(kx, ky, 6 * e, e, sfree, 314.1593, 0.003)
    logger.debug("---------------- Redhaff -------------------------")
    sdev = SM * SM2
    logger.debug("---------------- SRef -------------------------")
    sref = SRef(kx, ky, 2, 1, sfree)
    logger.debug("---------------- STrn -------------------------")
    strn = STrn(kx, ky, 9, 1, sfree)
    _ = sref * sdev * strn



if __name__ == "__main__":
    test_redhaff()
