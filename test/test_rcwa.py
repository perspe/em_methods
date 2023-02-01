import logging
from logging.config import fileConfig
import os
import unittest

import numpy as np
from scipy.sparse import diags

from em_methods.rcwa_core.rcwa_core import SFree, SMatrix, SRef, STrn
from em_methods.rcwa_core.rcwa_core import initialize_components, r_t_fields
from em_methods.rcwa import rcwa
from em_methods.grid import UniformGrid, Grid2D
from em_methods.smatrix import Layer1D, smm

# Get dev logger to control the logging level
base_path = os.path.dirname(os.path.abspath(__file__))
log_config = os.path.join(base_path, '..', 'em_methods', 'logging.ini')
fileConfig(log_config)
logger = logging.getLogger('dev_file')
# logger.setLevel(logging.INFO)

_single_test = True
_test_full = False
_test_uniform = False


class TestRCWA(unittest.TestCase):
    # @unittest.skipIf(_single_test, "Just testing one function")
    # def test_smatrix(self):
    #     """ Test Redhaffer Product """
    #     kx = np.array(np.diag([0]))
    #     ky = np.array(np.diag([0]))
    #     e = np.eye(1, 1, dtype=np.complex64)
    #     sfree = SFree(kx, ky)
    #     SM = SMatrix(kx, ky, 5.0449 * e, e, sfree, 314.1593, 0.005)
    #     SM2 = SMatrix(kx, ky, 6 * e, e, sfree, 314.1593, 0.003)
    #     sdev = SM * SM2
    #     sref = SRef(kx, ky, e * -1.4142, 2, 1, sfree)
    #     strn = STrn(kx, ky, e * 3, 9, 1, sfree)
    #     res = sref * sdev * strn
    #     res_s11 = np.array([[-0.3156 - 0.0437j, 0], [0, -0.3156 - 0.0437j]],
    #                        dtype=np.complex64)
    #     res_s12 = np.array([[1.2541 + 0.5773j, 0], [0, 1.2541 + 0.5773j]],
    #                        dtype=np.complex64)
    #     res_s21 = np.array([[0.5912 + 0.2721j, 0], [0, 0.5912 + 0.2721j]],
    #                        dtype=np.complex64)
    #     res_s22 = np.array([[0.2384 + 0.2113j, 0], [0, 0.2384 + 0.2113j]],
    #                        dtype=np.complex64)
    #     self.assertTrue(np.array_equal(np.round(res._S11, 4), res_s11),
    #                     f"{res._S11}")
    #     self.assertTrue(np.array_equal(np.round(res._S12, 4), res_s12),
    #                     f"{res._S12}")
    #     self.assertTrue(np.array_equal(np.round(res._S21, 4), res_s21),
    #                     f"{res._S21}")
    #     self.assertTrue(np.array_equal(np.round(res._S22, 4), res_s22),
    #                     f"{res._S22}")

    @unittest.skipIf(_single_test, "Just testing one function")
    def test_initialize_comps(self):
        """ Test Initialization """
        theta, phi, pte, ptm, lmb = 0, 0, 1, 0, 0.02
        p, q = 0, 0
        dx, dy = 0.0175, 0.015
        inc_med = (2, 1)
        trn_med = (9, 1)
        k0, sfree, Kx, Ky, Kz_ref, Kz_trn, _, p = initialize_components(
            theta, phi, lmb, (ptm, pte), p, q, dx, dy, inc_med, trn_med)
        # Check the results
        v0 = ([[0, -1j], [1j, 0]])
        self.assertEqual(round(k0, 4), 314.1593, f"{k0=}")
        self.assertTrue(np.array_equal(Kx.toarray(), np.array([[0]])), f"{Kx=}")
        self.assertTrue(np.array_equal(Ky.toarray(), np.array([[0]])), f"{Ky=}")
        self.assertTrue(np.array_equal(Kz_trn.toarray(), np.array([[3]])), f"{Kz_trn}")
        self.assertTrue(
            np.array_equal(np.round(Kz_ref.toarray(), 4),
                           np.array([[-1.4142]], dtype=np.complex64)),
            f"{Kz_ref=}")
        self.assertTrue(np.array_equal(sfree._V.toarray(), v0), f"{sfree._V=}")

    @unittest.skipIf(_single_test, "Just testing one function")
    def test_fields(self):
        """ Test field calculation """
        kx = diags([0], format="csc", dtype=np.complex64)
        ky = diags([0], format="csc", dtype=np.complex64)
        kz_ref = diags([-1.4142], format="csc", dtype=np.complex64)
        kz_trn = diags([3], format="csc", dtype=np.complex64)
        e = np.eye(1, 1, dtype=np.complex128)
        sfree = SFree(kx, ky)
        SM = SMatrix(kx, ky, 5.0449 * e, e, sfree, 314.1593, 0.005)
        SM2 = SMatrix(kx, ky, 6 * e, e, sfree, 314.1593, 0.003)
        sdev = SM * SM2
        sref = SRef(kx, ky, kz_ref, 2, 1, sfree)
        strn = STrn(kx, ky, kz_trn, 9, 1, sfree)
        res = sref * sdev * strn
        e_src = np.array([0, 1])
        r, t = r_t_fields(res, sref, strn, e_src, kx, ky, kz_ref, kz_trn)
        inc_med = (2, 1)
        trn_med = (9, 1)
        kz_inc = 1.4142
        R = -np.sum(np.real(kz_ref / kz_inc) @ r)
        T = np.sum(np.real(inc_med[1] * kz_trn / (trn_med[1] * kz_inc)) @ t)
        self.assertEqual(round(R, 4), 0.1015)
        self.assertEqual(round(T, 4), 0.8985)

    @unittest.skipIf(_single_test, "Just testing one function")
    @unittest.skipIf(not _test_uniform, "...")
    def test_uniform_unit(self):
        """ Conservation - Normal incidence 0 harmonics """
        theta, phi, pte, ptm, lmb = 0, 0, 1, 0, 0.02
        p, q = 0, 0
        layer_stack = [UniformGrid(0.1, xlims=[0., 0.0175], ylims=[0, 0.015])]
        R, T = rcwa(layer_stack, theta, phi, lmb, (ptm, pte), p, q, (1, 1),
                    (1, 1))
        logger.info(f"{R=}::{T=}")
        self.assertEqual(round(R, 1), 0)
        self.assertEqual(round(T, 1), 1)

    @unittest.skipIf(_single_test, "Just testing one function")
    @unittest.skipIf(not _test_uniform, "...")
    def test_uniform_multiple(self):
        """ Conservation - Normal incidence 1 harmonic """
        theta, phi, pte, ptm, lmb = 0, 0, 1, 0, 0.02
        p, q = 1, 1
        layer_stack = [UniformGrid(0.1, xlims=[0, 0.0175], ylims=[0, 0.015])]
        R, T = rcwa(layer_stack, theta, phi, lmb, (ptm, pte), p, q, (1, 1),
                    (1, 1))
        logger.info(f"{R=}::{T=}")
        self.assertEqual(round(R, 1), 0)
        self.assertEqual(round(T, 1), 1)

    @unittest.skipIf(_single_test, "Just testing one function")
    @unittest.skipIf(not _test_uniform, "...")
    def test_uniform_unit_angle(self):
        """ Conservation - 0 harmonic and angle """
        theta, phi, pte, ptm, lmb = 20, 0, 1, 0, 0.02
        p, q = 0, 0
        layer_stack = [UniformGrid(0.1)]
        R, T = rcwa(layer_stack, theta, phi, lmb, (ptm, pte), p, q, (1, 1),
                    (1, 1))
        logger.info(f"{R=}::{T=}")
        self.assertEqual(round(R, 1), 0)
        self.assertEqual(round(T, 1), 1)

    @unittest.skipIf(_single_test, "Just testing one function")
    @unittest.skipIf(not _test_uniform, "...")
    def test_uniform_multiple_angle(self):
        """ Conservation - 0 harmonic and angle """
        theta, phi, pte, ptm, lmb = 50, 0, 1, 0, 0.02
        p, q = 1, 1
        layer_stack = [UniformGrid(0.1, xlims=[0, 0.0175], ylims=[0, 0.015])]
        R, T = rcwa(layer_stack, theta, phi, lmb, (ptm, pte), p, q, (1, 1),
                    (1, 1))
        logger.info(f"{R=}::{T=}")
        self.assertEqual(round(R, 1), 0)
        self.assertEqual(round(T, 1), 1)

    """ EMPossible Tests """

    @unittest.skipIf(_single_test, "Just testing one function")
    def test_rcwa_complete_1(self):
        """ Complete test - 0 harmonic """
        theta, phi, pte, ptm, lmb = 0, 0, 1, 0, 0.02
        p, q = 0, 0
        bottom_layer = UniformGrid(0.003, 6)
        top_layer = Grid2D(512, 512, 0)
        top_layer._test_triangle()
        R, T = rcwa([top_layer, bottom_layer], theta, phi, lmb, (ptm, pte), p,
                    q, (2, 1), (9, 1))
        R = float(R)
        T = float(T)
        logger.info(f"{R=}::{T=}")
        with self.subTest(i=1):
            self.assertEqual(round(R, 3), 0.102, f"{R=}")
        with self.subTest(i=2):
            self.assertEqual(round(T, 3), 0.898, f"{T=}")

    # @unittest.skipIf(_single_test, "Just testing one function")
    def test_rcwa_complete_2(self):
        """ Complete test - 1 harmonics normal incidence """
        theta, phi, pte, ptm, lmb = 0, 0, 1, 0, 0.02
        p, q = 1, 1
        bottom_layer = UniformGrid(0.003,
                                   6,
                                   1,
                                   xlims=[0, 0.0175],
                                   ylims=[0, 0.015])
        top_layer = Grid2D(512, 512, 0, xlims=[0, 0.0175], ylims=[0, 0.015])
        top_layer._test_triangle()
        R, T = rcwa([top_layer, bottom_layer], theta, phi, lmb, (ptm, pte), p,
                    q, (2, 1), (9, 1))
        logger.info(f"{R=}::{T=}")
        self.assertFalse(
            round(R + T, 5) != 1,
            f"Conservation is not satisfied\n{R=}::{T=}::{R+T}")
        with self.subTest(i=1):
            self.assertEqual(round(R, 2), 0.1)
        with self.subTest(i=2):
            self.assertEqual(round(T, 2), 0.90)

    @unittest.skipIf(_single_test, "Just testing one function")
    def test_rcwa_complete_3(self):
        """ Complete test - 1 harmonic and angle """
        theta, phi, pte, ptm, lmb = 60, 30, 1 / np.sqrt(2), 1j / np.sqrt(
            2), 0.02
        p, q = 1, 1
        bottom_layer = UniformGrid(0.003,
                                   6,
                                   1,
                                   xlims=[0, 0.0175],
                                   ylims=[0, 0.015])
        top_layer = Grid2D(512, 512, 0, xlims=[0, 0.0175], ylims=[0, 0.015])
        top_layer._test_triangle()
        R, T = rcwa([top_layer, bottom_layer], theta, phi, lmb, (ptm, pte), p,
                    q, (2, 1), (9, 1))
        logger.info(f"{R=}::{T=}")
        self.assertFalse(
            round(R + T, 5) != 1,
            f"Conservation is not satisfied\n{R=}::{T=}::{R+T}")
        with self.subTest(i=1):
            self.assertEqual(round(R, 2), 0.09)
        with self.subTest(i=2):
            self.assertEqual(round(T, 2), 0.91)

    """ Structure conservation tests """

    @unittest.skipIf(_single_test, "Just testing one function")
    def test_circle(self):
        """ Normal incidence on a cylinder """
        theta, phi, pte, ptm, lmb = 0, 0, 1, 0, 0.02
        p, q = 0, 0
        grid = Grid2D(512, 512, 0.1)
        grid.add_circle(5, 1, 0.2, 0.2)
        R, T = rcwa([grid], theta, phi, lmb, (ptm, pte), p, q, (1, 1), (1, 1))
        logger.info(f"{R=}::{T=}")
        self.assertEqual(round(R + T, 1), 1)

    @unittest.skipIf(_single_test, "Just testing one function")
    def test_circle_angle(self):
        """ Angular incidence on a cylinder"""
        theta, phi, pte, ptm, lmb = 70, 0, 1, 0, 0.02
        p, q = 0, 0
        grid = Grid2D(512, 512, 0.1)
        grid.add_circle(5, 1, 0.2, 0.2)
        R, T = rcwa([grid], theta, phi, lmb, (ptm, pte), p, q, (1, 1), (1, 1))
        logger.info(f"{R=}::{T=}")
        self.assertEqual(round(R + T, 1), 1)

    @unittest.skipIf(not _test_full, "Testing fast functions")
    @unittest.skipIf(_single_test, "Just testing one function")
    def test_circle_heavy_angle(self):
        """ Angular incidence on a cylinder - 21 harmonics """
        theta, phi, pte, ptm, lmb = 70, 0, 1, 0, 0.02
        p, q = 2, 2
        grid = Grid2D(512, 512, 0.1)
        grid.add_circle(5, 1, 0.2, 0.2)
        R, T = rcwa([grid], theta, phi, lmb, (ptm, pte), p, q, (1, 1), (1, 1))
        logger.info(f"{R=}::{T=}")
        self.assertEqual(round(R + T, 1), 1)

    @unittest.skipIf(_single_test, "Just testing one function")
    def test_circle_abs(self):
        """ Angular incidence on a cylinder - 21 harmonics """
        theta, phi, pte, ptm, lmb = 70, 0, 1, 0, 0.02
        p, q = 4, 4
        grid = Grid2D(512, 512, 0.1)
        grid.add_circle(5 + 0.1, 1, 0.2, 0.2)
        R, T = rcwa([grid], theta, phi, lmb, (ptm, pte), p, q, (1, 1), (1, 1))
        logger.info(f"{R=}::{T=}")
        self.assertLessEqual(round(R + T, 1), 1)

    @unittest.skipIf(_single_test, "Just testing one function")
    @unittest.expectedFailure
    def test_edge_case(self):
        """ Teste edge case where wavelength = xlen = ylen """
        pte, ptm = 1, 0
        p, q = 1, 1
        grid = Grid2D(1024,
                      1024,
                      0.75,
                      xlims=[-0.75, 0.75],
                      ylims=[-0.75, 0.75])
        grid.add_circle(6, 1, 0.2)
        bottom_layer = UniformGrid(0.5,
                                   6,
                                   1,
                                   xlims=[-0.75, 0.75],
                                   ylims=[-0.75, 0.75])
        R, T = rcwa([grid, bottom_layer], 0, 0, 0.15, (ptm, pte), p, q, (1, 1),
                    (1, 1))
        self.assertEqual(round(R + T, 1), 1)

    @unittest.skipIf(_single_test, "Just testing one function")
    def test_uniform_harmonics(self):
        """ Test constant results with harmonics for uniform layers """
        pte, ptm, lmb = 1, 0, 500
        p, q = 0, 0
        bottom_layer = UniformGrid(100, 3**2, 1)
        top_layer = UniformGrid(150, 2.5**2, 1)
        R_test, T_test = rcwa([top_layer, bottom_layer], 0, 0, lmb, (ptm, pte),
                              0, 0, (1, 1), (1, 1))
        # Test for normal incicence
        for harm_i in range(1, 7):
            with self.subTest(f"{harm_i=}"):
                R_rcwa, T_rcwa = rcwa([top_layer, bottom_layer], 0, 0, lmb,
                                      (ptm, pte), p, q, (1, 1), (1, 1))
                self.assertEqual(R_rcwa, R_test)
                self.assertEqual(T_rcwa, T_test)
        R_test, T_test = rcwa([top_layer, bottom_layer], 70, 50, lmb,
                              (ptm, pte), 0, 0, (1, 1), (1, 1))
        # Test for angle
        for harm_i in range(1, 7):
            with self.subTest(f"{harm_i=}"):
                R_rcwa, T_rcwa = rcwa([top_layer, bottom_layer], 70, 0, lmb,
                                      (ptm, pte), p, q, (1, 1), (1, 1))
                self.assertEqual(round(R_rcwa, 5), round(R_test, 5))
                self.assertEqual(round(T_rcwa, 5), round(T_test, 5))

    """ Test vs SMM """

    @unittest.skipIf(_single_test, "Just testing one function")
    def test_scatmm_1l(self):
        pte, ptm, lmb = 1, 0, 500
        p, q = 0, 0
        layer1 = UniformGrid(100, 3**2, 1)
        R, T = rcwa([layer1], 0, 0, lmb, (ptm, pte), p, q, (1, 1), (1, 1))
        self.assertEqual(round(R, 2), 0.38)
        self.assertEqual(round(T, 2), 0.62)

    @unittest.skipIf(_single_test, "Just testing one function")
    def test_scatmm_angle(self):
        """ Test rcwa vs smm for a single layer (no harmonics) """
        pte, ptm, lmb = 1, 0, 500
        p, q = 0, 0
        bottom_layer = UniformGrid(100, 3**2, 1)
        s_layer1 = Layer1D("L1", 100, 3, 0)
        for theta_i in np.linspace(0, 89, 10):
            for phi_i in np.linspace(0, 360, 10):
                with self.subTest(f"{theta_i=}"):
                    R_smm, T_smm = smm([s_layer1], np.radians(theta_i), phi_i,
                                       lmb, (ptm, pte), (1, 1), (1, 1))
                    R_rcwa, T_rcwa = rcwa([bottom_layer], theta_i, phi_i, lmb,
                                          (ptm, pte), p, q, (1, 1), (1, 1))
                    self.assertEqual(round(R_rcwa, 4), round(R_smm, 4))
                    self.assertEqual(round(T_rcwa, 4), round(T_smm, 4))

    @unittest.skipIf(_single_test, "Just testing one function")
    def test_scatmm_1l_abs(self):
        """ Test rcwa vs smm for a single layer with absorption """
        pte, ptm, lmb = 1, 0, 200
        p, q = 0, 0
        layer1 = UniformGrid(10, (3 + 0.01j)**2, 1)
        s_layer1 = Layer1D("L1", 10, 3, 0.01)
        for theta_i in np.linspace(0, 89, 5):
            for phi_i in np.linspace(0, 360, 5):
                with self.subTest(f"{theta_i=}::{phi_i=}"):
                    R_smm, T_smm = smm([s_layer1], np.radians(theta_i), 0, lmb,
                                       (ptm, pte), (1, 1), (1, 1))
                    R_rcwa, T_rcwa = rcwa([layer1], theta_i, 0, lmb,
                                          (ptm, pte), p, q, (1, 1), (1, 1))
                    self.assertEqual(round(R_rcwa, 4), round(R_smm, 4))
                    self.assertEqual(round(T_rcwa, 4), round(T_smm, 4))

    @unittest.skipIf(_single_test, "Just testing one function")
    def test_scatmm_2l_abs(self):
        """ Test rcwa vs smm for a 2 layers with absorption pol and angle """
        pte, ptm, lmb = 1 / np.sqrt(2), 1j / np.sqrt(2), 200
        p, q = 0, 0
        layer1 = UniformGrid(400, (3 + 0.01j)**2, 1)
        layer2 = UniformGrid(150, (9 + 0.25j)**2, 1)
        s_layer1 = Layer1D("L1", 400, 3, 0.01)
        s_layer2 = Layer1D("L1", 150, 9, 0.25)
        for theta_i in np.linspace(0, 89, 5):
            for phi_i in np.linspace(0, 360, 5):
                with self.subTest(f"{theta_i=}::{phi_i=}"):
                    R_smm, T_smm = smm([s_layer1, s_layer2],
                                       np.radians(theta_i), np.radians(phi_i),
                                       lmb, (ptm, pte), (1, 1), (1, 1))
                    R_rcwa, T_rcwa = rcwa([layer1, layer2], theta_i, phi_i,
                                          lmb, (ptm, pte), p, q, (1, 1),
                                          (1, 1))
                    self.assertEqual(round(R_rcwa, 4), round(R_smm, 4))
                    self.assertEqual(round(T_rcwa, 4), round(T_smm, 4))

    """ Test vs FDTD """

    # @unittest.skipIf(_single_test, "Just testing one function")
    def test_fdtd_circle(self):
        """ Testing FDTD comparison """
        theta, phi, pte, ptm, lmb = 0, 0, 0, 1, 0.6
        p, q = 22, 22
        grid = Grid2D(1024 * 8,
                      1024 * 8,
                      0.1,
                      xlims=[-0.25, 0.25],
                      ylims=[-0.25, 0.25])
        grid.add_circle((4 + 0.15j)**2, 1, 0.1, 0)
        bottom_layer = UniformGrid(0.1, (4 + 0.15j)**2,
                                   xlims=[-0.25, 0.25],
                                   ylims=[-0.25, 0.25])
        R, T = rcwa([grid, bottom_layer], theta, phi, lmb, (ptm, pte), p, q,
                    (1, 1), (1, 1))
        print(f"{R=}::{T=}::{R+T}")
        self.assertLessEqual(round(R + T, 1), 1)

    """ Plot Tests """

    # @unittest.skipIf(_single_test, "Just testing one function")
    # def test_smoothness(self):
    #     """ Plot results for changing wvl """
    #     theta, phi, pte, ptm = 0, 0, 1, 0
    #     p, q = 15, 15
    #     grid = Grid2D(1024*2,
    #                   1024*2,
    #                   0.5,
    #                   xlims=[-1, 1],
    #                   ylims=[-1, 1])
    #     grid.add_square(3, 1, 100)
    #     bottom_layer = UniformGrid(0.75,
    #                                3,
    #                                xlims=[-1, 1],
    #                                ylims=[-1, 1])
    #     R_arr, T_arr = [], []
    #     lmb = np.linspace(0.02, 0.1, 10)
    #     for lmb_i in lmb:
    #         try:
    #             R, T = rcwa([grid, bottom_layer], theta, phi, lmb_i, (ptm, pte), p,
    #                         q, (1, 1), (1, 1))
    #             R_arr.append(R)
    #             T_arr.append(T)
    #         except np.linalg.LinAlgError as e:
    #             logger.error(f"Error: {lmb_i=}\n{e}")
    #             R_arr.append(np.nan)
    #             T_arr.append(np.nan)
    #     R_arr = np.array(R_arr)
    #     T_arr = np.array(T_arr)
    #     plt.plot(lmb, T_arr, lmb, R_arr, lmb, R_arr + T_arr)
    #     plt.show()
