from em_methods.optimization.pso import particle_swarm
import numpy as np
import unittest
import numpy as np


def test_func_1(x, y):
    return -np.exp(-(x**2)) * np.exp(-(y**2))


def test_func_2(x, y):
    return np.sin(x) * np.sin(y) / (x * y)


def test_func_3(x, y, z):
    return np.sin(x - 1) * np.sin(y - 2) * np.sin(z + 3) / ((x - 1) * (y - 2) * (z + 3))


def test_func_4(x, y):
    return 0.9 * np.exp(-((x - 3) ** 2)) * np.exp(-((y - 2) ** 2)) + np.exp(
            -(x**2)) * np.exp(-(y**2)) + 0.5*np.exp(-((x+3)**2))*np.exp(-(y+2)**2)


class Test_PSO_func(unittest.TestCase):
    def test_pso_single_run(self):
        """Test single run of PSO -- mostly for debugging"""
        fit, _, _, _ = particle_swarm(
            test_func_2,
            {"x": [-5, 5], "y": [-5, 5]},
            particles=5,
            progress=False,
            iterations=(5, 10, True),
        )
        self.assertEqual(round(fit, 1), 1, f"Best Value: {fit}")

    def test_pso_tolerance(self):
        """Test conditions to reach tolerance"""
        fit, _, _, _ = particle_swarm(
            test_func_2,
            {"x": [-50, 50], "y": [-50, 50]},
            progress=False,
            tolerance=(1e-5, 10),
            iterations=(25, 100, True),
        )
        self.assertEqual(round(fit, 1), 1, f"Best Value: {fit}")

    def test_pso_version(self):
        """Main test to check pso run between versions"""
        for i in range(10, 51):
            with self.subTest(i=i - 9):
                fit, _, _, _ = particle_swarm(
                    test_func_3,
                    {"x": [-i * 2, i * 2], "y": [-i * 2, i * 2], "z": [-i * 2, i * 2]},
                    progress=False,
                )
                self.assertEqual(round(fit, 1), 1, f"Best Value: {fit}")

    def test_close_peaks(self):
        """Test to differentiate between 2 close peaks"""
        for i in range(10, 51):
            with self.subTest(i=i - 9):
                fit, _, _, _ = particle_swarm(
                    test_func_4,
                    {
                        "x": [-i * 2, i * 2],
                        "y": [-i * 2, i * 2],
                    },
                    progress=False,
                )
                self.assertEqual(round(fit, 1), 1, f"Best Value: {fit}")
