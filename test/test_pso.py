from matplotlib.pyplot import logging
from em_methods.optimization.pso import particle_swarm
import numpy as np
import unittest
import random

logger = logging.getLogger()

def test_func_1(x, y):
    return -np.exp(-(x**2)) * np.exp(-(y**2))


def test_func_1_kwargs(x, y, test_arg=None):
    if test_arg is not None:
        raise Exception()
    return -np.exp(-(x**2)) * np.exp(-(y**2))


def test_func_2(x, y):
    return np.sin(x) * np.sin(y) / (x * y)


def test_func_3(x, y, z):
    return np.sin(x - 1) * np.sin(y - 2) * np.sin(z + 3) / ((x - 1) * (y - 2) * (z + 3))


def test_func_4(x, y):
    return (
        0.9 * np.exp(-((x - 3) ** 2)) * np.exp(-((y - 2) ** 2))
        + np.exp(-(x**2)) * np.exp(-(y**2))
        + 0.5 * np.exp(-((x + 3) ** 2)) * np.exp(-((y + 2) ** 2))
    )


class Test_PSO_func(unittest.TestCase):
    def test_pso_single_run(self):
        """Test single run of PSO with fixed seeds --- mostly for debugging"""
        logger.setLevel(logging.DEBUG)
        random.seed(1)
        np.random.seed(1)
        fit, _, _, gbest_arr = particle_swarm(
            test_func_2,
            {"x": [-5, 5], "y": [-5, 5]},
            particles=5,
            progress=False,
            iterations=(10, 100, False),
            inert_prop=(1, 1.5, False),
            export=True,
        )
        self.assertEqual(round(fit, 5), 0.98482, f"Best Value: {fit}")
        base_gbest_arr = [
            0.38653223765384287,
            0.38653223765384287,
            0.5259858201666858,
            0.5259858201666858,
            0.6991452032504347,
            0.6991452032504347,
            0.9848206261909154,
            0.9848206261909154,
            0.9848206261909154,
            0.9848206261909154,
        ]
        self.assertEqual(base_gbest_arr, gbest_arr)

    @unittest.expectedFailure
    def test_pso_func_kwargs(self):
        """
        Test extra function kwargs passed to the pso function
        This run should fail by passing False to test_func_1_kwargs
        """
        logger.setLevel(logging.DEBUG)
        _, _, _, _ = particle_swarm(
            test_func_1_kwargs,
            {"x": [-5, 5], "y": [-5, 5]},
            particles=5,
            progress=False,
            iterations=(10, 100, False),
            inert_prop=(1, 1.5, False),
            export=False,
            **{"test_arg": False},
        )

    def test_pso_full_run(self):
        """ Complete run of the PSO algorithm """
        logger.setLevel(logging.INFO)
        # Set a smaller than normal particles and iterations
        fit, _, _, _ = particle_swarm(
            test_func_2,
            {"x": [-10, 10], "y": [-10, 10]},
            particles=10,
            iterations=(25, 50, True),
            progress=False,
            export=False,
        )
        self.assertEqual(round(fit, 5), 1, f"Best Value: {fit}")

    def test_pso_version(self):
        """ Main test to check pso run between versions """
        logger.setLevel(logging.WARN)
        for i in range(10, 51):
            with self.subTest(i=i - 9):
                fit, _, _, _ = particle_swarm(
                    test_func_3,
                    {"x": [-i * 2, i * 2], "y": [-i * 2, i * 2], "z": [-i * 2, i * 2]},
                    particles=30,
                    progress=False,
                    tolerance=(0.00001, 20)
                )
                self.assertEqual(round(fit, 1), 1, f"Best Value: {fit}")

    def test_close_peaks(self):
        """ Test to differentiate between 2 close peaks """
        logger.setLevel(logging.WARN)
        for i in range(10, 51):
            with self.subTest(i=i - 9):
                fit, _, _, _ = particle_swarm(
                    test_func_4,
                    {
                        "x": [-i * 2, i * 2],
                        "y": [-i * 2, i * 2],
                    },
                    progress=False,
                    tolerance=(0.00001, 20)
                )
                self.assertEqual(round(fit, 1), 1, f"Best Value: {fit}")
