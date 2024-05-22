import unittest
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from em_methods.pv import single_diode_rp, luqing_liu_diode
from em_methods.optimization.pso import particle_swarm

# Override logger to always use debug
logger = logging.getLogger("sim")
logger.setLevel(logging.DEBUG)
BASETESTPATH: str = os.path.join("test", "pv")


class TestPV(unittest.TestCase):
    def test_single_diode_rp(self):
        """
        Test run the single diode equation
        """
        eta, rs, rsh, j0, jl = 1.5, 2, 3000, 1.5e-11, 21.7
        temp = 298
        voltage = np.linspace(0, 1.2, 100)
        j = single_diode_rp(voltage, jl, j0, rs, rsh, eta, temp)
        self.assertEqual(round(j[0], 4), 21.6855)

    def test_luqing_liu(self):
        """
        Test the luqing liu equation
        """
        eta, rs, rsh, temp = 1.5, 2, 3000, 298
        voc, jsc, jmpp = 1.09, 21.66, 20.34
        voltage = np.linspace(0, 1.2)
        current = luqing_liu_diode(voltage, jsc, jmpp, voc, rs, rsh, eta, temp)
        self.assertEqual(round(current[0], 4), 21.6596)

    def test_optimize_luqing(self):
        """
        Use the particle swarm algorithm to determine the
        best parameters of the luqing liu equation that fit experimental data
        """

        def optimize_function(exp_data, jsc, jmpp, voc, rs, rsh, eta, temp, n_cells):
            differences = []
            for rs_i, rsh_i, eta_i in zip(rs, rsh, eta):
                luqing_current = luqing_liu_diode(
                    exp_data.V, jsc, jmpp, voc, rs_i, rsh_i, eta_i, temp, n_cells
                )
                differences.append(
                    np.sqrt(np.mean((exp_data["j"] - luqing_current) ** 2))
                )
            return np.array(differences)

        test_file = os.path.join(BASETESTPATH, "iv_perovskite.txt")
        cell_area = 0.1963
        # Data and useful variables
        data = pd.read_csv(test_file, sep="\t")
        data["j"] = data["I"] * 1000 / cell_area
        data["p"] = data["j"] * data["V"]
        data["jabs"] = data["j"].abs()
        logger.debug(f"Imported data:\n{data}")
        # Variables for Luqing Liu equation
        jsc = data["j"].max()
        jmpp = data[data["p"] == data["p"].max()]["j"].values
        voc = data[data["jabs"] == data["jabs"].min()]["V"].values
        pso_parameters = {"eta": [1, 2], "rs": [0, 50], "rsh": [0, 1e10]}
        const_parameters = {
            "exp_data": data,
            "jsc": jsc,
            "jmpp": jmpp,
            "voc": voc,
            "temp": 298,
            "n_cells": 1,
        }
        logger.debug(const_parameters)
        r2, best_param, *_ = particle_swarm(
            optimize_function,
            pso_parameters,
            particles=50,
            iterations=(200, 200, False),
            maximize=False,
            progress=False,
            export_summary=False,
            **const_parameters,
        )
        print(best_param)
        best_current = luqing_liu_diode(
            data["V"], jsc, jmpp, voc, best_param[1], best_param[2], best_param[0], 298
        )
        best_current_cristina = luqing_liu_diode(
            data["V"], jsc, jmpp, voc, 2.85, 1e5, 1.4, 298
        )
        plt.scatter(data["V"], data["j"], 6, "r", label="Experimental")
        plt.plot(data["V"], best_current, label="Optimized")
        plt.plot(data["V"], best_current_cristina, label="Optimized Cristina")
        plt.annotate(f"R2={r2:.6f}", (0, 10))
        plt.legend()
        plt.grid()
        plt.show()
