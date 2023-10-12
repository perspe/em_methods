"""
Implementation of the particle swarm optimization algorithm
Functions:
    - particle_swarm: Implements the algorithm
"""
import logging
import os
from random import random
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger()


def _update_parameters(
    param, vel, max_param, min_param, inertia_w, ind_cog, soc_learning, pbest, gbest
):
    """
    Update equation for the particle swarm algorithm
    V_ij^(t+1) =
        learning rate : w*V_ij^t
        cognitive part : c1*r1*(pbest_ij - p_ij^t)
        social part : c2*r2*(gbest_ij - p_ij^t)
    Args:
        param - input variables (ij array - i particles, i parameters)
        vel - input velocities (ij array)
        max_param - maximum parameter values
        min_param - minimum parameter values
        inertia_w - inertia weight constant
        ind_cog - individual cognition parameter
        soc_learning - social learning parameter
        pbest - best set of parameters for a certain particle (ij array)
        gbest - best global set of parameters (i array)
    Return:
        Updated parameters and velocities
    """
    # Parameter Setup
    # --> Particles ↓ Parameters
    # | Param1_Part1 Param1_Part2 Param1_Part3 ...|
    # | Param2_Part1 Param2_Part2 Param2_Part3 ...|
    # | Param3_Part1 Param3_Part2 Param3_Part3 ...|
    # |      ...          ...         ...      ...|
    logger.debug("Update Properties --------------")
    logger.debug(f"Init values: {inertia_w}, {ind_cog}, {soc_learning}")
    logger.debug(f"vel=\n{vel}")
    logger.debug(f"pbest=\n{pbest}")
    logger.debug(f"gbest=\n{gbest}")
    r1 = random()
    r2 = random()
    logger.debug(f"R1:{r1}|R2:{r2}")
    max_param = np.broadcast_to(max_param[:, np.newaxis], param.shape)
    min_param = np.broadcast_to(min_param[:, np.newaxis], param.shape)
    logger.debug(f"min_param:\n{min_param}")
    logger.debug(f"max_param:\n{max_param}")
    logger.debug("Calculations -----------")
    # Update velocity
    part_1 = inertia_w * vel
    part_2 = ind_cog * r1 * (pbest - param)
    part_3 = soc_learning * r2 * (gbest - param)
    v_new = part_1 + part_2 + part_3
    logger.debug(f"part_1:\n{part_1}")
    logger.debug(f"part_2:\n{part_2}")
    logger.debug(f"part_3:\n{part_3}")
    # Update position
    param_new = param + v_new
    logger.debug(f"param_new:\n{param_new}")
    logger.debug(f"v_new:\n{v_new}")
    # Check if no parameters are outside the allowed ranges for the parameters
    mask_min = param_new < min_param
    mask_max = param_new > max_param
    logger.debug(f"mask_min:\n{mask_min}")
    logger.debug(f"mask_max:\n{mask_max}")
    param_new[mask_min] = min_param[mask_min]
    param_new[mask_max] = max_param[mask_max]
    logger.debug(f"Parameter Space:\n{param_new}")
    v_new[mask_min ^ mask_max] = 0
    # Option to invert velocity: v_new[mask_min ^ mask_max] = -v_new[mask_min ^ mask_max]
    logger.debug(f"Velocity space:\n{v_new}")
    return param_new, v_new


def _preview_results(
    ax,
    iteration: npt.NDArray,
    FoM: List[float],
    best_array: npt.NDArray,
    param_names: List[str],
):
    """Update the global plot preview of the results"""
    ax[0].clear()
    ax[0].plot(iteration, FoM)
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("FoM")
    ax[1].clear()
    ax[1].axis("off")
    ax[1].set_title("Best Results")
    for index, (best_item, item_name) in enumerate(zip(best_array, param_names)):
        ax[1].annotate(
            f"{item_name} = {best_item:.2g}",
            (0.05, 0.9 - index * 0.05),
            xycoords="axes fraction",
            fontfamily="Arial",
            fontsize=10,
        )
    plt.savefig("pso_update_res.png", dpi=200)
    logger.debug("Updated Summary Figure")


def particle_swarm(
    func,
    param_dict: Dict[str, List[float]],
    *,
    maximize: bool = True,
    inert_prop: Tuple[float, float, bool] = (0.9, 0.4, True),
    ind_cog: float = 1.45,
    soc_learning: float = 1.45,
    particles: int = 25,
    iterations: Tuple[int, int, bool] = (50, 100, True),
    tolerance: Tuple[float, int] = (0.05, 10),
    progress: bool = True,
    export: bool = False,
    basepath: str = "PSO_Results",
    **func_kwargs,
) -> Tuple[float, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
    """Implementation of the particle swarm algorithm
    Args:
        - func: optimization function
        - param_dict: dictionary with parameters and variation range
        - maximize: maximize or minimize the problem (default: maximize)
        - inert_prop: Inertial weight factor (start value, finish value, static/dynamic)
        - ind_cog: cognition index for particles (default = 1.45)
        - soc_learning: social learning index (default = 1.45)
        - particles: Number of particles (default: 25)
        - iteration: Define number of iterations (min, max, static/dynamic)
        - max_iterations: Max number of iterations (default = 100)
        - tolerance_percent
        - export: Export files with PSO data
        - basepath: Base path to save export and progress information
        - func_kwargs: Extra arguments to pass to the optimization function
    Return:
        - gfitness: Best value obtained
        - gbest: Best parameters
        - pbest: Best parameters for each particle
        - gbest_array: Array with the gfitness value for each iteration
    """
    # Create export path
    if export and not os.path.isdir(basepath):
        logger.info(f"Creating {basepath=}...")
        os.mkdir(basepath)
    min_iteration, max_iteration, iteration_check = iterations
    if max_iteration < min_iteration and iteration_check:
        raise Exception("max_iteration must be bigger than min_iteration")
    if not iteration_check:
        max_iteration = min_iteration
    # Create an array for the inertial factor variation (its max_iteration size)
    inert_factor_low, inert_factor_up, inert_sweep = inert_prop
    if inert_sweep:
        inert_factor = np.linspace(inert_factor_low, inert_factor_up, min_iteration)
        inert_factor_remaining = np.array(
            [inert_factor_up] * (max_iteration - min_iteration)
        )
        inert_factor = np.r_[inert_factor, inert_factor_remaining]
    else:
        inert_factor = np.ones(max_iteration) * inert_factor_low
    logger.debug(f"Inert_factor array:\n{inert_factor}")
    # Variable initialization
    param_names = list(param_dict.keys())
    vparam_names = [f"v{param_name_i}" for param_name_i in param_names]
    export_names = param_names.copy()
    export_names.append("FoM")
    export_names.extend(vparam_names)
    logger.debug(f"Parameters in study:\n{export_names}")
    param_max = np.array([p_max[1] for p_max in param_dict.values()])
    param_min = np.array([p_min[0] for p_min in param_dict.values()])
    # Random array with the start value for the parameters
    param_space = [
        np.random.uniform(param[0], param[1], size=(particles))
        for param in param_dict.values()
    ]
    param_space = np.stack(param_space)
    # Random array with the start value for the velocities
    vel_space = [
        np.random.uniform(
            -np.abs(max(param) - min(param)),
            np.abs(max(param) - min(param)),
            size=(particles),
        )
        for param in param_dict.values()
    ]
    vel_space = np.stack(vel_space)
    logger.debug(f"Initial Parameter Space:\n{param_space}")
    logger.debug(f"Initial Velocity Space:\n{vel_space}")
    # First run of the PSO outside loop
    iteration = 1
    func_input = {
        param_name: param_space[i] for i, param_name in enumerate(param_names)
    }
    func_results = func(**func_input, **func_kwargs)
    logger.debug(f"Initial Function Results:\n{func_results}")
    if maximize:
        fitness_arg = np.argmax(func_results)
    else:
        fitness_arg = np.argmin(func_results)
    # PSO optimization arrays (gfitness, pfitness, gbest, pbest, tol_array)
    tolerance_percent, tolerance_num = tolerance
    gfitness = func_results[fitness_arg]
    pfitness = func_results
    gbest = param_space[:, fitness_arg].flatten()
    tol_array = [0]
    gbest_array = [gfitness]
    pbest = param_space
    # Create figure handler to show the results
    if progress:
        _, ax = plt.subplots(
            1,
            2,
            figsize=(5, 4),
            gridspec_kw={"wspace": 0.1, "width_ratios": [0.7, 0.3]},
        )
        logger.debug(f"{np.arange(iteration)}::{gbest_array}")
        _preview_results(
            ax, np.arange(iteration), gbest_array, pbest[:, -1], param_names
        )
    # Export data
    if export:
        export_data = np.c_[param_space.T, func_results, vel_space.T]
        export_df = pd.DataFrame(export_data, columns=export_names)
        export_df.to_csv(
            os.path.join(basepath, f"pso_it{iteration:03d}.csv"), sep=" ", index=False
        )
    while iteration < max_iteration:
        # Check for the tolerance condition on the last tol_num values of the
        # tolerance array
        if iteration > min_iteration + tolerance_num:
            last_tolerances = np.array(tol_array)[-tolerance_num:]
            logger.debug(f"tol_array: {tol_array}\n{len(tol_array)}")
            logger.debug(f"last_tolerances: {last_tolerances}\n{len(last_tolerances)}")
            avg_tol = np.average(last_tolerances)
            logger.debug(f"avg_tol: {avg_tol}")
            if avg_tol < tolerance_percent:
                logger.warning(f"Tolerance reached at {iteration}... Exiting")
                break
        logger.info(f"PSO Running Iteration: {iteration}")
        param_space, vel_space = _update_parameters(
            param_space,
            vel_space,
            param_max,
            param_min,
            inert_factor[iteration - 1],
            ind_cog,
            soc_learning,
            pbest,
            gbest[:, np.newaxis],
        )
        # Update gbest and pbest
        func_input = {
            param_name: param_space[i] for i, param_name in enumerate(param_names)
        }
        func_results = func(**func_input, **func_kwargs)
        logger.debug(f"Function Results:\n{func_results}")
        gfitness_old = gfitness
        if maximize:
            fitness_candidate_ind = np.argmax(func_results)
            if func_results[fitness_candidate_ind] > gfitness:
                gfitness = func_results[fitness_candidate_ind]
                gbest = param_space[:, fitness_candidate_ind].flatten()
            pfitness_mask = func_results > pfitness
        else:
            fitness_candidate_ind = np.argmin(func_results)
            if func_results[fitness_candidate_ind] < gfitness:
                gfitness = func_results[fitness_candidate_ind]
                gbest = param_space[:, fitness_candidate_ind].flatten()
            pfitness_mask = func_results < pfitness
        # Add error values to array
        if gfitness_old == 0:
            gfitness_old = 1e-9
        tol_array.append((gfitness - gfitness_old) / gfitness_old)
        # Update gbest, pfitness and pbest
        logger.debug(f"Global best list:\n{gbest}")
        gbest_array.append(gfitness)
        # Update the FoM plot
        pfitness[pfitness_mask] = func_results[pfitness_mask]
        pbest[:, pfitness_mask] = param_space[:, pfitness_mask]
        logger.debug(f"Particle Best Values:\n{pfitness}")
        logger.debug(f"Particle Global best list:\n{pbest}")
        iteration += 1
        if progress:
            _preview_results(
                ax, np.arange(iteration), gbest_array, pbest[:, -1], param_names
            )
        if export:
            export_data = np.c_[param_space.T, func_results, vel_space.T]
            export_df = pd.DataFrame(export_data, columns=export_names)
            export_df.to_csv(
                os.path.join(basepath, f"pso_it{iteration:03d}.csv"),
                sep=" ",
                index=False,
            )
    logger.debug(
        f"Results:\ngfitness:{gfitness}\ngbest:\n{gbest}\npbest:\n{pbest}\ngbest_array:\n{gbest_array}"
    )
    return gfitness, gbest, pbest, gbest_array


def pso_iter_plt(
    param: str = "FoM",
    cmap: str = "PuOr",
    scalar: float = 1,
    export: bool = False,
    iter_num: float = 11,
    particle_num: float = 30,
    vmin: float = 70,
    vmax: float = 490,
    basepath: str = "D:\\Ivan S\\ATF_TSC\\structures\\voids\\PSO\\results\\PSO2",
    alpha: float = 1,
    colorbar: bool = True,
):
    g_matrix = []
    for j in range(particle_num):
        g_val = []
        fom = []
        for i in range(iter_num):
            iter = "{0:03}".format(i + 1)
            data = pd.read_csv(
                os.path.join(basepath, f"pso_it{str(iter)}.csv"), sep="\s+"
            )
            if param == "a_i":
                g_val.append(data[str(param)][j] * data[str("xy_radius")][j])
            else:
                g_val.append(data[str(param)][j])
            fom.append(data.FoM[j])
        g_matrix.append(np.array(g_val))
        upd_g_val = [x * scalar for x in g_val]
        h_scatt = plt.scatter(
            np.arange(1, iter_num + 1),
            upd_g_val,
            c=fom,
            s=30,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
        )
        plt.xlabel("Iteration")
    if scalar == 1e6:
        plt.ylabel(f"{param} (µm)")
    if scalar == 1e9:
        plt.ylabel(f"{param} (nm)")
    if scalar == 1:
        plt.ylabel(f"{param}")
    if colorbar:
        plt.colorbar(h_scatt).set_label("Jsc (A.m-2)")
    if export:
        if param == "a_i":
            plt.ylabel(f"pitch (µm)")
            plt.savefig(f"pitch.png", dpi=300)
        else:
            plt.savefig(f"{param}.png", dpi=300)


if __name__ == "__main__":

    def test_func(x, y):
        return -np.exp(-(x**2)) * np.exp(-(y**2))

    def test_func_2(x, y):
        return np.sin(x) * np.sin(y) / (x * y)

    def test_func_3(x, y):
        return np.sin(x * y)

    fit, gbest, pbest, _ = particle_swarm(
        test_func_3,
        {"x": [0, 3.14], "y": [0, 3.14]},
        maximize=True,
        export=True,
        progress=True,
    )
    print("----Results-----")
    print(fit, gbest, pbest, sep="\n")
