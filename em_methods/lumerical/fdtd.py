import logging
from multiprocessing import Manager, Queue
import os
import re
import shutil
from typing import Dict, List, Tuple, Union
from uuid import uuid4
import h5py
import time

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.constants as scc
from scipy.integrate import trapezoid

from em_methods.lumerical.lum_helper import (
    LumMethod,
    LumericalError,
    RunLumerical,
    _get_lumerical_results,
)
import lumapi

# Get module logger
logger = logging.getLogger("sim")

""" Helper functions """


def __read_autoshutoff(log_file: str) -> List:
    # Gather info from log and the delete it
    autoshut_off_re = re.compile("^[0-9]{0,3}\.?[0-9]+%")
    autoshut_off_list: List[Tuple[float, float]] = []
    with open(log_file, mode="r") as log:
        for log_line in log.readlines():
            match = re.search(autoshut_off_re, log_line)
            if match:
                autoshut_off_percent = float(log_line.split(" ")[0][:-1])
                autoshut_off_val = float(log_line.split(" ")[-1])
                autoshut_off_list.append((autoshut_off_percent, autoshut_off_val))
    logger.debug(f"Autoshutoff:\n{autoshut_off_list}")
    return autoshut_off_list


""" Main functions """


def fdtd_run(
    basefile: str,
    properties: Dict[str, Dict[str, float]],
    get_results: Dict[str, Dict[str, Union[str, List]]],
    *,
    get_info: Dict[str, str] = {},
    func=None,
    savepath: Union[None, str] = None,
    override_prefix: Union[None, str] = None,
    delete: bool = True,
    fdtd_kw={"hide": True},
    **kwargs,
):
    """
    Generic function to run lumerical files from python
    Steps: (Copy file to new location/Update Properties/Run/Extract Results)
    Args:
        basefile: Path to the original file
        properties: Dictionary with the property object and property names and values
        get_results: Dictionary with the results to extract
        get_info: Extra properties to extract from the simulation
        func: Function to run before the simulation
        savepath (default=.): Override default savepath for the new file
        override_prefix (default=None): Override prefix for the new file
        delete (default=False): Delete newly generated file
    Return:
        results: Dictionary with all the results
        time: Time to run the simulation
    """
    # Build the name of the new file and copy to a new location
    basepath, basename = os.path.split(basefile)
    savepath: str = savepath or basepath
    override_prefix: str = override_prefix or str(uuid4())[0:5]
    new_filepath: str = os.path.join(savepath, override_prefix + "_" + basename)
    logger.debug(f"new_filepath:{new_filepath}")
    shutil.copyfile(basefile, new_filepath)
    # Get logfile name
    log_file: str = os.path.join(
        savepath, f"{override_prefix}_{os.path.splitext(basename)[0]}_p0.log"
    )
    # Run simulation - the process is as follows
    # 1. Create Manager to Store the data (Manager seems to be more capable of handling large datasets)
    # 2. Create a process (RunLumerical) to run the lumerical file
    #       - This avoids problems when the simulation gives errors
    results = Manager().dict()
    run_process = RunLumerical(
        LumMethod.FDTD,
        results=results,
        solver="FDTD",
        log_queue=Queue(-1),
        filepath=new_filepath,
        properties=properties,
        get_results=get_results,
        get_info=get_info,
        func=func,
        lumerical_kw=fdtd_kw,
        **kwargs,
    )
    run_process.start()
    # check_thread = CheckRunState(log_file, run_process, process_queue)
    # check_thread.start()
    logger.debug("Run Process Started...")
    run_process.join()
    logger.debug(f"Simulation finished")
    results_keys = list(results.keys())
    if "runtime" not in results_keys:
        raise LumericalError("Simulation Finished Prematurely")
    autoshutoff = __read_autoshutoff(log_file)
    logger.debug(f"Autoshutoff: {autoshutoff[-1]}")
    if delete:
        logger.debug(f"Deleting unwanted files")
        os.remove(new_filepath)
        os.remove(log_file)
    if "analysis runtime" not in results_keys:
        raise LumericalError("Simulation Failed in Analysis")
    if "Error" in results_keys:
        raise LumericalError(results["Error"])
    # Extract data from process
    logger.debug(f"Simulation data:\n{results}")
    # Check for other possible runtime problems
    if "data" not in results_keys:
        raise LumericalError("No data available from simulation")
    data_results = results["data"]
    data_results["autoshutoff"] = autoshutoff
    logger.debug(f"Autoshutoff: {autoshutoff[-1]}")
    return (
        data_results,
        results["runtime"],
        results["analysis runtime"],
        results["data_info"],
    )

def rcwa_run(
    basefile: str,
    properties: Dict[str, Dict[str, float]],
    get_results: Dict[str, Dict[str, Union[str, List]]],
    *,
    get_info: Dict[str, str] = {},
    func=None,
    savepath: Union[None, str] = None,
    override_prefix: Union[None, str] = None,
    delete: bool = True,
    fdtd_kw={"hide": True},
    **kwargs,
):
    """
    Generic function to run lumerical files from python
    Steps: (Copy file to new location/Update Properties/Run/Extract Results)
    Args:
        basefile: Path to the original file
        properties: Dictionary with the property object and property names and values
        get_results: Dictionary with the results to extract
        get_info: Extra properties to extract from the simulation
        func: Function to run before the simulation
        savepath (default=.): Override default savepath for the new file
        override_prefix (default=None): Override prefix for the new file
        delete (default=False): Delete newly generated file
    Return:
        results: Dictionary with all the results
        time: Time to run the simulation
    """
    # Build the name of the new file and copy to a new location
    basepath, basename = os.path.split(basefile)
    savepath: str = savepath or basepath
    override_prefix: str = override_prefix or str(uuid4())[0:5]
    new_filepath: str = os.path.join(savepath, override_prefix + "_" + basename)
    logger.debug(f"new_filepath:{new_filepath}")
    shutil.copyfile(basefile, new_filepath)
    # Run simulation - the process is as follows
    # 1. Create Manager to Store the data (Manager seems to be more capable of handling large datasets)
    # 2. Create a process (RunLumerical) to run the lumerical file
    #       - This avoids problems when the simulation gives errors
    results = Manager().dict()
    run_process = RunLumerical(
        LumMethod.FDTD,
        results=results,
        solver="RCWA",
        log_queue=Queue(-1),
        filepath=new_filepath,
        properties=properties,
        get_results=get_results,
        get_info=get_info,
        func=func,
        lumerical_kw=fdtd_kw,
        **kwargs,
    )
    run_process.start()
    # check_thread = CheckRunState(log_file, run_process, process_queue)
    # check_thread.start()
    logger.debug("Run Process Started...")
    run_process.join()
    logger.debug(f"Simulation finished")
    results_keys = list(results.keys())
    if "runtime" not in results_keys:
        raise LumericalError("Simulation Finished Prematurely")
    if delete:
        logger.debug(f"Deleting unwanted files")
        os.remove(new_filepath)
    if "analysis runtime" not in results_keys:
        raise LumericalError("Simulation Failed in Analysis")
    if "Error" in results_keys:
        raise LumericalError(results["Error"])
    # Extract data from process
    logger.debug(f"Simulation data:\n{results}")
    # Check for other possible runtime problems
    if "data" not in results_keys:
        raise LumericalError("No data available from simulation")
    data_results = results["data"]
    return (
        data_results,
        results["runtime"],
        results["analysis runtime"],
        results["data_info"],
    )

def fdtd_run_large_data(
    basefile: str,
    properties: Dict[str, Dict[str, float]],
    get_results: Dict[str, Dict[str, Union[str, List]]],
    *,
    savepath: Union[None, str] = None,
    override_prefix: Union[None, str] = None,
    delete: bool = False,
    fdtd_kw: bool = {"hide": True},
):
    """
    Generic function to run lumerical files from python
    Steps: (Copy file to new location/Update Properties/Run/Extract Results)
    Args:
            basefile: Path to the original file
            properties: Dictionary with the property object and property names and values
            get_results: Dictionary with the properties to be calculated
            savepath (default=.): Override default savepath for the new file
            override_prefix (default=None): Override prefix for the new file
            delete (default=False): Delete newly generated file
    Return:
            results: Dictionary with all the results
            time: Time to run the simulation
    """
    # Build the name of the new file and copy to a new location
    basepath, basename = os.path.split(basefile)
    savepath: str = savepath or basepath
    override_prefix: str = override_prefix or str(uuid4())[0:5]
    new_filepath: str = os.path.join(savepath, override_prefix + "_" + basename)
    logger.debug(f"new_filepath:{new_filepath}")
    shutil.copyfile(basefile, new_filepath)
    # Update simulation properties, run and get results
    with lumapi.FDTD(filename=new_filepath, **fdtd_kw) as fdtd:
        # Update structures
        for structure_key, structure_value in properties.items():
            logger.debug(f"Editing: {structure_key}")
            fdtd.select(structure_key)
            for parameter_key, parameter_value in structure_value.items():
                logger.debug(f"Updating: {parameter_key} to {parameter_value}")
                fdtd.set(parameter_key, parameter_value)
        # Note: The double fdtd.runsetup() is important for when the setup scripts
        #       (such as the model script) depend on variables from other
        #       scripts. For example the model scripts needs the internal property
        #       of a layer generated from a structure group.
        #       The first run updates internally all the values
        #       The second run then updates all the structures with the updated values
        fdtd.runsetup()
        fdtd.runsetup()
        logger.debug(f"Running...")
        start_time = time.time()
        fdtd.run()
        fdtd_runtime = time.time() - start_time
        start_time = time.time()
        fdtd.runanalysis()
        analysis_runtime = time.time() - start_time
        results = _get_lumerical_results(fdtd, get_results)
    # Gather info from log and the delete it
    log_file: str = os.path.join(
        savepath, f"{override_prefix}_{os.path.splitext(basename)[0]}_p0.log"
    )
    autoshut_off_re = re.compile("^[0-9]{0,3}\.?[0-9]+%")
    autoshut_off_list: List[Tuple[float, float]] = []
    with open(log_file, mode="r") as log:
        for log_line in log.readlines():
            match = re.search(autoshut_off_re, log_line)
            if match:
                autoshut_off_percent = float(log_line.split(" ")[0][:-1])
                autoshut_off_val = float(log_line.split(" ")[-1])
                autoshut_off_list.append((autoshut_off_percent, autoshut_off_val))
    logger.debug(f"Autoshutoff:\n{autoshut_off_list}")
    logger.info(
        f"Simulation took: FDTD: {fdtd_runtime:0.2f}s | Analysis: {analysis_runtime:0.2f}s | Autoshutoff: {autoshut_off_list[-1]}"
    )
    if delete:
        os.remove(new_filepath)
        os.remove(log_file)
    return results, fdtd_runtime, analysis_runtime, autoshut_off_list


def fdtd_run_analysis(
    basefile: str,
    get_results: Dict[str, Dict[str, Union[str, List]]],
    fdtd_kw={"hide": True},
):
    """
    Generic function gather simulation data from already simulated files
    Args:
            basefile: Path to the original file
            get_results: Dictionary with the properties to be calculated
    Return:
            results: Dictionary with all the results
            time: Time to run the simulation
    """
    with lumapi.FDTD(filename=basefile, **fdtd_kw) as fdtd:
        fdtd.runanalysis()
        results = _get_lumerical_results(fdtd, get_results)
    return results


def fdtd_add_material(
    basefile: str,
    name: str,
    freq: npt.ArrayLike,
    permitivity: npt.NDArray[np.complex128],
    *,
    savefit: Union[None, str] = None,
    edit: bool = False,
    **kwargs,
):
    """
    Add a new material to the database of a specific file
    Args:
        - basefile: Name/path o the fsp file to add the material
        - name: Name for the new material in the DB (should be unique)
        - freq: Array with the frequency values (in Hz)
        - permitivity: complex array with the permitivity for the material
        - savefit (None or filename): Save the fit data to a new file
        - edit: edit an already present material in the D
        - kwargs: provide extra arguments for the material function (tolerance and max_coefficients)
    """
    basepath, _ = os.path.split(basefile)
    with lumapi.FDTD(filename=basefile, hide=True) as fdtd:
        if not edit:
            material = fdtd.addmaterial("Sampled 3D data")
            fdtd.setmaterial(material, "name", name)
            fdtd.setmaterial(name, "sampled data", np.c_[freq, permitivity])
        for key, value in kwargs.items():
            fdtd.setmaterial(name, key, value)
        if savefit is not None:
            fit_res = fdtd.getfdtdindex(name, np.array(freq), min(freq), max(freq))
            fit_res = fit_res.flatten()
            export_df = pd.DataFrame(
                {
                    "Wvl": scc.c / freq,
                    "n_og": np.real(np.sqrt(permitivity)),
                    "k_ok": np.imag(np.sqrt(permitivity)),
                    "n_fit": np.real(fit_res),
                    "k_fit": np.imag(fit_res),
                }
            )
            logger.debug(f"Export_array:\n{export_df}")
            export_df.to_csv(os.path.join(basepath, savefit), sep=" ", index=False)
        fdtd.save()


def fdtd_get_material(
    basefile: str,
    names: Union[str, List[str]],
    freq: npt.ArrayLike,
    *,
    fit=True,
    data=True,
    save: Union[None, str] = None,
    **kwargs,
):
    """
    Extract data from materials from the database
    Args:
        - basefile: Name/path o the fsp file to add the material
        - name: Names of the materials to extract from the database
        - freq: Array with the frequency values (in Hz)
        - fit: Extract the fitted data
        - data: Extract the original data
        - save: (None or filename): Save the fit data to a new file
        - kwargs: provide extra arguments for the material function (tolerance and max_coefficients)
    """
    basepath, _ = os.path.split(basefile)
    if isinstance(names, str):
        names: List[str] = [names]
    with lumapi.FDTD(filename=basefile, hide=True) as fdtd:
        return_res = pd.DataFrame({"Wvl": scc.c / freq})
        for name in names:
            for key, value in kwargs.items():
                fdtd.setmaterial(name, key, value)
            if fit:
                fit_res = fdtd.getfdtdindex(name, np.array(freq), min(freq), max(freq))
                return_res[f"n_fit_{name}"] = np.real(fit_res)
                return_res[f"k_fit_{name}"] = np.imag(fit_res)
            if data:
                base_data = fdtd.getindex(name, np.array(freq))
                return_res[f"n_og_{name}"] = np.real(base_data)
                return_res[f"k_og_{name}"] = np.imag(base_data)
        logger.debug(f"Export_array:\n{return_res.columns}")
        if save:
            return_res.to_csv(os.path.join(basepath, save), sep=" ", index=False)
    return return_res


""" Other functions """


def import_generation(gen_file):
    """
    Import and rearrange the necessary data
    Filter - (z, x, y)
    Generation - (z, x, y)
    """
    with h5py.File(gen_file, "r") as gen_file:
        gen_data = np.transpose(np.array(gen_file["G"]), (0, 2, 1))
        x = np.array(gen_file["x"]).transpose()
        y = np.array(gen_file["y"]).transpose()
        z = np.array(gen_file["z"]).transpose()

    return gen_data, x, y, z


def _average_generation(gen, x, y):
    """Calculate the y and x/y averages generation profiles"""
    norm_y = np.max(y) - np.min(y)
    y_gen = trapezoid(gen, y, axis=2) / norm_y
    norm_area = norm_y * (np.max(x) - np.min(x))
    xy_gen = trapezoid(trapezoid(gen, x, axis=1), y, axis=1) / norm_area
    return y_gen, xy_gen


def _export_hdf_data(filename, **kwargs):
    """Export data to an hdf5 file"""
    # Check for existing file and override
    if os.path.isfile(filename):
        os.remove(filename)

    # Add data do hdf datastructure
    with h5py.File(filename, "x") as file:
        for key, value in kwargs.items():
            file.create_dataset(key, data=value, dtype="double")


def average_generation(
    filename: str,
    mode: str = "2d",
    export_mode: str = "3d",
    export_name: Union[None, str] = None,
):
    generation, x, y, z = import_generation(filename)
    mode = mode.lower()
    if mode == "2d":
        average_gen, _ = _average_generation(generation, x.flatten(), y.flatten())
    elif mode == "1d":
        _, average_gen = _average_generation(generation, x.flatten(), y.flatten())
        export_gen = {"G": average_gen, "z": z}
    else:
        raise Exception("Unexpected mode (2d or 1d)")
    export_mode = export_mode.lower()
    if export_mode == "3d":
        average_gen = average_gen + np.zeros((len(x), len(z), len(y)))
        export_gen = {
            "G": np.transpose(average_gen, (1, 0, 2)),
            "x": x,
            "y": y,
            "z": z,
        }
    else:
        raise Exception("Unexpected export mode (3d)")
    if export_name is not None:
        _export_hdf_data(export_name, **export_gen)
    return export_gen
