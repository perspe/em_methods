import os
import shutil
from typing import Union, Dict, List, Tuple
from uuid import uuid4
import logging
import time
import re

import numpy as np
import numpy.typing as npt
import pandas as pd

import scipy.constants as scc

# Get module logger
logger = logging.getLogger("sim")

# Connect to Lumerical
import sys
# Determine the base path for lumerical
if os.name == "nt":
    LM_BASE: str = os.path.join("C:\\", "Program Files", "Lumerical")
elif os.name == "posix":
    LM_BASE: str = os.path.join("/opt", "Lumerical")
else:
    raise Exception("Operating system not supported...")
# Determine the newest version
LM_API: str = os.path.join("api", "python")
lm_dir_list: List[str] = os.listdir(LM_BASE)
if len(lm_dir_list) == 1:
    LUMAPI_PATH: str = os.path.join(LM_BASE, lm_dir_list[0], LM_API)
else:
    v_num: List[int] = [int(v_i[1:]) for v_i in lm_dir_list if v_i[0] == "v"]
    LUMAPI_PATH: str = os.path.join(LM_BASE, f"v{max(v_num)}", LM_API)
logger.debug(f"LUMAPI_PATH: {LUMAPI_PATH}")
sys.path.append(LUMAPI_PATH)
import lumapi

def fdtd_run(basefile: str,
             properties: Dict[str, Dict[str, float]],
             get_results: Dict[str, Dict[str, Union[str, List]]],
             *,
             savepath: Union[None, str] = None,
             override_prefix: Union[None, str] = None,
             delete: bool = False
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
    new_filepath: str = os.path.join(
        savepath, override_prefix + "_" + basename)
    logger.debug(f"new_filepath:{new_filepath}")
    shutil.copyfile(basefile, new_filepath)
    # Update simulation properties, run and get results
    results = {}
    with lumapi.FDTD(filename=new_filepath, hide=True) as fdtd:
        # Update structures
        for structure_key, structure_value in properties.items():
            logger.debug(f"Editing: {structure_key}")
            fdtd.select(structure_key)
            for parameter_key, parameter_value in structure_value.items():
                logger.debug(
                    f"Updating: {parameter_key} to {parameter_value}")
                fdtd.set(parameter_key, parameter_value)
        logger.debug(f"Running...")
        start_time = time.time()
        fdtd.run()
        fdtd_runtime = time.time() - start_time
        start_time = time.time()
        fdtd.runanalysis()
        analysis_runtime = time.time() - start_time
        logger.info(
            f"Simulation took: FDTD: {fdtd_runtime:0.2f}s | Analysis: {analysis_runtime:0.2f}s")
        # Obtain results
        get_results_info = list(get_results.keys())
        if "data" in get_results_info:
            for key, value in get_results["data"].items():
                if not isinstance(value, list):
                    value = [value]
                for value_i in value:
                    logger.debug(f"Getting result for: '{key}':'{value_i}'")
                    results["data."+key+"."+value_i] = fdtd.getdata(key, value_i)
        if "results" in get_results_info:
            for key, value in get_results["results"].items():
                if not isinstance(value, list):
                    value = [value]
                for value_i in value:
                    logger.debug(f"Getting result for: '{key}':'{value_i}'")
                    results["results."+key+"."+value_i] = fdtd.getresult(key, value_i)
    # Gather info from log and the delete it
    log_file: str = os.path.join(savepath, f"{override_prefix}_{os.path.splitext(basename)[0]}_p0.log")
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
    if delete:
        os.remove(new_filepath)
        os.remove(log_file)
    return results, fdtd_runtime, analysis_runtime, autoshut_off_list


def fdtd_add_material(basefile: str,
                      name: str,
                      freq: npt.ArrayLike,
                      permitivity: npt.NDArray[np.complex128],
                      *,
                      savefit: Union[None, str] = None,
                      edit: bool = False,
                      **kwargs):
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
            fdtd.setmaterial(name, "sampled data", np.c_[
                            freq, permitivity])
        for key, value in kwargs.items():
            fdtd.setmaterial(name, key, value)
        if savefit is not None:
            fit_res = fdtd.getfdtdindex(
                name, np.array(freq), min(freq), max(freq))
            fit_res = fit_res.flatten()
            export_df = pd.DataFrame(
                {"Wvl": scc.c/freq,
                "n_og":np.real(np.sqrt(permitivity)),
                "k_ok": np.imag(np.sqrt(permitivity)),
                "n_fit": np.real(fit_res),
                "k_fit": np.imag(fit_res)})
            logger.debug(f"Export_array:\n{export_df}")
            export_df.to_csv(os.path.join(basepath, savefit), sep=" ", index=False)
        fdtd.save()

def fdtd_get_material(basefile: str,
                      names: Union[str, List[str]],
                      freq: npt.ArrayLike,
                      *,
                      fit=True,
                      data=True,
                      save: Union[None, str] = None,
                      **kwargs):
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
        return_res = pd.DataFrame({"Wvl": scc.c/freq})
        for name in names:
            for key, value in kwargs.items():
                fdtd.setmaterial(name, key, value)
            if fit:
                fit_res = fdtd.getfdtdindex(
                    name, np.array(freq), min(freq), max(freq))
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