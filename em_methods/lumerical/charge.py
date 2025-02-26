from dataclasses import dataclass
import logging
from multiprocessing import Manager, Queue
import os
import shutil
from typing import Dict, Union, List
from uuid import uuid4

import h5py
import numpy as np
import pandas as pd
from scipy.constants import c, e, h, k
from scipy.integrate import trapz

from PyAstronomy import pyaC
from em_methods.lumerical.lum_helper import (
    LumMethod,
    LumericalError,
    RunLumerical,
    _get_lumerical_results,
)
from em_methods.lumerical.fdtd import fdtd_run
import lumapi


# Get module logger
logger = logging.getLogger("dev")


@dataclass(frozen=False)
class SimInfo:
    SolarGenName: Union[str, List[str]]
    GenName: Union[str, List[str]]
    SCName: Union[str, List[str]]
    Cathode: str
    Anode: str

    @property
    def SolarGenName_List(self) -> List[str]:
        return (
            self.SolarGenName
            if isinstance(self.SolarGenName, list)
            else [self.SolarGenName]
        )

    @property
    def GenName_List(self) -> List[str]:
        return self.GenName if isinstance(self.GenName, list) else [self.GenName]

    @property
    def SCName_List(self) -> List[str]:
        return self.SCName if isinstance(self.SCName, list) else [self.SCName]


""" Main functions """


def charge_run(
    basefile: str,
    properties: Dict[str, Dict[str, float]],
    get_results,
    *,
    get_info: Dict[str, str] = {},
    func=None,
    savepath: Union[None, str] = None,
    override_prefix: Union[None, str] = None,
    delete: bool = False,
    device_kw={"hide": True},
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
        LumMethod.CHARGE,
        results=results,
        log_queue=Queue(-1),
        filepath=new_filepath,
        properties=properties,
        get_results=get_results,
        get_info=get_info,
        func=func,
        lumerical_kw=device_kw,
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
    return (
        results["data"],
        results["runtime"],
        results["analysis runtime"],
        results["data_info"],
    )


def charge_run_analysis(basefile: str, get_results, device_kw={"hide": True}):
    """
    Generic function to gather simulation data from already simulated files
    Args:
        basefile: Path to the original file
        get_results: Dictionary with the results to extract

    Return:
        results: Dictionary with all the results
    """
    with lumapi.DEVICE(filename=basefile, **device_kw) as charge:
        results = _get_lumerical_results(charge, get_results)
        charge.close()
    return results


def extract_iv_data(results, names):
    """
    Obtains the performance metrics of a solar cell
    Args:
            results: Dictionary with all the results from the charge_run function
            names: SimInfo dataclass structure about the simulation (e.g. SimInfo("solar_generation_PVK", "G_PVK.mat", "Perovskite", "ITO_top", "ITO"))
            regime: "am" or "dark" for illuminated IV or dark IV
    Returns:
            If the curve is illuminated: PCE, FF, Voc, Jsc, current_density, voltage, stop, P
            OR
            If the curve is dark: current_density, voltage
    """

    current = np.array(results["results.CHARGE." + str(names.Cathode)]["I"])
    voltage = np.array(
        results["results.CHARGE." + str(names.Cathode)]["V_" + str(names.Cathode)]
    )
    Lx = results["func_output"][0]
    Ly = results["func_output"][1]

    return current, voltage, Lx, Ly


def iv_curve(
    voltage, current=None, Lx=None, Ly=None, regime="am", current_density=None
):
    if current is None and current_density is None:
        raise ValueError("Either current or current_density must be provided.")

    if current_density is not None:
        if current_density[0] > 0:
            current_density = np.array(current_density) * -1
        current_density = np.array(current_density)

    else:
        if Lx is None or Ly is None:
            raise ValueError(
                "Lx and Ly must be provided when current_density is not given."
            )
        Lx = Lx * 100  # from m to cm
        Ly = Ly * 100  # from m to cm
        area = Ly * Lx  # area in cm^2
        current_density = (np.array(current) * 1000) / area  # mA/cm2

    if (
        len(current_density) == 1 and len(current_density[0]) != 1
    ):  # CHARGE I output is not always consistent
        current_density = current_density[0]
        voltage = [float(arr[0]) for arr in voltage]
    elif voltage.ndim == 2:
        voltage = [float(arr[0]) for arr in voltage]
    if current_density.ndim == 2:
        current_density = [arr[0] for arr in current_density]

    Ir = 1000  # W/m²
    if regime == "am":
        abs_voltage_min = min(np.absolute(voltage))  # volatage value closest to zero
        if abs_voltage_min in voltage:
            Jsc = current_density[np.where(voltage == abs_voltage_min)[0][0]]
        elif -abs_voltage_min in voltage:
            Jsc = current_density[np.where(voltage == -abs_voltage_min)[0][0]]

        voltage = np.array(voltage)
        current_density = np.array(current_density)
        is_unordered = not np.all(voltage[:-1] <= voltage[1:])

        if is_unordered:  # if it is unordered, then it will be ordered
            sorted_indices_voltage = np.argsort(voltage)
            voltage = voltage[sorted_indices_voltage]
            current_density = current_density[sorted_indices_voltage]
            _, unique_indices = np.unique(voltage, return_index=True)
            voltage = voltage[unique_indices]
            current_density = current_density[unique_indices]
        Voc, stop = pyaC.zerocross1d(voltage, current_density, getIndices=True)

        try:
            stop = stop[0]
            Voc = Voc[0]
        except IndexError:
            stop = np.nan
            Voc = np.nan
        try:
            vals_v = np.linspace(min(voltage), max(voltage), 100)
            new_j = np.interp(vals_v, voltage, current_density)
            P = [vals_v[x] * abs(new_j[x]) for x in range(len(vals_v)) if new_j[x] < 0]
            FF = abs(max(P) / (Voc * Jsc))
            PCE = ((FF * Voc * abs(Jsc) * 10**-3) / (Ir * (10**-4))) * 100
        except ValueError:
            P = np.nan
            FF = np.nan
            PCE = np.nan

        return PCE, FF, Voc, Jsc, current_density, voltage

    elif regime == "dark":
        return current_density, voltage


def IQE(
    active_region_list,
    properties,
    charge_file,
    path,
    fdtd_file,
    wl=[i for i in range(300, 1001, 50)],
    min_edge=None,
    avg_mode=False,
    B=None,
):
    current_jsc = [[] for _ in range(len(active_region_list))]
    Jsc_g = [[] for _ in range(len(active_region_list))]
    Jph_g = [[] for _ in range(len(active_region_list))]
    results_dir = os.path.join(
        path, "IQE_results_" + str(os.path.splitext(fdtd_file)[0])[:-3]
    )  # remove the sufix _qe from the file's name
    os.makedirs(results_dir, exist_ok=True)

    for wvl in wl:
        current_Jsc, current_Jph = run_fdtd_and_charge_EQE(
            active_region_list,
            properties,
            charge_file,
            path,
            fdtd_file,
            wvl * 10**-9,
            min_edge=min_edge,
            avg_mode=avg_mode,
            B=B,
        )
        for i in range(len(active_region_list)):
            Jsc_g[i].append(current_Jsc[i])
            Jph_g[i].append(current_Jph[i])
            if np.isnan(current_Jph[i]) or current_Jph[i] == 0:
                print(
                    f"Warning: Jph[{i}] is NaN or 0 at wavelength {wvl}, setting IQE to NaN"
                )
                iqe_value = np.nan
            else:
                iqe_value = -current_Jsc[i] / current_Jph[i]
                print(f"Computed IQE[{i}] at {wvl} nm: {iqe_value}")

            # Ensure IQE is saved as a scalar
            iqe_scalar = (
                iqe_value[0] if isinstance(iqe_value, (list, np.ndarray)) else iqe_value
            )
            current_jsc[i] = (
                current_Jsc[i][0]
                if isinstance(current_Jsc[i], (list, np.ndarray))
                else current_Jsc[i]
            )
            # Save IQE
            result_file = os.path.join(
                results_dir, f"IQE_{active_region_list[i].SCName}.csv"
            )
            data = pd.DataFrame(
                {
                    "wavelength": [wvl],
                    "IQE": [iqe_scalar],
                    "Jsc": [-current_jsc[i][0]],
                    "Jph": [current_Jph[i]],
                }
            )
            if os.path.exists(result_file):
                data.to_csv(result_file, mode="a", header=False, index=False)
            else:
                data.to_csv(result_file, index=False, header=True)

    IQE_values = [
        [-Jsc_g[i][j] / Jph_g[i][j] for j in range(len(wl))]
        for i in range(len(active_region_list))
    ]
    return IQE_values, Jsc_g, Jph_g, wl


def IQE_tandem(path, fdtd_file, active_region_list, properties, run_abs: bool = True):
    """
    Calculates the total internal quantum efficiency (IQE) and total absorption for a tandem solar cell configuration.
    The function extracts absorption data, interpolates it to a common wavelength grid, and then processes IQE data
    from previously computed results in imbedded folder called "IQE_results + fdtd_file name".

    Args:
            path: directory where the FDTD and CHARGE files exist.
            fdtd_file: String FDTD file name.
            active_region_list: list with SimInfo dataclasses containing the details of each active region in the simulation
                                (e.g. [SimInfo("solar_generation_Si", "G_Si.mat", "Si", "AZO", "ITO_bottom"),
                                      SimInfo("solar_generation_PVK", "G_PVK.mat", "Perovskite", "ITO_top", "ITO")]).
            properties: Dictionary with the property object and property names and values.
            run_abs: Boolean flag indicating whether to recompute absorption before processing IQE data.

    Returns:
            total_iqe: 1D numpy array containing the minimum IQE values across all active regions at each wavelength.
            total_abs: 1D numpy array containing the total absorption across all active regions.
            all_wvl_new[0]: 1D numpy array representing the common wavelength grid used for interpolation.
    """
    all_abs, all_wvl = [], []
    all_iqe, all_iqe_wvl = [], []
    all_wvl_new = []
    min_max_wvl = float("inf")
    for names in active_region_list:
        if run_abs:
            abs_extraction(
                names, path, fdtd_file, properties=properties
            )  # will calc the abs
        results_path = os.path.join(path, names.SolarGenName)
        abs_data = pd.read_csv(results_path + ".csv")
        all_abs.append(abs_data["abs"])
        all_wvl.append(abs_data["wvl"])
    for i, _ in enumerate(active_region_list):
        if max(all_wvl[i]) < min_max_wvl:
            min_max_wvl = max(all_wvl[i])
    for i, _ in enumerate(active_region_list):  # all abs should have the same size
        all_wvl_new.append(
            np.linspace(min(all_wvl[i]) * 10**9, min_max_wvl * 10**9, 1001)
        )
        all_abs[i] = np.interp(all_wvl_new[i], all_wvl[i] * 10**9, all_abs[i])
    total_abs = sum(all_abs)
    iqe_path = os.path.join(path, "IQE_results_" + str(os.path.splitext(fdtd_file)[0]))
    for names in active_region_list:
        results_path = os.path.join(iqe_path, "IQE_" + names.SCName)
        iqe_data = pd.read_csv(results_path + ".csv")
        all_iqe.append(iqe_data["IQE"])
        all_iqe_wvl.append(iqe_data["wavelength"])
    for i, _ in enumerate(active_region_list):
        all_iqe[i] = np.interp(all_wvl_new[i], all_iqe_wvl[i], all_iqe[i])
    all_iqe = np.array(all_iqe)  # Convert list of arrays to 2D NumPy array
    total_iqe = np.min(all_iqe, axis=0)  # min IQE at each wavlength
    return total_iqe, total_abs, all_wvl_new[0], all_iqe, all_abs


def _import_generation(gen_file):
    """Import the necessary data
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
    y_gen = trapz(gen, y, axis=2) / norm_y
    norm_area = norm_y * (np.max(x) - np.min(x))
    xy_gen = trapz(trapz(gen, x, axis=1), y, axis=1) / norm_area

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


def __prepare_gen(fdtd_handler, active_regions, override_freq):
    """Function to preprocess the files necessary in get_gen function"""
    for names in active_regions:
        for gen_obj, file in zip(names.SolarGenName_List, names.GenName_List):
            g_name = file.replace(".mat", "")
            fdtd_handler.select(gen_obj)
            fdtd_handler.set("export filename", g_name)
        if override_freq is not None:
            fdtd_handler.setglobalsource("wavelength start", override_freq)
            fdtd_handler.setglobalsource("wavelength stop", override_freq)


def get_gen(
    fdtd_file: str,
    properties,
    active_regions: List[SimInfo],
    avg_mode: bool = False,
    override_freq: Union[None, float] = None,
):
    """
    Alters the cell design ("properties"), simulates the FDTD file, and creates the generation rate .mat file(s)
    (in same directory as FDTD file)
    Args:
        fdtd_file: String FDTD file name
        properties: Dictionary with the property object and property names and values
        active_regions: list with SimInfo dataclassses with the details of the simulations
            [SimInfo("solar_generation_Si", "G_Si.mat", "Si", "AZO", "ITO_bottom")]
        avg_mode: bool that determines whether or not the generation rate is averaged in y (necessary for light-trapping)
        override_freq: Override frequencies in the file (necessary for QE calculations)
    """
    basepath, basename = os.path.split(fdtd_file)
    results = {"data": {}, "results": {}}
    for active_region in active_regions:
        results["data"].update(
            {
                active_subregion: "Jsc"
                for active_subregion in active_region.SolarGenName_List
            }
        )
        results["results"].update(
            {
                active_subregion: "Pabs_total"
                for active_subregion in active_region.SolarGenName_List
            }
        )
    logger.debug(f"Results to obtain in get_gen: {results}")
    res, *_ = fdtd_run(
        fdtd_file,
        properties,
        results,
        func=__prepare_gen,
        delete=True,
        **{"active_regions": active_regions, "override_freq": override_freq},
    )
    logger.debug(f"Get Gen run_fdtd results: {res}")
    if avg_mode:
        logger.debug("Averaged Genration -> 3d to 2d")
        for active_region in active_regions:
            for genregion in active_region.GenName_List:
                generation_name = os.path.join(basepath, genregion)
                logger.debug(f"New Generation Name: {generation_name}")
                gen_data, x, y, z = _import_generation(generation_name)
                y_gen, _ = _average_generation(gen_data, x.flatten(), y.flatten())
                y_gen_3d = y_gen + np.zeros((len(x), len(z), len(y))
                export_3d_averaged_data = {
                    "G": np.transpose(y_gen_3d, (1, 0, 2)),
                    "x": x,
                    "y": y,
                    "z": z,
                }
                _export_hdf_data(
                    generation_name, **export_3d_averaged_data
                )
    return res


def __set_iv_parameters(
    charge,
    bias_regime: str,
    name: SimInfo,
    v_max,
    method_solver,
    def_sim_region=None,
    B=None,
    v_single_point=None,
    generation: str = None,
    min_edge=None,
    range_num_points=101,
):
    """
    Imports the generation rate into new CHARGE file, creates the simulation region based on the generation rate,
    sets the iv curve parameters and ensure correct solver is selected (e.g. start range, stop range...).
    Has the possibility to prepare the CHAGE file for a full IV curve ending at v_max Volts or a single voltage point (v_single_point).
    Can also input the radiative recombination value in the semiconductor material defined by "SCName" in the SimInfo dataclass.
    Args:
            charge: as in "with lumapi.DEVICE(...) as charge"
            bias_regime: (str) "forward" or "reverse" bias regime
            name: SimInfo dataclass structure about the simulation (e.g. SimInfo("solar_generation_PVK", "G_PVK.mat", "Perovskite", "ITO_top", "ITO"))
            path: (str) CHARGE file directory
            v_max: (float) determines the maximum voltage calculated in the IV curve
            method_solver: (str) defines de method for solving the drift diffusion equations in CHARGE: "GUMMEL" or "NEWTON"
            def_sim_region: (str) input that defines if it is necessary to create a new simulation region. Possible input values include '2D', '2d', '3D', '3d'.
                        A simulation region will be defined accordingly to the specified dimentions. If no string is introduced then no new simulation region
                        will be created
            B: (float) Radiative recombination coeficient. By default it is None.
            v_single_point: (float) If anything other than None, overrides v_max and the current response of the cell is calculated at v_single_point Volts.
            generation: (str) Toogles on or off if the generation is deleted from the CHARGE file when running with a single point - useful for getting
                    illuminated and dark band diagrams at a set voltage
    Returns:

            Lx,Ly: (float) dimentions of solar cell surface area normal to the incident light direction in meters
    """

    valid_dimensions = {"2d", "3d"}
    if def_sim_region is not None and def_sim_region.lower() not in valid_dimensions:
        raise LumericalError(
            "def_sim_region must be one of '2D', '2d', '3D', '3d' or have no input"
        )
    charge.switchtolayout()
    if isinstance(name.GenName, list):
        print("2T")
        terminal = 2
        for i in range(
            0, len(name.GenName)
        ):  # ADDS ALL THE GENERATIONS IN THE REGION LIST ARRAY
            charge.addimportgen()
            charge.set("name", str(name.GenName[i][:-4]))
            # Import generation file path
            charge.set("volume type", "solid")
            charge.set("volume solid", str(name.SCName[i]))
            charge.importdataset(name.GenName[i])
            charge.save()
    else:
        print("4T")
        terminal = 4
        charge.addimportgen()
        charge.set("name", str(name.GenName[:-4]))
        print(str(name.GenName[:-4]))
        # Import generation file path
        charge.set("volume type", "solid")
        charge.set("volume solid", str(name.SCName))
        charge.importdataset(name.GenName)
        charge.save()
    if def_sim_region is not None:
        # Defines boundaries for simulation region
        charge.select("geometry::" + name.Anode)
        z_max = charge.get("z max")
        charge.select("geometry::" + name.Cathode)
        z_min = charge.get("z min")
        if terminal == 2:
            charge.select(
                "CHARGE::" + str(name.GenName[0][:-4])
            )  # ALL SIMULATIION REGIONS SHOULD HAVE THE SAME THICKENESS, SELECTING [0] IS THE SAME AS ANY OTHER
        elif terminal == 4:
            charge.select("CHARGE::" + str(name.GenName[:-4]))
        x_span = charge.get("x span")
        x = charge.get("x")
        y_span = charge.get("y span")
        y = charge.get("y")
        # Creates the simulation region (2D or 3D)
        # charge.addsimulationregion()

        if terminal == 2:
            sim_name = "2Terminal"
        elif terminal == 4:
            sim_name = name.SCName
        charge.addsimulationregion()
        charge.set("name", sim_name)

        if "2" in def_sim_region:
            charge.set("dimension", "2D Y-Normal")
            charge.set("x", x)
            charge.set("x span", x_span)
            charge.set("y", y)
        elif "3" in def_sim_region:
            charge.set("dimension", "3D")
            charge.set("x", x)
            charge.set("x span", x_span)
            charge.set("y", y)
            charge.set("y span", y_span)
        charge.select(str(sim_name))
        charge.set("z max", z_max)
        charge.set("z min", z_min)
        charge.select("CHARGE")
        charge.set("simulation region", sim_name)
        charge.save()
    # Defining solver parameters
    if bias_regime == "forward":
        charge.select("CHARGE")
        charge.set("solver type", method_solver)
        charge.set("enable initialization", True)
        if min_edge is not None:
            charge.set("min edge length", min_edge)
        # charge.set("init step size",1) #unsure if it works properly
        charge.save()
        # Setting sweep parameters
        charge.select("CHARGE::boundary conditions::" + str(name.Cathode))
        if v_single_point is not None:
            charge.set("sweep type", "single")
            charge.save()
            charge.set("voltage", v_single_point)
            charge.save()
            print("single")
            if terminal == 2 and generation is not None:
                for i in range(0, len(name.GenName)):
                    charge.select("CHARGE::" + str(name.GenName[i][:-4]))
                    charge.delete()
                    charge.save()
            elif terminal == 4 and generation is not None:
                charge.select("CHARGE::" + str(name.GenName[:-4]))
                charge.delete()
                charge.save()
        else:
            charge.set("sweep type", "range")
            charge.save()
            charge.set("range start", 0)
            charge.set("range stop", v_max)
            charge.set("range num points", range_num_points)
            charge.set("range backtracking", "enabled")
            charge.save()
    # Reverse bias regime
    elif bias_regime == "reverse":
        charge.select("CHARGE")
        charge.set("solver type", "GUMMEL")
        charge.set("enable initialization", False)
        charge.save()
        # Setting sweep parameters
        charge.select("CHARGE::boundary conditions::" + str(name.Cathode))
        charge.set("sweep type", "range")
        charge.save()
        charge.set("range start", 0)
        charge.set("range stop", -1)
        charge.set("range num points", 21)
        charge.set("range backtracking", "enabled")
        charge.save()
        # Determining simulation region dimentions
    if def_sim_region is not None:
        sim_region = sim_name
    else:
        sim_region = charge.getnamed("CHARGE", "simulation region")
    charge.select(sim_region)
    Lx = charge.get("x span")
    if "3D" not in charge.get("dimension"):
        charge.select("CHARGE")
        Ly = charge.get("norm length")
    else:
        charge.select(str(sim_region))
        Ly = charge.get("y span")
    if B is not None:
        charge.select("materials::" + name.SCName + "::" + name.SCName)
        charge.set("recombination.radiative.copt.constant", B)
        charge.save()
    return Lx, Ly

def run_fdtd_and_charge(active_region_list, properties, charge_file, path, fdtd_file, v_max = 1.5, run_FDTD = True, def_sim_region=None, save_csv = False, B = None,  method_solver = "NEWTON", v_single_point = None, avg_mode: bool = False, min_edge= None, range_num_points = 101):
    """ 
    Runs the FDTD and CHARGE files for the multiple active regions defined in the active_region_list
    It utilizes helper functions for various tasks like running simulations, extracting IV curve performance metrics PCE, FF, Voc, Jsc
    Args:
            active_region_list: list with SimInfo dataclassses with the details of the simulation 
                            (e.g. [SimInfo("solar_generation_Si", "G_Si.mat", "Si", "AZO", "ITO_bottom"),
                                    SimInfo("solar_generation_PVK", "G_PVK.mat", "Perovskite", "ITO_top", "ITO")])
            properties: (Dictionary) with the property object and property names and values                    
            charge_file: (str) name of CHARGE file
            path: (str) directory where the FDTD and CHARGE files exist
            fdtd_file: (str)  name of FDTD file
            v_max: (float) determines the maximum voltage calculated in the IV curve
            run_FDTD: (bool) that determines whether or not a new Generation profile is created
            def_sim_region: (str) input that defines if it is necessary to create a new simulation region. Possible input values include '2D', '2d', '3D', '3d'.
                        A simulation region will be defined accordingly to the specified dimentions. If no string is introduced then no new simulation region 
                        will be created
            save_csv: (bool) determines wether or not the Current Density and Voltage (simulation outputs) are saved in csv file with name: "{names.SCName}_IV_curve.csv" 
            B: (float) Radiative recombination coeficient. By default it is None. Input options: 
                        -None: Simulation will use values in the charge files.
                        -[value1, value2, ...]: Simulation will use the values in the B array. Note that value1 will correspond to the 1st object in the active_region_list array
                        -[value1, True, ...]: Simulation will calculate the B value based on the current FTDT file for the second object in the active_region_list array
            method_solver: (str) defines de method for solving the drift diffusion equations in CHARGE: "GUMMEL" or "NEWTON" 
            v_single_point: (float) If anything other than None, overrides v_max and the current response of the cell is calculated at v_single_point Volts.
    Returns:
            PCE, FF, Voc, Jsc, Current_Density, Voltage: np.arrays with the power convercion efficiency[%], fill factor[0-1], open circuit voltage[V], 
                        short cirucit current[mA/cm2], current density array [mA/cm2], voltage array[V] for all the materials in the active_region_list. 
                        (e.g. if the active_region_list has 2 materials: PCE = [pce1, pce2], ... Voltage = [[...],[...]] )
            
    """ 
    
    pce_array, ff_array, voc_array, jsc_array, current_density_array, voltage_array = [], [], [], [], [], []
    valid_solver = {"GUMMEL", "NEWTON"}
    if method_solver.upper() not in valid_solver:
        raise LumericalError("method_solver must be 'GUMMEL' or 'NEWTON' or any case variation, or have no input")
    charge_path = os.path.join(path, charge_file)
    if run_FDTD:
        _ = get_gen(path, fdtd_file, properties, active_region_list, avg_mode = avg_mode)
    if B == None:
        B = [None for _ in range(0, len(active_region_list))]
    if min_edge == None:
        min_edge = [None for _ in range(0, len(active_region_list))]
    if not isinstance(v_max, (list, np.ndarray)):
        v_max = [v_max for _ in range(0, len(active_region_list))] #if the v_max is just a float then it assumes that scalar for all the active regions
    if not isinstance(range_num_points, (list, np.ndarray)):
        range_num_points = [int(range_num_points) for _ in range(0, len(active_region_list))]
    results = None
    for names in active_region_list:
        if B[active_region_list.index(names)] == True: #checks if it will calculate B for that object
            B[active_region_list.index(names)] = extract_B_radiative([names], path, fdtd_file, charge_file, properties = properties , run_abs = False)[0] #B value is calculated based on last FDTD for that index
            print(f'The B values: {B}')
        conditions_dic = {"bias_regime":"forward","name": names, "v_max": v_max[active_region_list.index(names)],"def_sim_region":def_sim_region,"B":B[active_region_list.index(names)], 
                          "method_solver": method_solver.upper(), "v_single_point": v_single_point, "min_edge": min_edge[active_region_list.index(names)], "range_num_points" : range_num_points[active_region_list.index(names)]  }
        get_results = {"results": {"CHARGE": str(names.Cathode)}}  # get_results: Dictionary with the properties to be calculated
        try:
            start_time = time.time()
            results = charge_run(charge_path, properties, get_results, 
                                func= __set_iv_parameters, delete = True, device_kw={"hide": True},**conditions_dic)
            run_time = time.time() - start_time
        except LumericalError:
            try: 
                start_time = time.time()          
                logger.warning("Retrying simulation")
                results = charge_run(charge_path, properties, get_results, 
                               func= __set_iv_parameters, delete = True,  device_kw={"hide": True} ,**conditions_dic)
                run_time = time.time() - start_time
            except LumericalError:
                pce, ff, voc, jsc, current_density, voltage = (np.nan for _ in range(6))
                pce_array.append(pce)
                ff_array.append(ff)
                voc_array.append(voc)
                jsc_array.append(jsc)
                current_density_array.append(current_density)
                voltage_array.append(voltage)
                print(
                    f"Semiconductor {names.SCName}, cathode {names.Cathode}\n Voc = {voc_array[-1]:.3f}V \n Jsc =  {jsc_array[-1]:.4f} mA/cm² \n FF = {ff_array[-1]:.3f} \n PCE = {pce_array[-1]:.3f}%"
                )
                continue

        current, voltage, Lx, Ly = extract_iv_data(results[0], names)
        pce, ff, voc, jsc, current_density, voltage = iv_curve(
            voltage, current=current, Lx=Lx, Ly=Ly
        )

        if np.isnan(voc) and not np.isnan(jsc):
            print(
                f"V_max = {max_volt[active_regions.index(names)]} V, which might be too small. Trying {max_volt[active_regions.index(names)]+0.2} V"
            )
            max_volt[active_regions.index(names)] = (
                max_volt[active_regions.index(names)] + 0.2
            )

            conditions_dic = {
                "bias_regime": "forward",
                "name": names,
                "v_max": max_volt[active_regions.index(names)],
                "def_sim_region": def_sim_region,
                "B": rad_recombination_coeff[active_regions.index(names)],
                "method_solver": charge_solver.upper(),
                "v_single_point": v_single_point,
                "min_edge": min_edge[active_regions.index(names)],
                "range_num_points": range_num_points[active_regions.index(names)],
            }
            get_results = {
                "results": {"CHARGE": str(names.Cathode)}
            }  # get_results: Dictionary with the properties to be calculated
            try:
                start_time = time.time()
                results = charge_run(
                    charge_path,
                    properties,
                    get_results,
                    func=__set_iv_parameters,
                    delete=True,
                    device_kw={"hide": True},
                    **conditions_dic,
                )
                run_time = time.time() - start_time
            except LumericalError:
                pce, ff, voc, jsc, current_density, voltage = (np.nan for _ in range(6))
                pce_array.append(pce)
                ff_array.append(ff)
                voc_array.append(voc)
                jsc_array.append(jsc)
                current_density_array.append(current_density)
                voltage_array.append(voltage)
                print(
                    f"Semiconductor {names.SCName}, cathode {names.Cathode}\n Voc = {voc_array[-1]:.3f}V \n Jsc =  {jsc_array[-1]:.4f} mA/cm² \n FF = {ff_array[-1]:.3f} \n PCE = {pce_array[-1]:.3f}%"
                )
                continue

            current, voltage, Lx, Ly = extract_iv_data(results[0], names)
            pce, ff, voc, jsc, current_density, voltage = iv_curve(
                voltage, current=current, Lx=Lx, Ly=Ly
            )

        pce_array.append(pce)
        ff_array.append(ff)
        voc_array.append(voc)
        jsc_array.append(jsc)
        current_density_array.append(current_density)
        voltage_array.append(voltage)
        print(
            f"Semiconductor {names.SCName}, cathode {names.Cathode}\n Voc = {voc_array[-1]:.3f}V \n Jsc =  {jsc_array[-1]:.4f} mA/cm² \n FF = {ff_array[-1]:.3f} \n PCE = {pce_array[-1]:.3f}% \n Run time = {run_time/60} mins "
        )
        if save_csv:
            df = pd.DataFrame({"Current_Density": current_density, "Voltage": voltage})
            csv_path = os.path.join(basepath, f"{names.SCName}_IV_curve.csv")
            df.to_csv(csv_path, sep="\t", index=False)

    return (
        pce_array,
        ff_array,
        voc_array,
        jsc_array,
        current_density_array,
        voltage_array,
    )


def run_fdtd_and_charge_EQE(
    active_region_list,
    properties,
    charge_file,
    path,
    fdtd_file,
    freq,
    def_sim_region="2d",
    B=None,
    method_solver="GUMMEL",
    v_single_point=0,
    v_max=0,
    min_edge=None,
    avg_mode: bool = False,
):
    """
    UNTESTED


    Runs the FDTD and CHARGE files for the multiple active regions defined in the active_region_list
    It utilizes helper functions for various tasks like running simulations, extracting IV curve performance metrics PCE, FF, Voc, Jsc
    Args:
            active_region_list: list with SimInfo dataclassses with the details of the simulation
                            (e.g. [SimInfo("solar_generation_Si", "G_Si.mat", "Si", "AZO", "ITO_bottom"),
                                    SimInfo("solar_generation_PVK", "G_PVK.mat", "Perovskite", "ITO_top", "ITO")])
            properties: Dictionary with the property object and property names and values
            charge_file: name of CHARGE file
            path: (str) directory where the FDTD and CHARGE files exist
            fdtd_file: (str)  name of FDTD file
            freq: (float) incident photon frequency[Hz]
            def_sim_region: (str) input that defines if it is necessary to create a new simulation region. Possible input values include '2D', '2d', '3D', '3d'.
                        A simulation region will be defined accordingly to the specified dimentions. If no string is introduced then no new simulation region
                        will be created
            B: (float) Radiative recombination coeficient. By default it is None.
            method_solver: (str) defines de method for solving the drift diffusion equations in CHARGE: "GUMMEL" or "NEWTON"
            v_single_point: (float) If anything other than None, overrides v_max and the current response of the cell is calculated at v_single_point Volts.

    Returns:
            Jsc: array with the dimention of the active_region_list, containing the CHARGE output for a freq in A/m2
            Jph: array with the dimention of the active_region_list, containing the FDTD output for a freq in A/m2

    """
    valid_solver = {"GUMMEL", "NEWTON"}
    if method_solver.upper() not in valid_solver:
        raise LumericalError(
            "method_solver must be 'GUMMEL' or 'NEWTON' or any case variation, or have no input"
        )
    charge_path = os.path.join(path, charge_file)
    if B == None:
        B = [None for _ in range(0, len(active_region_list))]
    if min_edge == None:
        min_edge = [None for _ in range(0, len(active_region_list))]
    if not isinstance(v_single_point, (list, np.ndarray)):
        v_single_point = [v_single_point for _ in range(0, len(active_region_list))]
    if not isinstance(v_max, (list, np.ndarray)):
        v_max = [v_max for _ in range(0, len(active_region_list))]
    Jsc = []
    charge_path = os.path.join(path, charge_file)
    Jph = get_gen(
        path,
        fdtd_file,
        properties,
        active_region_list,
        avg_mode=avg_mode,
        override_freq=freq,
        quantum_efficiency=True,
    )  # A/m2
    results = None
    for names in active_region_list:
        B_temp = [B[i] for i in range(0, len(B))]
        if (
            B_temp[active_region_list.index(names)] == True
        ):  # checks if it will calculate B for that object
            B_temp[active_region_list.index(names)] = extract_B_radiative(
                [names],
                path,
                fdtd_file,
                charge_file,
                properties=properties,
                run_abs=False,
            )[
                0
            ]  # B value is calculated based on last FDTD for that index
            print(f"The B values: {B_temp}")
        conditions_dic = {
            "bias_regime": "forward",
            "name": names,
            "v_max": v_max[active_region_list.index(names)],
            "def_sim_region": def_sim_region,
            "B": B_temp[active_region_list.index(names)],
            "method_solver": method_solver.upper(),
            "v_single_point": v_single_point[active_region_list.index(names)],
            "min_edge": min_edge[active_region_list.index(names)],
        }
        get_results = {
            "results": {"CHARGE": str(names.Cathode)}
        }  # get_results: Dictionary with the properties to be calculated
        try:
            results = charge_run(
                charge_path,
                properties,
                get_results,
                func=__set_iv_parameters,
                delete=True,
                device_kw={"hide": True},
                **conditions_dic,
            )
        except LumericalError:
            try:
                logger.warning("Retrying simulation")
                results = charge_run(
                    charge_path,
                    properties,
                    get_results,
                    func=__set_iv_parameters,
                    delete=True,
                    device_kw={"hide": True},
                    **conditions_dic,
                )
            except LumericalError:
                current_density = np.nan
                Jsc.append(current_density)
                continue
        current = results[0]["results.CHARGE." + str(names.Cathode)]["I"][0]  # A
        Lx = results[0]["func_output"][0]
        Ly = results[0]["func_output"][1]
        Lx = Lx * 100  # from m to cm
        Ly = Ly * 100  # from m to cm
        area = Ly * Lx  # area in cm^2
        current_density = (np.array(current) * 1000) / area  # mA/cm2
        # print(current_density)
        Jsc.append(current_density * 10)  # A/m2
    return Jsc, Jph  # arrays with information about all active materials


def run_fdtd_and_charge_multi(
    active_region_list,
    properties,
    charge_file,
    path,
    fdtd_file,
    times_to_run=3,
    v_max=1.5,
    run_FDTD=True,
    def_sim_region=None,
    save_csv=False,
    B=None,
    method_solver="NEWTON",
    v_single_point=None,
):  # needs to be updated
    """
    UNTESTED


    Runs the FDTD and CHARGE files for the multiple active regions defined in the active_region_list multiple times (3)
    It utilizes helper functions for various tasks like running simulations, extracting IV curve performance metrics PCE, FF, Voc, Jsc
    Args:
            active_region_list: list with SimInfo dataclassses with the details of the simulation
                            (e.g. [SimInfo("solar_generation_Si", "G_Si.mat", "Si", "AZO", "ITO_bottom"),
                                    SimInfo("solar_generation_PVK", "G_PVK.mat", "Perovskite", "ITO_top", "ITO")])
            properties: (Dictionary) with the property object and property names and values
            charge_file: (str) name of CHARGE file
            path: (str) directory where the FDTD and CHARGE files exist
            fdtd_file: (str)  name of FDTD file
            v_max: (float) determines the maximum voltage calculated in the IV curve
            run_FDTD: (bool) that determines whether or not a new Generation profile is created
            def_sim_region: (str) input that defines if it is necessary to create a new simulation region. Possible input values include '2D', '2d', '3D', '3d'.
                        A simulation region will be defined accordingly to the specified dimentions. If no string is introduced then no new simulation region
                        will be created
            save_csv: (bool) determines wether or not the Current Density and Voltage (simulation outputs) are saved in csv file with name: "{names.SCName}_IV_curve.csv"
            B: (float) Radiative recombination coeficient. By default it is None.
            method_solver: (str) defines de method for solving the drift diffusion equations in CHARGE: "GUMMEL" or "NEWTON"
            v_single_point: (float) If anything other than None, overrides v_max and the current response of the cell is calculated at v_single_point Volts.
    Returns:
            PCE, FF, Voc, Jsc: array of np.arrays with active_region_list size with the averaged (after times_to_run times) power convercion efficiency[%], fill factor[0-1], open circuit voltage[V],
                        short cirucit current[mA/cm2], for all the materials in the active_region_list.
                        (e.g. if the active_region_list has 2 materials: PCE = [pce1_avg, pce2_avg] )
    """
    (
        PCE,
        FF,
        Voc,
        Jsc,
        PCE_temp,
        FF_temp,
        Voc_temp,
        Jsc_temp,
        Current_Density,
        Voltage,
    ) = ([], [], [], [], [], [], [], [], [], [])
    charge_path = os.path.join(path, charge_file)
    get_gen(path, fdtd_file, properties, active_region_list)
    results = None
    get_results = {
        "results": {"CHARGE": str(names.Cathode)}
    }  # get_results: Dictionary with the properties to be calculated
    for names in active_region_list:
        for i in range(times_to_run):
            try:
                results = charge_run(
                    charge_path,
                    properties,
                    get_results,
                    func=__set_iv_parameters,
                    delete=True,
                    device_kw={"hide": True},
                    **{
                        "bias_regime": "forward",
                        "name": names,
                        "v_max": v_max,
                        "def_sim_region": def_sim_region,
                        "B": B[active_region_list.index(names)],
                        "method_solver": method_solver,
                        "v_single_point": v_single_point,
                    },
                )
            except LumericalError:
                try:
                    logger.warning("Retrying simulation")
                    results = charge_run(
                        charge_path,
                        properties,
                        get_results,
                        func=__set_iv_parameters,
                        delete=True,
                        device_kw={"hide": True},
                        **{
                            "bias_regime": "forward",
                            "name": names,
                            "v_max": v_max,
                            "def_sim_region": def_sim_region,
                            "B": B[active_region_list.index(names)],
                            "method_solver": method_solver,
                            "v_single_point": v_single_point,
                        },
                    )
                except LumericalError:
                    pce, ff, voc, jsc, current_density, voltage, stop, p = (
                        np.nan for _ in range(8)
                    )
                    PCE_temp.append(pce)
                    FF_temp.append(ff)
                    Voc_temp.append(voc)
                    Jsc_temp.append(jsc)
                    # Current_Density.append(current_density)
                    # Voltage.append(voltage)
                    print(
                        f"Semiconductor {names.SCName}, cathode {names.Cathode}\n Voc = {Voc[-1]:.3f}V \n Jsc =  {Jsc[-1]:.4f} mA/cm² \n FF = {FF[-1]:.3f} \n PCE = {PCE[-1]:.3f}%"
                    )
                    continue
            pce, ff, voc, jsc, current_density, voltage, stop, p = iv_curve(
                results[0], "am", names
            )
            PCE_temp.append(pce)
            FF_temp.append(ff)
            Voc_temp.append(voc)
            Jsc_temp.append(jsc)
            # Current_Density.append(current_density)
            # Voltage.append(voltage)
            print(
                f"Semiconductor {names.SCName}, cathode {names.Cathode}\n Voc = {Voc[-1]:.3f}V \n Jsc =  {Jsc[-1]:.4f} mA/cm² \n FF = {FF[-1]:.3f} \n PCE = {PCE[-1]:.3f}%"
            )
        pce_temp = np.average(PCE_temp)
        ff_temp = np.average(FF_temp)
        voc_temp = np.average(Voc_temp)
        jsc_temp = np.average(Jsc_temp)
    PCE.append(pce_temp)
    FF.append(ff_temp)
    Voc.append(voc_temp)
    Jsc.append(jsc_temp)
    return PCE, FF, Voc, Jsc


def band_diagram(
    active_region_list,
    properties,
    charge_file,
    path,
    fdtd_file,
    def_sim_region=None,
    v_single_point: int = 0,
    B=None,
    method_solver="GUMMEL",
    min_edge=None,
    generation=None,
):
    """
    Extracts the band diagram in a simulation region at v_single_point Volts. ATTENTION the monitor has to be placed preemptively in CHARGE with the desired geometry and position
    Args:
            active_region_list: list with SimInfo dataclassses with the details of the simulation
                            (e.g. [SimInfo("solar_generation_Si", "G_Si.mat", "Si", "AZO", "ITO_bottom"),
                                    SimInfo("solar_generation_PVK", "G_PVK.mat", "Perovskite", "ITO_top", "ITO")])
            properties: (Dictionary) with the property object and property names and values
            charge_file: (str) name of CHARGE file
            path: (str) directory where the FDTD and CHARGE files exist
            def_sim_region: (str) input that defines if it is necessary to create a new simulation region. Possible input values include '2D', '2d', '3D', '3d'.
                        A simulation region will be defined accordingly to the specified dimentions. If no string is introduced then no new simulation region
                        will be created
            B: (float) Radiative recombination coeficient. By default it is None.
            method_solver: (str) defines de method for solving the drift diffusion equations in CHARGE: "GUMMEL" or "NEWTON"
            v_single_point: (float) If anything other than None, overrides v_max and the current response of the cell is calculated at v_single_point Volts.
    Returns:
            Thickness, Ec, Ev, Efn, Efp: array of np.arrays with active_region_list size
                        (e.g. if the active_region_list has 2 materials: Ec = [[...]], [...]] )
    """
    valid_solver = {"GUMMEL", "NEWTON"}
    if method_solver.upper() not in valid_solver:
        raise LumericalError(
            "method_solver must be 'GUMMEL' or 'NEWTON' or any case variation, or have no input"
        )
    if min_edge == None:
        min_edge = [None for _ in range(0, len(active_region_list))]
    if B == None:
        B = [None for _ in range(0, len(active_region_list))]

    charge_path = os.path.join(path, charge_file)
    Ec, Ev, Efn, Efp, Thickness = [], [], [], [], []
    for names in active_region_list:
        if (
            B[active_region_list.index(names)] == True
        ):  # checks if it will calculate B for that object
            B[active_region_list.index(names)] = extract_B_radiative(
                [names],
                path,
                fdtd_file,
                charge_file,
                properties=properties,
                run_abs=False,
            )[
                0
            ]  # B value is calculated based on last FDTD for that index
            print(f"The B values: {B}")
        conditions_dic = {
            "bias_regime": "forward",
            "name": names,
            "v_max": None,
            "def_sim_region": def_sim_region,
            "B": B[active_region_list.index(names)],
            "method_solver": method_solver.upper(),
            "v_single_point": v_single_point,
            "min_edge": min_edge[active_region_list.index(names)],
            "generation": generation,
        }
        get_results = {
            "results": {"CHARGE::monitor": "bandstructure"}
        }  # get_results: Dictionary with the properties to be calculated

        results = charge_run(
            charge_path,
            properties,
            get_results,
            func=__set_iv_parameters,
            delete=True,
            device_kw={"hide": True},
            **conditions_dic,
        )
        bandstructure = results[0]
        ec = bandstructure["results.CHARGE::monitor.bandstructure"]["Ec"].flatten()
        thickness = bandstructure["results.CHARGE::monitor.bandstructure"][
            "z"
        ].flatten()
        ev = bandstructure["results.CHARGE::monitor.bandstructure"]["Ev"].flatten()
        efn = bandstructure["results.CHARGE::monitor.bandstructure"]["Efn"].flatten()
        efp = bandstructure["results.CHARGE::monitor.bandstructure"]["Efp"].flatten()
        Ec.append(ec)
        Ev.append(ev)
        Efn.append(efn)
        Efp.append(efp)
        Thickness.append(thickness)
    return Thickness, Ec, Ev, Efn, Efp


def abs_extraction(names, path, fdtd_file, properties={}):
    """
    Extracts the Absoprtion spectrum of material in "names" and creates csv file with format: 'wvl':abs['lambda'].flatten(), 'pabs':abs['Pabs_total']
    Args:
            names: SimInfo.SCname from SimInfo dataclassses with the details of the simulation
            properties: (Dictionary) with the property object and property names and values
            path: (str) directory where the FDTD and CHARGE files exist
            fdtd_file: (str)  name of FDTD file
            properties: (Dictionary) with the property object and property names and values

    """
    fdtd_path = os.path.join(path, fdtd_file)
    override_prefix: str = str(uuid4())[0:5]
    new_fdtd_file = override_prefix + "_" + fdtd_file
    new_filepath_fdtd: str = os.path.join(path, new_fdtd_file)
    shutil.copyfile(fdtd_path, new_filepath_fdtd)
    log_file_fdtd: str = os.path.join(
        path, f"{override_prefix}_{os.path.splitext(fdtd_file)[0]}_p0.log"
    )
    with lumapi.FDTD(filename=new_filepath_fdtd, hide=True) as fdtd:
        for structure_key, structure_value in properties.items():
            fdtd.select(structure_key)
            for parameter_key, parameter_value in structure_value.items():
                fdtd.set(parameter_key, parameter_value)
        fdtd.save()
        fdtd.run()
        fdtd.runanalysis(names.SolarGenName)
        abs = fdtd.getresult(names.SolarGenName, "Pabs_total")
        fdtd.switchtolayout()
        fdtd.close()
    results = pd.DataFrame({"wvl": abs["lambda"].flatten(), "pabs": abs["Pabs_total"]})
    results_path = os.path.join(path, names.SolarGenName)
    results.to_csv(
        results_path + ".csv", header=("wvl", "abs"), index=False
    )  # saves the results in file in path
    os.remove(new_filepath_fdtd)
    # os.remove(log_file_fdtd)


# untested:
def charge_extract(names, path, charge_file, properties={}):
    """
    Extracts the Band gap, lenght and intrinsic carrier density of material in "names"
    Args:
            names: SimInfo.SCname from SimInfo dataclassses with the details of the simulation
            properties: (Dictionary) with the property object and property names and values
            path: (str) directory where the FDTD and CHARGE files exist
            charge_file: (str) name of CHARGE file
            properties: (Dictionary) with the property object and property names and values

    Returns:
            Eg:(float) Band gap  [eV]
            L:(float) [m]
            ni:(float) [1/cm3]

    """
    charge_path = os.path.join(path, charge_file)
    override_prefix: str = str(uuid4())[0:5]
    new_charge_file = override_prefix + "_" + charge_file
    new_filepath_charge: str = os.path.join(path, new_charge_file)
    shutil.copyfile(charge_path, new_filepath_charge)
    log_file_charge: str = os.path.join(
        path, f"{override_prefix}_{os.path.splitext(charge_file)[0]}_p0.log"
    )
    with lumapi.DEVICE(filename=new_filepath_charge, hide=True) as charge:
        charge.switchtolayout()
        for structure_key, structure_value in properties.items():
            charge.select(structure_key)
            for parameter_key, parameter_value in structure_value.items():
                charge.set(parameter_key, parameter_value)
        charge.save()
        charge.select("materials::" + names.SCName + "::" + names.SCName)
        T = 300
        Eg = charge.get("electronic.gamma.Eg.constant")
        mn = charge.get("electronic.gamma.mn.constant")
        mp = charge.get("electronic.gamma.mp.constant")
        mn = mn * 9.11 * 10**-31
        mp = mp * 9.11 * 10**-31
        Nc = 2 * ((2 * np.pi * mn * k * T / (h**2)) ** 1.5) * 10**-6
        Nv = 2 * ((2 * np.pi * mp * k * T / (h**2)) ** 1.5) * 10**-6
        ni = ((Nv * Nc) ** 0.5) * (np.exp(-Eg * e / (2 * k * T)))
        charge.select("geometry::" + names.SCName)
        L = charge.get("z span")  # in m
        charge.close()
    os.remove(new_filepath_charge)
    # os.remove(log_file_charge)
    return Eg, L, ni


def adjust_abs(energy, abs_data, Eg):
    """
    Function that cutsoff absorption data below bandgap. Useful when there is poor FDTD fitting to calculate the absorption.
    Args:
        energy: (array) energy values in eV
        abs_data: (array) absorption data [0-1]
        Eg: band gap [eV]
    Returns:
            array with absorption cutoff below bandgap
    """
    return [0 if e < Eg else abs_data[i] for i, e in enumerate(energy)]


def phi_bb(E):
    """
    Function that derives the blackbody spectrum at 300K in a given energy range defined by E[eV]
    Args:
        E: (array) energy in ev
    Returns:
        array with balck body spectrum at 300K
    """
    h_ev = h / e
    k_b = k / e
    return (
        (2 * np.pi * E**2)
        / (h_ev**3 * c**2)
        * (np.exp(E / (k_b * 300)) - 1) ** -1
    )  # (eV.s.m2)-1


def extract_B_radiative(
    active_region_list, path, fdtd_file, charge_file, properties={}, run_abs=True
):
    """
    Function that extracts the radiative constant (B) for each material in the active_region_list according to the SQ but with an FDTD derived absorption

    Args:
        active_region_list: list with SimInfo dataclassses with the details of the simulation
                            (e.g. [SimInfo("solar_generation_Si", "G_Si.mat", "Si", "AZO", "ITO_bottom"),
                                    SimInfo("solar_generation_PVK", "G_PVK.mat", "Perovskite", "ITO_top", "ITO")])
        properties: (Dictionary) with the property object and property names and values
        charge_file: (str) name of CHARGE file
        path: (str) directory where the FDTD and CHARGE files exist
        fdtd_file: (str) name of FDTD file
        run_abs: (bool) determines whether or not absorption need to be recalculated. For most cases it is necessary
    Returns:
        B_list: (array) with the B constant [cm3/s] for all the mateials in the active_region_list
    """
    B_list = []
    for names in active_region_list:
        results_path = os.path.join(path, names.SolarGenName)
        if run_abs:
            abs_extraction(names, path, fdtd_file, properties)
        elif not os.path.isfile(results_path + ".csv"):
            abs_extraction(names, path, fdtd_file, properties)
        else:
            abs_data = pd.read_csv(results_path + ".csv")
        abs_data = pd.read_csv(results_path + ".csv")
        Eg, L, ni = charge_extract(names, path, charge_file, properties)
        if (
            len(abs_data["wvl"]) == 1
        ):  # for when only one wvl is concidered such as the IQE and EQE measurements
            energy = 1240 / (np.array(abs_data["wvl"])[0] * 1e9)
            # print('energy',energy)
            bb_spectrum = phi_bb(energy)
            # print('bb',bb_spectrum)
            # print('abs',np.array(abs_data['abs'])[0])
            Jo = e * (-bb_spectrum * np.array(abs_data["abs"])[0])
        else:
            wvl = np.linspace(
                min(abs_data["wvl"]), max(abs_data["wvl"]), 70000
            )  # wvl in m
            abs_data = np.interp(wvl, abs_data["wvl"], abs_data["abs"])
            energy = 1240 / (wvl * 1e9)
            abs_data = adjust_abs(
                energy, abs_data, Eg
            )  # new absorption with cut off below Eg
            bb_spectrum = phi_bb(energy)
            Jo = e * np.trapz(-bb_spectrum * abs_data, energy)  # A/m2
        Jo = Jo * 0.1  # mA/cm2
        B = Jo * 10**-5 / (e * (ni**2) * (L))
        B_list.append(B)
    return B_list


def _get_replacement(lst):
    for i in range(len(lst)):
        if lst[i] < 0:
            lst[i] = 0
    return lst


def get_iv_4t(folder, pvk_v, pvk_iv, si_v, si_iv):
    """Plots the IV curve of a tandem solar cell, in 4-terminal configuration, with 2 subcells in parallel.
    Note: Bottom subcell is divided into 2 cells, connected in series, to double the VOC.
    Args:
            folder: folder where the .txt files are stored
            pvk_voltage_file: name of the PVK voltage file
            pvk_current_file: name of the PVK current file
            si_voltage_file: name of the Silicon voltage file
            si_current_file: name of the Silicon current file
    """
    # PEROVSKITE______________________
    pvk_iv = _get_replacement(np.array(pvk_iv).flatten())

    # plt.figure(figsize=(5.5,5))
    # plt.plot(pvk_v, pvk_iv, label = 'PVK subcell', c = 'steelblue')

    # Determine Voc,pvk
    Voc_pvk = pvk_v[np.where(pvk_iv == 0)[0]][0]

    # SILICON__________________________
    # plt.plot(si_v, _get_replacement(si_iv), label = 'Si subcell', c = 'yellowgreen', linestyle = '--')

    si_v = si_v * 2
    si_iv = _get_replacement(si_iv / 2)
    # plt.plot(si_v, si_iv, label = '2 series Si subcell', c = 'yellowgreen')

    # Determine Voc,Si
    Voc_si = si_v[np.where(si_iv == 0)[0]][0]

    # TANDEM____________________________

    # Determine Voc
    Voc = min(Voc_pvk, Voc_si)

    # Determine voltage
    tan_v = np.array([v for i, v in enumerate(pvk_v) if i % 2 == 0])
    tan_leng = len(
        tan_v[tan_v <= Voc]
    )  # number of elements in the array that are less than Voc

    # Determine current
    pvk_iv = [v for i, v in enumerate(pvk_iv) if i % 2 == 0]
    tandem_iv = si_iv[: len(pvk_iv)] + pvk_iv
    tan_i = []

    for i in range(len(tandem_iv)):
        if i <= (tan_leng - 1):
            tan_i.append(tandem_iv[i])
        else:
            tan_i.append(0)
    leng = min(len(tan_v), len(tan_i))

    # PLOT Tandem IV curve
    # plt.plot(tan_v[:leng], tan_i[:leng], label = 'PSTSC', c = 'crimson', linestyle='--')
    # plt.legend()
    # plt.ylabel('Current Density [mA/cm2]')
    # plt.xlabel('Voltage [V]')
    # plt.ylim(0,30)
    # plt.xlim(-0.1,2)
    # plt.grid(linestyle=':')

    # plt.savefig(os.path.join(folder, "iv_curve_4t.svg"), dpi = 300, bbox_inches = 'tight')

    # Get IV curve variables
    voltage = si_v[: len(tandem_iv)].flatten()
    current_density = tandem_iv
    Ir = 1000  # W/m²

    # DETERMINE JSC
    abs_voltage_min = min(np.absolute(voltage))  # voltage value closest to zero
    if abs_voltage_min in voltage:
        Jsc = current_density[np.where(voltage == abs_voltage_min)[0]]
        Isc = current_density[np.where(voltage == abs_voltage_min)[0]]
    elif -abs_voltage_min in voltage:
        Jsc = current_density[np.where(voltage == -abs_voltage_min)[0]]
        Isc = current_density[np.where(voltage == -abs_voltage_min)[0]]

    # DETERMINE VOC THROUGH INTERPOLATION

    vals_v = np.linspace(min(voltage), max(voltage), 100)
    new_j = np.interp(vals_v, voltage, current_density.flatten())
    P = [vals_v[x] * abs(new_j[x]) for x in range(len(vals_v)) if new_j[x] > 0]
    # calculate the power for all points [W]
    FF = abs(max(P) / (Voc * Isc))
    PCE = ((FF * Voc * abs(Isc) * 10) / (Ir)) * 100

    print(
        f"FF = {FF[0]:.2f}, PCE = {PCE[0]:.2f} %, Voc = {Voc:.2f} V, Isc = {Isc[0]:.2f} mA/cm2"
    )
    return vals_v, new_j

    # OLD CODE
    """
    def plot(PCE, FF, Voc, Jsc, current_density, voltage, stop, regime:str, P): #NOT FINISHED -> regime should not be an input, the performance metrics should be enough
  
    Plots the IV curve
    Args:
            PCE, FF, Voc, Jsc: Perfomance metrics obtained from the iv_curve funtion.  
            current_density, voltage: Arrays obtained from the iv_curve funtion
            P: Array obtained from the iv_curve funtion with the Power values through out the iv curve.
            stop: Voc position in voltage array 
            regime: "am" or "dark" for illuminated IV or dark IV
    
    fig, ax = plt.subplots()
    if regime == 'am' and stop is not np.nan and Voc is not np.nan:
        plt.plot(voltage[: stop + 2], current_density[: stop + 2], "o-")
        plt.plot(Voc, 0, "o", color="red")
        props = dict(boxstyle="round", facecolor="white", alpha=0.5)
        ax.add_patch(
                Rectangle(
                    (
                        voltage[find_index(P, max(P))],
                        current_density[find_index(P, max(P))],
                    ),
                    -voltage[find_index(P, max(P))],
                    -current_density[find_index(P, max(P))],
                    facecolor="lightsteelblue",
                )
            )
        textstr = f"Voc = {Voc:.3f}V \n Jsc =  {Jsc:.4f} mA/cm² \n FF = {FF:.3f} \n PCE = {PCE:.3f}%"
        plt.text(
            0.05,
            0.80,
            textstr,
            transform=ax.transAxes,
            fontsize=17,
            verticalalignment="top",
            bbox=props,
            )
        plt.ylabel("Current density [mA/cm²] ")
        plt.xlabel("Voltage [V]")
        plt.axhline(0, color="gray", alpha=0.3)
        plt.axvline(0, color="gray", alpha=0.3)
        plt.grid()
        plt.show()

    if regime == 'dark':
        plt.plot(voltage, current_density, "o-")
        plt.ylabel("Current density [mA/cm²] ")
        plt.xlabel("Voltage [V]")
        plt.axhline(0, color="gray", alpha=0.3)
        plt.axvline(0, color="gray", alpha=0.3)
        plt.grid()
        plt.show()
    """

    """
    
def find_index(wv_list, n):
    index = [i for i in range(0, len(wv_list)) if wv_list[i] == n]
    return index[0]
    """
