from dataclasses import dataclass
import logging
from multiprocessing import Manager, Queue
import os
import shutil
from typing import Dict, Union, List, Tuple
from uuid import uuid4

import h5py
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from PyAstronomy import pyaC
from em_methods.lumerical.lum_helper import (
    LumMethod,
    LumericalError,
    RunLumerical,
    _get_lumerical_results,
)
from em_methods.lumerical.fdtd import fdtd_run, fdtd_run_analysis
from em_methods.formulas.physiscs import (
    rad_recombination_coeff,
    intrinsic_carrier_density,
)
import lumapi


# Get module logger
logger = logging.getLogger("dev")


@dataclass()
class SimInfo:
    """
    Structure with the connection properties between FDTD and CHARGE
    """

    SolarGenName: Union[str, List[str]]
    GenName: Union[str, List[str]]
    SCName: Union[str, List[str]]
    Cathode: str
    Anode: str
    RadCoeff: Union[float, bool, List[Union[float, bool]]] = False
    """ Extract properties always as list """

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

    @property
    def RadCoeff_List(self) -> List[Union[float, bool]]:
        return self.RadCoeff if isinstance(self.RadCoeff, list) else [self.RadCoeff]

    @property
    def simObjects(self):
        return len(self.SolarGenName_List)

    def getNamedGeneration(self, genname):
        """This extracts the extension from the generation filename"""
        return genname[:-4]


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


def __charge_extract_iv_data(results, active_region):
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

    current = np.array(results["results.CHARGE." + active_region.Cathode]["I"])
    voltage = np.array(
        results["results.CHARGE." + active_region.Cathode]["V_" + active_region.Cathode]
    )
    x_span = results["func_output"][0]
    y_span = results["func_output"][1]
    return current, voltage, x_span, y_span


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
    properties: Dict,
    active_regions: List[SimInfo],
    avg_mode: bool = False,
    override_freq: Union[None, float] = None,
    run_fdtd: bool = False,
    fdtd_kw: Dict = {},
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
        override_freq: Override frequencies in the file (necessary for EQE calculations)
        run_fdtd: Wether to run fdtd or simply import generation data
        fdtd_kw: Extra arguments to pass to fdtd_run and lumapi.FDTD (fdtd_kw)
    """
    basepath, _ = os.path.split(fdtd_file)
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
    fdtd_base_args = {
        "basefile": fdtd_file,
        "properties": properties,
        "get_results": results,
        "func": __prepare_gen,
    }
    if run_fdtd:
        res, *_ = fdtd_run(
            **fdtd_base_args,
            **fdtd_kw,
            **{"active_regions": active_regions, "override_freq": override_freq},
        )
    else:
        res, *_ = fdtd_run_analysis(fdtd_file, results, fdtd_kw=fdtd_kw)
    logger.debug(f"Get Gen run_fdtd results: {res}")
    if avg_mode:
        logger.debug("Averaged Genration -> 3d to 2d")
        for active_region in active_regions:
            for genregion in active_region.GenName_List:
                generation_name = os.path.join(basepath, genregion)
                logger.debug(f"New Generation Name: {generation_name}")
                gen_data, x, y, z = _import_generation(generation_name)
                y_gen, _ = _average_generation(gen_data, x.flatten(), y.flatten())
                y_gen_3d = y_gen + np.zeros((len(x), len(z), len(y)))
                export_3d_averaged_data = {
                    "G": np.transpose(y_gen_3d, (1, 0, 2)),
                    "x": x,
                    "y": y,
                    "z": z,
                }
                _export_hdf_data(generation_name, **export_3d_averaged_data)
    return res


def set_sim_region(
    charge_handler,
    xy_size: Union[str, Tuple[float, float]],
    anode: str,
    cathode: str,
    xy_center: Tuple[float, float] = (0, 0),
    dimensions: str = "2d",
) -> Tuple[str, float, float]:
    """
    Helper function for run_fdtd_and_charge to setup the simulation region
    Args:
        xy_size: (str) simulation object to base simulation region dimensions
                 ((float, float)) specific values to use
        anode, cathode
        xy_center: Values for the centering of the simulation region
        dimensions: '2d' or '3d'
    Returns:
        sim_name: Name of the simulation region
        x_span, y_span: x and y size
    """
    if dimensions.lower() not in ["2d", "3d"]:
        raise LumericalError(
            "def_sim_region must be one of '2D', '2d', '3D', '3d' or None"
        )
    if isinstance(xy_size, str):
        logger.debug(f"Defining simulation region based on object: {xy_size}")
        charge_handler.select(f"CHARGE::{xy_size}")
        x = charge_handler.get("x")
        y = charge_handler.get("y")
        x_span: float = charge_handler.get("x span")
        y_span: float = charge_handler.get("y span")
        sim_name: str = "user_" + xy_size
    elif isinstance(xy_size, tuple):
        x_span = xy_size[0]
        y_span = xy_size[1]
        x = xy_center[0]
        y = xy_center[1]
        sim_name = "user_xysize"
    else:
        raise Exception("Invalid input for xy_size")
    logger.debug(f"Created Simulation Region: {sim_name}")
    charge_handler.select("geometry::" + anode)
    z_max = charge_handler.get("z max")
    charge_handler.select("geometry::" + cathode)
    z_min = charge_handler.get("z min")
    # Create and resize simulation region
    charge_handler.addsimulationregion()
    charge_handler.set("name", sim_name)
    # Define simulation region as 2D or 3D
    if "2" in dimensions:
        charge_handler.set("dimension", "2D Y-Normal")
    elif "3" in dimensions:
        charge_handler.set("dimension", "3D")
        charge_handler.set("y span", y_span)
    charge_handler.set("x", x)
    charge_handler.set("x span", x_span)
    charge_handler.set("y", y)
    charge_handler.set("z max", z_max)
    charge_handler.set("z min", z_min)
    charge_handler.select("CHARGE")
    charge_handler.set("simulation region", sim_name)
    charge_handler.save()
    return sim_name, x_span, y_span


def set_mesh_conditions(charge_handler, min_edge: float):
    """Helper function to override mesh conditions for CHARGE"""
    charge_handler.select("CHARGE")
    logger.debug(f"Overriding min edge length to: {min_edge}")
    charge_handler.set("min edge length", min_edge)
    charge_handler.save()
    pass


def set_bias_regime(
    charge_handler,
    cathode: str,
    *,
    bias_regime: str = "forward",
    method_solver: str = "NEWTON",
    voltage: Union[float, Tuple[float, float]] = (0, 1.5),
    voltage_points: int = 10,
    is_voltage_range: bool = True,
) -> None:
    """
    Helper function to run_fdtd_and_charge
    This function is focused on defining the bias regime and simulation voltage range
    Args:
        charge_handler, cathode
        bias_regime (forward, backward), method_solver (NEWTON, GUMMEL)
        is_voltage_range: simulate for 1 point or range
        voltage: point to simulate or range for simulation
        voltage_points: Number of voltage points to run
    """
    # Pre-run checks
    if is_voltage_range and voltage_points < 1:
        raise Exception("with is_voltage_range True, voltage_points should be > 0")
    charge_handler.select("CHARGE")
    if bias_regime == "forward":
        logger.debug("Running for forward bias regime")
        charge_handler.set("solver type", method_solver)
        charge_handler.set("enable initialization", True)
    elif bias_regime == "reverse":
        logger.debug("Running for reverse bias regime")
        logger.info("Reverse Regime only runs for the Gummel solver")
        charge_handler.set("solver type", "GUMMEL")
        charge_handler.set("enable initialization", False)
    charge_handler.select("CHARGE::boundary conditions::" + cathode)
    if not is_voltage_range and isinstance(voltage, (int, float)):
        charge_handler.set("sweep type", "single")
        charge_handler.set("voltage", voltage)
    elif not is_voltage_range:
        raise Exception("is_voltage_range False requires voltage to be (int, float)")
    if not isinstance(voltage, tuple):
        voltage: Tuple[float, float] = (0, voltage)
    if is_voltage_range:
        charge_handler.set("sweep type", "range")
        charge_handler.set("range start", voltage[0])
        charge_handler.set("range stop", voltage[1])
        charge_handler.set("range num points", voltage_points)
        charge_handler.set("range backtracking", "enabled")
    charge_handler.save()


def set_rad_coeff(
    charge_handler,
    material: str,
    rad_coeff: Union[float, bool],
    gen_wvl=None,
    gen_abs=None,
) -> None:
    """
    Set Radiative Recombination coefficient for 'material'
    The value is set based on the value of rad_coeff
    Args:
        rad_coeff:
            float: Override the value in CHARGE
            bool: Use default value in CHARGE (False) or calculates from gen_wvl, gen_abs (True)
        gen_wvl, gen_abs: wavelength and absorption to calculate rad_coeff
    """
    if not rad_coeff:
        logger.debug("Using Default Radiative Recombination Coeff in CHARGE")
        charge_handler.save()
        return
    charge_handler.select("geometry::" + material)
    z_span = charge_handler.get("z span")  # in m
    material_db = charge_handler.get("material")
    logger.debug(f"Changing rad_recombination_coeff for: {material_db}")
    charge_handler.select(f"materials::{material_db}::{material_db}")
    if isinstance(rad_coeff, float):
        logger.debug("Using provided Radiative Recombination Coefficient")
        charge_handler.set("recombination.radiative.copt.constant", rad_coeff)
        charge_handler.save()
        return
    if (
        rad_coeff is True
        and isinstance(gen_wvl, np.ndarray)
        and isinstance(gen_abs, np.ndarray)
    ):
        logger.debug("Calculating Radiative Recombination Coefficient from FDTD data")
        bandgap = charge_handler.get("electronic.gamma.Eg.constant")
        mn = charge_handler.get("electronic.gamma.mn.constant")
        mp = charge_handler.get("electronic.gamma.mp.constant")
        ni = intrinsic_carrier_density(mn, mp, bandgap)
        logger.debug(f"Extracted data: {bandgap}::{z_span}::{mn}::{mp}::{ni}")
        rad_coeff = rad_recombination_coeff(gen_wvl, gen_abs, bandgap, z_span, ni)
        logger.debug(f"Calculated B coefficient: {rad_coeff}")
        charge_handler.set("recombination.radiative.copt.constant", rad_coeff)
        charge_handler.save()
    else:
        raise LumericalError("Incompatible variables, rad_coeff, gen_wvl, gen_abs")


def __set_iv_parameters(
    charge,
    active_region: SimInfo,
    override_bias_regime_args: Dict = {},
    def_sim_region: Union[None, str] = "2d",
    min_edge=None,
    fdtd_results: Dict = {},
):
    """
    Function that aggregates the behaviour of:
        set_rad_coeff, set_sim_region, set_bias_regime, set_mesh_conditions
    This is the default function for __set_iv_parameters
    Args:
        charge: Handler for the charge file
        active_region: SimInfo with the charge/fdtd connecting information
        override_bias_regime_args: Override values for the set_bias_regime
        min_edge: Argument for set_mesh_conditions
        def_sim_region: Create sim_region or use the one already in the file
    Return:
        xspan, yspan: (float) x and y solar cell dimensions (m) normal to the incident light
    """
    charge.switchtolayout()
    active_region_info_list = zip(
        active_region.GenName_List,
        active_region.SCName_List,
        active_region.SolarGenName_List,
        active_region.RadCoeff_List,
    )
    for genname, scname, solar_gen_name, rad_coeff in active_region_info_list:
        logger.debug(
            f"Updating Charge information for: {genname}::{scname}::{solar_gen_name}"
        )
        charge.addimportgen()
        charge.set("name", genname[:-4])
        charge.set("volume type", "solid")
        charge.set("volume solid", scname)
        charge.importdataset(genname)
        charge.save()
        gen_wvl = fdtd_results[f"results.{solar_gen_name}.Pabs_total"][
            "lambda"
        ].flatten()
        gen_abs = fdtd_results[f"results.{solar_gen_name}.Pabs_total"][
            "Pabs_total"
        ].flatten()
        set_rad_coeff(charge, scname, rad_coeff, gen_wvl, gen_abs)
    # Defines boundaries for simulation region
    if def_sim_region is not None:
        _, xspan, yspan = set_sim_region(
            charge,
            active_region.GenName_List[0][:-4],
            active_region.Anode,
            active_region.Cathode,
            dimensions=def_sim_region,
        )
    else:
        sim_region = charge.getnamed("CHARGE", "simulation region")
        charge.select(sim_region)
        xspan = charge.get("x span")
        yspan = charge.get("y span")
    # Defining solver parameters
    set_bias_regime(charge, active_region.Cathode, **override_bias_regime_args)
    return xspan, yspan


def run_fdtd_and_charge(
    active_regions: Union[SimInfo, List[SimInfo]],
    base_properties: Dict,
    charge_file: str,
    fdtd_file: str,
    *,
    charge_extra_properties: Dict = {},
    fdtd_extra_properties: Dict = {},
    run_fdtd: bool = True,
    def_sim_region: Union[None, str] = "2d",
    override_bias_regime_args: Union[Dict, List[Dict]] = {
        "voltage": 1.5,
        "voltage_points": 101,
    },
    min_edge: Union[List[float], None, List[None]] = None,
    avg_mode: bool = False,
    save_csv=False,
    savepath: str = ".",
    charge_kw: Dict = {"device_kw": {"hide": True}, "delete": True},
    fdtd_kw: Dict = {"fdtd_kw": {"hide": True}, "delete": True},
):
    """
    Runs the FDTD and CHARGE files for the multiple active regions defined in the active_regions
    The main results are the IV parameters for the solar cell
    Args:
        active_regions: SimInfo list with regions for simulation
        base_properties: Properties to change in charge and fdtd
        charge_file: path to CHARGE file
        fdtd_file: path FDTD file
        charge/fdtd_extra_properties: Additional properties to change in fdtd and charge
        run_FDTD: (bool) Calculate generation profile or use already calculated
        def_sim_region: (None | str)
            None: Do not create simulation region
            str: Create Simulation region from '2D' '3D'
        override_bias_regime_args: Arguments for set_bias_regime function
            (voltage, voltage_points, is_voltage_range)
        avg_mode (bool): Simplify generation to 2D or calculate 3D
        min_edge: Charge setting for the size of the simulation region
        save_csv: (bool) Save Current Density and Voltage in csv file "{names.SCName}_IV_curve.csv"
        savepath: Directory to save results data
        charge_kw: Pass arguments for charge_run
        fdtd_kw: Pass arguments for fdtd_run
    Returns:
        PCE (%), FF[0-1], Voc[V], Jsc[mA/cm2], Current_Density[mA/cm2], Voltage[V]:
    """
    # Perform pre-run checks
    if not isinstance(active_regions, list):
        active_regions = [active_regions]
    if not isinstance(override_bias_regime_args, list):
        override_bias_regime_args = [override_bias_regime_args]
    if min_edge is None:
        min_edge = [min_edge]
    if len(active_regions) != len(override_bias_regime_args) or len(
        active_regions
    ) != len(min_edge):
        raise Exception(
            "active_regions | override_bias_regime_args | min_edge should have the same size"
        )
    fdtd_extra_properties.update(base_properties)
    logger.debug(f"Final FDTD Properties: {fdtd_extra_properties}")
    gen_results = get_gen(
        fdtd_file,
        fdtd_extra_properties,
        active_regions,
        run_fdtd=run_fdtd,
        avg_mode=avg_mode,
        fdtd_kw=fdtd_kw,
    )
    pce_array, ff_array, voc_array, jsc_array, current_density_array, voltage_array = (
        [] for _ in range(6)
    )
    iv_variables = [
        pce_array,
        ff_array,
        voc_array,
        jsc_array,
        current_density_array,
        voltage_array,
    ]
    results = None
    for active_region, override_bias_regime_i, min_edge_i in zip(
        active_regions, override_bias_regime_args, min_edge
    ):
        conditions_dic = {
            "active_region": active_region,
            "override_bias_regime_args": override_bias_regime_i,
            "def_sim_region": def_sim_region,
            "min_edge": min_edge_i,
            "fdtd_results": gen_results,
        }
        charge_extra_properties.update(base_properties)
        charge_kw.update(
            {
                "basefile": charge_file,
                "properties": base_properties,
                "get_results": {"results": {"CHARGE": active_region.Cathode}},
                "func": __set_iv_parameters,
            }
        )
        logger.debug(
            f"Charge runconditions:\n{charge_extra_properties}\n{charge_kw}\n{conditions_dic}"
        )
        results = charge_run(**charge_kw, **conditions_dic)
        current, voltage, x_span, y_span = __charge_extract_iv_data(
            results[0], active_region
        )
        iv_results = iv_curve(voltage, current, x_span, y_span)
        for variable, var_result in zip(iv_variables, iv_results):
            variable.append(var_result)
        logger.info(
            f"""
Semiconductor:{active_region.SCName}
Cathode {active_region.Cathode}
Voc = {voc_array[-1]:.3f}V
Jsc =  {jsc_array[-1]:.4f} mA/cm²
FF = {ff_array[-1]:.3f}
PCE = {pce_array[-1]:.3f}%
"""
        )
        if save_csv:
            df = pd.DataFrame(
                {"Current_Density": jsc_array[-1], "Voltage": voc_array[-1]}
            )
            csv_path = os.path.join(savepath, f"{active_region.SCName}_IV_curve.csv")
            df.to_csv(csv_path, sep="\t", index=False)
    return tuple(iv_variables)


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
