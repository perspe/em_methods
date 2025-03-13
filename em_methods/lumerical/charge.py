from dataclasses import dataclass
from functools import partial, update_wrapper
import logging
from multiprocessing import Manager, Queue
import os
import shutil
from typing import Dict, List, Tuple, Union
from uuid import uuid4

import numpy as np
import numpy.typing as npt
import pandas as pd

from em_methods import Units
from em_methods.formulas.physiscs import (
    intrinsic_carrier_density,
    rad_recombination_coeff,
)
from em_methods.lumerical.fdtd import average_generation, fdtd_run, fdtd_run_analysis
from em_methods.lumerical.lum_helper import (
    LumMethod,
    LumericalError,
    RunLumerical,
    _get_lumerical_results,
)
from em_methods.utilities import iv_parameters
import lumapi


# Get module logger
logger = logging.getLogger("sim")


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

    def update_rad_coefficient(self, new_rad_coefficient):
        self.RadCoeff = new_rad_coefficient


""" Helper Functions """


def __prepare_gen(
    fdtd_handler, active_regions, override_wavelength, wavelength_units: Units
):
    """Function to preprocess the files necessary in get_gen function"""
    for names in active_regions:
        for gen_obj, file in zip(names.SolarGenName_List, names.GenName_List):
            g_name = file.replace(".mat", "")
            fdtd_handler.select(gen_obj)
            fdtd_handler.set("export filename", g_name)
        if override_wavelength is not None:
            fdtd_handler.setglobalsource(
                "wavelength start",
                override_wavelength * wavelength_units.convertTo(Units.M),
            )
            fdtd_handler.setglobalsource(
                "wavelength stop",
                override_wavelength * wavelength_units.convertTo(Units.M),
            )


def __get_gen(
    fdtd_file: str,
    properties: Dict,
    active_regions: List[SimInfo],
    avg_mode: bool = False,
    override_wavelength: Union[None, float] = None,
    wavelength_units: Units = Units.NM,
    run_fdtd: bool = True,
    fdtd_kw: Dict = {"fdtd_kw": {"hide": True}},
):
    """
    Alters the cell design ("properties"), simulates the FDTD file, and creates the generation rate .mat file(s)
    (in same directory as FDTD file)
    Args:
        fdtd_file: String FDTD file name
        properties: Dictionary with the property object and property names and values
        active_regions: list with SimInfo details of the simulations
            [SimInfo("solar_generation_Si", "G_Si.mat", "Si", "AZO", "ITO_bottom")]
        avg_mode: average generation rate is averaged in y (for LT)
        override_wavelength: Override wavelength in the file (for EQE calculations)
        wavelength_units: Provided units for wavelength
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
            **{
                "active_regions": active_regions,
                "override_wavelength": override_wavelength,
                "wavelength_units": wavelength_units,
            },
        )
    else:
        if "fdtd_kw" in fdtd_kw.keys():
            fdtd_kw = fdtd_kw["fdtd_kw"]
        else:
            fdtd_kw = {}
        res = fdtd_run_analysis(fdtd_file, results, fdtd_kw=fdtd_kw)
    logger.debug(f"Get Gen run_fdtd results: {res}")
    if avg_mode:
        logger.debug("Averaged Genration -> 3d to 2d")
        for active_region in active_regions:
            for genregion in active_region.GenName_List:
                generation_name = os.path.join(basepath, genregion)
                average_generation(
                    generation_name,
                    mode="2d",
                    export_mode="3d",
                    export_name=generation_name,
                )
    return res


""" Helper functions to set parameters in CHARGE """


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
    # Correction factor for y_span in 2D simulations
    if "2" in dimensions:
        y_span = charge_handler.get("norm length")
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
        if "3D" not in charge.get("dimension"):
            charge.select("CHARGE")
            charge.get("norm length")
        else:
            yspan = charge.get("y span")
    if min_edge is not None:
        set_mesh_conditions(charge, min_edge)
    # Defining solver parameters
    set_bias_regime(charge, active_region.Cathode, **override_bias_regime_args)
    return xspan, yspan


""" Main Run Functions (run_fdtd_and_charge and charge_run) """


def charge_run(
    basefile: str,
    properties: Dict[str, Dict[str, float]],
    get_results,
    *,
    get_info: Dict[str, str] = {},
    func=None,
    savepath: Union[None, str] = None,
    override_prefix: Union[None, str] = None,
    delete: bool = True,
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


def run_fdtd_and_charge(
    active_regions: Union[SimInfo, List[SimInfo]],
    base_properties: Dict,
    charge_file: str,
    fdtd_file: str,
    *,
    def_sim_region: Union[None, str] = "2d",
    min_edge: Union[List[float], None, List[None]] = None,
    override_bias_regime_args: Union[Dict, List[Dict]] = {},
    override_get_gen_args: Dict = {},
    charge_extra_properties: Dict = {},
    fdtd_extra_properties: Dict = {},
    charge_get_extra_results: Dict = {},
    charge_kw: Dict = {},
    fdtd_kw: Dict = {},
):
    """
    Runs the FDTD and CHARGE files for the multiple active regions defined in the active_regions
    The main results are the IV parameters for the solar cell
    Args:
        active_regions: SimInfo list with FDTD and CHARGE connecting information
        base_properties: Properties to change in charge and fdtd
        charge_file: path to CHARGE file
        fdtd_file: path FDTD file
        def_sim_region: (None | str)
            None: Do not create simulation region
            str: Create Simulation region from '2D' '3D'
        min_edge: Charge setting for the size of the simulation region
        override_get_gen_args: Override arguments for get_gen
        override_bias_regime_args: Arguments for set_bias_regime function
            (voltage, voltage_points, is_voltage_range)
        charge/fdtd_extra_properties: Additional properties to change in fdtd and charge
        charge_get_extra_results: Adicional results to extract from charge
        charge_kw: Pass arguments for charge_run
        fdtd_kw: Pass arguments for fdtd_run
    Returns:
        list with result for each active region
    """
    # Perform pre-run checks
    if not isinstance(active_regions, list):
        active_regions = [active_regions]
    if not isinstance(override_bias_regime_args, list):
        override_bias_regime_args = [override_bias_regime_args] * len(active_regions)
    if min_edge is None:
        min_edge = [min_edge] * len(active_regions)
    if len(active_regions) != len(override_bias_regime_args) or len(
        active_regions
    ) != len(min_edge):
        raise Exception(
            "active_regions | override_bias_regime_args | min_edge should have the same size"
        )
    # def_sim_region -> override_get_gen_args["avg_mode"] == True
    if def_sim_region == "2d":
        if not "avg_mode" in override_get_gen_args.keys():
            logger.warning(
                f"def_sim_region = '2d' -> avg_mode = True in override_get_gen_args"
            )
            override_get_gen_args.update({"avg_mode": True})
        elif (
            "avg_mode" in override_get_gen_args.keys()
            and not override_get_gen_args["avg_mode"]
        ):
            logger.warning(
                f"def_sim_region = '2d' -> avg_mode = True in override_get_gen_args"
            )
            override_get_gen_args["avg_mode"] = True
    fdtd_extra_properties.update(base_properties)
    logger.debug(f"Final FDTD Properties: {fdtd_extra_properties}")
    gen_results = __get_gen(
        fdtd_file,
        fdtd_extra_properties,
        active_regions,
        fdtd_kw=fdtd_kw,
        **override_get_gen_args,
    )
    results = []
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
        get_results = {"results": {"CHARGE": active_region.Cathode}}
        get_results.update(charge_get_extra_results)
        charge_kw.update(
            {
                "basefile": charge_file,
                "properties": charge_extra_properties,
                "get_results": get_results,
                "func": __set_iv_parameters,
            }
        )
        logger.debug(
            f"Charge runconditions:\n{charge_extra_properties}\n{charge_kw}\n{conditions_dic}"
        )
        result = charge_run(**charge_kw, **conditions_dic)
        results.append(result)
    logger.debug("Run Successfuly")
    return gen_results, results


def run_iqe(
    active_regions: Union[SimInfo, List[SimInfo]],
    base_properties: Dict,
    charge_file: str,
    fdtd_file: str,
    wavelengths: npt.ArrayLike,
    /,
    wavelength_units: Units =  Units.NM,
    **override_fdtd_and_charge_args,
):
    """
    Calculate the IQE for wavelengths
    active_regions: Regions in the simulation to run the simulation
    base_properties: Properties to change in FDTD and CHARGE
    charge_file/fdtd_file: Path for fdtd and charge files
    wavelengths: ArrayLike with wavelengths to run simulations
    wavelength_units: Units for the wavelength
    override_fdtd_and_charge_args: Override other run_fdtd_and_charge args
    """
    if not isinstance(active_regions, list):
        active_regions = [active_regions]
    base_run_fdtd_and_charge = {
        "active_regions": active_regions,
        "base_properties": base_properties,
        "charge_file": charge_file,
        "fdtd_file": fdtd_file,
    }
    if "override_bias_regime_args" in override_fdtd_and_charge_args.keys():
        if override_fdtd_and_charge_args["override_bias_regime_args"][
            "is_voltage_range"
        ]:
            logger.warning("IQE simulations require constant voltage")
    override_fdtd_and_charge_args.update(
        {"override_bias_regime_args": {"voltage": 0, "is_voltage_range": False}}
    )
    override_fdtd_and_charge_args.update(base_run_fdtd_and_charge)
    iqe_results = {active_region.SCName: [] for active_region in active_regions}
    for wavelength in wavelengths:
        logger.debug(f"Running IQE for: {wavelength}")
        if "override_get_gen_args" in override_fdtd_and_charge_args.keys():
            logger.warning(
                "Overriding any provided override_wavelength in override_get_gen_args"
            )
            override_fdtd_and_charge_args["override_get_gen_args"].update(
                {
                    "override_wavelength": wavelength,
                    "wavelength_units": wavelength_units,
                }
            )
        else:
            override_fdtd_and_charge_args.update(
                {
                    "override_get_gen_args": {
                        "override_wavelength": wavelength,
                        "wavelength_units": wavelength_units,
                    }
                }
            )
        fdtd_res, charge_res = run_fdtd_and_charge(**override_fdtd_and_charge_args)
        for active_region, charge_res_i in zip(active_regions, charge_res):
            x_span, y_span = charge_res_i[0]["func_output"]
            area = x_span * y_span
            jph = 0
            # TODO: What happens for multiple gens at the same time?
            for sgname in active_region.SolarGenName_List:
                jph += fdtd_res[f"data.{sgname}.Jsc"]
            isc = charge_res_i[0][f"results.CHARGE.{active_region.Cathode}"][
                "I"
            ].flatten()[0]
            jsc = - isc / area
            iqe_results[active_region.SCName].append(jsc / jph)
            logger.info(f"IQE ({wavelength}): {jsc/jph} ({jsc:.3g}/{jph:.3g})")
    iqe_results = {key: np.array(value) for key, value in iqe_results.items()}
    return iqe_results


""" Alias Functions """

""" Alias function to get bandstructure results """
run_bandstructure = partial(
    run_fdtd_and_charge,
    charge_get_extra_results={"results": {"CHARGE::monitor": "bandstructure"}},
    override_bias_regime_args={"voltage": 0, "is_voltage_range": False},
)
update_wrapper(run_bandstructure, run_fdtd_and_charge)


""" Functions to extract rund_fdtd_and_charge results """


def run_fdtd_and_charge_to_iv(results, cathodes: Union[str, List[str]]) -> List[Tuple]:
    """
    Shortcut function to convert results from run_fdtd_and_charge to IV parameters
    """
    if isinstance(cathodes, str):
        cathodes = [cathodes]
    if not isinstance(results, list):
        results = [results]
    return_res = []
    for result_i, cathode in zip(results, cathodes):
        logger.debug(f"Calculating Results for {cathode}")
        result_i = result_i[0]
        current = np.array(result_i[f"results.CHARGE.{cathode}"]["I"])
        voltage = np.array(result_i[f"results.CHARGE.{cathode}"][f"V_{cathode}"])
        x_span = result_i["func_output"][0]
        y_span = result_i["func_output"][1]
        current_density = current.flatten() * 1e3 / (x_span * y_span * 1e4)
        voltage = voltage.flatten()
        res_i = iv_parameters(voltage, current_density)
        return_res.append(res_i)
    return return_res


def run_bandstructure_to_bands(results: Union[List[Dict], Dict]) -> List[Tuple]:
    """
    Shortcut function to convert results from run_fdtd_and_charge to IV parameters
    """
    if not isinstance(results, list):
        results = [results]
    return_res = []
    for results_i in results:
        results_i = results_i[0]
        ec = results_i["results.CHARGE::monitor.bandstructure"]["Ec"].flatten()
        thickness = results_i["results.CHARGE::monitor.bandstructure"]["z"].flatten()
        ev = results_i["results.CHARGE::monitor.bandstructure"]["Ev"].flatten()
        efn = results_i["results.CHARGE::monitor.bandstructure"]["Efn"].flatten()
        efp = results_i["results.CHARGE::monitor.bandstructure"]["Efp"].flatten()
        return_res.append((thickness, ec, ev, efn, efp))
    return return_res


""" Compatibility Functions """


def run_fdtd_and_charge_legacy(
    active_region_list,
    properties,
    charge_file,
    path,
    fdtd_file,
    v_max=1.5,
    run_FDTD=True,
    def_sim_region=None,
    B=None,
    method_solver="NEWTON",
    v_single_point=None,
    avg_mode=False,
    min_edge=None,
    range_num_points=101,
    save_csv=False,
):
    charge_file = os.path.join(path, charge_file)
    fdtd_file = os.path.join(path, fdtd_file)
    if not isinstance(active_region_list, list):
        active_region_list = [active_region_list]
    if B == None:
        B = [None for _ in range(0, len(active_region_list))]
    # Adapt v_max / v_single_point / range_num_points to override_bias_regime
    if not isinstance(v_single_point, (list, np.ndarray)):
        v_single_point = [v_single_point for _ in range(0, len(active_region_list))]
    if not isinstance(v_max, (list, np.ndarray)):
        v_max = [v_max for _ in range(0, len(active_region_list))]
    if not isinstance(range_num_points, (list, np.ndarray)):
        range_num_points = [
            int(range_num_points) for _ in range(0, len(active_region_list))
        ]
    override_bias_regime_list = []
    for active_region, b_i in zip(active_region_list, B):
        active_region.update_rad_coefficient(b_i)
    for v_i, v_single_i, range_i in zip(v_max, v_single_point, range_num_points):
        if v_single_i is not None:
            override_bias_regime_list.append(
                {
                    "voltage": v_single_i,
                    "is_voltage_range": False,
                    "method_solver": method_solver,
                }
            )
        else:
            override_bias_regime_list.append(
                {
                    "voltage": v_i,
                    "is_voltage_range": True,
                    "voltage_points": range_i,
                    "method_solver": method_solver,
                }
            )
    _, results = run_fdtd_and_charge(
        active_region_list,
        properties,
        charge_file,
        fdtd_file,
        def_sim_region=def_sim_region,
        min_edge=min_edge,
        override_bias_regime_args=override_bias_regime_list,
        override_get_gen_args={"run_fdtd": run_FDTD, "avg_mode": avg_mode},
    )
    logger.debug(f"Simulated Results: {results}")
    final_results = []
    for result, active_region in zip(results, active_region_list):
        results_i = run_fdtd_and_charge_to_iv(result, active_region.Cathode)
        final_results.append(results_i[0])
        if save_csv:
            monitor = f"results.CHARGE.{active_region.Cathode}"
            df = pd.DataFrame(
                {
                    "Current_Density": result[0][monitor]["I"].flatten(),
                    "Voltage": result[0][monitor][
                        f"V_{active_region.Cathode}"
                    ].flatten(),
                }
            )
            csv_path = os.path.join(path, f"{active_region.SCName}_IV_curve.csv")
            df.to_csv(csv_path, sep="\t", index=False)
    logger.debug(f"Obtained Results: {final_results}")
    corrected_res = []
    for res in zip(*final_results):
        corrected_res.append(list(res))
    return corrected_res
