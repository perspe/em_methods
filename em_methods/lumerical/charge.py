import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Union, Dict, List
import logging
from uuid import uuid4
import shutil
import sys
from dataclasses import dataclass 
from matplotlib.patches import Rectangle
from PyAstronomy import pyaC
from multiprocessing import Queue, Manager
from em_methods.lumerical.lum_helper import (
    RunLumerical,
    _get_lumerical_results,
    LumericalError,
    LumMethod,
)

# Get module logger
logger = logging.getLogger("sim")

# Connect to Lumerical
# Determine the base path for lumerical
if os.name == "nt":
    LM_BASE: str = os.path.join("C:\\", "Program Files", "Lumerical")
elif os.name == "posix":
    LM_BASE: str = os.path.join("/opt", "lumerical")
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
if os.name == "nt":
    os.add_dll_directory(LUMAPI_PATH)
import lumapi
import pandas as pd
from scipy.constants import h, c, k, e


@dataclass
class SimInfo:
    SolarGenName: str
    GenName: str
    SCName: str
    Cathode: str
    Anode: str


""" Main functions """


def charge_run(
    basefile: str,
    properties: Dict[str, Dict[str, float]],
    get_results,
    *,
    get_info: Dict[str, str] ={},
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
            savepath (default=.): Override default savepath for the new file
            override_prefix (default=None): Override prefix for the new file
            delete (default=False): Delete newly generated file
            names: SimInfo dataclass structure about the simulation (e.g. SimInfo("solar_generation_PVK", "G_PVK.mat", "Perovskite", "ITO_top", "ITO"))
            func: optional funtion
            get_info: Dictionary with additional data to extract from the CHARGE file 
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
    # 1. Create Queue to Store the data
    # 2. Create a process (RunLumerical) to run the lumerical file
    #       - This avoids problems when the simulation gives errors
    # 3. Create a Thread to check run state
    #       - If thread finds error then it kill the RunLumerical process
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
    return results["data"], results["runtime"], results["analysis runtime"], results["data_info"]


def charge_run_analysis(basefile: str, names, device_kw={"hide": True}):
    """
    Generic function to gather simulation data from already simulated files
    Args:
            basefile: Path to the original file
            names: SimInfo dataclass structure about the simulation (e.g. SimInfo("solar_generation_PVK", "G_PVK.mat", "Perovskite", "ITO_top", "ITO"))

    Return:
            results: Dictionary with all the results
    """
    get_results = {"results": {"CHARGE": str(names.Cathode)}}
    with lumapi.DEVICE(filename=basefile, **device_kw) as charge:
        results = _get_lumerical_results(charge, get_results)
        charge.close()
    return results


def find_index(wv_list, n):
    index = [i for i in range(0, len(wv_list)) if wv_list[i] == n]
    return index[0]


def iv_curve(results, regime, names):
    """
    Obtains the performance metrics of a solar cell
    Args:
            results: Dictionary with all the results from the charge_run function
            names: SimInfo dataclass structure about the simulation (e.g. SimInfo("solar_generation_PVK", "G_PVK.mat", "Perovskite", "ITO_top", "ITO"))
            regime: "am" or "dark" for illuminated IV or dark IV
    Returns: 
            PCE, FF, Voc, Jsc, current_density, voltage, stop, P 
            OR
            current_density, voltage
    """

    current = np.array(results["results.CHARGE." + str(names.Cathode)]["I"])
    voltage = np.array(results["results.CHARGE." + str(names.Cathode)]["V_" + str(names.Cathode)])
    #a = results["results.CHARGE.monitor.bandstructure"]["Ec"]
    Lx = results["func_output"][0]
    Ly = results["func_output"][1]

    if ((len(current) == 1 and len(current[0]) != 1) ):  # charge I output is not always consistent
         current = current[0]
         voltage = [float(arr[0]) for arr in voltage]
    elif (voltage.ndim == 2):
         voltage = [float(arr[0]) for arr in voltage]
         current = [arr[0] for arr in current]

    Lx = Lx * 100  # from m to cm
    Ly = Ly * 100  # from m to cm
    area = Ly * Lx  # area in cm^2
    current_density = (np.array(current) * 1000) / area
    Ir = 1000  # W/m²
    if regime == "am":
        # DETERMINE JSC
        abs_voltage_min = min(np.absolute(voltage))  # volatage value closest to zero
        if abs_voltage_min in voltage:

            Jsc = current_density[np.where(voltage == abs_voltage_min)[0][0]]
            Isc = current[np.where(voltage == abs_voltage_min)[0][0]]
            # Jsc = current_density[voltage.index(abs_voltage_min)]
            # Isc = current[voltage.index(abs_voltage_min)]
        elif -abs_voltage_min in voltage:
            # the position in the array of Jsc and Isc should be the same
            Jsc = current_density[np.where(voltage == -abs_voltage_min)[0][0]]
            Isc = current[np.where(voltage == -abs_voltage_min)[0][0]]
            # Jsc = current_density[voltage.index(-abs_voltage_min)]
            # Isc = current[voltage.index(-abs_voltage_min)]

        # DETERMINE VOC THROUGH INTERPOLATION
        
        Voc, stop = pyaC.zerocross1d(np.array(voltage), np.array(current_density), getIndices=True)
        try:
            stop = stop[0]
            Voc = Voc[0]
        except IndexError: 
            stop = np.nan
            Voc = np.nan
        try:
            vals_v = np.linspace(min(voltage), max(voltage), 100)
            new_j = np.interp(vals_v, voltage, current)
            P = [vals_v[x] * abs(new_j[x]) for x in range(len(vals_v)) if new_j[x] < 0 ]
            #P = [voltage[x] * abs(current[x]) for x in range(len(voltage)) if current[x] < 0 ]  # calculate the power for all points [W]
            FF = abs(max(P) / (Voc * Isc))
            PCE = ((FF * Voc * abs(Isc)) / (Ir * (area * 10**-4))) * 100
        except ValueError:
            P = np.nan
            FF = np.nan
            PCE = np.nan

        return PCE, FF, Voc, Jsc, current_density, voltage, stop, P

    elif regime == "dark":
        return current_density, voltage
       

def plot(PCE, FF, Voc, Jsc, current_density, voltage, stop, regime:str, P): #NOT FINISHED -> regime should not be an input, the performance metrics should be enough
    """
    Plots the IV curve
    Args:
            PCE, FF, Voc, Jsc: Perfomance metrics obtained from the iv_curve funtion.  
            current_density, voltage: Arrays obtained from the iv_curve funtion
            P: Array obtained from the iv_curve funtion with the Power values through out the iv curve.
            stop: Voc position in voltage array 
            regime: "am" or "dark" for illuminated IV or dark IV
    """
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


def get_gen(path, fdtd_file, properties, active_region_list):
    """
    Alters the cell design ("properties"), simulates the FDTD file, and creates the generation rate .mat file(s)
    (in same directory as FDTD file)
    Args:
            path: directory where the FDTD and CHARGE files exist
            fdtd_file: String FDTD file name
            properties: Dictionary with the property object and property names and values
            active_region_list: list with SimInfo dataclassses with the details of the simulation 
                                (e.g. [SimInfo("solar_generation_Si", "G_Si.mat", "Si", "AZO", "ITO_bottom"),
                                        SimInfo("solar_generation_PVK", "G_PVK.mat", "Perovskite", "ITO_top", "ITO")])
    """
    fdtd_path = os.path.join(path, fdtd_file)
    override_prefix: str = str(uuid4())[0:5]
    new_filepath: str = os.path.join(path, override_prefix + "_" + fdtd_file)
    shutil.copyfile(fdtd_path, new_filepath)
    log_file: str = os.path.join(
        path, f"{override_prefix}_{os.path.splitext(fdtd_file)[0]}_p0.log"
    )
    with lumapi.FDTD(filename=new_filepath, hide=True) as fdtd:
        # CHANGE CELL GEOMETRY
        for structure_key, structure_value in properties.items():
            fdtd.select(structure_key)
            for parameter_key, parameter_value in structure_value.items():
                fdtd.set(parameter_key, parameter_value)
        fdtd.save()


        # EXPORT GENERATION FILES
        for names in active_region_list:
            if isinstance(names.GenName, list): #if it is 2T:
                for i in range(0, len(names.GenName)):
                    gen_obj = names.SolarGenName[i] 
                    file = names.GenName[i]
                    g_name = file.replace(".mat", "")
                    fdtd.select(str(gen_obj))
                    fdtd.set("export filename", str(g_name))
            else: #if it is 4T:
                gen_obj = names.SolarGenName  # Solar Generation analysis object name 
                file = names.GenName # generation file name
                g_name = file.replace(".mat", "")
                fdtd.select(str(gen_obj))
                fdtd.set("export filename", str(g_name))
        fdtd.run()
        fdtd.runanalysis()
        fdtd.save()
        fdtd.close()
        os.remove(new_filepath)
        #os.remove(log_file)


def get_gen_eqe(path, fdtd_file, properties, active_region_list, freq):
    """
    Alters the cell design ("properties"), simulates the FDTD file, and creates the generation rate .mat file(s)
    (in same directory as FDTD file)
    Args:
            path: directory where the FDTD and CHARGE files exist
            fdtd_file: String FDTD file name
            properties: Dictionary with the property object and property names and values
            active_region_list: list with SimInfo dataclassses with the details of the simulation 
                                (e.g. [SimInfo("solar_generation_Si", "G_Si.mat", "Si", "AZO", "ITO_bottom"),
                                        SimInfo("solar_generation_PVK", "G_PVK.mat", "Perovskite", "ITO_top", "ITO")])
    """
    fdtd_path = os.path.join(path, fdtd_file)
    override_prefix: str = str(uuid4())[0:5]
    new_filepath: str = os.path.join(path, override_prefix + "_" + fdtd_file)
    shutil.copyfile(fdtd_path, new_filepath)

    with lumapi.FDTD(filename=new_filepath, hide=True) as fdtd:
        # CHANGE CELL GEOMETRY
        for structure_key, structure_value in properties.items():
            fdtd.select(structure_key)
            for parameter_key, parameter_value in structure_value.items():
                fdtd.set(parameter_key, parameter_value)
        for names in active_region_list:    
            fdtd.setglobalsource('wavelength start', freq)
            fdtd.setglobalsource('wavelength stop', freq)
            fdtd.run()
            fdtd.runanalysis(names.SolarGenName)
            jph_pvk = fdtd.getdata(active_region_list[0].SolarGenName, "Jsc")
        



        # EXPORT GENERATION FILES
        for names in active_region_list:
            gen_obj = names.SolarGenName  # Solar Generation analysis object name
            file = names.GenName # generation file name
            g_name = file.replace(".mat", "")
            fdtd.select(str(gen_obj))
            fdtd.set("export filename", str(g_name))
        fdtd.run()
        fdtd.runanalysis()
        fdtd.save()
        fdtd.close()
        os.remove(new_filepath)
        return jph_pvk

def __set_iv_parameters(charge, bias_regime: str, name: SimInfo, path:str, v_max, method_solver, def_sim_region=None, B = None, v_single_point = None):
    """ 
    Imports the generation rate into new CHARGE file, creates the simulation region based on the generation rate, 
    sets the iv curve parameters and ensure correct solver is selected (e.g. start range, stop range...)
    Args:
            charge: as in "with lumapi.DEVICE(...) as charge
            bias_regime: forward or reverse bias
            name: SimInfo dataclass structure about the simulation (e.g. SimInfo("solar_generation_PVK", "G_PVK.mat", "Perovskite", "ITO_top", "ITO"))
            path: CHARGE file directory
            v_max: maximum voltage calculated in the IV curve
            def_sim_region: optional input that defines if it is necessary to create a new simulation region. Possible input values include '2D', '2d', '3D', '3d'.
                        A simulation region will be defined accordingly to the specified dimentions. If no string is introduced then no new simulation region 
                        will be created
    Returns:
            
            Lx,Ly: dimentions of solar cell surface area normal to the incident light direction
    """
    valid_dimensions = {"2d", "3d"}
    if def_sim_region is not None and def_sim_region.lower() not in valid_dimensions:
        raise LumericalError("def_sim_region must be one of '2D', '2d', '3D', '3d' or have no input")
    charge.switchtolayout()
    if isinstance(name.GenName, list):
        print("2T") 
        terminal = 2
        for i in range(0, len(name.GenName)): #ADDS ALL THE GENERATIONS IN THE REGION LIST ARRAY
            charge.addimportgen()
            charge.set("name", str(name.GenName[i][:-4]))
            print(str(name.GenName[i][:-4]))
            # Import generation file path
            charge.set("volume type", "solid")
            charge.set("volume solid", str(name.SCName[i]))
            charge.importdataset(name.GenName[i])
            charge.save()
    # Create "Import generation rate" objects
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
            charge.select("CHARGE::" + str(name.GenName[0][:-4])) #ALL SIMULATIION REGIONS SHOULD HAVE THE SAME THICKENESS, SELECTING [0] IS THE SAME AS ANY OTHER
        elif  terminal == 4:
            charge.select("CHARGE::" + str(name.GenName[:-4]))
        x_span = charge.get("x span")
        x = charge.get("x")
        y_span = charge.get("y span")
        y = charge.get("y")
        # Creates the simulation region (2D or 3D)
        #charge.addsimulationregion()

        if terminal == 2: 
            sim_name = "2Terminal"            
        elif terminal == 4:
            sim_name=name.SCName
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
        #charge.set("init step size",1) #unsure if it works properly
        charge.save()
        # Setting sweep parameters
        charge.select("CHARGE::boundary conditions::" + str(name.Cathode))
        if v_single_point is not None:
            charge.set("sweep type", "single")
            charge.save()
            charge.set("voltage", v_single_point)
            charge.save()
            print("single")
            charge.select("CHARGE::"+ str(name.GenName[:-4]))
            charge.delete()
            charge.save()
        else:
            charge.set("sweep type", "range")
            charge.save()
            charge.set("range start", 0)
            charge.set("range stop", v_max)
            charge.set("range num points", 41)
            charge.set("range backtracking", "enabled")
            charge.save()
    #Reverse bias regime
    elif bias_regime == "reverse":
        charge.select("CHARGE")
        charge.set("solver type", "GUMMEL")
        charge.set("enable initialization", False)
        charge.save()
        #Setting sweep parameters
        charge.select("CHARGE::boundary conditions::" + str(name.Cathode))
        charge.set("sweep type", "range")
        charge.save()
        charge.set("range start", 0)
        charge.set("range stop", -1)
        charge.set("range num points", 21)
        charge.set("range backtracking", "enabled")
        charge.save()
    #Determining simulation region dimentions
    if def_sim_region is not None:
        sim_region = sim_name
    else:
        sim_region = charge.getnamed("CHARGE","simulation region")
    charge.select(sim_region)
    Lx = charge.get("x span")
    if "3D" not in charge.get("dimension"):
        charge.select("CHARGE")
        Ly = charge.get("norm length")
    else:
        charge.select(str(sim_region))
        Ly = charge.get("y span")
    if B is not None:
        charge.select("materials::"+ name.SCName + "::" + name.SCName)
        charge.set("recombination.radiative.copt.constant", B)
        charge.save()
    return Lx, Ly
    

def run_fdtd_and_charge(active_region_list, properties, charge_file, path, fdtd_file, v_max = 1.5, run_FDTD = True, def_sim_region=None, plot=False, B = None,  method_solver = "NEWTON", v_single_point = None ):
    """ 
    Runs the FDTD and CHARGE files for the multiple active regions defined in the active_region_list
    It utilizes helper functions for various tasks like running simulations, extracting IV curve performance metrics PCE, FF, Voc, Jsc
    Args:
            active_region_list: list with SimInfo dataclassses with the details of the simulation 
                            (e.g. [SimInfo("solar_generation_Si", "G_Si.mat", "Si", "AZO", "ITO_bottom"),
                                    SimInfo("solar_generation_PVK", "G_PVK.mat", "Perovskite", "ITO_top", "ITO")])
            properties: Dictionary with the property object and property names and values                    
            charge_file: name of CHARGE file
            path: directory where the FDTD and CHARGE files exist
            def_sim_region: optional input that defines if it is necessary to create a new simulation region. Possible input values include '2D', '2d', '3D', '3d'.
                            A simulation region will be defined accordingly to the specified dimentions. If no string is introduced then no new simulation region 
                            will be created
    """ 

    PCE = []
    FF = []
    Voc = []
    Jsc = []
    Current_Density = []
    Voltage = []
    charge_path = os.path.join(path, charge_file)
    if run_FDTD:
        get_gen(path, fdtd_file, properties, active_region_list)
    if B == None:
        B = [None for _ in range(0, len(active_region_list))]
    results = None
    for names in active_region_list:
        get_results = {"results": {"CHARGE": str(names.Cathode)}}  # get_results: Dictionary with the properties to be calculated
        try:
            results = charge_run(charge_path, properties, get_results, 
                                func= __set_iv_parameters, delete = True, device_kw={"hide": True},**{"bias_regime":"forward","name": names, "path": path, "v_max": v_max,"def_sim_region":def_sim_region,"B":B[active_region_list.index(names)], "method_solver": method_solver, "v_single_point": v_single_point })
        except LumericalError:
            try:            
                logger.warning("Retrying simulation")
                results = charge_run(charge_path, properties, get_results, 
                               func= __set_iv_parameters, delete = True,  device_kw={"hide": True} ,**{"bias_regime":"forward","name": names, "path": path, "v_max": v_max,"def_sim_region":def_sim_region,"B":B[active_region_list.index(names)],  "method_solver": method_solver, "v_single_point": v_single_point})
            except LumericalError:
                pce, ff, voc, jsc, current_density, voltage, stop, p = (np.nan for _ in range(8))
                PCE.append(pce)
                FF.append(ff)
                Voc.append(voc)
                Jsc.append(jsc)
                Current_Density.append(current_density)
                Voltage.append(voltage)
                print(f"Semiconductor {names.SCName}, cathode {names.Cathode}\n Voc = {Voc[-1]:.3f}V \n Jsc =  {Jsc[-1]:.4f} mA/cm² \n FF = {FF[-1]:.3f} \n PCE = {PCE[-1]:.3f}%")
                continue


        pce, ff, voc, jsc, current_density, voltage, stop, p = iv_curve( results[0],"am", names)
        PCE.append(pce)
        FF.append(ff)
        Voc.append(voc)
        Jsc.append(jsc)
        Current_Density.append(current_density)
        Voltage.append(voltage)
        print(f"Semiconductor {names.SCName}, cathode {names.Cathode}\n Voc = {Voc[-1]:.3f}V \n Jsc =  {Jsc[-1]:.4f} mA/cm² \n FF = {FF[-1]:.3f} \n PCE = {PCE[-1]:.3f}%")
        if plot:
            plot(pce, ff, voc, jsc, current_density, voltage, stop, 'am',p) 
    
    if len(active_region_list) > 1 :
        return PCE, FF, Voc, Jsc, Current_Density, Voltage
    else:
        return pce, ff, voc, jsc, current_density, voltage
    


def run_fdtd_and_charge_EQE(active_region_list, properties, charge_file, path, fdtd_file, freq, v_max = 1.5, def_sim_region=None, B = None, method_solver = "NEWTON", v_single_point = 0):
    """ 
    Runs the FDTD and CHARGE files for the multiple active regions defined in the active_region_list
    It utilizes helper functions for various tasks like running simulations, extracting IV curve performance metrics PCE, FF, Voc, Jsc
    Args:
            active_region_list: list with SimInfo dataclassses with the details of the simulation 
                            (e.g. [SimInfo("solar_generation_Si", "G_Si.mat", "Si", "AZO", "ITO_bottom"),
                                    SimInfo("solar_generation_PVK", "G_PVK.mat", "Perovskite", "ITO_top", "ITO")])
            properties: Dictionary with the property object and property names and values                    
            charge_file: name of CHARGE file
            path: directory where the FDTD and CHARGE files exist
            def_sim_region: optional input that defines if it is necessary to create a new simulation region. Possible input values include '2D', '2d', '3D', '3d'.
                            A simulation region will be defined accordingly to the specified dimentions. If no string is introduced then no new simulation region 
                            will be created
    """ 
    Jsc = []
    Current_Density = []
    charge_path = os.path.join(path, charge_file)
    jph_pvk = get_gen_eqe(path, fdtd_file, properties, active_region_list, freq)
    results = None
    for names in active_region_list:
        get_results = {"results": {"CHARGE": str(names.Cathode)}}
        try:
            results = charge_run(charge_path, properties, get_results, 
                                func= __set_iv_parameters, delete = True, device_kw={"hide": True},**{"bias_regime":"forward","name": names, "path": path, "v_max": v_max,"def_sim_region":def_sim_region,"B":B[active_region_list.index(names)], "method_solver": method_solver, "v_single_point": v_single_point })
        except LumericalError:
            try:            
                logger.warning("Retrying simulation")
                results = charge_run(charge_path, properties, get_results, 
                                func= __set_iv_parameters, delete = True, device_kw={"hide": True},**{"bias_regime":"forward","name": names, "path": path, "v_max": v_max,"def_sim_region":def_sim_region,"B":B[active_region_list.index(names)], "method_solver": method_solver, "v_single_point": v_single_point })
            except LumericalError:
                current_density = np.nan
                Current_Density.append(current_density)
                continue

        current = results[0]["results.CHARGE." + str(names.Cathode)]["I"][0]
        Lx = results[0]["func_output"][0]
        Ly = results[0]["func_output"][1]
        Lx = Lx * 100  # from m to cm
        Ly = Ly * 100  # from m to cm
        area = Ly * Lx  # area in cm^2
        current_density = (np.array(current) * 1000) / area #mA/cm2
        print(current_density)
        current_density = current_density*10 #A/m2
        jsc = current_density
        # plot(pce, ff, voc, jsc, current_density, voltage, stop, 'am',p) 

    return jsc, jph_pvk

def run_fdtd_and_charge_multi(active_region_list, properties, charge_file, path, fdtd_file, def_sim_region=None): #needs to be updated
    """ 
    Runs the FDTD and CHARGE files for the multiple active regions defined in the active_region_list
    It utilizes helper functions for various tasks like running simulations, extracting IV curve performance metrics PCE, FF, Voc, Jsc
    Args:
            active_region_list: list with SimInfo dataclassses with the details of the simulation 
                            (e.g. [SimInfo("solar_generation_Si", "G_Si.mat", "Si", "AZO", "ITO_bottom"),
                                    SimInfo("solar_generation_PVK", "G_PVK.mat", "Perovskite", "ITO_top", "ITO")])
            properties: Dictionary with the property object and property names and values                    
            charge_file: name of CHARGE file
            path: directory where the FDTD and CHARGE files exist
            def_sim_region: optional input that defines if it is necessary to create a new simulation region. Possible input values include '2D', '2d', '3D', '3d'.
                            A simulation region will be defined accordingly to the specified dimentions. If no string is introduced then no new simulation region 
                            will be created
    """ 
    PCE = []
    FF = []
    Voc = []
    Jsc = []

    PCE_temp = []
    FF_temp = []
    Voc_temp = []
    Jsc_temp = []

    Current_Density = []
    Voltage = []

    charge_path = os.path.join(path, charge_file)
    get_gen(path, fdtd_file, properties, active_region_list)
    results = None
    for names in active_region_list:
        for i in range(3):
            try:
                results = charge_run(charge_path, properties, names, 
                                    func= __set_iv_parameters, **{"bias_regime":"forward","name": names, "path": path, "def_sim_region":def_sim_region})
            except LumericalError:
                try:            
                    logger.warning("Retrying simulation")
                    results = charge_run(charge_path, properties, names, 
                                func= __set_iv_parameters, **{"bias_regime":"forward","name": names, "path": path, "def_sim_region":def_sim_region})
                except LumericalError:
                    pce, ff, voc, jsc, current_density, voltage, stop, p = (np.nan for _ in range(8))
                    PCE_temp.append(pce)
                    FF_temp.append(ff)
                    Voc_temp.append(voc)
                    Jsc_temp.append(jsc)
                    
                    
                    
                    Current_Density.append(current_density)
                    Voltage.append(voltage)
                    print(f"Semiconductor {names.SCName}, cathode {names.Cathode}\n Voc = {Voc[-1]:.3f}V \n Jsc =  {Jsc[-1]:.4f} mA/cm² \n FF = {FF[-1]:.3f} \n PCE = {PCE[-1]:.3f}%")
                    continue

            pce, ff, voc, jsc, current_density, voltage, stop, p = iv_curve( results[0],"am", names)
            PCE_temp.append(pce)
            FF_temp.append(ff)
            Voc_temp.append(voc)
            Jsc_temp.append(jsc)
            Current_Density.append(current_density)
            Voltage.append(voltage)
            print(f"Semiconductor {names.SCName}, cathode {names.Cathode}\n Voc = {Voc[-1]:.3f}V \n Jsc =  {Jsc[-1]:.4f} mA/cm² \n FF = {FF[-1]:.3f} \n PCE = {PCE[-1]:.3f}%")
        PCE = np.average(PCE_temp)
        FF = np.average(FF_temp)
        Voc = np.average(Voc_temp)
        Jsc = np.average(Jsc_temp)

        
        
        # plot(pce, ff, voc, jsc, current_density, voltage, stop, 'am',p) 
    
    
    return pce, ff, voc, jsc, current_density, voltage #change to upper case when running more than one material



def band_diagram(active_region_list,charge_file, path, properties, def_sim_region = None, v_max = None, V_band_diagram = 1):
    charge_path = os.path.join(path, charge_file)
    #get_results = {"CHARGE": {"monitor": " bandstructure"}}
    Ec, Ev, Efn, Efp, Thickness = [], [], [], [], []
    for names in active_region_list:
        get_results = {"results": {"CHARGE::monitor": "bandstructure"}}  # get_results: Dictionary with the properties to be calculated
        results = charge_run(charge_path, properties, get_results, 
                                func= __set_iv_parameters, delete = True, device_kw={"hide": True},**{"bias_regime":"forward","name": names, "path": path, "v_max": v_max , "single_point":True,"def_sim_region":def_sim_region, "V_band_diagram":V_band_diagram })
        bandstructure =results[0]
        ec= bandstructure['results.CHARGE::monitor.bandstructure']["Ec"].flatten()
        thickness = bandstructure['results.CHARGE::monitor.bandstructure']["z"].flatten()
        ev = bandstructure['results.CHARGE::monitor.bandstructure']["Ev"].flatten()
        efn = bandstructure['results.CHARGE::monitor.bandstructure']["Efn"].flatten()
        efp = bandstructure['results.CHARGE::monitor.bandstructure']["Efp"].flatten()
        Ec.append(ec)
        Ev.append(ev)
        Efn.append(efn)
        Efp.append(efp)
        Thickness.append(thickness)
    if len(active_region_list) > 1 :
        return Thickness,Ec, Ev, Efn, Efp
    else:
        return thickness, ec, ev, efn, efp

def abs_extraction(names, path, fdtd_file, properties = {}): 
    fdtd_path = os.path.join(path, fdtd_file)
    override_prefix: str = str(uuid4())[0:5]
    new_fdtd_file = override_prefix + "_" + fdtd_file
    new_filepath_fdtd: str = os.path.join(path, new_fdtd_file)
    shutil.copyfile(fdtd_path, new_filepath_fdtd)
    log_file_fdtd: str = os.path.join(
        path, f"{override_prefix}_{os.path.splitext(fdtd_file)[0]}_p0.log"
    )
    with lumapi.FDTD(filename = new_filepath_fdtd, hide = True) as fdtd: 
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
    results = pd.DataFrame({'wvl':abs['lambda'].flatten(), 'pabs':abs['Pabs_total']})
    results_path = os.path.join(path, names.SolarGenName)
    results.to_csv(results_path +'.csv', header = ('wvl', 'abs'), index = False) #saves the results in file in path
    os.remove(new_filepath_fdtd)
    os.remove(log_file_fdtd)
    
    
#untested: 
def charge_extract( names, path, charge_file, properties = {}):
    charge_path = os.path.join(path, charge_file)
    override_prefix: str = str(uuid4())[0:5]
    new_charge_file = override_prefix + "_" + charge_file
    new_filepath_charge: str = os.path.join(path, new_charge_file)
    shutil.copyfile(charge_path, new_filepath_charge)
    log_file_charge: str = os.path.join(
        path, f"{override_prefix}_{os.path.splitext(charge_file)[0]}_p0.log"
    )
    with lumapi.DEVICE(filename=new_filepath_charge, hide = True) as charge:
        charge.switchtolayout()
        for structure_key, structure_value in properties.items():
            charge.select(structure_key)
            for parameter_key, parameter_value in structure_value.items():
                charge.set(parameter_key, parameter_value)
        charge.save()
        charge.select("materials::"+ names.SCName + "::" + names.SCName)
        T = 300
        Eg = charge.get("electronic.gamma.Eg.constant")
        mn = charge.get("electronic.gamma.mn.constant")
        mp = charge.get("electronic.gamma.mp.constant")
        mn = mn*9.11*10**-31
        mp = mp*9.11*10**-31
        Nc = 2*((2*np.pi*mn*k*T/(h**2))**1.5)*10**-6
        Nv = 2*((2*np.pi*mp*k*T/(h**2))**1.5)*10**-6
        ni = ((Nv*Nc)**0.5)*(np.exp(-Eg*e/(2*k*T)))
        charge.select("geometry::" + names.SCName)
        L = charge.get("z span") # in m 
        charge.close()
    os.remove(new_filepath_charge)
    #os.remove(log_file_charge)
    return Eg, L, ni

def adjust_abs(energy, new_abs, Eg):
    return [0 if e < Eg else new_abs[i] for i, e in enumerate(energy)]

def phi_bb(E):
    h_ev = h/e 
    k_b = k / e 
    return ((2*np.pi*E**2)/(h_ev**3*c**2) * (np.exp(E/(k_b*300))-1)**-1) #(eV.s.m2)-1

def extract_B_radiative(active_region_list, path, fdtd_file, charge_file, properties = {} , run_abs = True):
        B_list = []
        for names in active_region_list:
            results_path = os.path.join(path, names.SolarGenName)
            if run_abs == True:
                abs_extraction(names, path, fdtd_file, properties)
            else: 
                try:
                    abs_data = pd.read_csv(results_path +'.csv')
                except FileNotFoundError: 
                    abs_extraction(names, path, fdtd_file, properties)
            abs_data = pd.read_csv(results_path +'.csv')   
            wvl = np.linspace(min(abs_data['wvl']), max(abs_data['wvl']), 70000) #wvl in m
            abs = np.interp(wvl, abs_data['wvl'],abs_data['abs'])
            energy = 1240/(wvl*1e9)
            Eg, L, ni = charge_extract( names, path, charge_file, properties)
            abs = adjust_abs(energy, abs, Eg) #new absorption with cut off below Eg
            bb_spectrum = phi_bb(energy)
            Jo  = e * np.trapz(-bb_spectrum*abs, energy) #A/m2
            Jo = Jo* 0.1 #mA/cm2
            B = Jo*10**-5/(e*(ni**2)*(L))
            B_list.append(B)
        return B_list 



def _get_replacement(lst):
    for i in range(len(lst)):
        if lst[i]<0:
            lst[i] = 0
    return(lst)

def get_iv_4t(folder, pvk_voltage_file, pvk_current_file, si_voltage_file, si_current_file):
    '''Plots the IV curve of a tandem solar cell, in 4-terminal configuration, with 2 subcells in parallel.
    Note: Bottom subcell is divided into 2 cells, connected in series, to double the VOC.
    Args:
            folder: folder where the .txt files are stored
            pvk_voltage_file: name of the PVK voltage file
            pvk_current_file: name of the PVK current file
            si_voltage_file: name of the Silicon voltage file
            si_current_file: name of the Silicon current file
    '''
    # PEROVSKITE______________________
    pvk_v = np.array(pd.read_csv(os.path.join(folder, f"{pvk_voltage_file}.txt"), header = None)).flatten()
    pvk_iv = - pd.read_csv(os.path.join(folder, f"{pvk_current_file}.txt"), header = None)
    pvk_iv = _get_replacement(np.array(pvk_iv).flatten())

    plt.figure(figsize=(5.5,5))
    plt.plot(pvk_v, pvk_iv, label = 'PVK subcell', c = 'steelblue')

    # Determine Voc,pvk
    Voc_pvk = pvk_v[np.where(pvk_iv == 0)[0]][0]

    # SILICON__________________________
    si_v = np.array(pd.read_csv(os.path.join(folder, f"{si_voltage_file}.txt"), header = None))*2 # multiplied to get 2 series cells
    si_iv = - (np.array(pd.read_csv(os.path.join(folder, f"{si_current_file}.txt"), header = None))/2).flatten()
    si_iv = _get_replacement(si_iv)
    plt.plot(si_v, si_iv, label = '2 series Si subcell', c = 'yellowgreen')

    # Determine Voc,Si
    Voc_si = si_v[np.where(si_iv == 0)[0]][0]

    # TANDEM____________________________

    # Determine Voc
    Voc = min(Voc_pvk, Voc_si)

    # Determine voltage
    tan_v = np.array([v for i, v in enumerate(pvk_v) if i % 2 == 0])
    tan_leng = len(tan_v[tan_v<=Voc]) #number of elements in the array that are less than Voc

    # Determine current
    pvk_iv = [v for i, v in enumerate(pvk_iv) if i % 2 == 0]
    tandem_iv = si_iv[:len(pvk_iv)]+pvk_iv
    tan_i = []

    for i in range(len(tandem_iv)):
        if i<=(tan_leng-1):
            tan_i.append(tandem_iv[i])
        else:
            tan_i.append(0)
    leng = min(len(tan_v), len(tan_i))

    # PLOT Tandem IV curve
    plt.plot(tan_v[:leng], tan_i[:leng], label = 'PSTSC', c = 'crimson', linestyle='--')
    plt.legend()
    plt.ylabel('Current Density [mA/cm2]')
    plt.xlabel('Voltage [V]')
    plt.ylim(0,30)
    plt.xlim(-0.1,2)
    plt.grid(linestyle=':')

    plt.savefig(os.path.join(folder, "iv_curve_4t.svg"), dpi = 300, bbox_inches = 'tight')

    # Get IV curve variables
    voltage = si_v[:len(tandem_iv)].flatten()
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
    PCE = ((FF * Voc * abs(Isc)*10) / (Ir)) * 100

    print(f'FF = {FF[0]:.2f}, PCE = {PCE[0]:.2f} %, Voc = {Voc:.2f} V, Isc = {Isc[0]:.2f} mA/cm2')