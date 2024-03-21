import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Union, Dict, List, Tuple
import logging
from uuid import uuid4
import shutil
import sys
from dataclasses import dataclass 
from matplotlib.patches import Rectangle
from PyAstronomy import pyaC
from multiprocessing import Queue
from em_methods.lumerical.lum_helper import (
    CheckRunState,
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
    names: SimInfo,
    *,
    get_info: Dict[str, str] ={},
    func=None,
    savepath: Union[None, str] = None,
    override_prefix: Union[None, str] = None,
    delete: bool = False,
    device_kw={"hide": False},
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
    get_results = {"results": {"CHARGE": str(names.Cathode)}
    }  # get_results: Dictionary with the properties to be calculated
    process_queue = Queue()
    run_process = RunLumerical(
        LumMethod.CHARGE,
        proc_queue=process_queue,
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
    check_thread = CheckRunState(log_file, run_process, process_queue)
    check_thread.start()
    logger.debug("Run Process Started...")
    run_process.join()
    if process_queue.empty():
        raise LumericalError("Simulation Finished Prematurely")
    # Extract data from process
    data = []
    while not process_queue.empty():
        data.append(process_queue.get())
    # Check for other possible runtime problems
    if len(data) < 2:
        raise LumericalError("Error Running simulation")
    elif len(data) == 2:
        raise LumericalError("Error acquiring data (problem in get_results)")
    elif len(data) == 3:
        raise LumericalError("Error acquiring results (problem in get_info)")
    elif len(data) > 4:
        raise LumericalError("Unknown problem")
    charge_runtime, analysis_runtime, results, data_info = tuple(data)
    if delete:
        os.remove(new_filepath)
        os.remove(log_file)

    
    return results, charge_runtime, analysis_runtime, data_info


def charge_run_analysis(basefile: str, names, device_kw={"hide": True}):
    """
    Generic function gather simulation data from already simulated files
    Args:
            basefile: Path to the original file
            names: SimInfo dataclass structure about the simulation (e.g. SimInfo("solar_generation_PVK", "G_PVK.mat", "Perovskite", "ITO_top", "ITO"))

    Return:
            results: Dictionary with all the results
            time: Time to run the simulation
    """
    get_results = {"results": {"CHARGE": str(names.Cathode)}}
    with lumapi.DEVICE(filename=basefile, **device_kw) as charge:
        results = _get_lumerical_results(charge, get_results)
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
    Lx = results["func_output"][0]
    Ly = results["func_output"][1]

    if (len(current) == 1 and len(current[0]) != 1):  # charge I output is not always consistent
         current = current[0]
         voltage = [float(arr[0]) for arr in voltage]

    Lx = Lx * 100  # from m to cm
    Ly = Ly * 100  # from m to cm
    area = Ly * Lx  # area in cm^2
    current_density = (np.array(current) * 1000) / area
    Ir = 1000  # W/m²

    if regime == "am":
        # DETERMINE JSC (ROUGH)
        abs_voltage_min = min(np.absolute(voltage))  # volatage value closest to zero
        if abs_voltage_min in voltage:
            Jsc = current_density[np.where(voltage == abs_voltage_min)[0]][0]
            Isc = current[np.where(voltage == abs_voltage_min)[0]][0]
            # Jsc = current_density[voltage.index(abs_voltage_min)]
            # Isc = current[voltage.index(abs_voltage_min)]
        elif -abs_voltage_min in voltage:
            # the position in the array of Jsc and Isc should be the same
            Jsc = current_density[np.where(voltage == -abs_voltage_min)[0]][0]
            Isc = current[np.where(voltage == -abs_voltage_min)[0]][0]
            # Jsc = current_density[voltage.index(-abs_voltage_min)]
            # Isc = current[voltage.index(-abs_voltage_min)]

        # DETERMINE VOC THROUGH INTERPOLATION
        Voc, stop = pyaC.zerocross1d(np.array(voltage), np.array(current_density), getIndices=True)
        stop = stop[0]
        Voc = Voc[0]
        P = [voltage[x] * abs(current[x]) for x in range(len(voltage)) if current[x] < 0 ]  # calculate the power for all points [W]
        FF = abs(max(P) / (Voc * Isc))
        PCE = ((FF * Voc * abs(Isc)) / (Ir * (area * 10**-4))) * 100
        return PCE, FF, Voc, Jsc, current_density, voltage, stop, P

    elif regime == "dark":
        return current_density, voltage
        

def plot(PCE, FF, Voc, Jsc, current_density, voltage, stop, regime:str,P): #NOT FINISHED -> regime should not be an input, the performance metrics should be enough
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
    if regime == 'am':
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

    with lumapi.FDTD(filename=new_filepath, hide=True) as fdtd:
        # CHANGE CELL GEOMETRY
        for structure_key, structure_value in properties.items():
            fdtd.select(structure_key)
            for parameter_key, parameter_value in structure_value.items():
                fdtd.set(parameter_key, parameter_value)
        fdtd.save()

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

def __set_iv_parameters(charge, bias_regime: str, name: SimInfo, path:str, def_sim_region=None):
    """ 
    Imports the generation rate into new CHARGE file, creates the simulation region based on the generation rate, 
    sets the iv curve parameters and ensure correct solver is selected (e.g. start range, stop range...)
    Args:
            charge: as in "with lumapi.DEVICE(...) as charge
            bias_regime: forward or reverse bias
            name: SimInfo dataclass structure about the simulation (e.g. SimInfo("solar_generation_PVK", "G_PVK.mat", "Perovskite", "ITO_top", "ITO"))
            path: CHARGE file directory
            def_sim_region: optional input that defines if it is necessary to create a new simulation region. Possible input values include '2D', '2d', '3D', '3d'.
                        A simulation region will be defined accordingly to the specified dimentions. If no string is introduced then no new simulation region 
                        will be created
    Returns:
            
            Lx,Ly: dimentions of solar cell surface area normal to the incident light direction
    """
    valid_dimensions = {"2d", "3d"}
    if def_sim_region is not None and def_sim_region.lower() not in valid_dimensions:
        raise ValueError("def_sim_region must be one of '2D', '2d', '3D', '3d' or have no input")
    charge.switchtolayout()
    # Create "Import generation rate" objects
    charge.addimportgen()
    charge.set("name", str(name.GenName[:-4]))
    # Import generation file path
    charge.set("volume type", "solid")
    charge.set("volume solid", str(name.SCName))
    charge.importdataset(os.path.join(path, name.GenName))
    charge.save()
    if def_sim_region is not None: 
        # Defines boundaries for simulation region -UNTESTED
        charge.select("geometry::" + name.Anode)  # anode
        z_max = charge.get("z max")
        charge.select("geometry::" + name.Cathode)  # cathode
        z_min = charge.get("z min")
        charge.select("CHARGE::" + str(name.GenName[:-4]))  # solar generation
        x_span = charge.get("x span")
        x = charge.get("x")
        y_span = charge.get("y span")
        y = charge.get("y")
        # Creates the simulation region (2D or 3D)
        charge.addsimulationregion()
        charge.set("name", name.SCName) 
        if "2" in def_sim_region: 
            charge.set("dimension", 2)
            charge.set("x", x)
            charge.set("x span", x_span)
            charge.set("y", y)
        elif "3" in def_sim_region:
            charge.set("dimension", 3)
            charge.set("x", x)
            charge.set("x span", x_span)
            charge.set("y", y)
            charge.set("x span", y_span)
        charge.select(str(name.SCName))
        charge.set("z max", z_max)
        charge.set("z min", z_min)
        charge.select("CHARGE")
        charge.set("simulation region", name.SCName)
        charge.save()
    # Defining solver parameters
    if bias_regime == "forward":
        charge.select("CHARGE")
        charge.set("solver type", "NEWTON")
        charge.set("enable initialization", True)
        charge.set("init step size",1) #unsure if it works properly
        charge.save()
        # Setting sweep parameters
        print("we are in the forward section")
        charge.select("CHARGE::boundary conditions::" + str(name.Cathode))
        charge.set("sweep type", "range")
        charge.save()
        charge.set("range start", 0)
        charge.set("range stop", 1.5)
        charge.set("range num points", 11)
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
        sim_region = name.SCName
    else:
        sim_region = charge.getnamed("CHARGE","simulation region")
    charge.select(sim_region)
    Lx = charge.get("x span")
    if "3D" not in charge.get("dimension"):
        print("This is a 2D simulation")
        charge.select("CHARGE")
        Ly = charge.get("norm length")
    else:
        print("This is a 3D simulation")
        charge.select(str(sim_region))
        Ly = charge.get("y span")
    return Lx, Ly
    

def run_fdtd_and_charge(active_region_list, properties, charge_file, path, fdtd_file, def_sim_region=None):
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
    #get_gen(path, fdtd_file, properties, active_region_list)
    for names in active_region_list:
        results = charge_run(charge_path, properties, names,
            func= __set_iv_parameters, **{"bias_regime":"forward","name": names, "path": path, "def_sim_region":def_sim_region})
        pce, ff, voc, jsc, current_density, voltage, stop, p = iv_curve( results[0],"am", names)
        PCE.append(pce)
        FF.append(ff)
        Voc.append(voc)
        Jsc.append(jsc)
        Current_Density.append(current_density)
        Voltage.append(voltage)
        print(f"Semiconductor {names.SCName}, cathode {names.Cathode}\n Voc = {Voc[-1]:.3f}V \n Jsc =  {Jsc[-1]:.4f} mA/cm² \n FF = {FF[-1]:.3f} \n PCE = {PCE[-1]:.3f}%")
        plot(pce, ff, voc, jsc, current_density, voltage, stop, 'am',p) 
        print(Jsc)
    
    return PCE, FF, Voc, Jsc, Current_Density, Voltage
