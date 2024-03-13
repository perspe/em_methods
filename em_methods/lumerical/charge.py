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
            names: one array element of the gen_mat dictionary
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
    check_thread = CheckRunState(log_file, run_process)
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
            names: one of the array elements in the gen_mat dictionary with the absorbers and corresponding strings
            {material_1: ["solar_gen_1", "1.mat", "geometry_name_1", "cathode_1", "anode_1"],
            material_2: ["solar_gen_2", "2.mat", "geometry_name_2", "cathode_2", "anode_2"]}
            names would thus be (for instance): ["solar_gen_2", "2.mat", "geometry_name_2", "cathode_2", "anode_2"]

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


def plot_iv_curve(results, regime, names):

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
    fig, ax = plt.subplots()
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
        print(voltage)
        print(current_density)
        Voc, stop = pyaC.zerocross1d(np.array(voltage), np.array(current_density), getIndices=True)
        stop = stop[0]
        Voc = Voc[0]
        props = dict(boxstyle="round", facecolor="white", alpha=0.5)

        plt.plot(voltage[: stop + 2], current_density[: stop + 2], "o-")
        plt.plot(Voc, 0, "o", color="red")

        P = [
            voltage[x] * abs(current[x]) for x in range(len(voltage)) if current[x] < 0
        ]  # calculate the power for all points [W]

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
        FF = abs(max(P) / (Voc * Isc))
        PCE = ((FF * Voc * abs(Isc)) / (Ir * (area * 10**-4))) * 100
        textstr = f"Voc = {Voc:.3f}V \n Jsc =  {Jsc:.4f} mA/cm² \n FF = {FF:.3f} \n PCE = {PCE:.3f}%"
        print(textstr)
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
        return PCE, FF, Voc, Jsc

    elif regime == "dark":
        plt.plot(voltage, current_density, "o-")
        plt.ylabel("Current density [mA/cm²] ")
        plt.xlabel("Voltage [V]")
        plt.axhline(0, color="gray", alpha=0.3)
        plt.axvline(0, color="gray", alpha=0.3)
        plt.grid()
        plt.show()


def get_gen(path, fdtd_file, properties, active_region_list):
    """Alters the cell design ("properties"), simulates the FDTD file, and creates the generation rate .mat file(s)
    (in same directory as FDTD file)
    Args:
        path: String of folder path of FDTD and CHARGE files (must be in same folder);
        fdtd_file: String FDTD file name;
        properties: Dictionary with the property object and property names and values
        gen_mat: Dictionary with the absorbers and corresponding strings
        {material_1: ["solar_gen_1", "1.mat", "geometry_name_1", "cathode_1", "anode_1"],
        material_2: ["solar_gen_2", "2.mat", "geometry_name_2", "cathode_2", "anode_2"]}
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

def __set_iv_parameters(charge, bias_regime: str, name: SimInfo, path:str):
    """ Imports the generation rate into new CHARGE file, creates the simulation region based on the generation rate, 
        sets the iv curve parameters and ensure correct solver is selected (e.g. start range, stop range...)
    Args:
        basefile: directory of file
        bias_regime: forward or reverse bias
        name: SimInfo dataclass structure about the simulation (e.g. SimInfo("solar_generation_PVK", "G_PVK.mat", "Perovskite", "ITO_top", "ITO"))
    """
    def_sim_region = "yes"
    charge.switchtolayout()
    
    # Create "Import generation rate" objects
    charge.addimportgen()
    charge.set("name", str(name.GenName[:-4]))
    print(name.GenName[:-4])

    # Import generation file path
    charge.set("volume type", "solid")
    charge.set("volume solid", str(name.SCName))
    charge.importdataset(os.path.join(path, name.GenName))
    charge.save()

    if def_sim_region == "yes":  # Is it necessary to define a simulation region
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

        # Creates the simulation region (2D or 3D)-UNTESTED
        charge.addsimulationregion()
        charge.set("name", name.SCName)  # defines the name of the simulation region as the name of the semiconductor in question
        if True:
            charge.set("dimension", 2)
            charge.set("x", x)
            charge.set("x span", x_span)
            charge.set("y", y)

        if False:
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
        # charge.set("init step size",5) #rather small init step. If sims take too long to start look into changing it to a larger value
        charge.save()

        # setting sweep parameters
        print("we are in the forward section")
        charge.select("CHARGE::boundary conditions::" + str(name.Cathode))
        charge.set("sweep type", "range")
        charge.save()
        charge.set("range start", 0)
        charge.set("range stop", 1.5)
        charge.set("range num points", 11)
        charge.set("range backtracking", "enabled")
        charge.save()

    # reverse bias
    elif bias_regime == "reverse":
        charge.select("CHARGE")
        charge.set("solver type", "GUMMEL")
        charge.set("enable initialization", False)
        charge.save()

        # setting sweep parameters
        charge.select("CHARGE::boundary conditions::" + str(name.Cathode))
        charge.set("sweep type", "range")
        charge.save()
        charge.set("range start", 0)
        charge.set("range stop", -1)
        charge.set("range num points", 21)
        charge.set("range backtracking", "enabled")
        charge.save()
    
    charge.select(name.SCName)
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
    

    





def updt_gen(path, charge_file, gen_mat, bias_regime, properties):
    """Alters the cell DESIGN ("properties"), IMPORTS the generation rate .mat file(s) (in same directory as FDTD file)
    and creates the simulation regions
    Args:
        path: String of folder path of FDTD and CHARGE files (must be in same folder);
        charge_file: String DEVICE file name;
        gen_mat: Dictionary with the absorbers and corresponding strings
        {material_1: ["solar_gen_1", "1.mat", "geometry_name_1", "cathode_1", "anode_1"],
        material_2: ["solar_gen_2", "2.mat", "geometry_name_2", "cathode_2", "anode_2"]}
    """

    charge_path = os.path.join(path, charge_file)
    override_prefix: str = str(uuid4())[0:5]
    new_filepath: str = os.path.join(path, override_prefix + "_" + charge_file)
    shutil.copyfile(charge_path, new_filepath)

    PCE = [] 
    PCE.append( charge_run(
                    new_filepath,
                    properties,
                    names,
                    func=__set_iv_parameters,
                    **{"bias_regime": bias_regime, "name": names},
                )[3]
            )
    print(f"Semiconductor {names[2]}, cathode {names[3]}, PCE = {PCE[-1]}")

    return PCE
