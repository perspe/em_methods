import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from typing import Union, Dict, List, Tuple
import logging
from uuid import uuid4
import shutil
import time
import re
import numpy.typing as npt
import pandas as pd
from matplotlib.patches import Rectangle
from datetime import datetime

import scipy.constants as scc

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
import lumapi



def _get_charge_results(fdtd_handler: lumapi.DEVICE, get_results: Dict[str, Dict[str, float]]) -> Dict:
    """
    Alias function to extract results from FDTD file (to avoid code redundancy)
    """
    # Obtain results
    results = {}
    get_results_info = list(get_results.keys())
    if "data" in get_results_info:
        for key, value in get_results["data"].items():
            if not isinstance(value, list):
                value = [value]
            for value_i in value:
                logger.debug(f"Getting result for: '{key}':'{value_i}'")
                results["data."+key+"."+value_i] = fdtd_handler.getdata(key, value_i)
    if "results" in get_results_info:
        for key, value in get_results["results"].items():
            if not isinstance(value, list):
                value = [value]
            for value_i in value:
                logger.debug(f"Getting result for: '{key}':'{value_i}'")
                results["results."+key+"."+value_i] = fdtd_handler.getresult(key, value_i)
    return results

""" Main functions """

def charge_run(basefile: str,
             properties: Dict[str, Dict[str, float]],
             get_results: Dict[str, Dict[str, Union[str, List]]],
             *,
             savepath: Union[None, str] = None,
             override_prefix: Union[None, str] = None,
             delete: bool = False,
             device_kw: bool = {"hide": False}
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
    with lumapi.DEVICE(filename=new_filepath, **device_kw) as charge:
        
        # Update structures
        for structure_key, structure_value in properties.items():
            logger.debug(f"Editing: {structure_key}")
            charge.select(structure_key)
            for parameter_key, parameter_value in structure_value.items():
                logger.debug(
                    f"Updating: {parameter_key} to {parameter_value}")
                charge.set(parameter_key, parameter_value)
        # Note: The double fdtd.runsetup() is important for when the setup scripts
        #       (such as the model script) depend on variables from other
        #       scripts. For example the model scripts needs the internal property
        #       of a layer generated from a structure group.
        #       The first run updates internally all the values
        #       The second run then updates all the structures with the updated values
        # charge.runsetup()
        # charge.runsetup()
        logger.debug(f"Running...")
        start_time = time.time()
        charge.run("CHARGE")
        charge_runtime = time.time() - start_time
        start_time = time.time()
        charge.runanalysis()
        analysis_runtime = time.time() - start_time
        logger.info(
            f"Simulation took: CHARGE: {charge_runtime:0.2f}s | Analysis: {analysis_runtime:0.2f}s")
        results = _get_charge_results(charge, get_results)
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

    #Getting IV curve"
    #cathode_name = input("Input the cathode name:\n")
    #get_results_charge = {"results":{"CHARGE":str(cathode_name)}}   
    
    iv_curve(new_filepath, 
             properties, 
             get_results,
             device_kw={"hide": True})
    
    return results, charge_runtime, analysis_runtime, autoshut_off_list

def charge_run_analysis(basefile: str,
                      get_results: Dict[str, Dict[str, Union[str, List]]],
                      device_kw={"hide": True}):
    """
    Generic function gather simulation data from already simulated files
    Args:
            basefile: Path to the original file
            get_results: Dictionary with the properties to be calculated
    Return:
            results: Dictionary with all the results
            time: Time to run the simulation
    """
    with lumapi.DEVICE(filename=basefile, **device_kw) as charge:
        results = _get_charge_results(charge, get_results)
    return results

def find_index(wv_list, n):        
    index = [i for i in range(0,len(wv_list)) if wv_list[i] == n] 
    return index[0]

def iv_curve(basefile: str,
             get_results: Dict[str, Dict[str, Union[str, List]]],
             *,
             device_kw={"hide": True}):
    
    #obtains IV curve from already run simulation
    results = charge_run_analysis(basefile, get_results, device_kw)
    cathode_name = get_results['results']['CHARGE']
    current = list(results['results.CHARGE.'+cathode_name]['I'])
    voltage = list(results['results.CHARGE.'+cathode_name]['V_'+cathode_name])

    if len(current) == 1 and len(current[0]) != 1: #charge I output is not always concistent
        current = current[0]
        voltage = [float(arr[0]) for arr in voltage] 
    
    print("Attention the Voc results are determined through finding the abs min fo the current density. \n That is, they are not the result of an interpolation and depend on the \n data aquisition process -> make sure to collect a lot of points around Voc to get a more accurate value ")
    print("\n Make sure the name of the simulation region is correct in the script")
    
    with lumapi.DEVICE(filename=basefile, **device_kw) as charge:
        charge.select("simulation region_1")
        Lx = charge.get("x span")
        if "3D" not in charge.get("dimension"):
            print("This is a 2D simulation")
            charge.select("CHARGE")
            Ly = charge.get("norm length")
        else:
            print("This is a 3D simulation")
            charge.select("simulation region_1")
            Ly = charge.get("y span")
    Lx = Lx * 100 #from m to cm
    Ly = Ly * 100 #from m to cm
    area = Ly*Lx #area in cm^2 
    current_density = list((np.array(current)*1000)/area)
    fig, ax = plt.subplots()
    Ir = 1000 #W/m²



    #DETERMINE JSC (ROUGH)
    abs_voltage_min = min(np.absolute(voltage)) #volatage value closest to zero
    if abs_voltage_min in voltage:
        #Jsc = current_density[np.where(voltage == abs_voltage_min)[0]]
        #Isc = current[np.where(voltage == abs_voltage_min)[0]]
        Jsc = current_density[voltage.index(abs_voltage_min)]
        Isc = current[voltage.index(abs_voltage_min)]   
    elif -abs_voltage_min in voltage: 
        #the position in the array of Jsc and Isc should be the same
        #Jsc = current_density[np.where(voltage == -abs_voltage_min)[0]]
        #Isc = current[np.where(voltage == -abs_voltage_min)[0]]
        Jsc = current_density[voltage.index(-abs_voltage_min)]
        Isc = current[voltage.index(-abs_voltage_min)]   
 

    #DETERMINE VOC (ROUGH) through J   
    abs_current_min = min(np.absolute(current_density)) #voltage value closest to zero
    if abs_current_min in current_density:
        #Voc = voltage[np.where(current_density == abs_current_min)[0]]
        Voc = voltage[current_density.index(abs_current_min)]
    elif -abs_current_min in current_density: 
        #Voc = voltage[np.where(current_density == -abs_current_min)[0]]
        Voc = voltage[current_density.index(-abs_current_min)]
    print(Voc) #V

    Voc_index = voltage.index(Voc)

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)              
    plt.plot(voltage, current_density, 'o-')     
    # plt.plot(voltage[:Voc_index+2], current_density[:Voc_index+2], 'o-')

    P = [voltage[x]*abs(current[x]) for x in range(len(voltage)) if current[x]<0] #calculate the power for all points [W]


    ax.add_patch(Rectangle((voltage[find_index(P, max(P))],current_density[find_index(P, max(P))]),
                                                    -voltage[find_index(P, max(P))], -current_density[find_index(P, max(P))],
                                                    facecolor='lightsteelblue'))
    FF = abs(max(P)/(Voc*Isc))
    PCE = ((FF*Voc*abs(Isc))/(Ir*(area*10**-4)))*100
    
    textstr = 'Voc = ' + str(np.round(Voc, 3)) + ' V' + '\n' 'Jsc = '+ str(np.round(Jsc,6)) + ' mA/cm²' + "\n" "FF = " + str(np.round(FF,3)) + "\n" "PCE = " + str(np.round(PCE, 3)) + " %"
    print(textstr)
    plt.text(0.05, 0.80, textstr, transform=ax.transAxes, fontsize=17, verticalalignment='top', bbox=props)
    plt.ylabel('Current density [mA/cm²] ')
    plt.xlabel('Voltage [V]')
    plt.axhline(0, color='gray',alpha = 0.3)
    plt.axvline(0, color='gray',alpha = 0.3)        
    plt.grid()
    # plt.savefig(os.path.split(basefile)[1][:-3]+"_IV_"+ str(datetime.now())+".png")
    plt.show()

def get_gen(fdtd_file, properties):
    """ Alters the cell design ("properties"), simulates the FDTD file, and creates the generation rate .mat file(s) (in same directory as FDTD file)
    Args:
        properties: Dictionary with the property object and property names and values
        fdtd_file: String of path to the FDTD file
    """
    with lumapi.FDTD(filename = fdtd_file, hide = True) as fdtd:
        for structure_key, structure_value in properties.items():
            fdtd.select(structure_key)
            for parameter_key, parameter_value in structure_value.items():
                fdtd.set(parameter_key, parameter_value)
        fdtd.run()
        fdtd.runanalysis()
        fdtd.switchtolayout()
        fdtd.save()
        fdtd.close()

def updt_gen(path, charge_file, properties, gen_mat): 
    """ Alters the cell DESIGN ("properties"), IMPORTS the generation rate .mat file(s) (in same directory as FDTD file) and simulates the CHARGE file
    Args:
        properties: Dictionary with the property object and property names and values;
        path: String of folder path of FDTD and CHARGE files (must be in same folder);
        charge_file: String DEVICE file name;
        gen_mat: Dictionary with the absorbers and corresponding strings {material_1: ["1.mat", "geometry_name_1"], material_2: ["2.mat", "geometry_name_2"]}
    """
    # TODO(TO BE TESTED)
    charge_path =  str(path)+"\\"+str(charge_file)

    basepath, basename = os.path.split(charge_path)
    savepath: str = savepath or basepath
    override_prefix: str = override_prefix or str(uuid4())[0:5]
    new_filepath: str = os.path.join(
        savepath, override_prefix + "_" + basename)
    logger.debug(f"new_filepath:{new_filepath}")
    shutil.copyfile(path, new_filepath)

    with lumapi.DEVICE(filename = charge_path, hide = True) as charge:

        # CHANGE CELL GEOMETRY
        for structure_key, structure_value in properties.items():
            charge.select(structure_key)
            for parameter_key, parameter_value in structure_value.items():
                charge.set(parameter_key, parameter_value)
        charge.save()

        # CHANGE GENERATION PROFILE                
        for mat, names in gen_mat.items():
            file = names[0]
            obj = names[1]
            # Create "Import generation rate" objects
            g_name = file.replace('.mat', '')
            charge.addimportgen()
            charge.set("name", g_name)
            # Import generation file path
            charge.set('import file path', str(path)+'\\'+str(file))
            charge.set("volume type", "solid")
            charge.set("volume solid",str(obj))
            charge.save()
        charge.run("CHARGE")
        charge.switchtolayout()
        charge.save()
        charge.close()
    

    
    






















