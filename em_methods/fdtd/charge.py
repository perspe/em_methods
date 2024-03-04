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
from PyAstronomy import pyaC

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
             bias_regime:str,  
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

    
    cathode_name = get_results['results']['CHARGE']
    __set_iv_parameters(basefile, cathode_name, bias_regime)
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

    
    current_forward, voltage_forward = iv_curve(new_filepath, 
             get_results,
             device_kw={"hide": True})
    
    # voltage_reverse, current_reverse = iv_curve(new_filepath, 
    #          properties,
    #          get_results,
    #          device_kw={"hide": True})
    
    #voltage = voltage_reverse[::-1]+voltage_forward
    #current = current_reverse[::-1]+current_forward

    voltage = voltage_forward
    current = current_forward

    print(voltage)
    print(current)

    PCE = __plot_iv_curve(basefile, current, voltage)
    

    return results, charge_runtime, analysis_runtime, autoshut_off_list, PCE

def __charge_run_analysis(basefile: str,
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
    
    """ Plots the iv curve of the cell in CHARGE. 
    Args:
        basefile: Path of the CHARGE file (e.g. C:\\Users\\MonicaDyreby\\Documents\\Planar silicon solar cell\\solar_cell.ldev")
        bias_regime: defines the regime (i.e. forward or reverse) in lower case
        get retults: Dictionary with the cathode (e.g. "base") results path inside CHARGE (e.g. {"results":{"CHARGE":"base"}})
        
        full example: iv_curve(r"C:\\Users\\MonicaDyreby\\Documents\\Planar silicon solar cell\\solar_cell.ldev", {},{"results":{"CHARGE":"base"}})
    """

    #obtains IV curve from already run simulation
    cathode_name = get_results['results']['CHARGE']
    results = __charge_run_analysis(basefile, get_results, device_kw)
    current = list(results['results.CHARGE.'+cathode_name]['I'])
    voltage = list(results['results.CHARGE.'+cathode_name]['V_'+cathode_name])

    if len(current) == 1 and len(current[0]) != 1: #charge I output is not always concistent
        current = current[0]
        voltage = [float(arr[0]) for arr in voltage] 
    
    return np.array(current), np.array(voltage)
    
    

def __plot_iv_curve(basefile:str, current, voltage):
    
    logger.warning("Attention the Voc results are determined through finding the abs min fo the current density. That is, they are not the result of an interpolation and depend on the \n data aquisition process -> make sure to collect a lot of points around Voc to get a more accurate value")
    logger.warning("Make sure the name of the simulation region is correct in the script")
    
    with lumapi.DEVICE(filename=basefile, hide = True) as charge:
        charge.select("simulation region")
        Lx = charge.get("x span")
        if "3D" not in charge.get("dimension"):
            print("This is a 2D simulation")
            charge.select("CHARGE")
            Ly = charge.get("norm length")
        else:
            print("This is a 3D simulation")
            charge.select("simulation region")
            Ly = charge.get("y span")
    Lx = Lx * 100 #from m to cm
    Ly = Ly * 100 #from m to cm
    area = Ly*Lx #area in cm^2 
    current_density = (np.array(current)*1000)/area
    fig, ax = plt.subplots()
    Ir = 1000 #W/m²

    #DETERMINE JSC (ROUGH)
    abs_voltage_min = min(np.absolute(voltage)) #volatage value closest to zero
    if abs_voltage_min in voltage:
        Jsc = current_density[np.where(voltage == abs_voltage_min)[0]][0]
        Isc = current[np.where(voltage == abs_voltage_min)[0]][0]
        #Jsc = current_density[voltage.index(abs_voltage_min)]
        #Isc = current[voltage.index(abs_voltage_min)]   
    elif -abs_voltage_min in voltage: 
        #the position in the array of Jsc and Isc should be the same
        Jsc = current_density[np.where(voltage == -abs_voltage_min)[0]][0]
        Isc = current[np.where(voltage == -abs_voltage_min)[0]][0]
        #Jsc = current_density[voltage.index(-abs_voltage_min)]
        #Isc = current[voltage.index(-abs_voltage_min)]   
 

    #DETERMINE VOC THROUGH INTERPOLATION
    Voc = pyaC.zerocross1d(voltage, current_density, getIndices=False)[0]       
    
    #DETERMINE VOC (ROUGH) through J    
    # abs_current_min = min(np.absolute(current_density)) #voltage value closest to zero
    # if abs_current_min in current_density:
    #     Voc = voltage[np.where(current_density == abs_current_min)[0]][0]
    #     #Voc = voltage[current_density.index(abs_current_min)]
    # elif -abs_current_min in current_density: 
    #     Voc = voltage[np.where(current_density == -abs_current_min)[0]][0]
    #     #Voc = voltage[current_density.index(-abs_current_min)]
    # print(Voc) #V

    #Voc_index = voltage.index(Voc)

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.plot(voltage[:-12], current_density[:-12], 'o-') 
    plt.plot(Voc, 0, 'o', color = "red")       
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
    return PCE


def __set_iv_parameters(basefile:str, cathode_name:str, bias_regime:str):
    """ Sets the iv curve parameters and ensure correct solver is selected (e.g. start range, stop range...)
    Args:
        basefile: directory of file
        cathode_name: name of the cathode associated with the subcell in question
        subcell_name: name of the subcell (i.e Si or Perovskite)
    """
    with lumapi.DEVICE(filename=basefile, hide = True) as charge: #set the electode sweep as range (forward bias)
        charge.switchtolayout()
        #setting sweep parameters
        if (bias_regime == "forward"):
            charge.select("CHARGE")
            charge.set("solver type","NEWTON")
            charge.set("enable initialization", True)
            charge.set("init step size",5) #rather small init step. If sims take too long to start look into changing it to a larger value
            charge.save()
            
            
            charge.select("CHARGE::boundary conditions::"+str(cathode_name))
            charge.set("sweep type","range")
            charge.save()
            charge.set("range start",0) 
            charge.set("range stop",2) 
            charge.set("range num points",21) 
            charge.set("range backtracking","enabled")
            charge.save()

        #reverse bias
        if(bias_regime == "reverse"):
            charge.select("CHARGE")
            charge.set("solver type","GUMMEL")
            charge.set("enable initialization", False)
            charge.save()

            charge.select("CHARGE::boundary conditions::"+str(cathode_name))
            charge.set("sweep type","range")
            charge.save()
            charge.set("range start",0) 
            charge.set("range stop",-1) 
            charge.set("range num points",21)
            charge.set("range backtracking","enabled")
            charge.save()


def get_gen(path, fdtd_file, properties, gen_mat): 
    """ Alters the cell design ("properties"), simulates the FDTD file, and creates the generation rate .mat file(s) (in same directory as FDTD file)
    Args:
        path: String of folder path of FDTD and CHARGE files (must be in same folder);
        fdtd_file: String FDTD file name;
        properties: Dictionary with the property object and property names and values
        gen_mat: Dictionary with the absorbers and corresponding strings {material_1: ["solar_gen_1", "1.mat", "geometry_name_1"], material_2: ["solar_gen_2", "2.mat", "geometry_name_2"]}
    """
    fdtd_path =  str(path)+"\\"+str(fdtd_file)

    basepath, basename = os.path.split(fdtd_path)
    override_prefix: str = str(uuid4())[0:5]
    new_filepath: str = os.path.join(
        basepath, override_prefix + "_" + basename)
    new_path = shutil.copyfile(fdtd_path, new_filepath)
    
    with lumapi.FDTD(filename = new_path, hide = True) as fdtd:
        # CHANGE CELL GEOMETRY
        for structure_key, structure_value in properties.items():
            fdtd.select(structure_key)
            for parameter_key, parameter_value in structure_value.items():
                fdtd.set(parameter_key, parameter_value)
        fdtd.save()

        # EXPORT GENERATION FILES
        for mat, names in gen_mat.items():
            gen_obj = names[0] # Solar Generation analysis object name
            file = names[1] # generation file name
            g_name = file.replace('.mat', '')
            fdtd.select(str(gen_obj))
            fdtd.set("export filename", str(g_name))
        fdtd.run()
        fdtd.runanalysis()
        fdtd.save()
        fdtd.close()

def updt_gen(path, charge_file, gen_mat): 
    """ Alters the cell DESIGN ("properties"), IMPORTS the generation rate .mat file(s) (in same directory as FDTD file) and simulates the CHARGE file
    Args:
        path: String of folder path of FDTD and CHARGE files (must be in same folder);
        charge_file: String DEVICE file name;
        gen_mat: Dictionary with the absorbers and corresponding strings {material_1: ["solar_gen_1", "1.mat", "geometry_name_1"], material_2: ["solar_gen_2", "2.mat", "geometry_name_2"]}
    """
    charge_path =  str(path)+"\\"+str(charge_file)

    basepath, basename = os.path.split(charge_path)
    override_prefix: str = str(uuid4())[0:5]
    new_filepath: str = os.path.join(
        basepath, override_prefix + "_" + basename)
    new_path = shutil.copyfile(charge_path, new_filepath)

    with lumapi.DEVICE(filename = new_path, hide = True) as charge:

        # CHANGE GENERATION PROFILE                
        for mat, names in gen_mat.items():
            file = names[1] #Generation file name
            obj = names[2] # Generation CHARGE object name
            
            # Create "Import generation rate" objects
            g_name = file.replace('.mat', '')
            charge.addimportgen()
            charge.set("name", g_name)
            
            # Import generation file path
            charge.set("volume type", "solid")
            charge.set("volume solid",str(obj))
            charge.importdataset(str(path)+'\\'+str(file))
            charge.save()
        charge.close()



def get_tandem_results(path, fdtd_file, charge_file, properties, gen_mat, bias_regime, get_results) #TODO get more results (FF, Voc, Isc, ...)
    get_gen(path, fdtd_file, properties, gen_mat)
    updt_gen(path, charge_file, gen_mat)
    basefile = str(path)+"\\"+str(charge_file)
    results, charge_runtime, analysis_runtime, autoshut_off_list, pce = charge_run(basefile, bias_regime, properties, get_results)
    return pce