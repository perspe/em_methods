from threading import Thread
from multiprocessing import Process, Queue
import os
import sys
from typing import List, Dict, Union
import logging
import time

# Get module logger
logger = logging.getLogger("dev")

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


class CheckRunState(Thread):
    """
    Thread to read Lumerical Log file and
    detect if error occured...
    If so kill running process
    """

    def __init__(self, logfile: str, lum_process: Process):
        super().__init__(daemon=True)
        self.logfile: str = logfile
        self.lum_process: Process = lum_process

    def run(self):
        while True:
            if not os.path.isfile(self.logfile):
                logger.debug("Logfile not detected")
                time.sleep(5)
                continue
            curr_state: int = self._check_state()
            if curr_state == -1:
                logger.critical("Error detected in simulation")
                self.lum_process.terminate()
                break
            elif curr_state == 1:
                logger.info("Simulation Finished Successfully")
                break
            time.sleep(5)

    def _check_state(self):
        """Determine running state for file"""
        with open(self.logfile, "r") as logfile:
            log_data: str = logfile.read()
        if "Simulation completed successfully" in log_data:
            return 1
        elif "Error" in log_data:
            return -1
        else:
            return 0


class RunLumerical(Process):
    """
    Main Process to run the lumerical file.
    This method avoids problems when the simulation has
        any kind of error and just gets stuck.
    By running in a different process it is possible to kill
        the process and still keep running any script
    """

    def __init__(
        self,
        queue: Queue,
        filepath: str,
        properties: Dict[str, Dict[str, float]],
        get_results: Dict[str, Dict[str, Union[str, List]]],
        lum_kw: Dict,
        func=None,
        **kwargs,
    ):
        super().__init__()
        self.queue = queue
        self.filepath = filepath
        self.properties = properties
        self.get_results = get_results
        self.func = func
        self.lum_kw = lum_kw
        self.kwargs = kwargs

    def run(self):
        with lumapi.DEVICE(filename=self.filepath, **self.lum_kw) as charge:
            charge.switchtolayout()
            # Update structures
            for structure_key, structure_value in self.properties.items():
                logger.debug(f"Editing: {structure_key}")
                charge.select(structure_key)
                for parameter_key, parameter_value in structure_value.items():
                    logger.debug(f"Updating: {parameter_key} to {parameter_value}")
                    charge.set(parameter_key, parameter_value)
            charge.save()
            if self.func is not None:
                logger.debug(f"Running External function {self.func.__name__}\n{self.kwargs}...")
                self.func(charge, **self.kwargs)
            
            # Note: The double fdtd.runsetup() is important for when the setup scripts
            #       (such as the model script) depend on variables from other
            #       scripts. For example the model scripts needs the internal property
            #       of a layer generated from a structure group.
            #       The first run updates internally all the values
            #       The second run then updates all the structures with the updated values
            charge.runsetup()
            charge.runsetup()
            logger.debug(f"Running...")
            start_time = time.time()
            charge.switchtolayout()
            charge.run("CHARGE")
            charge_runtime = time.time() - start_time
            self.queue.put(charge_runtime)
            start_time = time.time()
            charge.runanalysis()
            analysis_runtime = time.time() - start_time
            self.queue.put(analysis_runtime)
            logger.info(
                f"Simulation took: CHARGE: {charge_runtime:0.2f}s | Analysis: {analysis_runtime:0.2f}s"
            )
            results = _get_lumerical_results(charge, self.get_results)
            self.queue.put(results)


class LumericalError(Exception):
    """
    Class to aggregate all lumerical exception errors
    """

    def __init__(self, message) -> None:
        self.message = message
        super().__init__(self.message)


def _get_lumerical_results(
    lum_handler, get_results: Dict[str, Dict[str, Union[List, str]]]
) -> Dict:
    """
    Alias function to extract results from Lumerical file (to avoid code redundancy)
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
                results["data." + key + "." + value_i] = lum_handler.getdata(
                    key, value_i
                )
    if "results" in get_results_info:
        for key, value in get_results["results"].items():
            if not isinstance(value, list):
                value = [value]
            for value_i in value:
                logger.debug(f"Getting result for: '{key}':'{value_i}'")
                results["results." + key + "." + value_i] = lum_handler.getresult(
                    key, value_i
                )
    return results
