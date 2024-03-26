from threading import Thread
from multiprocessing import Process, Queue
import os
import sys
from typing import List, Dict, Union, Callable, Any
import logging
import logging.handlers
import time
from enum import Enum

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

class LumMethod(Enum):
    """ Enum class with the different Lumerical Simulation methods """
    CHARGE = 1
    DEVICE = 1
    FDTD = 2
    HEAT = 3

def _run_method(method: LumMethod) -> Union[None,Callable]:
    """ Function that determines the Solver to be used for the simulation """
    logger.debug(f"Method: {method}")
    if method == LumMethod.CHARGE or method == LumMethod.DEVICE:
        return lumapi.DEVICE
    elif method == LumMethod.FDTD:
        return lumapi.FDTD
    else:
        return None


# class CheckRunState(Thread):
#     """
#     Thread to check Lumerical Run State. Kill if error occurred
#     Args:
#         logfile
#     """

#     def __init__(self, logfile: str, lum_process: Process, process_queue: Queue):
#         super().__init__(daemon=True)
#         self.logfile: str = logfile
#         self.lum_process: Process = lum_process
#         self.lum_queue: Queue = process_queue
#         if self.lum_process.pid is None:
#             raise Exception("RunLumerical process has not yet started...")
#         # Avoid problems when simulation deletes logfile
#         self._logfile_exists = False

#     def run(self):
#         while True:
#             # is_lum_running = psutil.pid_exists(self.lum_process.pid)
#             if not self.lum_process.is_alive() and not self._logfile_exists:
#                 # Process terminated before creating logfile
#                 logger.debug("RunLumerical process terminated prematurely")
#                 break
#             if not self._logfile_exists and not os.path.isfile(self.logfile):
#                 # Waiting for logfile
#                 logger.debug("Waiting for logfile")
#                 time.sleep(2)
#                 continue
#             # Cue to indicate that simulation has started
#             self._logfile_exists = True
#             if not self.lum_process.is_alive():
#                 logger.info("Simulation Finished Successfully")
#                 #if self.lum_process.is_alive():
#                 #    self.lum_process.terminate()
#                 #    self.lum_queue.close()
#                 break
#             if self._check_state() == -1:
#                 # Wait for when the Solver terminated with error
#                 # but is still waiting for extra data
#                 time.sleep(10)
#                 # Kill process if still alive
#                 if self.lum_process.is_alive():
#                     self.lum_process.terminate()
#                     #self.lum_queue.close()
#                 logger.critical("Error detected in simulation (Done Cleanup)")

#                 break
#             else:
#                 # Simulation is running wait 5 seconds to recheck
#                 time.sleep(5)
#                 continue

#     def _check_state(self) -> int:
#         """Determine running state for file"""
#         with open(self.logfile, "r") as logfile:
#             log_data: str = logfile.read()
#         if "Error" in log_data:
#             return -1
#         else:
#             return 0


class RunLumerical(Process):
    """
    Main Process to run the lumerical file.
    This method avoids problems when the simulation has
        any kind of error and just gets stuck.
    By running in a different process it is possible to kill
        the process and still keep running any script
    Args:
        proc_queue: Queue for the main process, to extract simulation results
        log_queue: Queue to log information into logfile (often = Queue(-1))
        filepath: Lumerical file path
        properties: List of properties to change
        get_results: List of results to extract from lumerical
        get_info: Extra properties in the simulation to extrac information
        func: Extra function to run inside lumerical
        lumerical_kw: Extra arguments for the lumerical.PROGRAM()
        kwargs: Extra arguments for func
    """

    def __init__(
        self,
        method: LumMethod,
        *,
        proc_queue: Queue,
        log_queue: Queue,
        filepath: str,
        properties: Dict[str, Dict[str, float]],
        get_results: Dict[str, Dict[str, Union[str, List]]],
        get_info: Dict[str, str],
        func: Union[Callable, None]=None,
        lumerical_kw: Dict,
        **kwargs,
    ):
        super().__init__()
        log_handler = logging.handlers.QueueHandler(log_queue)
        logger.addHandler(log_handler)
        self.method = method
        self.queue = proc_queue
        self.filepath = filepath
        self.properties = properties
        self.get_results = get_results
        self.get_info = get_info
        self.func = func
        self.lum_kw = lumerical_kw
        self.kwargs = kwargs

    def run(self):
        lum_run_function = _run_method(self.method)
        if lum_run_function is None:
            raise LumericalError(
                    f"Invalid Method {self.method}: {[method_i for method_i in LumMethod]}"
                    )
        logger.debug("Starting Lumerical")
        with lum_run_function(filename=self.filepath, **self.lum_kw) as lumfile:
            # Guarantee simulation file is not in simulation mode
            lumfile.switchtolayout()
            # Update structures
            for structure_key, structure_value in self.properties.items():
                logger.debug(f"Editing: {structure_key}")
                lumfile.select(structure_key)
                for parameter_key, parameter_value in structure_value.items():
                    logger.debug(f"Updating: {parameter_key} to {parameter_value}")
                    lumfile.set(parameter_key, parameter_value)
            lumfile.runsetup()
            lumfile.save()
            results = {}
            if self.func is not None:
                logger.debug(f"""
                             Running External function:
                             Name:\"{self.func.__name__}\"
                             Args:{self.kwargs}
                             """)
                func_output: Any = self.func(lumfile, **self.kwargs)
                results["func_output"] = func_output
            else:
                results["func_output"] = None
            # Note: The double lumfile.runsetup() is important for when the setup scripts
            #       (such as the model script) depend on variables from other
            #       scripts. For example the model scripts needs the internal property
            #       of a layer generated from a structure group.
            #       The first run updates internally all the values
            #       The second run then updates all the structures with the updated values
            lumfile.runsetup()
            lumfile.runsetup()
            logger.debug(f"Running...")
            start_time = time.time()
            lumfile.switchtolayout()
            lumfile.run("CHARGE")
            charge_runtime = time.time() - start_time
            self.queue.put(charge_runtime)
            start_time = time.time()
            lumfile.runanalysis()
            analysis_runtime = time.time() - start_time
            self.queue.put(analysis_runtime)
            logger.info(
                f"Simulation took: CHARGE: {charge_runtime:0.2f}s | Analysis: {analysis_runtime:0.2f}s"
            )
            # lumfile.save()
            # # Necessary for device to give time to store data
            # if self.method == LumMethod.DEVICE or self.method == LumMethod.CHARGE:
            #     time.sleep(3)
            # Try get results
            # If data cannot be accessed pass the error to the user
            try:
                lum_results = _get_lumerical_results(lumfile, self.get_results)
                results.update(lum_results)
            except lumapi.LumApiError as lum_error:
                results = lum_error
            self.queue.put(results)
            info_data = self.get_info.copy()
            for info_obj, info_property in self.get_info.items():
                lumfile.select(info_obj)
                info_data[info_obj] = lumfile.get(info_property)
            self.queue.put(info_data)
            # Guarantee the file is properly closed
            lumfile.close()
            

            


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
                logger.debug(results["data."+key+"."+value_i])
    if "results" in get_results_info:
        for key, value in get_results["results"].items():
            if not isinstance(value, list):
                value = [value]
            for value_i in value:
                logger.debug(f"Getting result for: '{key}':'{value_i}'")
                results["results." + key + "." + value_i] = lum_handler.getresult(
                    key, value_i
                )
                logger.debug(results["results."+key+"."+value_i])
    return results
