from threading import Thread
from uuid import uuid4
import shutil
import re
import tempfile
from multiprocessing import Process, Queue
import os
import sys
import numpy as np
from typing import List, Dict, Union, Callable, Any, Tuple
import logging
import logging.handlers
import time
from enum import Enum
import pickle
from dataclasses import dataclass

# Get module logger
# Using any logger (such as dev) that log into
# files may cause some problems in the end of the
# program
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

""" Objects for internal management """


class LumMethod(Enum):
    """Enum class with the different Lumerical Simulation methods"""

    CHARGE = 1
    DEVICE = 1
    FDTD = 2
    HEAT = 3


@dataclass(frozen=True)
class Job:
    """Dataclass with Job Information"""

    ID: str
    METHOD: LumMethod
    SOLVER: str
    STORE_PATH: str
    FILEPATH: str
    LOGFILE: str

    def clear_files(self, sim_file: bool, log_file: bool):
        if os.path.exists(self.FILEPATH) and sim_file:
            os.remove(self.FILEPATH)
        if os.path.exists(self.LOGFILE) and log_file:
            os.remove(self.LOGFILE)


class LumericalError(Exception):
    """
    Class to aggregate all lumerical exception errors
    """

    def __init__(self, message) -> None:
        self.message = message
        super().__init__(self.message)


""" Helper Functions """


def __read_autoshutoff(log_file: str) -> List:
    # Gather info from log and the delete it
    autoshut_off_re = re.compile(r"^[0-9]{0,3}.?[0-9]+%")
    autoshut_off_list: List[Tuple[float, float]] = []
    with open(log_file, mode="r") as log:
        for log_line in log.readlines():
            match = re.search(autoshut_off_re, log_line)
            if match and not "initialized" in log_line:
                autoshut_off_percent = float(log_line.split(" ")[0][:-1])
                autoshut_off_val = float(log_line.split(" ")[-1])
                autoshut_off_list.append((autoshut_off_percent, autoshut_off_val))
    logger.debug(f"Autoshutoff:\n{autoshut_off_list}")
    return autoshut_off_list


def __close_simulation(results) -> None:
    """This function checks simulation results for any error"""
    results_keys = list(results.keys())
    if "runtime" not in results_keys:
        raise LumericalError("Simulation Finished Prematurely")
    if "analysis runtime" not in results_keys:
        raise LumericalError("Simulation Failed in Analysis")
    data = results["data"]
    if isinstance(data, str) and data != "Success":
        raise LumericalError(f"No data available from simulation {data}")


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
                results["data." + key + "." + value_i] = lum_handler.getdata(
                    key, value_i
                )
    if "results" in get_results_info:
        for key, value in get_results["results"].items():
            if not isinstance(value, list):
                value = [value]
            for value_i in value:
                results["results." + key + "." + value_i] = lum_handler.getresult(
                    key, value_i
                )
    if "source" in get_results_info:
        lum_handler.select(get_results["source"])
        f_min = lum_handler.get("frequency start")
        f_max = lum_handler.get("frequency stop")
        f_points = lum_handler.getglobalmonitor("frequency points")
        logger.debug(f"fmin: {f_min} | fmax: {f_max} | fpoint: {f_points}")
        freq = np.linspace(f_min, f_max, int(f_points))
        sourcepower = lum_handler.sourcepower(freq)
        results["source"] = {"freq": freq, "Psource": sourcepower}
    return results


def _run_method(method: LumMethod) -> Union[None, Callable]:
    """Function that determines the Solver to be used for the simulation"""
    logger.debug(f"Method: {method}")
    if method == LumMethod.CHARGE or method == LumMethod.DEVICE:
        return lumapi.DEVICE
    elif method == LumMethod.FDTD:
        return lumapi.FDTD
    else:
        return None


""" Generic function to run lumerical """


def lumerical_batch(
    basefile: str,
    properties: List[Dict],
    get_results: Dict[str, Dict[str, Union[str, List]]],
    *,
    get_info: Dict[str, str] = {},
    func=None,
    savepath: Union[None, str] = None,
    delete: bool = True,
    delete_log: bool = True,
    solver="FDTD",
    method=LumMethod.FDTD,
    lumerical_kw={"hide": True},
    **kwargs,
):
    """
    This function is similar to lumerical_run, but it can create multiple runs
    at the same time. See fdtd_run for full description of args.
    Note!!: This is made to be run via slurm, that can schedule and manage
    runs and resources, please be careful when running this function
    """
    # Build the name of the new file and copy to a new location
    basepath, basename = os.path.split(basefile)
    savepath: str = savepath or basepath
    jobs, process_list = [], []
    res_queue = Queue()
    with tempfile.TemporaryDirectory() as tmp:
        for index, prop in enumerate(properties):
            jobid = f"run{index}"
            new_filepath: str = os.path.join(savepath, jobid + basename)
            shutil.copyfile(basefile, new_filepath)
            log_file: str = os.path.join(
                savepath, f"{jobid}{os.path.splitext(basename)[0]}_p0.log"
            )
            tmp_file: str = os.path.join(tmp, f"{jobid}.pkl")
            job = Job(jobid, method, solver, tmp_file, new_filepath, log_file)
            logger.debug(f"Running Job: {job}")
            jobs.append(job)
            run_process = RunLumerical(
                job_info=job,
                res_queue=res_queue,
                log_queue=Queue(-1),
                properties=prop,
                get_results=get_results,
                get_info=get_info,
                func=func,
                lumerical_kw=lumerical_kw,
                **kwargs,
            )
            # Run Simulation
            run_process.start()
            logger.debug("Run Process Started...")
            process_list.append(run_process)
        for process in process_list:
            process.join()
        logger.debug(f"Simulations finished")
        return_res = [None] * len(jobs)
        while not res_queue.empty():
            jobid, result = res_queue.get()
            logger.debug(f"Simulation data:\n{jobid}\n---------\n{result}")
            for index, job in enumerate(jobs):
                if job.ID == jobid:
                    __close_simulation(result)
                    with open(job.STORE_PATH, "rb") as file:
                        data_results = pickle.load(file)
                    if job.SOLVER == "FDTD":
                        autoshutoff = __read_autoshutoff(job.LOGFILE)
                        logger.debug(f"Autoshutoff: {autoshutoff[-1]}")
                        result["autoshutoff"] = autoshutoff
                    result["data"] = data_results
                    return_res[index] = result
        logger.debug("All jobs were ordered properly...Clearing all files")
        for job in jobs:
            job.clear_files(delete, delete_log)
        return tuple(return_res)


def lumerical_run(
    basefile: str,
    properties: Dict[str, Dict[str, float]],
    get_results: Dict[str, Dict[str, Union[str, List]]],
    *,
    get_info: Dict[str, str] = {},
    func=None,
    savepath: Union[None, str] = None,
    override_prefix: Union[None, str] = None,
    delete: bool = True,
    delete_log: bool = True,
    solver="FDTD",
    method=LumMethod.FDTD,
    lumerical_kw={"hide": True},
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
        delete_log (default=False): Delete newly generated log file
        solver (default=FDTD): Choose solver to run in simulation
        lumerical_kw: Pass arguments for lumapi.FDTD
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
    log_file: str = os.path.join(
        savepath, f"{override_prefix}_{os.path.splitext(basename)[0]}_p0.log"
    )
    # Run simulation - the process is as follows
    # 1. Create Temporary Directory to Store all the core simulation data
    # 2. Create a process (RunLumerical) to run the lumerical file
    #       - This allows creating multiple RunLumerical processes
    with tempfile.TemporaryDirectory() as tmp:
        tmp_file: str = os.path.join(tmp, f"{override_prefix}.pkl")
        job = Job(override_prefix, method, solver, tmp_file, new_filepath, log_file)
        res_queue = Queue()
        run_process = RunLumerical(
            job_info=job,
            res_queue=res_queue,
            log_queue=Queue(-1),
            properties=properties,
            get_results=get_results,
            get_info=get_info,
            func=func,
            lumerical_kw=lumerical_kw,
            **kwargs,
        )
        # Run Simulation
        run_process.start()
        logger.debug("Run Process Started...")
        run_process.join()
        logger.debug(f"Simulation finished")
        _, results = res_queue.get()
        logger.debug(f"Simulation data:\n{results}")
        __close_simulation(results)
        # Extract Data
        with open(tmp_file, "rb") as file:
            data_results = pickle.load(file)
        logger.debug(f"Read Results: {data_results.keys()}")
        # Add autoshutoff to results - Only for FDTD
        if solver == "FDTD":
            autoshutoff = __read_autoshutoff(log_file)
            data_results["autoshutoff"] = autoshutoff
            logger.info(f"Autoshutoff: {autoshutoff[-1]}")
        job.clear_files(delete, delete_log)
    return (data_results, results["runtime"], results["analysis runtime"])


""" Core process to run lumerical job """


class RunLumerical(Process):
    """
    Main Process to run the lumerical file.
    This method avoids problems when the simulation has
        any kind of error and just gets stuck.
    By running in a different process it is possible to kill
        the process and still keep running any script
    Also it allows to create batch functions to run multiple files
        at the same time
    Args:
        job_info: Job object with information of the job (id, filename, logfile)
        res_queue: Queue for the main process, to extract simulation results
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
        *,
        job_info: Job,
        res_queue: Queue,
        log_queue: Queue,
        properties: Dict[str, Dict[str, float]],
        get_results: Dict[str, Dict[str, Union[str, List]]],
        get_info: Dict[str, str] = {},
        func: Union[Callable, None] = None,
        lumerical_kw: Dict = {},
        **kwargs,
    ):
        super().__init__()
        log_handler = logging.handlers.QueueHandler(log_queue)
        logger.addHandler(log_handler)
        self.job_info = job_info
        self.log_queue = log_queue
        self.res_queue = res_queue
        self.properties = properties
        self.get_results = get_results
        self.get_info = get_info
        self.func = func
        self.lum_kw = lumerical_kw
        self.kwargs = kwargs

    def run(self):
        sim_data = {}
        results = {}
        lum_run_function = _run_method(self.job_info.METHOD)
        if lum_run_function is None:
            raise LumericalError(
                f"Invalid Method {self.job_info.METHOD}: {[method_i for method_i in LumMethod]}"
            )
        logger.debug("Starting Lumerical")
        with lum_run_function(
            filename=self.job_info.FILEPATH, **self.lum_kw
        ) as lumfile:
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
            if self.func is not None:
                logger.debug(f"""
                             Running External function:
                             Name:\"{self.func.__name__}\"
                             Args:{self.kwargs}
                             """)
                func_output: Any = self.func(lumfile, **self.kwargs)
                sim_data["func_output"] = func_output
            else:
                sim_data["func_output"] = None
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
            lumfile.run(self.job_info.SOLVER)
            runtime = time.time() - start_time
            results["runtime"] = runtime
            start_time = time.time()
            lumfile.runanalysis()
            analysis_runtime = time.time() - start_time
            results["analysis runtime"] = analysis_runtime
            logger.info(
                f"Simulation took: {str(self.job_info.METHOD)}: {runtime:0.2f}s | Analysis: {analysis_runtime:0.2f}s"
            )
            lumfile.save()
            # Get data from object that is not results (!! Integrate into get_lumerical_results)
            info_data = self.get_info.copy()
            for info_obj, info_property in self.get_info.items():
                lumfile.select(info_obj)
                info_data[info_obj] = lumfile.get(info_property)
            sim_data["data_info"] = info_data
            # Try get results
            # If data cannot be accessed pass the error to the main loop
            try:
                lum_results = _get_lumerical_results(lumfile, self.get_results)
                sim_data.update(lum_results)
                logger.debug("Extracted Results Successfully...")
                results["data"] = "Success"
                logger.debug(f"Saving data to pkl file to {self.job_info.STORE_PATH}")
                with open(self.job_info.STORE_PATH, "wb") as file:
                    pickle.dump(sim_data, file)
                logger.debug(f"Saved Data")
            except lumapi.LumApiError as lum_error:
                results["data"] = f"Error: {lum_error}"
            except:
                logger.warning("Unexpected Error... Please Try to Find Error Source")
                results["data"] = "Error: Unexpected Error"
            finally:
                self.res_queue.put((self.job_info.ID, results))


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
