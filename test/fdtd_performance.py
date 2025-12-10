import unittest
import os
import logging
from em_methods.lumerical.fdtd import fdtd_run, fdtd_run_analysis, fdtd_add_material, rcwa_run, filtered_pabs
import multiprocessing
import lumapi
import numpy as np
import time

# Override logger to always use debug
logger = logging.getLogger('sim')
logger.setLevel(logging.INFO)

BASETESTPATH: str = os.path.join("test", "performance")

CORES = multiprocessing.cpu_count()

logger.info(f"Available cores: {CORES}")

def setResources(handle, processes, threads, capacity):
    """ Function to change the number of processes in lumerical """
    handle.setresource("FDTD", 1, "processes", processes)
    handle.setresource("FDTD", 1, "threads", threads)
    handle.setresource("FDTD", 1, "capacity", capacity)

class TestFDTD(unittest.TestCase):
    def test_single_run(self):
        """ Make a single run of a test file with everything default """
        fdtd_file: str = os.path.join(BASETESTPATH, "spherical_dielectric_particle.fsp")
        processes = np.arange(2, CORES, 2)
        threads = CORES/processes
        capacity = np.ones_like(threads)
        runtime_data = []
        for proc_i, thread_i, cap_i in zip(processes, threads, capacity):
            logger.info(f"Running for: {proc_i}::{thread_i}::{cap_i}")
            setResources_args = {"processes": proc_i, "threads": thread_i, "capacity": cap_i}
            _, runtime, _, _ = fdtd_run(fdtd_file, {}, {}, func=setResources, delete=True, fdtd_kw={"hide": True}, **setResources_args)
            logger.info(f"Run took: {runtime} s")
            runtime_data.append(runtime)
        data = np.c_[processes, threads, capacity, np.array(runtime_data)]
        np.savetxt("runtimedata.txt", data)