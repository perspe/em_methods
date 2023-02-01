import logging
from logging.config import fileConfig
import os
import sys
import numpy as np

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_path, ".."))

from em_methods.grid import Grid2D, UniformGrid
from em_methods.rcwa import rcwa



# Get dev logger to control the logging level
log_config = os.path.join(base_path, '..', 'em_methods', 'logging.ini')
fileConfig(log_config)
logger = logging.getLogger('dev')
logger.setLevel(logging.WARNING)

## Convergence for a circle with without absorption

pte, ptm = 1, 0
grid = Grid2D(1024*8, 1024*8, 0.75, xlims=[-0.75, 0.75], ylims=[-0.75, 0.75])
grid.add_circle(6, 1, 0.2)
bottom_layer = UniformGrid(0.5, 6, 1, xlims=[-0.75, 0.75], ylims=[-0.75, 0.75])
R_arr, T_arr = [], []
x = np.arange(40)
for i in x:
    print("Harmonics", i, sep=" ")
    try:
        R, T = rcwa([grid, bottom_layer], 0, 0,
                    0.7, (ptm, pte), i, i, (1, 1), (1, 1))
        R_arr.append(R)
        T_arr.append(T)
    except:
        print("Error")
        R_arr.append(np.nan)
        T_arr.append(np.nan)

R_arr = np.array(R_arr)
T_arr = np.array(T_arr)
np.savetxt(base_path + "/RCWA/convergence_circle.txt", np.c_[x, R_arr, T_arr])
