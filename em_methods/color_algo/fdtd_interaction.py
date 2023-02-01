import os
import numpy as np
from color_scripts import satcolorgen,colordist
from em_methods.optimization.pso import particle_swarm
import logging
import shutil
import sys
sys.path.append(r"C:\\Program Files\Lumerical\v202\api\python")
import lumapi
import math

# TODO: Convert to the general logging method
#Set logging level
logging.basicConfig(**{
    "level": logging.ERROR,
    "format": "%(asctime)s [%(levelname)s] %(filename)s:%(funcName)s:%(lineno)d:%(message)s"
})

# E costume colocar variaveis constantes no topo do codigo em letra
# maiuscula. Assim fica tambem tudo num ponto fácil de usar
COLOR: str = "#FF69B4"  # must be hexcode
SATURATION: bool = True
BASEFILE: str = "nanopillar.fsp"
PARTICLES=25
INERTIA=(0.9,0.4,True)
SOCIAL = 1.49
COGNITION = 1.49
MIN_ITER = 50   
MAX_ITER = 100
TOLERANCE: int = 10



param_dict = {"tSpiro": [50e-9, 200e-9],
              "tITO": [50e-9, 200e-9],
              "pillar_width": [20e-9, 500e-9],
              "pillar_height": [0e-9, 500e-9],
              "pillar_pitch": [1, 3]
}

def batchSim(tSpiro, tITO, pillar_width, pillar_height, pillar_pitch, basepath: str="PSO_OPT"):
    """
    Run a batch of simulations in sequency, to avoid rw issues, for the provided array data
    """
    # # Check for path to store simulations in a different folder
    if not os.path.isdir(basepath):
        os.mkdir(basepath)
        
    generator = list(enumerate(zip(tSpiro, tITO, pillar_width, pillar_height, pillar_pitch)))
    basefile: str = os.path.join(basepath, "pso_opt_file")
    fom_list = []
    
    # First create all the files and set the properties for the pso iteration
    for index, (tSpiro_i, tITO_i, tp_width, tp_height, tp_pitch) in generator:
        logging.debug(f"Properties: {tSpiro_i=}::{tITO_i=}::{tp_width=}::{tp_height=}::{tp_pitch=}")
        shutil.copyfile(BASEFILE, f"{basefile}_{index}.fsp")
        with lumapi.FDTD(filename=f"{basefile}_{index}.fsp",hide=True) as fdtd:
            fdtd.select("::model")
            fdtd.set("tSpiro", tSpiro_i)
            fdtd.set("tITO", tITO_i)
            fdtd.set('pillar_width', tp_width)
            fdtd.set('pillar_height', tp_height)
            fdtd.set('pillar_pitch', tp_pitch)
            fdtd.run()
            fdtd.runanalysis("solar_gen")
            T = fdtd.getresult("monitor", "T")["T"].tolist()
            jsc = fdtd.getdata("solar_gen", "Jsc")
            solar = fdtd.solar(1)[380:781:5]*1e-9  # obtains the solar spectrum
            fdtd.close() # The license error is back, added this one to try and fix it 
        del fdtd # Seriously doubt this... but there is still a stray error hard to identify
        wavelength = np.arange(380, 781, 5)
        T = T * solar.flatten()
        Tdict = dict()  # creates an empty dictionary for the reflection
        for i in range(len(wavelength)): # for wvl_i in wavelength: - numpy arrays sao iteraveis
            # pairs each wavelength to a reflection
            Tdict[wavelength[i]] = T[i] 
        # Este warning no spyder devese a usares o from golorgen import *
        color = satcolorgen(T)  # generates saturated colors
        # calculates color distance
        color_distance = colordist(color, COLOR)
        # calculates FOM
        if color_distance == 0:
            fom_list.append(jsc)
        else:
            fom_list.append(jsc/math.sqrt(color_distance))
        logging.debug(f"{jsc=}::{color_distance}::{fom_list[-1]}")
        logging.debug(f"Written file:{basefile}_{index}.fsp")
    return np.array(fom_list)


gfitness, gbest, pbest, gbest_array =\
    particle_swarm(batchSim, param_dict,\
                       tolerance=TOLERANCE,\
                           inert_prop=INERTIA,\
                               ind_cog=COGNITION,\
                                   soc_learning=SOCIAL,\
                                       min_iterations=MIN_ITER,\
                                           max_iterations=MAX_ITER,\
                                               particles=PARTICLES,\
                               export=True, progress=True)


np.save('gfitness.npy', gfitness)
np.save('gbest.npy', gbest)
np.save('pbest.npy', pbest)
np.save('gbest_array.npy', gbest_array)

################################################################################
#daqui para baixo é um copy paste para guardar a melhor iteração separadamente
tSpiro, tITO, pillar_width, pillar_height, pillar_pitch=[*gbest]
tSpiro=[tSpiro]
tITO=[tITO]
pillar_width=[pillar_width]
pillar_height=[pillar_height]
pillar_pitch=[pillar_pitch]
basepath='BEST'
if not os.path.isdir(basepath):
    os.mkdir(basepath)
    
generator = list(enumerate(zip(tSpiro, tITO, pillar_width, pillar_height, pillar_pitch)))
basefile: str = os.path.join(basepath, "pso_opt_file")
fom_list = []

# First create all the files and set the properties for the pso iteration
for index, (tSpiro_i, tITO_i, tp_width, tp_height, tp_pitch) in generator:
    logging.debug(f"Properties: {tSpiro_i=}::{tITO_i=}::{tp_width=}::{tp_height=}::{tp_pitch=}")
    shutil.copyfile(BASEFILE, f"{basefile}_{index}.fsp")
    with lumapi.FDTD(filename=f"{basefile}_{index}.fsp",hide=True) as fdtd:
        fdtd.select("::model")
        fdtd.set("tSpiro", tSpiro_i)
        fdtd.set("tITO", tITO_i)
        fdtd.set('pillar_width', tp_width)
        fdtd.set('pillar_height', tp_height)
        fdtd.set('pillar_pitch', tp_pitch)
        fdtd.run()
        fdtd.runanalysis("solar_gen")
        T = fdtd.getresult("monitor", "T")["T"].tolist()
        jsc = fdtd.getdata("solar_gen", "Jsc")
        solar = fdtd.solar(1)[380:781:5]*1e-9  # obtains the solar spectrum
    del fdtd # Seriously doubt this... but there is still a stray error hard to identify
    wavelength = np.arange(380, 781, 5)
    T = T * solar.flatten()
    Tdict = dict()  # creates an empty dictionary for the reflection
    for i in range(len(wavelength)): # for wvl_i in wavelength: - numpy arrays sao iteraveis
        # pairs each wavelength to a reflection
        Tdict[wavelength[i]] = T[i] 
    # Este warning no spyder devese a usares o from golorgen import *
    color = satcolorgen(T)  # generates saturated colors
    # calculates color distance
    color_distance = colordist(color, COLOR)
    # calculates FOM
    if color_distance == 0:
        fom_list.append(jsc)
    else:
        fom_list.append(jsc/math.sqrt(color_distance))
    logging.debug(f"{jsc=}::{color_distance}::{fom_list[-1]}")
    logging.debug(f"Written file:{basefile}_{index}.fsp")
