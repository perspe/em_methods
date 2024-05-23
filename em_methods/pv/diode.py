import numpy as np
from pandas.io.feather_format import pd
from scipy.optimize import fsolve
from scipy.special import lambertw
import scipy.constants as scc
import logging

from typing import Union, Any
import numpy.typing as npt

logger = logging.getLogger("sim")

"""
Diode Equations
"""

def _single_diode_rp(
    j: float,
    jl: float,
    j0: float,
    v: float,
    rs: float,
    rsh: float,
    eta: float,
    temp: float,
    n_cells: int = 1,
) -> float:
    """
    Main expression for the single diode Rp equation
    This is organized such that it return 0, for valid
    values of j
    """
    vt = scc.k * temp / scc.e
    v = n_cells * v
    rs = n_cells * rs
    rsh = n_cells * rsh
    return (
        jl - j0 * (np.exp(((v + j * rs)) / (eta * vt)) - 1) - ((v + j * rs) / rsh) - j
    )


def single_diode_rp(
    v: npt.NDArray[np.floating],
    jl: float,
    j0: float,
    rs: float,
    rsh: float,
    eta: float,
    temp: float,
    n_cells: int = 1,
    ) -> npt.NDArray[np.floating]:
    """
    Advanced configuration of the diode equation, that takes
    into account losses due to contact resistance and electron
    surfaces, the current flow resistance and the electrode
    resistance.
    This method calculates the values via the 0s of the transcendental equation
    Args:
        jl (mA/cm2): short-circuit current for the cell
        j0 (mA/cm2): Saturation current density
        v (V): voltage
        rs (Ohm.cm2): diode series resistance
        rsh (Ohm.cm2): diode parallel resistance
        eta: ideality factor
        temp (K): temperature
        n_cells: number of identical cell in the solar module
    Return:
        current (A)
    """
    current = np.zeros_like(v)
    # Use j in A/cm2 to facilitate calculations
    jl /= 1000
    j0 /= 1000
    for index, voltage in enumerate(v):
        zeros = fsolve(
            _single_diode_rp, 0, args=(jl, j0, voltage, rs, rsh, eta, temp, n_cells)
        )
        if len(zeros) > 1:
            logger.warning("More than 1 zero determined when solving diode rp equation")
        current[index] = zeros[0]
    return current*1000

def single_diode_rp_lambert(
    v: npt.NDArray[np.floating],
    jl: float,
    j0: float,
    rs: float,
    rsh: float,
    eta: float,
    temp: float,
) -> npt.NDArray[np.floating]:
    """
    Solution to the single diode Rp equation via the LambertW functions
    Args:
        v (V): voltage array
        jl, j0 (mA/cm2): short-circuit and saturation currents
        rs, rsh (Ohm.cm2): Series and shunt resistance for the cell
        eta: cell ideality factor
        temp (K): Cell temperature
    return:
        j (mA/cm2): current density
    """
    # Avoid overflow in the exponentials
    jl /= 1000
    j0 /= 1000
    # Simplification terms
    vt = scc.k * temp / scc.e
    nvt = eta*vt
    rs_rsh = rs+rsh
    # Calculate terms in the equation
    term_1 = v/rs_rsh
    lambertw_e = rsh*(rs*jl+rs*j0+v)/(nvt*rs_rsh)
    term_2 = lambertw(rs*j0*rsh*np.exp(lambertw_e)/(nvt*rs_rsh))*nvt/rs
    term_3 = rsh*(j0+jl)/rs_rsh
    return (-term_1-term_2+term_3)*1000


def luqing_liu_diode(
    voltage: npt.NDArray[np.floating],
    jsc: float,
    jmpp: float,
    voc: float,
    rs: float,
    rsh: float,
    eta: float,
    temp: float,
    n_cells: int = 1,
) -> npt.NDArray[np.floating]:
    """
    Luqing Liu version of the diode equation. This equation has 3 main
    assumptions
    1. Rs << Rph
    2. I0<<Isc
    3. Curve behaviour for I=0 and V=0
    Args:
        voltage (array - V): voltages to determine the current
        jsc (mA/cm2): cell short-circuit current
        jmpp (mA/cm2): cell maximum power point current
        voc (V): cell open-circuit voltage
        rs (Ohm): cell series resistance
        rph (Ohm): cell parallel resistance
        eta: cell ideality factor
        temp (K): cell temperature
        n_cells: number of similar cells
    Return:
        current_density (mA/cm2)
    """
    current = np.zeros_like(voltage)
    if rsh == 0 or jsc == 0:
        return current
    vt = scc.k * temp / scc.e
    impp = jmpp / jsc
    gamma = 1 - (voc / (jsc * rsh / 1000))
    theta = 0.77 * impp * gamma
    m = (
        1
        - (1 / gamma)
        + (voc / ((n_cells * eta * vt) + (theta * gamma * jsc * rs / 1000)))
    )
    logger.debug(f"{gamma}\n{m}\n{theta}")
    for index, v_i in enumerate(voltage):
        v = v_i / voc
        i = 1 - (1 - gamma) * v - gamma * ((((v**2) + 1) / 2) ** m)
        j = i * jsc
        current[index] = j
    return current
