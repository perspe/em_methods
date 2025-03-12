from typing import Tuple, Union, overload

import numpy as np
import numpy.typing as npt
import scipy.constants as scc
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d


def intrinsic_carrier_density(
    mn: float, mp: float, bandgap: float, temperature: float = 300
):
    """
    Extracts the Bandgap, z_span and intrinsic carrier density of material in "names"
    Args:
        mn, mp: electron and hole effective masses
        bandgap (eV), temperature (K)
    Returns:
        intrinsic carrier density
    """
    mn = mn * 9.11e-31
    mp = mp * 9.11e-31
    Nc = 2 * ((2 * np.pi * mn * scc.k * temperature / (scc.h**2)) ** 1.5) * 10**-6
    Nv = 2 * ((2 * np.pi * mp * scc.k * temperature / (scc.h**2)) ** 1.5) * 10**-6
    ni = ((Nv * Nc) ** 0.5) * (np.exp(-bandgap * scc.e / (2 * scc.k * temperature)))
    return ni


@overload
def blackbody_spectrum(energy: float, temperature: float = 300.0) -> float:
    ...


@overload
def blackbody_spectrum(
    energy: npt.NDArray, temperature: Union[float, npt.NDArray] = 300.0
) -> npt.NDArray:
    ...


@overload
def blackbody_spectrum(
    energy: Union[float, npt.NDArray], temperature: npt.NDArray
) -> npt.NDArray:
    ...


def blackbody_spectrum(
    energy: Union[float, npt.NDArray], temperature: Union[float, npt.NDArray] = 300.0
) -> Union[float, npt.NDArray]:
    """
    Function that derives the blackbody spectrum at 300K in a given energy range defined by E[eV]
    Args:
        energy: (array) energy in ev
    Returns:
        Black Body Spectrum (eV.s.m2)-1
    """
    h_ev = scc.h / scc.e
    k_b = scc.k / scc.e
    return (
        (2 * np.pi * energy**2)
        / (h_ev**3 * scc.c**2)
        * (np.exp(energy / (k_b * temperature)) - 1) ** -1
    )


def _adjust_abs(
    energy: npt.NDArray,
    absorption: npt.NDArray,
    bandgap: float,
    interp_points: int = 10000,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Function that cuts absorption data below bandgap.
    Useful when there is poor FDTD fitting to calculate the absorption.
    Args:
        energy: energy values in eV
        abs_data: absorption data [0-1]
        bandgap (in eV)
    Returns:
        array with absorption cutoff below bandgap
    """
    interp_abs = interp1d(energy, absorption)
    new_energy = np.linspace(np.min(energy), np.max(energy), interp_points)
    new_abs = interp_abs(new_energy)
    cutoff_mask = new_energy > bandgap
    return new_energy[cutoff_mask], new_abs[cutoff_mask]


def rad_recombination_coeff(
    wavelength: Union[float, npt.NDArray],
    absorption: Union[float, npt.NDArray],
    bandgap: float,
    z_span: float,
    edensity: float,
) -> float:
    """
    Calculate the radiative constant (B) based on the SQ limit,
    but considering the FDTD-derived absorption
    """
    energy = 1240.0 / (wavelength * 1e9)
    if not isinstance(energy, (int, float)) and not isinstance(
        absorption, (int, float)
    ):
        energy, absorption = _adjust_abs(energy, absorption, bandgap)
        dark_current: float = scc.e * trapezoid(
            blackbody_spectrum(energy) * absorption, energy
        )
    elif isinstance(energy, float) and isinstance(absorption, float):
        dark_current = scc.e * (blackbody_spectrum(energy) * absorption)
    dark_current *= 0.1  # mA/cm2
    return dark_current * 1e-5 / (scc.e * (edensity**2) * (z_span))
