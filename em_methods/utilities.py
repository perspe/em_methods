from typing import Dict, List, Union, Tuple
from scipy.interpolate import interp1d
from scipy.integrate import trapz, cumulative_trapezoid
import scipy.constants as scc
import pandas as pd
import os
from pathlib import Path
import numpy as np
from em_methods import Units

# Get some module paths
file_path = Path(os.path.abspath(__file__))
parent_path = file_path.parent
data_path = os.path.join(parent_path, "data")


def jsc_files(
    filename: Union[List[str], str],
    *,
    wvl_units: Units = Units.NM,
    percent: bool = False,
    **read_csv_args
) -> Tuple[pd.Series, Dict[str, pd.DataFrame]]:
    """
    Calculate Jsc for a single or multiple files
    Args:
        filename: file or list of files
        wvl_units: wavelength units (default nm)
        percent: EQE or absorption files in %
        **read_csv_args: pass args for read_csv
    """
    # Add extra parameters to read_csv_args
    if "keep_default_na" not in read_csv_args.keys():
        read_csv_args.update({"keep_default_na": False})
    # Create the factors for calculation
    percent_factor: int = 100 if percent else 1
    wvl_factor = wvl_units.convertTo(Units.NM)
    units_factor = (scc.h * scc.c) / (scc.e * 1e-10)
    # Import spectrum and interpolate
    solar_spectrum = pd.read_csv(
        os.path.join(data_path, "solar_data.csv"), sep=" ", names=["WVL", "IRR"]
    )
    int_spectrum = interp1d(solar_spectrum["WVL"], solar_spectrum["IRR"])
    # Dictionary for cumulative results
    jsc_sum: Dict = {}
    results: Dict = {}
    if isinstance(filename, str):
        filelist: List[str] = [filename]
    else:
        filelist: List[str] = filename
    for file in filelist:
        data = pd.read_csv(file, **read_csv_args)
        wvl = data.iloc[:, 0]
        abs = data.iloc[:, 1] / percent_factor
        # Integration
        results[os.path.basename(file)] = trapz(
            int_spectrum(wvl) * abs * wvl * wvl_factor / units_factor,
            wvl * wvl_factor,
        )
        cumsum = cumulative_trapezoid(
            int_spectrum(wvl) * abs * wvl * wvl_factor / units_factor,
            wvl * wvl_factor,
            initial=0,
        )
        int_Abs = (
            int_spectrum(wvl)
            * abs
            * wvl
            * wvl_units.convertTo(Units.M)
            / (scc.h * scc.c)
        )  # units: m-2.nm-1.s-1
        jsc_sum[os.path.basename(file)] = pd.DataFrame(
            np.c_[wvl, abs, int_spectrum(wvl), int_Abs, cumsum],
            columns=["WVL", "ABS", "Int_WVL", "Int_Absorption", "Cumsum"],
        )
    return pd.Series(results), jsc_sum


# Functions to calculate the Lambertian absorption and current
def bulk_absorption(
    wav,
    n_data,
    k_data,
    thickness: float,
    *,
    wav_units: Units = Units.NM,
    thickness_units: Units = Units.NM,
    pass_type: str = "lambert"
):
    """
    Bulk absorption of a Lambertian Scatterer
    Args:
        wavelength (default in nm)/n/k: Refractive index information for the material
        thickness (float default in nm): thickness of the material
        wav_units (Units): wavelength units
        thickness_units (Units): units for the thickness
        pass_type (str): Type of path enhancement (single/double pass or lambertian)
    """
    if pass_type == "single":
        pass_coefficient: int = 1
        n_data = 1
    elif pass_type == "double":
        pass_coefficient: int = 2
        n_data = 1
    elif pass_type == "lambert":
        pass_coefficient: int = 4
    else:
        raise Exception("Unknown pass_type: available (single, double, lambert)")
    tptm = np.exp(
        -pass_coefficient
        * thickness
        * 4
        * np.pi
        * k_data
        * thickness_units.convertTo(wav_units)
        / wav
    )
    rf = 1 - 1 / n_data**2
    return (1 - tptm) / (1 - rf * tptm)


def lambertian_thickness(
    thicknesses,
    wavelength,
    n,
    k,
    *,
    thickness_units: Units = Units.NM,
    wav_units: Units = Units.NM,
    pass_type: str = "lambert"
):
    """
    Calculate the Lambertian limit for a range of thicknesses
    Args:
        thicknesses (default in nm): Array with the values of thicknesses to calculate
        wavelength (default in nm): Array with the range of wavelengths
        n/k: refractive index values (should match the wavelengths)
        thickness_units (Units): define units for the thickness
        wav_units (Units): define the units for the wavelength
        pass_type (str): Type of path enhancement (single/double pass or lambertian)
    Returns:
        Array with the Lambertian Jsc
    """
    solar_spectrum = pd.read_csv(
        os.path.join(data_path, "solar_data.csv"), sep=" ", names=["WVL", "IRR"]
    )
    astm_interp = interp1d(solar_spectrum["WVL"], solar_spectrum["IRR"])
    # Convert general units to nm
    wavelength *= wav_units.convertTo(Units.NM)
    thicknesses *= thickness_units.convertTo(Units.NM)
    # The factor includes the conversion from nm to m
    # It also includes the final conversion from A/m2 to mA/cm2
    wvl_units = (scc.h * scc.c) / (scc.e * 1e-10)
    return np.array(
        [
            trapz(
                bulk_absorption(wavelength, n, k, t_i, pass_type=pass_type)
                * astm_interp(wavelength)
                * wavelength
                / wvl_units,
                wavelength,
            )
            for t_i in thicknesses
        ]
    )
