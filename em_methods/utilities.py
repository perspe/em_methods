from typing import Dict, List, Union, Tuple
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid, cumulative_trapezoid
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
    input_data: Union[List[Union[str, pd.DataFrame]], str, pd.DataFrame],
    *,
    am_spectrum: str = "am1.5",
    wvl_units: Units = Units.NM,
    percent: bool = False,
    **read_csv_args,
) -> Tuple[pd.Series, Dict[str, pd.DataFrame]]:
    """
    Calculate Jsc for a single or multiple files
    Args:
        filename: file or list of files
        am_spectrum: incate what spectrum to use
        wvl_units: wavelength units (default nm)
        percent: EQE or absorption files in %
        **read_csv_args: pass args for read_csv
    """
    # Add extra parameters to read_csv_args
    if isinstance(input_data, (str, pd.DataFrame)):
        input_list: List[Union[str, pd.DataFrame]] = [input_data]
    else:
        input_list: List[Union[str, pd.DataFrame]] = input_data
    if "keep_default_na" not in read_csv_args.keys():
        read_csv_args.update({"keep_default_na": False})
    # Create the factors for calculation
    percent_factor: int = 100 if percent else 1
    wvl_factor = wvl_units.convertTo(Units.NM)
    units_factor = (scc.h * scc.c) / (scc.e * 1e-10)
    # Import spectrum and interpolate
    if "am1.5" in am_spectrum.lower():
        solar_spectrum = pd.read_csv(
            os.path.join(data_path, "solar_data_am1.5.csv"), sep=" ", names=["WVL", "IRR"])
    elif "am0" in am_spectrum.lower():
        solar_spectrum = pd.read_csv(
            os.path.join(data_path, "solar_data_am0.csv"), sep=",", names=["WVL", "IRR"])
    int_spectrum = interp1d(solar_spectrum["WVL"], solar_spectrum["IRR"])
    # Dictionary for cumulative results
    jsc_sum: Dict = {}
    results: Dict = {}
    for item in input_list:
        if isinstance(item, str):
            if os.path.isfile(item):
                data = pd.read_csv(item, **read_csv_args)
                source_name = os.path.basename(item)
            else:
                raise ValueError(f"The string {item} is not a valid path.")
        elif isinstance(item, pd.DataFrame):
            data = item
            source_name = "DataFrame"
        else:
            raise ValueError(f"Unsupported type: {type(item)}")
        wvl = data.iloc[:, 0]
        abs = data.iloc[:, 1] / percent_factor
        # Integration
        results[source_name] = trapezoid(
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
        jsc_sum[source_name] = pd.DataFrame(
            np.c_[wvl, abs, int_spectrum(wvl), int_Abs, cumsum],
            columns=["wvl", "Abs", "Int_wvl", "Int_Abs", "Cumsum"],
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
    pass_type: str = "lambert",
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
    pass_type: str = "lambert",
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
        os.path.join(data_path, "solar_data_am1.5.csv"), sep=" ", names=["WVL", "IRR"]
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
            trapezoid(
                bulk_absorption(wavelength, n, k, t_i, pass_type=pass_type)
                * astm_interp(wavelength)
                * wavelength
                / wvl_units,
                wavelength,
            )
            for t_i in thicknesses
        ]
    )
