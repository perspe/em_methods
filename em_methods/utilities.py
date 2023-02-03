from typing import Dict, List, Union, Tuple
from scipy.interpolate import interp1d
from scipy.integrate import trapz, cumulative_trapezoid
import scipy.constants as scc
import pandas as pd
import os
from pathlib import Path
import numpy as np

# Get some module paths
file_path = Path(os.path.abspath(__file__))
parent_path = file_path.parent
data_path = os.path.join(parent_path, "data")


def jsc_files(
    filename: Union[List[str], str],
    wvl_units: str = "nm",
    percent: bool = False,
    **read_csv_args
) -> Tuple[pd.Series, Dict[str, pd.DataFrame]]:
    """
    Calculate Jsc for a single or multiple files
    Args:
        filename: file or list of files
        wvl_units: wavelength units (default nm - available [nm, um])
        percent: EQE or absorption files in %
        **read_csv_args: pass args for read_csv
    """
    # Add extra parameters to read_csv_args
    if "keep_default_na" not in read_csv_args.keys():
        read_csv_args.update({"keep_default_na": False})
    # Create the factors for calculation
    percent_factor: int = 100 if percent else 1
    if wvl_units == "nm":
        wvl_factor: float = 1
    elif wvl_units == "um":
        wvl_factor: float = 1e-3
    else:
        raise Exception("Not acceptable units (available are: nm, um)")
    wvl_units = (scc.h * scc.c) / (scc.e * 1e-10)
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
            int_spectrum(wvl) * abs * wvl * wvl_factor / wvl_units,
            wvl * wvl_factor,
        )
        cumsum = cumulative_trapezoid(
            int_spectrum(wvl) * abs * wvl * wvl_factor / wvl_units,
            wvl * wvl_factor,
            initial=0,
        )
        jsc_sum[os.path.basename(file)] = pd.DataFrame(
            np.c_[wvl, abs, cumsum], columns=["WVL", "ABS", "Cumsum"]
        )
    return pd.Series(results), jsc_sum
