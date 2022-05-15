from typing import Dict, List, Union
from scipy.interpolate import interp1d
from scipy.integrate import trapz
import scipy.constants as scc
import pandas as pd
import os
from pathlib import Path

# Get some module paths
file_path = Path(os.path.abspath(__file__))
parent_path = file_path.parent
data_path = os.path.join(parent_path, "data")


def jsc_files(filename: Union[List[str], str],
              **read_csv_args) -> Union[pd.Series, float]:
    """ Calculate the Jsc for filename """
    # Import spectrum and interpolate
    solar_spectrum = pd.read_csv(os.path.join(data_path, "solar_data.csv"),
                                 sep=" ",
                                 names=["WVL", "IRR"])
    int_spectrum = interp1d(solar_spectrum.WVL, solar_spectrum.IRR)
    factor = (scc.h * scc.c) / (scc.e * 1e-10)
    # Import data
    if isinstance(filename, list):
        results: Dict = {}
        for file in filename:
            data = pd.read_csv(file, **read_csv_args)
            wvl = data.iloc[:, 0]
            abs = data.iloc[:, 1]
            # Make the integration
            results[os.path.basename(file)] = trapz(
                int_spectrum(wvl) * abs * wvl / factor, wvl)
        return pd.Series(results)
    else:
        data = pd.read_csv(filename, **read_csv_args)
        wvl = data.iloc[:, 0]
        abs = data.iloc[:, 1]
        # Make the integration
        Jsc = trapz(int_spectrum(wvl) * abs * wvl / factor, wvl)
        return Jsc
