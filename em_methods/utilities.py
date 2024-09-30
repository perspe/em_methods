from typing import Dict, List, Union, Tuple
from scipy.interpolate import interp1d
from scipy.integrate import trapz, cumulative_trapezoid
import scipy.constants as scc
import pandas as pd
import os
from pathlib import Path
import numpy as np
from em_methods import Units
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors

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

def plot_IV_curves(v , j, color, labels, voc, pce, ff, jsc, j_max, v_max, y_lim = -1, label_vary = [], unit = "", parameters = None, legend = None, save_path = None):
    """
    Plots the IV curve(s) provided in the inputs. Has the capacity to plot several curves with the same color range 
    (useful when plotting several IV curves and changing only one variable) 
    Args:
            v: dictionary with the voltage arrays
            j: dictionary with the current density arrays in mA/cm2
            color: dictionary with the colors for each IV curve
            labels: dictionary with the labels for each IV curve
            voc, pce, ff, jsc: dictionaries with the IV curve metrics for each curve
            j_max: scalar which determines the maximum heigth of the plot in the y scale
            v_max: scalar which determines the maximum heigth of the plot in the x scale
            label_vary: array with the names/values of a variable that might be changing in the same IV curve set, empty by default
            unit: string with the unit of the variable that might be changing in the same IV curve set, empty by default
            parameters: array with the parameters that might be changing in the same IV curve set, empty by default
            y_lim: scalar which determines the maximum heigth of the plot in the y scale   
            legend: 'out' or 'no' or None. By default 'out' When 'out' the legend is displayed outside the plot. When 'no' the legend is not displayed. 
                    When None the legend is displayed inside the plot at random position.
            save_path: string with the path where the plot should be saved. If None, the plot is not saved.   
            
            input format example (any number of curves is allowed): 
                labels = {"label_1":label_1, "label_2":label_2, "label_3":label_3 }
                color = {"color_1":color_1, "color_2":color_2, "color_3":color_3 }
                voc = {"voc_1":voc_1, "voc_2" : voc_2, "voc_3": voc_3}
                v = {"v_1":v_1, "v_2":v_2, "v_3":v_3}
                j = {"j_1":j_1, "j_2":j_2, "j_3":j_3} 
                pce = {"pce_1":pce_1, "pce_2": pce_2, "pce_3": pce_3}
                ff = {"ff_1":ff_1, "ff_2":ff_2, "ff_3": ff_3}
                jsc = {"jsc_1":jsc_1, "jsc_2":jsc_2, "jsc_3":jsc_3}
    """
    def colorFader(c1,c2 = '#FFFFFF',mix=0): 
        c1=np.array(mpl.colors.to_rgb(c1))
        c2=np.array(mpl.colors.to_rgb(c2))
        return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

    fig, ax = plt.subplots()
    textstr_name = f"Voc =\nJsc =\nFF  =\nPce ="
    textstr_units = f"  V \n  mA/cm² \n\n  %"
    vary_true = 0
    for i in range(1,len(v.keys())+1):
        v_key = f'v_{i}'
        j_key = f'j_{i}'
        color_key = f'color_{i}'
        label_key = f'label_{i}'
        voc_key = f'voc_{i}'
        jsc_key = f'jsc_{i}'
        pce_key = f'pce_{i}'
        ff_key = f'ff_{i}'
        
        if isinstance(voc[voc_key], (list, tuple, np.ndarray)): 
            vary_true = vary_true + 1
            for z in range(0, len(voc[voc_key])):
                plt.plot(v[v_key][z], j[j_key][z], "-", color=colorFader(color[color_key],mix=(z)/len(voc[voc_key])) ,label = f"{labels[label_key]} {label_vary[z]} {unit}" )
                plt.plot(voc[voc_key][z], 0, "o", color=colorFader(color[color_key],mix=(z)/len(voc[voc_key])))
            if parameters is not None: 
                vary_true = vary_true - 1
                textstr = f"     {voc[voc_key][parameters]:05.2f}\n     {abs(jsc[jsc_key][parameters]):05.2f}\n     {ff[ff_key][parameters]:05.2f} \n     {pce[pce_key][parameters]:05.2f}"
                color[color_key] = colorFader(color[color_key],mix=(voc[voc_key].index(voc[voc_key][parameters]))/len(voc[voc_key]))
            else: 
                textstr = ""    
                
    
        else:
            plt.plot(v[v_key], j[j_key], "-", color= color[color_key], label = labels[label_key])
            plt.plot(voc[voc_key], 0, "o", color=color[color_key])
            textstr = f"     {voc[voc_key]:05.2f}\n     {abs(jsc[jsc_key]):05.2f}\n     {ff[ff_key]:05.2f} \n     {pce[pce_key]:05.2f}"
        x_position = 1.00 + (0.20*(i)) - (0.20*vary_true)
        plt.text(
            x_position,
            0.80,
            textstr,
            transform=ax.transAxes,
            fontsize=17,
            verticalalignment="top",
            color = color[color_key] 
        )
    x_position  = x_position + 0.20
    plt.text(
        1.05,
        0.80,
        textstr_name,
        transform=ax.transAxes,
        fontsize=17,
        verticalalignment="top",
        color = "black" 
        )
    plt.text(
        x_position,
        0.80,
        textstr_units,
        transform=ax.transAxes,
        fontsize=17,
        verticalalignment="top",
        color = "black" 
        )
    if legend is None:    
        plt.legend()
    elif legend == "out":
        plt.legend(bbox_to_anchor=(1.05, 0.4), loc='upper left')
    elif legend == 'no':
        print('no legend')
    plt.ylabel("Current density [mA/cm²] ")
    plt.xlabel("Voltage [V]")
    plt.grid(True, linestyle='--')
    ax.set_ylim(y_lim, j_max)
    ax.set_xlim(0, v_max)
    if save_path is not None:
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    plt.show()

def band_plotting(ec, ev, efn, efp, thickness, color, labels, legend='out', biased = False, save_path= None):
    """
    Plots the Band Diagrams provided in the inputs. This code is modular: it can plot results from CHARGE and AFORS-HET without manipulating the data previously.
    Args:
            ec: dictionary with the conduction band arrays
            ev: dictionary with the valence band arrays
            color: dictionary with the colors for each IV curve
            labels: dictionary with the labels for each IV curve
            legend: 'out' or 'no' or None. By default 'out' When 'out' the legend is displayed outside the plot. When 'no' the legend is not displayed. 
                    When None the legend is displayed inside the plot at random position.
            save_path: string with the path where the plot should be saved. If None, the plot is not saved.     
            
            input format example (any number of curves is allowed): 
                labels = {"label_1":label_1, "label_2":label_2, "label_3":label_3, "label_4":label_4 }
                color = {"color_1":color_1, "color_2":color_2, "color_3":color_3, "color_4":color_4 }
                ec = {"ec_1":ec_1, "ec_2":ec_2, "ec_3":ec_3, "ec_4":ec_4}
                ev = {"ev_1":ev_1, "ev_2":ev_2, "ev_3":ev_3, "ev_4":ev_4}
                efn = {"efn_1":efn_1, "efn_2":efn_2, "efn_3":efn_3, "efn_4":efn_4}
                efp = {"efp_1":efp_1, "efp_2":efp_2, "efp_3":efp_3, "efp_4":efp_4}   
                thickness = {"thickness_1":thickness_1, "thickness_2":thickness_2, "thickness_3":thickness_3, "thickness_4":thickness_4}
    """ 
    fig, ax = plt.subplots(1,1, figsize=(10,6))
    for i in range(1,len(ec.keys())+1):
        ec_key = f'ec_{i}'
        ev_key = f'ev_{i}'
        efn_key = f'efn_{i}'
        efp_key = f'efp_{i}'
        thickness_key = f'thickness_{i}'
        color_key = f'color_{i}'
        label_key = f'label_{i}'
        corrector = 0
        if biased == False:
            if efp[efp_key][0] != 0: #method of finding the afors data
                corrector = efp[efp_key][0]
                thickness_fix = thickness[thickness_key][::-1]*1e+4
            else:
                thickness_fix = np.array(thickness[thickness_key]) * 1e6 - thickness[thickness_key][0] * 1e6
            plt.plot(thickness_fix , ec[ec_key]-corrector, "-", color= color[color_key], label = labels[label_key])
            plt.plot(thickness_fix, ev[ev_key]-corrector, "-", color= color[color_key])
            plt.plot(thickness_fix , efn[efn_key]-corrector, "--", color= color[color_key])
            plt.plot(thickness_fix , efp[efp_key]-corrector, "--", color= color[color_key])
        else: 
            if ec[ec_key][len(ec[ec_key])/2] < 0: # if the middle of the conduction band is negative, then it is affors
                corrector = efp[efp_key][len(efp[efp_key])/2]/2
                thickness_fix = thickness[thickness_key][::-1]*1e+4
            else:
                thickness_fix = np.array(thickness[thickness_key]) * 1e6 - thickness[thickness_key][0] * 1e6
            plt.plot(thickness_fix , ec[ec_key]-corrector, "-", color= color[color_key], label = labels[label_key])
            plt.plot(thickness_fix, ev[ev_key]-corrector, "-", color= color[color_key])
            plt.plot(thickness_fix , efn[efn_key]-corrector, "--", color= color[color_key])
            plt.plot(thickness_fix , efp[efp_key]-corrector, "--", color= color[color_key]) 

    if legend is None:    
        plt.legend()
    elif legend == "out":
        plt.legend(bbox_to_anchor=(1.05, 0), loc='upper left')
    elif legend == 'no':
        print('no legend')
    plt.ylabel("[eV]")
    plt.xlabel("thickness [um]")
    plt.grid(True, linestyle='--')
    if save_path is not None:
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    plt.show()
