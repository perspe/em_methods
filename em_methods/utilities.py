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
from PyAstronomy import pyaC
from itertools import product
import shutil
from em_methods.optimization.pso import particle_swarm
from em_methods.lumerical.charge import  iv_curve, _import_generation
from em_methods.pv.diode import luqing_liu_diode

#dont know if necessary yet
import h5py
from mpl_toolkits.mplot3d import Axes3D


# Get some module paths
file_path = Path(os.path.abspath(__file__))
parent_path = file_path.parent
data_path = os.path.join(parent_path, "data")


def jsc_files(
    filename: Union[List[str], str],
    *,
    wvl_units: Units = Units.NM,
    percent: bool = False,
    **read_csv_args,
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


def __colorFader(c1, c2="#FFFFFF", mix=[0]):
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return [mpl.colors.to_hex((1 - mix_i) * c1 + mix_i * c2) for mix_i in mix]


def plot_IV(
    voltages: List,
    currents: List,
    colors: List[str],
    labels: List[str],
    voc_list: List[float],
    pce_list: List[float],
    ff_list: List[float],
    jsc_list: List[float],
    xmax: float,
    ylim: Tuple[float, float],
    sublabels: List[str] = [],
    unit: str = "",
    parameters: Union[None, float, List[Union[float, None]]] = None,
    legend_args: Union[None, Dict] = None,
    save_path: Union[None, str] = None,
    save_args: Dict = {"dpi": 300, "bbox_inches": "tight"},
    ax=None,
):
    """
    Plots the IV curve(s) provided in the inputs.
    Has the capacity to plot several curves with the same color range
    (useful when plotting several IV curves and changing only one variable)
    Args:
        voltages: list with voltages to plot
        currents: list with currents to plot
        colors: list with colors for profiles
        labels: list with labels for the profiles
        voc_list, pce_list, ff_list, jsc_list: list with the IV curve metrics for each curve
        xmax: maximum width of the plot in the y scale
        ylim: tuple with the minimum and maximum values for the plot height
        sublabels: Sublevels of the different labels
        unit: Units of the changing variables in sublabels
        parameters: what labels to show
        legend_args: Argument to pass to legend
        save_path: string with the path where the plot should be saved. If None, the plot is not saved.
        save_args: Arguments to pass to save_function
        ax: Outside axes to save the plot
    """
    show_plot: bool = True
    if ax is None:
        _, ax = plt.subplots()
        show_plot = False
    if isinstance(parameters, (float, int)) or parameters is None:
        parameters = [parameters]
    label_list = product(labels, sublabels)
    color_list = []
    for color_i in colors:
        label_colors = __colorFader(color_i, mix=np.linspace(0.1, 0.9, len(sublabels)))
        color_list.extend(label_colors)
    iter_vars = zip(
        voltages,
        currents,
        color_list,
        label_list,
        voc_list,
        pce_list,
        ff_list,
        jsc_list,
    )
    x_position = 1.15
    for volt_i, j_i, color_i, label_i, voc_i, pce_i, ff_i, jsc_i in iter_vars:
        label_key, label_var = label_i
        plt.plot(
            volt_i, j_i, "-", color=color_i, label=f"{label_key} {label_var} {unit}"
        )
        plt.plot(voc_i, 0, "o", color=color_i)
        if float(label_var) not in parameters:
            continue
        textstr = f"     {voc_i:05.2f}\n     {abs(jsc_i):05.2f}\n     {ff_i:05.2f} \n     {pce_i:05.2f}"
        plt.text(
            x_position,
            0.80,
            textstr,
            transform=ax.transAxes,
            fontsize=17,
            verticalalignment="top",
            color=color_i,
        )
        x_position += 0.2
    textstr_name = "$V_{oc}$ =\n$J_{sc}$ =\nFF  =\nPCE ="
    plt.text(
        1.05,
        0.80,
        textstr_name,
        transform=ax.transAxes,
        fontsize=17,
        verticalalignment="top",
        color="black",
    )
    textstr_units = f"  V \n  mA/cm² \n  %\n  %"
    plt.text(
        x_position + 0.05,
        0.80,
        textstr_units,
        transform=ax.transAxes,
        fontsize=17,
        verticalalignment="top",
        color="black",
    )
    if legend_args is None:
        plt.legend()
    else:
        plt.legend(**legend_args)
    plt.ylabel("Current density [mA/cm²] ")
    plt.xlabel("Voltage [V]")
    ax.set_ylim(ylim)
    ax.set_xlim(0, xmax)
    if save_path is not None:
        plt.savefig(save_path, **save_args)
    if show_plot:
        plt.show()

def plot_IV_curves(
    v,
    j,
    color,
    labels,
    voc,
    pce,
    ff,
    jsc,
    y_max,
    x_max,
    y_min=-1,
    label_vary=[],
    unit="",
    parameters=None,
    legend=None,
    save_path=None,
):
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

    def colorFader(c1, c2="#FFFFFF", mix=0):
        c1 = np.array(mpl.colors.to_rgb(c1))
        c2 = np.array(mpl.colors.to_rgb(c2))
        return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)

    fig, ax = plt.subplots()
    textstr_name = "$V_{oc}$ =\n$J_{sc}$ =\nFF  =\nPCE ="
    textstr_units = f"  V \n  mA/cm² \n  %\n  %"
    vary_true = 0
    for i in range(1, len(v.keys()) + 1):
        v_key = f"v_{i}"
        j_key = f"j_{i}"
        color_key = f"color_{i}"
        label_key = f"label_{i}"
        voc_key = f"voc_{i}"
        jsc_key = f"jsc_{i}"
        pce_key = f"pce_{i}"
        ff_key = f"ff_{i}"

        if isinstance(voc[voc_key], (list, tuple, np.ndarray)):
            vary_true = vary_true + 1
            for z in range(0, len(voc[voc_key])):
                plt.plot(
                    v[v_key][z],
                    j[j_key][z],
                    "-",
                    color=colorFader(color[color_key], mix=(z) / len(voc[voc_key])),
                    label=f"{labels[label_key]} {label_vary[z]} {unit}",
                )
                plt.plot(
                    voc[voc_key][z],
                    0,
                    "o",
                    color=colorFader(color[color_key], mix=(z) / len(voc[voc_key])),
                )
            if parameters is not None:
                vary_true = vary_true - 1
                textstr = f"     {voc[voc_key][parameters]:05.2f}\n     {abs(jsc[jsc_key][parameters]):05.2f}\n     {ff[ff_key][parameters]:05.2f} \n     {pce[pce_key][parameters]:05.2f}"
                color[color_key] = colorFader(
                    color[color_key],
                    mix=(voc[voc_key].index(voc[voc_key][parameters]))
                    / len(voc[voc_key]),
                )
            else:
                textstr = ""

        else:
            plt.plot(
                v[v_key], j[j_key], "-", color=color[color_key], label=labels[label_key]
            )
            plt.plot(voc[voc_key], 0, "o", color=color[color_key])
            textstr = f"     {voc[voc_key]:05.2f}\n     {abs(jsc[jsc_key]):05.2f}\n     {ff[ff_key]:05.2f} \n     {pce[pce_key]:05.2f}"
        x_position = 1.00 + (0.20 * (i)) - (0.20 * vary_true)
        plt.text(
            x_position,
            0.80,
            textstr,
            transform=ax.transAxes,
            fontsize=17,
            verticalalignment="top",
            color=color[color_key],
        )
    x_position = x_position + 0.20
    plt.text(
        1.05,
        0.80,
        textstr_name,
        transform=ax.transAxes,
        fontsize=17,
        verticalalignment="top",
        color="black",
    )
    plt.text(
        x_position,
        0.80,
        textstr_units,
        transform=ax.transAxes,
        fontsize=17,
        verticalalignment="top",
        color="black",
    )
    if legend is None:
        plt.legend()
    elif legend == "out":
        plt.legend(bbox_to_anchor=(1.05, 0.4), loc="upper left")
    elif legend == "no":
        print("no legend")
    plt.ylabel("Current density [mA/cm²] ")
    plt.xlabel("Voltage [V]")
    plt.grid(True, linestyle="--")
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, x_max)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_IV_curves_v2(
    v,
    j,
    color = {},
    labels = {},
    voc = {},
    pce = {},
    ff = {},
    jsc = {},
    y_max = 15,
    x_max = 2,
    y_min=-1,
    label_vary=[],
    unit="",
    parameters=None,
    legend=None,
    save_path=None,
):
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
    colors_matplotlib = plt.rcParams['axes.prop_cycle'].by_key()['color']
    def colorFader(c1, c2="#FFFFFF", mix=0):
        c1 = np.array(mpl.colors.to_rgb(c1))
        c2 = np.array(mpl.colors.to_rgb(c2))
        return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)

        
    fig, ax = plt.subplots()
    textstr_name = "$V_{oc}$ =\n$J_{sc}$ =\nFF  =\nPCE ="
    textstr_units = f"  V \n  mA/cm² \n  %\n  %"
    vary_true = 0

    for i in range(1, len(v.keys()) + 1):
        v_key = f"v_{i}"
        j_key = f"j_{i}"
        color_key = f"color_{i}"
        label_key = f"label_{i}"
        voc_key = f"voc_{i}"
        jsc_key = f"jsc_{i}"
        pce_key = f"pce_{i}"
        ff_key = f"ff_{i}"
        if color == {}: 
            color[color_key] = colors_matplotlib[(i - 1) % len(colors_matplotlib)]
            print("Hey",color[color_key])
            erase_color = True
        else: 
            erase_color = False
        if voc == {} or ff == {} or pce == {} or jsc == {}:
            print('heyyyy', type(v[v_key]))
            print(v[v_key])
            print(j[j_key])
            pce[pce_key], ff[ff_key], voc[voc_key], jsc[jsc_key], _, _ = iv_curve( np.array(v[v_key]), current_density = np.array(j[j_key]))
            erase_iv = True
        else: 
            erase_iv = False
        if labels == {}:
            labels[label_key] = ""
            legend = 'no'
            erase_label = True
        else:
            erase_label = False 


        if isinstance(voc[voc_key], (list, tuple, np.ndarray)): #UNTESTED
            vary_true = vary_true + 1
            for z in range(0, len(voc[voc_key])):
                plt.plot(
                    v[v_key][z],
                    j[j_key][z],
                    "-",
                    color=colorFader(color[color_key], mix=(z) / len(voc[voc_key])),
                    label=f"{labels[label_key]} {label_vary[z]} {unit}",
                )
                plt.plot(
                    voc[voc_key][z],
                    0,
                    "o",
                    color=colorFader(color[color_key], mix=(z) / len(voc[voc_key])),
                )
            if parameters is not None:
                vary_true = vary_true - 1
                textstr = f"     {voc[voc_key][parameters]:05.2f}\n     {abs(jsc[jsc_key][parameters]):05.2f}\n     {ff[ff_key][parameters]:05.2f} \n     {pce[pce_key][parameters]:05.2f}"
                color[color_key] = colorFader(
                    color[color_key],
                    mix=(voc[voc_key].index(voc[voc_key][parameters]))
                    / len(voc[voc_key]),
                )
            else:
                textstr = ""

        else:
            plt.plot(
                v[v_key], j[j_key], "-", color=color[color_key], label=labels[label_key]
            )
            plt.plot(voc[voc_key], 0, "o", color=color[color_key])
            textstr = f"     {voc[voc_key]:05.2f}\n     {abs(jsc[jsc_key]):05.2f}\n     {ff[ff_key]:05.2f} \n     {pce[pce_key]:05.2f}"
        x_position = 1.00 + (0.20 * (i)) - (0.20 * vary_true)
        plt.text(
            x_position,
            0.80,
            textstr,
            transform=ax.transAxes,
            fontsize=17,
            verticalalignment="top",
            color=color[color_key],
        )
        if erase_color:
            color = {} 
        if erase_iv: 
            voc, jsc, pce, ff = {}, {}, {}, {}
        if erase_label: 
            labels = {}
    x_position = x_position + 0.20
    plt.text(
        1.05,
        0.80,
        textstr_name,
        transform=ax.transAxes,
        fontsize=17,
        verticalalignment="top",
        color="black",
    )
    plt.text(
        x_position,
        0.80,
        textstr_units,
        transform=ax.transAxes,
        fontsize=17,
        verticalalignment="top",
        color="black",
    )
    if legend is None:
        plt.legend()
    elif legend == "out":
        plt.legend(bbox_to_anchor=(1.05, 0.4), loc="upper left")
    elif legend == "no":
        print("no legend")
    plt.ylabel("Current density [mA/cm²] ")
    plt.xlabel("Voltage [V]")
    plt.grid(True, linestyle="--")
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, x_max)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def band_plotting(
    ec,
    ev,
    efn,
    efp,
    thickness,
    color,
    labels,
    legend="out",
    biased=False,
    save_path=None,
):
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
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for i in range(1, len(ec.keys()) + 1):
        ec_key = f"ec_{i}"
        ev_key = f"ev_{i}"
        efn_key = f"efn_{i}"
        efp_key = f"efp_{i}"
        thickness_key = f"thickness_{i}"
        color_key = f"color_{i}"
        label_key = f"label_{i}"
        corrector = 0
        if biased == False:
            if efp[efp_key][0] != 0:  # method of finding the afors data
                corrector = efp[efp_key][0]
                thickness_fix = thickness[thickness_key][::-1] * 1e4
            else:
                thickness_fix = (
                    np.array(thickness[thickness_key]) * 1e6
                    - thickness[thickness_key][0] * 1e6
                )
            plt.plot(
                thickness_fix,
                ec[ec_key] - corrector,
                "-",
                color=color[color_key],
                label=labels[label_key],
            )
            plt.plot(thickness_fix, ev[ev_key] - corrector, "-", color=color[color_key])
            plt.plot(
                thickness_fix, efn[efn_key] - corrector, "--", color=color[color_key]
            )
            plt.plot(
                thickness_fix, efp[efp_key] - corrector, "--", color=color[color_key]
            )
        else:
            if (
                ec[ec_key][len(ec[ec_key]) / 2] < 0
            ):  # if the middle of the conduction band is negative, then it is affors
                corrector = efp[efp_key][len(efp[efp_key]) / 2] / 2
                thickness_fix = thickness[thickness_key][::-1] * 1e4
            else:
                thickness_fix = (
                    np.array(thickness[thickness_key]) * 1e6
                    - thickness[thickness_key][0] * 1e6
                )
            plt.plot(
                thickness_fix,
                ec[ec_key] - corrector,
                "-",
                color=color[color_key],
                label=labels[label_key],
            )
            plt.plot(thickness_fix, ev[ev_key] - corrector, "-", color=color[color_key])
            plt.plot(
                thickness_fix, efn[efn_key] - corrector, "--", color=color[color_key]
            )
            plt.plot(
                thickness_fix, efp[efp_key] - corrector, "--", color=color[color_key]
            )

    if legend is None:
        plt.legend()
    elif legend == "out":
        plt.legend(bbox_to_anchor=(1.05, 0), loc="upper left")
    elif legend == "no":
        print("no legend")
    plt.ylabel("[eV]")
    plt.xlabel("thickness [um]")
    plt.grid(True, linestyle="--")
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_generation_3d(
        path: str,
        gen_file: str,
        save_fig: bool = False,
        figsize: Tuple[float, float] = (7, 7),
        transparent: bool = True,
        fontsize:int = 18,
        ):
    
    """
    Plots a 3D visualization of the generation rate data from a given file.

    Args:

        path : str - Directory path where the generation data file is located.
        gen_file : str - Name of the generation data file to be loaded and visualized.
        save_fig : bool, optional - Whether to save the generated figure as an image file. Default is False.
        figsize : tuple of (float, float), optional - Size of the figure in inches (width, height). Default is (7, 7).
        transparent : bool, optional - Whether the figure background should be transparent. Default is True.
        fontsize : int, optional - Font size for axis labels and title. Default is 18.

    """
    
    # Load generation data
    generation_name = os.path.join(path, gen_file)
    gen_data, x, y, z = _import_generation(generation_name)
    
    # Create 3D meshgrid for plotting
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Flatten arrays for scatter plotting
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()
    G_flat = gen_data.flatten()
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    if transparent:
        fig.patch.set_alpha(0)  # transparent background
        ax.set_facecolor((1, 1, 1, 0))  #subplot background is also transparent
     
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams["font.family"] = "Arial"

    scatter = ax.scatter(X_flat*1e9, Y_flat*1e9, Z_flat*1e9, c=G_flat, cmap='viridis', marker='o')
    ax.set_xlabel('X Coordinate [nm]')
    ax.set_ylabel('Y Coordinate [nm]')
    ax.set_zlabel('Z Coordinate [nm]')
    ax.set_title('3D Generation Rate Visualization')
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Generation Rate')
    if save_fig:
        plt.savefig(os.path.join(path, str(gen_file) + "plot.png"))


def iv_parameters(voltage, current_density, area, current=[]):
    "area: in cm2"
    Ir = 1000  # W/m²
    current_density = np.array(current_density)
    if current == []:
        current = current_density * area * 10**-3
    else:
        current = np.array(current)
    abs_voltage_min = min(np.absolute(voltage))  # volatage value closest to zero
    if abs_voltage_min in voltage:
        Jsc = current_density[np.where(voltage == abs_voltage_min)[0]][0]
        Isc = current[np.where(voltage == abs_voltage_min)[0]][0]
        # Jsc = current_density[voltage.index(abs_voltage_min)]
        # Isc = current[voltage.index(abs_voltage_min)]
    elif -abs_voltage_min in voltage:
        # the position in the array of Jsc and Isc should be the same
        Jsc = current_density[np.where(voltage == -abs_voltage_min)[0]][0]
        Isc = current[np.where(voltage == -abs_voltage_min)[0]][0]
        # Jsc = current_density[voltage.index(-abs_voltage_min)]
        # Isc = current[voltage.index(-abs_voltage_min)]

    Voc, stop = pyaC.zerocross1d(
        np.array(voltage), np.array(current_density), getIndices=True
    )
    Voc = Voc[0]
    P = [
        voltage[x] * abs(current[x]) for x in range(len(voltage)) if current[x] < 0
    ]  # calculate the power for all points [W]
    vals_v = np.linspace(min(voltage), max(voltage), 100)
    new_j = np.interp(vals_v, voltage, current)
    P = [vals_v[x] * abs(new_j[x]) for x in range(len(vals_v)) if new_j[x] < 0]

    FF = abs(max(P) / (Voc * Isc))

    PCE = ((FF * Voc * abs(Isc)) / (Ir * (area * 10**-4))) * 100
    Jsc = -current_density[0]
    return abs(FF), abs(PCE), abs(Jsc), abs(Voc)

#2Terminal IV curve
def fom_func(voc_temp, voltage, charge_current, diode_func_current):
    """
    Define the figure of merit for determining the eta of the IV curve.
    Args:
        voltage: (array) voltage array of IV curve
        charge_current: (array) current of IV curve in mA/cm2 (positive)
        diode_func_current: (array) current of IV curve in mA/cm2 (positive) from diode model
    Returns:
        Absolute difference between charge and diode current 
    """
    voltage_linspace = np.linspace(0, voc_temp, 500)
    charge_current_interp = np.interp(voltage_linspace,voltage, charge_current)
    diode_func_current_interp = np.interp(voltage_linspace, voltage, diode_func_current)
    dif  = sum(abs(charge_current_interp - diode_func_current_interp)) 
    return dif

def calc_jmpp(voltage_charge,current_density_charge): 
    """
    Calculates the maximum power point current density from the simulated IV curve
    Args:
        voltage_charge: (array) voltage array of IV curve
        current_density_charge: (array) current of IV curve in mA/cm2 (negative)
    Returns:
        Maximum power point current density, jmpp (float)
    """
    vals_v = np.linspace(min(voltage_charge), max(voltage_charge), 100)
    new_j = np.interp(vals_v, voltage_charge, current_density_charge) #current_density should be negative
    if new_j[0]>0: 
        new_j = np.array(new_j)*-1
    P = [vals_v[x] * abs(new_j[x]) for x in range(len(vals_v)) if new_j[x] < 0 ]
    jmpp = new_j[P.index(max(P))]
    return jmpp

def calc_R(voc_temp,voltage_charge,current_density_charge):
    """
    Calculates the series and shunt resistances of the simulated IV curve, trhough the deriviative
    Args:
        voc_temp: (float) voc volatge from simulated IV curve 
        voltage_charge: (array) voltage array of IV curve 
        current_density_charge: (array) current of IV curve in mA/cm2 (negative) 
    Returns:
        Shunt and Series resistance (float)
    """
    #closest_index = (np.abs(voltage_charge - voc_temp)).argmin()
    
    #dV_dJ = np.gradient(voltage_charge ,current_density_charge)
    
    
    near_sc_index = (np.abs(voltage_charge - 0)).argmin()  # Closest index to V = 0
    voltage_charge_temp = voltage_charge[:near_sc_index+5]
    current_density_charge_temp = current_density_charge[:near_sc_index+5]
    rsh_derivative = abs(np.gradient(voltage_charge_temp ,current_density_charge_temp)).mean() 
    
    #rsh_derivative = abs(dV_dJ[near_sc_index])


    near_voc_index = (np.abs(voltage_charge - voc_temp)).argmin()  # Closest index to Voc
    voltage_charge_temp = voltage_charge[near_voc_index-5:near_voc_index]
    current_density_charge_temp = current_density_charge[near_voc_index-5:near_voc_index]
    rs_derivative = abs(np.gradient(voltage_charge_temp ,current_density_charge_temp)).mean()
    
    
    #rs_derivative = abs(dV_dJ[near_voc_index])
    
    return rsh_derivative, rs_derivative

def pso_func( eta_pso, path,  active_region_list, voltage_charge, current_density_charge):
    """
    Evaluates the Figure of Merit matrix for a photovoltaic device simulation based on 
    the given PSO (Particle Swarm Optimization) parameter values and IV curve characteristics.

    Args:
        eta_pso: (array-like) Array of PSO parameter values (diode ideality factor).
        path: (str) Path to IV curve .csv files generated by run_fdtd_and_charge function in charge.py (not used explicitly in the function)
        active_region_list: (list) List of SimInfo objects in the photovoltaic device (not used explicitly in the function).
        voltage_charge: (array) Voltage array of the IV curve.
        current_density_charge: (array) Current density array of the IV curve in mA/cm² (negative values).

    Returns:
        FoM_matrix: (list) A list of Figure of Merit (FoM) values for the simulated IV curves.
    """
    FoM_matrix = []
    generator = list(enumerate(zip(eta_pso)))
    pce_temp, ff_temp, voc_temp, jsc_temp, _, _ =  iv_curve(voltage_charge, current_density = current_density_charge)
    jmpp_temp = calc_jmpp(voltage_charge,current_density_charge)
    rsh_derivative, rs_derivative =  calc_R(voc_temp,voltage_charge,current_density_charge)

    for index, (eta_pso) in generator:
        diode_func_current = luqing_liu_diode(
            voltage = voltage_charge,
            jsc = -jsc_temp,
            jmpp =  -jmpp_temp,
            voc = voc_temp,
            rs = rs_derivative,
            rsh = rsh_derivative,
            eta = eta_pso[0],
            temp = 300,
            n_cells =1,
            )
        FoM_matrix.append(fom_func(voc_temp, voltage_charge, -current_density_charge, diode_func_current))
    return FoM_matrix

def run_pso(folder, active_region_list, param_dict, voltage, current):
    """
    Executes a Particle Swarm Optimization (PSO) algoritm to optimize the Figure of Merit

    Args:
        folder: (str) Path to the base directory where PSO results and plots will be saved.
        active_region_list: (list) List of SimInfo objects representing the active regions in the photovoltaic device.
        param_dict: (list of dict) List of parameter dictionaries for each active region to guide the PSO optimization process.
        voltage: (array) List of voltage arrays for the IV curves, corresponding to each active region.
        current: (array) List of current density arrays for the IV curves (in mA/cm², negative values), corresponding to each active region.

    Returns:
        best_FoM: (list) Best Figure of Merit (FoM) values for each active region.
        best_param: (list) Best parameter sets from the PSO optimization for each active region.
        best_param_particle: (list) Best-performing particle parameter sets for each iteration of the PSO process.
        best_FoM_iter: (list) FoM values at each iteration of the PSO for all active regions.

    Notes:
        - Creates a subdirectory for each active region within the base folder to save results.
        - Moves the PSO progress plot (`pso_update_res.png`) to the respective subdirectory.
    """
    best_FoM, best_param,best_param_particle, best_FoM_iter  =[], [], [], [] 
    for i, arl in enumerate(active_region_list):
        voltage_charge = voltage[i]
        current_density_charge = current[i]
        newpath = os.path.join(folder,arl.SCName)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        best_FoM_temp, best_param_temp, best_param_particle_temp, best_FoM_iter_temp = particle_swarm(pso_func, param_dict[i],  
            particles = 30, iterations = (40, 40, True), export = True, maximize = False, basepath = newpath,
           **{"path":folder, "active_region_list": arl, "voltage_charge": voltage_charge, "current_density_charge": current_density_charge})
        code_direct = os.getcwd()
        src_folder = code_direct if os.path.samefile(code_direct, folder) else folder
        src_path = os.path.join(code_direct, "pso_update_res.png" )
        dst_path = os.path.join(newpath, "pso_update_res.png" )
        shutil.move(src_path, dst_path)
        best_FoM.append(best_FoM_temp)
        best_param.append(best_param_temp)
        best_param_particle.append(best_param_particle_temp)
        best_FoM_iter.append(best_FoM_iter_temp)
    return best_FoM, best_param, best_param_particle, best_FoM_iter

def plot_2T(folder, active_region_list,  param_dict):
    """
    Generates a 2-terminal IV curve by combining data from individual subcell simulations and running optimization routines.

    Args:
        folder: (str) Path to the directory containing subcell IV curve CSV files.
        active_region_list: (list) List of `SimInfo` objects representing the active regions in the photovoltaic device.
        param_dict: (list of dict) List of parameter dictionaries for each active region to guide the PSO optimization process.

    Returns:
        voltage: (array) Voltage values of the combined 2-terminal IV curve.
        current_density: (array) Current density values of the combined 2-terminal IV curve (mA/cm²).

    Notes:
        - The function scales the voltage by a factor of 2 to represent the combined 2-terminal configuration.
        - Results are saved in a CSV file named "2T_IV_curve.csv" in the provided folder.

    Example Usage:
        plot_2T(folder, active_region_list,  param_dict)
        param_dict= [{'eta_pso': [0.9,4]},{'eta_pso': [0.9,3]}]
        Perovskite = SimInfo("solar_generation_PVK", "G_PVK.mat", "Perovskite", "interlayer","ITO" )
        Si = SimInfo("solar_generation_Si","G_Si.mat", "Si", "AZO", "interlayer")
        region= [Perovskite, Si]
        voltage, current_density = plot_2T("C:\Downloads\simulation_results", region, param_dict)
    """    
    
    pce, ff, voc, jsc, jmpp, rsh_derivative, rs_derivative  = [],[],[],[],[],[],[]
    voltage_charge_iv_curve = [0]
    voltage = []
    current_density = []
    for _, arl in enumerate(active_region_list): 
        voltage_charge = pd.read_csv(os.path.join(folder, f"{arl.SCName}_IV_curve.csv"), delimiter='\t')["Voltage"]
        if np.array(voltage_charge)[-1] > np.array(voltage_charge_iv_curve)[-1]:
            voltage_charge_iv_curve = voltage_charge
        current_density_charge = pd.read_csv(os.path.join(folder, f"{arl.SCName}_IV_curve.csv"), delimiter='\t')["Current_Density"]
        pce_temp, ff_temp, voc_temp, jsc_temp, _, _ =  iv_curve(voltage_charge, current_density = current_density_charge)
        jmpp_temp = calc_jmpp(voltage_charge,current_density_charge)
        rsh_derivative_temp, rs_derivative_temp =  calc_R(voc_temp,voltage_charge,current_density_charge)    
        pce.append(pce_temp)
        ff.append(ff_temp)
        voc.append(voc_temp)
        jsc.append(jsc_temp)
        jmpp.append(jmpp_temp)
        rsh_derivative.append(rsh_derivative_temp)
        rs_derivative.append(rs_derivative_temp)
        voltage.append(voltage_charge)
        current_density.append(current_density_charge)
    best_FoM, best_param, best_param_particle, best_FoM_iter = run_pso(folder, active_region_list, param_dict, voltage, current_density)
    
    for i, arl in enumerate(active_region_list):
        diode_func_current_temp = luqing_liu_diode(
            voltage = voltage[i],
            jsc = -jsc[i],
            jmpp =  -jmpp[i],
            voc = voc[i],
            rs = rs_derivative[i],
            rsh = rsh_derivative[i],
            eta = best_param[i],
            temp = 300,
            n_cells =1,
            )
        df = pd.DataFrame({"Current_Density": diode_func_current_temp, "Voltage": voltage[i]})
        csv_path = os.path.join(folder, f"{arl.SCName}_diode_fit_IV_curve.csv")
        df.to_csv(csv_path, sep = '\t', index = False)
        
    diode_func_current = luqing_liu_diode(
            voltage = np.array(voltage_charge_iv_curve)*2,
            jsc = min(abs(np.array(jsc))),
            jmpp =  min(jmpp),
            voc = sum(voc),
            rs = sum(rs_derivative),
            rsh = min(rsh_derivative),
            eta = sum(best_param),
            temp = 300,
            n_cells =1,
            )
    v_tandem = np.array(voltage_charge_iv_curve)*2
    j_tandem = diode_func_current
    df = pd.DataFrame({"Current_Density": j_tandem, "Voltage": v_tandem })
    csv_path = os.path.join(folder, "2T_IV_curve.csv")
    df.to_csv(csv_path, sep = '\t', index = False)
    return v_tandem, j_tandem, voltage, current_density

def plot_4T(folder,active_region_list):
    voltage_charge_iv_curve = 0
    voltage = []
    current_density = []    
    voc=  []
    for i, arl in enumerate(active_region_list): 
        voltage.append(pd.read_csv(os.path.join(folder, f"{arl.SCName}_IV_curve.csv"), delimiter='\t')["Voltage"])
        if np.array(voltage[i])[-1] > np.array(voltage_charge_iv_curve):
            voltage_charge_iv_curve = np.array(voltage[i])[-1]
        current_density.append(pd.read_csv(os.path.join(folder, f"{arl.SCName}_IV_curve.csv"), delimiter='\t')["Current_Density"])
        pce_temp, ff_temp, voc_temp, jsc_temp, _, _ =  iv_curve(voltage[i], current_density = current_density[i])
        voc.append(voc_temp)
    for i, _ in enumerate(active_region_list):     
        new_voltage = np.linspace(min(voltage[i]), voltage_charge_iv_curve, 2001)
        current_density[i] = np.interp(new_voltage, voltage[i], current_density[i])
        voltage[i] = new_voltage
    smallest_voc_material = voc.index(min(voc)) 
    j_tandem =  sum(np.array(current_density))
    v_tandem = voltage[smallest_voc_material]
    #pce_tandem, ff_tandem, voc_tandem, jsc_tandem, _, _ = iv_curve([], voltage[smallest_voc_material], 1, 1, "am", current_density = j_tandem)
    df = pd.DataFrame({"Current_Density": j_tandem, "Voltage": v_tandem })
    csv_path = os.path.join(folder, "4T_IV_curve.csv")
    df.to_csv(csv_path, sep = '\t', index = False)
    return v_tandem, j_tandem, voltage, current_density



"""
def plot_4T(folder, active_region_list,  param_dict):
    
    Generates a 4-terminal IV curve by combining data from individual subcell simulations and running optimization routines.

    Args:
        folder: (str) Path to the directory containing subcell IV curve CSV files.
        active_region_list: (list) List of `SimInfo` objects representing the active regions in the photovoltaic device.
        param_dict: (list of dict) List of parameter dictionaries for each active region to guide the PSO optimization process.

    Returns:
        voltage: (array) Voltage values of the combined 2-terminal IV curve.
        current_density: (array) Current density values of the combined 2-terminal IV curve (mA/cm²).

    Notes:
        - The function scales the voltage by a factor of 2 to represent the combined 2-terminal configuration.
        - Results are saved in a CSV file named "2T_IV_curve.csv" in the provided folder.

    Example Usage:
        plot_2T(folder, active_region_list,  param_dict)
        param_dict= [{'eta_pso': [0.9,4]},{'eta_pso': [0.9,3]}]
        Perovskite = SimInfo("solar_generation_PVK", "G_PVK.mat", "Perovskite", "interlayer","ITO" )
        Si = SimInfo("solar_generation_Si","G_Si.mat", "Si", "AZO", "interlayer")
        region= [Perovskite, Si]
        voltage, current_density = plot_2T("C:\Downloads\simulation_results", region, param_dict)
       
    
    pce, ff, voc, jsc, jmpp, rsh_derivative, rs_derivative  = [],[],[],[],[],[],[]
    voltage_charge_iv_curve = [0]
    voltage = []
    current = []
    for _, arl in enumerate(active_region_list): 
        voltage_charge = pd.read_csv(os.path.join(folder, f"{arl.SCName}_IV_curve.csv"), delimiter='\t')["Voltage"]
        if np.array(voltage_charge)[-1] > np.array(voltage_charge_iv_curve)[-1]:
            voltage_charge_iv_curve = voltage_charge
        current_density_charge = pd.read_csv(os.path.join(folder, f"{arl.SCName}_IV_curve.csv"), delimiter='\t')["Current_Density"]
        pce_temp, ff_temp, voc_temp, jsc_temp, _, _ =  iv_curve([], voltage_charge, 1, 1, "am", current_density = current_density_charge)
        jmpp_temp = calc_jmpp(voltage_charge,current_density_charge)
        rsh_derivative_temp, rs_derivative_temp =  calc_R(voc_temp,voltage_charge,current_density_charge)    
        pce.append(pce_temp)
        ff.append(ff_temp)
        voc.append(voc_temp)
        jsc.append(jsc_temp)
        jmpp.append(jmpp_temp)
        rsh_derivative.append(rsh_derivative_temp)
        rs_derivative.append(rs_derivative_temp)
        voltage.append(voltage_charge)
        current.append(current_density_charge)
    best_FoM, best_param, best_param_particle, best_FoM_iter = run_pso(folder, active_region_list, param_dict, voltage, current)
    diode_func_current = luqing_liu_diode(
            voltage = np.array(voltage_charge_iv_curve),
            jsc = sum(abs(np.array(jsc))),
            jmpp =  sum(jmpp),
            voc = min(voc),
            rs = sum(rs_derivative), #sum matches LT-spice calc and makes logical sence
            rsh = min(rsh_derivative),
            eta = min(best_param), #min matches LT-spice calc
            temp = 300,
            n_cells =1,
            )
    voltage = np.array(voltage_charge_iv_curve)
    current_density = diode_func_current
    df = pd.DataFrame({"Current_Density": current_density, "Voltage": voltage})
    csv_path = os.path.join(folder, "4T_IV_curve.csv")
    df.to_csv(csv_path, sep = '\t', index = False)
    print("RS",rs_derivative)
    print("RSH",rsh_derivative)
    print("eta",best_param)
    return voltage, current_density, rs_derivative, rsh_derivative, best_param


def plot_IV_curve(folder, active_region_list, param_dict, terminal_type="2T"):
    Generates a 2-terminal or 4-terminal IV curve by combining data from individual subcell simulations and running optimization routines.

    Args:
        folder: (str) Path to the directory containing subcell IV curve CSV files.
        active_region_list: (list) List of `SimInfo` objects representing the active regions in the photovoltaic device.
        param_dict: (list of dict) List of parameter dictionaries for each active region to guide the PSO optimization process.
        terminal_type: (str) Type of IV curve to generate ("2T" or "4T").

    Returns:
        voltage: (array) Voltage values of the combined IV curve.
        current_density: (array) Current density values of the combined IV curve (mA/cm²).

    Notes:
        - Results are saved in a CSV file named "2T_IV_curve.csv" or "4T_IV_curve.csv" in the provided folder.
    
    # Initialize variables
    pce, ff, voc, jsc, jmpp, rsh_derivative, rs_derivative = [], [], [], [], [], [], []
    voltage_charge_iv_curve = [0]
    voltage = []
    current = []

    # Process each active region
    for _, arl in enumerate(active_region_list):
        # Read voltage and current density data
        voltage_charge = pd.read_csv(os.path.join(folder, f"{arl.SCName}_IV_curve.csv"), delimiter='\t')["Voltage"]
        if np.array(voltage_charge)[-1] > np.array(voltage_charge_iv_curve)[-1]:
            voltage_charge_iv_curve = voltage_charge
        current_density_charge = pd.read_csv(os.path.join(folder, f"{arl.SCName}_IV_curve.csv"), delimiter='\t')["Current_Density"]

        # Calculate performance metrics
        pce_temp, ff_temp, voc_temp, jsc_temp, _, _ = iv_curve([], voltage_charge, 1, 1, "am", current_density=current_density_charge)
        jmpp_temp = calc_jmpp(voltage_charge, current_density_charge)
        rsh_derivative_temp, rs_derivative_temp = calc_R(voc_temp, voltage_charge, current_density_charge)

        # Append metrics to lists
        pce.append(pce_temp)
        ff.append(ff_temp)
        voc.append(voc_temp)
        jsc.append(jsc_temp)
        jmpp.append(jmpp_temp)
        rsh_derivative.append(rsh_derivative_temp)
        rs_derivative.append(rs_derivative_temp)
        voltage.append(voltage_charge)
        current.append(current_density_charge)

    # Run optimization
    best_FoM, best_param, best_param_particle, best_FoM_iter = run_pso(folder, active_region_list, param_dict, voltage, current)

    # Generate IV curve based on terminal type
    if terminal_type == "2T":
        diode_func_current = luqing_liu_diode(
            voltage=np.array(voltage_charge_iv_curve) * 2,
            jsc=min(abs(np.array(jsc))),
            jmpp=min(jmpp),
            voc=sum(voc),
            rs=sum(rs_derivative),
            rsh=min(rsh_derivative),
            eta=min(best_param),
            temp=300,
            n_cells=1,
        )
        voltage = np.array(voltage_charge_iv_curve) * 2
        output_file = "2T_IV_curve.csv"

    elif terminal_type == "4T":
        diode_func_current = luqing_liu_diode(
            voltage=np.array(voltage_charge_iv_curve),
            jsc=sum(abs(np.array(jsc))),
            jmpp=sum(jmpp),
            voc=min(voc),
            rs=sum(rs_derivative),
            rsh=min(rsh_derivative),
            eta=min(best_param),
            temp=300,
            n_cells=1,
        )
        voltage = np.array(voltage_charge_iv_curve)
        output_file = "4T_IV_curve.csv"

    else:
        raise ValueError("Invalid terminal_type. Must be '2T' or '4T'.")

    # Save results to CSV
    current_density = diode_func_current
    df = pd.DataFrame({"Current_Density": current_density, "Voltage": voltage})
    csv_path = os.path.join(folder, output_file)
    df.to_csv(csv_path, sep='\t', index=False)

    print("RS", rs_derivative)
    print("RSH", rsh_derivative)
    print("eta", best_param)

    return voltage, current_density
    """