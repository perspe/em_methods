from typing import List, Union
import scipy.constants as scc
import numpy.typing as npt
import numpy as np

city: str = "Sines"

# Constants
MW_CO2: float = 44.01  # g/mol
MW_H2: float = 2.01568  # g/mol
WH_TO_MJ = 0.0036

H2_KG_CITY = {
    "Sines": {"constant": 27.34758880209588, "variable": 23.8878},
    "Edmonton": {"constant": 19.64191460924014, "variable": 17.7885},
    "Crystal Brook": {"constant": 25.081214039491254, "variable": 22.2923},
}


def hydrogen_mass_to_volume(
    mass: Union[npt.NDArray, float],
    temp: Union[npt.NDArray, float],
    pressure: Union[npt.NDArray, float],
) -> Union[npt.NDArray, float]:
    """
    Convert hydrogen mass (kg) to mass (m3)
    Args:
        mass: Hydrogen volume in kg
        temp: Temperature (in K)
        pressure: Pressure (in Pa)
    """
    molar_mass_h2 = 0.002016
    n = mass * scc.R * temp / pressure
    return n / molar_mass_h2


def hydrogen_volume_to_mass(
    volume: Union[npt.NDArray, float],
    temp: Union[npt.NDArray, float],
    pressure: Union[npt.NDArray, float],
) -> Union[npt.NDArray, float]:
    """
    Convert hydrogen volume (m3) to mass (kg)
    Args:
        volume: Hydrogen volume in m3
        temp: Temperature (in K)
        pressure: Pressure (in Pa)
    """
    molar_mass_h2 = 0.002016
    n = (pressure * volume) / (scc.R * temp)
    return n * molar_mass_h2


def fuel_properties(
    heating_value: float,
    MW: float,
    density_25C: float,
    h2_kg_value: float,
    C_atoms: int,
    stoichiometry: float,
    cost: float,
    carbon_tax: float,
) -> List[float]:
    """
    Calculates various properties and revenues for a given fuel.

    Args:
        heating_value: MJ/kg, lower heating value LHV (Net calorific value).
        MW: g/mol, molecular weight.
        density_25C: g/cm3, density at standard temperature and pressure (25ºC and 1 atm).
        h2_kg_value: mass of H produced in kg/Wpeak for a given city and electrical supply
        C_atoms: Number of C atoms in one fuel molecule.
        stoichiometry: For each H2 molecule, generates x molecules of derivative.
        cost: $/kg, most recent price of fuel
        carbon_tax: $/g CO2 emissions, carbon tax rate.

    Returns:
        List[float]: A list of calculated properties and revenues.
    """
    mol: float = h2_kg_value / MW_H2 * stoichiometry  # mol/Wpeak
    g: float = mol * MW  # g/Wpeak
    n_atom_cm3: float = g / density_25C  # cm3/Wpeak
    l: float = n_atom_cm3 / 1000  # L/Wpeak
    l_pressurized: float = l * 1.01325 / 250  # L/Wpeak if stored at 250 bar
    calorific_value: float = mol * MW * heating_value * 0.27777777777778  # Wh/Wpeak
    revenue: float = cost / 1000 * g  # $/Wpeak
    revenue_after_tax: float = revenue - (
        carbon_tax * mol * MW_CO2 * C_atoms
    )  # $/Wpeak
    loss_tax: float = 100 - (100 * revenue_after_tax / revenue)  # in % of revenue loss
    return [
        g,
        l,
        l_pressurized,
        calorific_value,
        revenue,
        revenue_after_tax,
        loss_tax,
    ]


def electrochemical(
    h2_production: Union[npt.NDArray, float],
    ec_consumed: Union[npt.NDArray, float],
    water_consumed: Union[npt.NDArray, float],
    system_losses: Union[npt.NDArray, float],
    h2_heating_value: Union[npt.NDArray, float],
    pv_yield: Union[npt.NDArray, float],
    ec_capacity: float = 0.5,
    units: str = "relative",
    temp: float = 298.15,
    pressure: float = scc.atm,
):
    """
    Args:
        h2_production (Nm3/h): volume of H2 produced per hour
        ec_consumed (Wh/h): energy consumed by the EC per hour
        water_consumed (L/Wh): water consumed per Wh of energy consumed
        system_losses (%): EC system losses in the electronic parts
        h2_heating_value (MJ/kg): lower heating value of hydrogen, aka energy density
        pv_yield (same as 'units'): PV energy yield considering temperature, reflection and system losses per hour
        ec_capacity (Wh/h): Capacity of each electrochemical cell
        units ('relative' or 'absolute'): W/Wpeak or W
    """
    if units == "relative":
        capacity = ec_capacity
    elif units == "absolute":
        capacity = ec_capacity * 1e6
    else:
        raise Exception("Invalid value in 'units' (use 'relative' or 'absolute'")
    consumption_losses_ph = ec_consumed * (1 + system_losses / 100)
    ec_units = ec_capacity / ec_consumed
    # Ratio between energy supplied and required
    supply_ratio = pv_yield / (ec_units * consumption_losses_ph)
    supply_ratio = np.clip(supply_ratio, 0, 1)
    supply_energy = ec_units * ec_consumed * supply_ratio
    # H2 production
    h2_volume = ec_units * h2_production * supply_ratio
    water_consumption = supply_energy * water_consumed * 1000
    h2_mass = hydrogen_volume_to_mass(h2_volume, temp, pressure)
    h2_calorific_power = h2_mass / 1000 * h2_heating_value / WH_TO_MJ
    return (
        supply_ratio,
        supply_energy,
        h2_mass,
        h2_calorific_power,
        water_consumption,
    )
