from typing import List

city: str = "Sines"

# Constants
MW_CO2: float = 44.01  # g/mol
MW_H2: float = 2.01568  # g/mol

H2_KG_CITY = {
    "Sines": {"constant": 27.34758880209588, "variable": 23.8878},
    "Edmonton": {"constant": 19.64191460924014, "variable": 17.7885},
    "Crystal Brook": {"constant": 25.081214039491254, "variable": 22.2923},
}

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
