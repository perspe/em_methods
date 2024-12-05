from typing import List, Union
import scipy.constants as scc
import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys


# User inputs

city: str = "Sines"  # choose between Sines, Edmonton and Crystal Brook
EC_supply: str = "variable"  # choose between constant and variable
storage_pressure: float = 250   # bar

# Constants

MW_CO2: float = 44.01  # g/mol
MW_H2: float = 2.01568  # g/mol
WH_TO_MJ = 0.0036 
USD_TO_EUR = 0.950365  # Nov 2024

# Carbon tax for each country, in $/g CO2 emissions, September 2024

if city == 'Sines':
    carbon_tax = 82.16/10e6   # https://www.portugalresident.com/carbon-tax-raised-to-e74-429-t-co2-as-impact-of-high-crude-price-eases/
elif city == 'Edmonton':
    carbon_tax = 80/10e6      # https://femmesetvilles.org/canada-carbon-tax-rebate-confirmed-payment-eligibility-dates/#:~:text=On%20April%201%2C%202024%2C%20the,and%20other%20fuel%2Drelated%20expenses.
elif city == 'Crystal Brook':
    carbon_tax = 46.65/10e6   # https://economics.uq.edu.au/article/2024/04/australia-now-has-shadow-carbon-price#:~:text=In%202024%2C%20the%20shadow%20carbon,and%20costs%20of%20rule%20changes.
else:
    print('Invalid city')
    sys.exit()    


def H2_kg(city, EC_supply):
    
    '''
    Gives the mass of H2 produced in kg/Wpeak, in a certain city --> Results of script 4.3 and 4.4
    
    city: 'Sines', 'Edmonton' or 'Crystal Brook'
    EC_supply: 'constant' or 'variable'
    
    '''
    values = {'Sines': {'constant': 33.01493556,'variable': 27.088164},
    'Edmonton': {'constant': 23.68395914,'variable':20.34863355},
    'Crystal Brook': {'constant': 30.37134349,'variable': 25.4350396}}

    if city in values and EC_supply in values[city]:
        return values[city][EC_supply]
    else:
        raise ValueError("Invalid city or EC_supply value")

H2_kg = H2_kg(city, EC_supply)



def hydrogen_mass_to_volume(
    mass: Union[npt.NDArray, float],
    temp: Union[npt.NDArray, float],
    pressure: Union[npt.NDArray, float],
) -> Union[npt.NDArray, float]:
    """
    Convert hydrogen mass (kg) to volume (m3)
    Args:
        mass: Hydrogen volume in kg
        temp: Temperature (in K)
        pressure: Pressure (in Pa)
    """
    n = mass * scc.R * temp / pressure
    return n / MW_H2


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
    n = (pressure * volume) / (scc.R * temp)
    return n * MW_H2


def fuel_properties(
    heating_value: float,
    MW: float,
    density_25C: float,
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
        C_atoms: Number of C atoms in one fuel molecule.
        stoichiometry: For each H2 molecule, generates x molecules of derivative.
        cost: $/kg, most recent price of fuel
        carbon_tax: $/g CO2 emissions, carbon tax rate.

    Returns:
        List[float]: A list of calculated properties and revenues.
    """
    mol: float = H2_kg / MW_H2 * stoichiometry  # mol/Wpeak
    g: float = mol * MW  # g/Wpeak
    Ncm3: float = g / density_25C  # cm3/Wpeak
    l: float = Ncm3 / 1000  # L/Wpeak
    l_pressurized: float = l * 1.01325 / storage_pressure  # L/Wpeak if stored at 250 bar
    calorific_value: float = mol * MW * heating_value / 1000 / WH_TO_MJ  # Wh/Wpeak
    revenue: float = cost / 1000 * g  # $/Wpeak
    revenue_after_tax: float = revenue - (
        carbon_tax * mol * MW_CO2 * C_atoms
    )  # $/Wpeak
    loss_tax: float = 100 - (100 * revenue_after_tax / revenue)  # in % of revenue loss
    # Round values
    l, l_pressurized = [round(val,4) for val in [l, l_pressurized]]
    g, calorific_value, loss_tax = [round(val,2) for val in [g, calorific_value, loss_tax]]
      
    return [
        g,
        l,
        l_pressurized,
        calorific_value,
        revenue,
        revenue_after_tax,
        loss_tax,
    ]


# Examples of fuels/chemicals derived from H2

H2_data = fuel_properties(119.96, 2.01568, 8.988e-5, 0, 1, 5.26, carbon_tax)        # September 2024
NH3_data = fuel_properties(18.646, 17.03022, 0.769e-3, 0, 0.6666667, 0.7288, carbon_tax)
CH4_data = fuel_properties(50.00, 16.04236, 0.717e-3, 1, 0.5, 1.9170, carbon_tax)
CH3OH_data = fuel_properties(19.930, 32.04, 0.792, 1, 0.5, 0.6202, carbon_tax)
kerosene_data = fuel_properties(43.00, 170, 0.82, 13.5, 0.0689655172, 1.1950, carbon_tax)
O2_data = fuel_properties(0, 15.999, 1.429e-3, 0, 0.5, 0.2185, carbon_tax)             # June 2024

gs = [H2_data[0], O2_data[0], NH3_data[0], CH4_data[0], CH3OH_data[0], kerosene_data[0]]
Ls = [H2_data[1], O2_data[1], NH3_data[1], CH4_data[1], CH3OH_data[1], kerosene_data[1]]
Ls_pressurized = [H2_data[2], O2_data[2], NH3_data[2], CH4_data[2], CH3OH_data[2], kerosene_data[2]]
calorific_values = [H2_data[3], O2_data[3], NH3_data[3], CH4_data[3], CH3OH_data[3], kerosene_data[3]]
revenues_USD = [H2_data[4], O2_data[4], NH3_data[4], CH4_data[4], CH3OH_data[4], kerosene_data[4]]
revenues_after_tax_USD = [H2_data[5], O2_data[5], NH3_data[5], CH4_data[5], CH3OH_data[5], kerosene_data[5]]
losses_tax = [H2_data[6], O2_data[6], NH3_data[6], CH4_data[6], CH3OH_data[6], kerosene_data[6]]   


# Convert USD to EUR
revenues_eur = [round(rev * USD_TO_EUR,4) for rev in revenues_USD]
revenues_after_tax_eur = [round(rev * USD_TO_EUR,2) for rev in revenues_after_tax_USD] 



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
  
# Print

print('Hydrogen fuel derivatives\n')
print('Chemicals: hydrogen, oxygen, ammonia, methane, methanol, kerosene')
print('      -->     H2       O2      NH3      CH4     CH3OH  C12H26−C15H32')
print('')
print(f'mass of product (g/Wpeak):\n{gs}\n')
print(f'volume of product at 25C and 1 atm (L/Wpeak):\n{Ls}\n')
print(f'volume of product at 25C and {storage_pressure} bar (L/Wpeak):\n{Ls_pressurized}\n')
print(f'calorific values (Wh/Wpeak):\n{calorific_values}\n')
print(f'revenue (€/Wpeak):\n{revenues_eur}\n')
print(f'revenue + carbon tax (€/Wpeak):\n{revenues_after_tax_eur}\n')
print(f'revenue loss due to carbon tax (%):\n{losses_tax}\n')

    
# Plot

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3.25, 7.55))
plt.rcParams['figure.dpi'] = 300
palette = 'mako_r'

bars = ['hydrogen', 'oxygen', 'ammonia', 'methane', 'methanol', 'kerosene']
alpha_values = [0.167, 0.333, 0.5, 0.667, 0.833, 1]
alpha_values2 = [0.6, 0.8, 1]
bar_width = 0.9
number_size = 10.5
label_font = 10.5
size_title = 10.5

ax1.set_title('Calorific value (Wh/W$_{{p}}$$^{{PV}}$/year)',fontsize=size_title)
palette = sns.color_palette(palette, len(calorific_values))
for i, (value, label, color) in enumerate(zip(calorific_values, bars, palette), start=1):
    ax1.bar(i, value, label=label, width=bar_width, color=color)
    ax1.text(i, value, f'{round(value)}', ha='center', va='bottom', fontsize=number_size)
ax1.set_ylim(0,2300) #2700
ax1.set_xticks([])
ax1.set_yticks([])
    
ax2.set_title('Mass production (g/W$_{{p}}$$^{{PV}}$/year)',fontsize=size_title)
palette = sns.color_palette(palette, len(gs))
for i, (value, label, color) in enumerate(zip(gs, bars, palette), start=1):
    ax2.bar(i, value, label=label, width=bar_width, color=color)
    ax2.text(i, value, f'{round(value)}', ha='center', va='bottom', fontsize=number_size)
ax2.set_ylim(0,250) #310
ax2.set_xticks([])
ax2.set_yticks([])

ax3.set_title('Revenue (€/W$_{{p}}$$^{{PV}}$/year)',fontsize=size_title)
palette = sns.color_palette(palette, len(revenues_after_tax_eur))
for i, (value, label, color) in enumerate(zip(revenues_after_tax_eur, bars, palette), start=1):
    ax3.bar(i, value, label=label, width=bar_width, color=color)
    ax3.text(i, value, f'{value}', ha='center', va='bottom', fontsize=number_size,color='black')
ax3.set_xticks(range(1, len(bars) + 1))
ax3.set_xticklabels(bars, rotation=45, ha='right', fontsize=label_font)
ax3.text(2.9,0.220,f'Global price, Nov 2024\nCarbon tax = {round(carbon_tax*0.91*10e6,1)} €/ton CO$_2$', fontsize=7.5) #0.117
ax3.set_ylim(0,0.26) #0.33
ax3.set_yticks([])

plt.tight_layout()
plt.subplots_adjust(wspace=0)
plt.show()


