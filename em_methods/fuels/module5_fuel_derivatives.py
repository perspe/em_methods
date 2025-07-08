import matplotlib.pyplot as plt
import seaborn as sns

# Constants
CITY = 'Sines'
STORAGE_PRESSURE = 250  # bar
R = 8.3144598  # J/(K.mol)
T = 298.15  # K (25ºC)
P = 101325  # PA (1 atm)
MW_CO2 = 44.01  # g/mol
MW_H2 = 2.01568  # g/mol

# Carbon tax (USD per gram of CO2) for each city as of September 2024
CARBON_TAX_RATES = {
    'Sines': 82.16 / 1e6,
    'Edmonton': 80 / 1e6,
    'Crystal Brook': 46.65 / 1e6
}

def get_h2_kg(city: str, ec_supply: str) -> float:
    """
    Returns the mass of H2 produced in kg/Wpeak for a given city and EC supply mode.
    
    Args:
    - city: City name ('Sines', 'Edmonton', or 'Crystal Brook').
    - ec_supply: Supply mode ('constant' or 'variable').
    
    Returns:
    - H2 mass in kg/Wpeak.
    
    Raises:
    - ValueError: If city or EC supply mode is invalid.
    """
    values = {
        'Sines': {'constant': 33.01493556, 'variable': 27.088164},
        'Edmonton': {'constant': 23.68395914, 'variable': 20.34863355},
        'Crystal Brook': {'constant': 30.37134349, 'variable': 25.4350396}
    }

    try:
        return values[city][ec_supply]
    except KeyError:
        raise ValueError(f"Invalid city '{city}' or EC supply '{ec_supply}'.")

def get_carbon_tax(city: str) -> float:
    """
    Returns the carbon tax for a given city.
    
    Args:
    - city: City name ('Sines', 'Edmonton', or 'Crystal Brook').
    
    Returns:
    - Carbon tax in $/g CO2.
    
    Raises:
    - ValueError: If city is not recognized.
    """
    try:
        return CARBON_TAX_RATES[city]
    except KeyError:
        raise ValueError(f"Invalid city '{city}'.")

def fuel_derivatives(H2_kg: float, heating_value: float, mw: float, density_stp: float, c_atoms: int, 
                     stoichiometry: float, cost: float, carbon_tax: float) -> list:
    """
    Calculates various fuel-related derivatives for a specific fuel.
    
    Args:
    - H2_kg: Mass of H2 in kg/Wpeak.
    - heating_value: Lower heating value (MJ/kg).
    - mw: Molecular weight (g/mol).
    - density_stp: Density at standard temperature and pressure (g/cm³).
    - c_atoms: Number of carbon atoms in one fuel molecule.
    - stoichiometry: Stoichiometric factor (mol of derivative per mol of H2).
    - cost: Fuel cost in USD/kg.
    - carbon_tax: Carbon tax rate in $/g CO2.
    
    Returns:
    - List of calculated values: [g/Wpeak, L/Wpeak, pressurized L/Wpeak, calorific value (Wh/Wpeak), revenue (USD/Wpeak), 
      revenue after tax (USD/Wpeak), tax loss (%)]. 
    """
    mol = H2_kg / MW_H2 * stoichiometry  # mol/Wpeak
    g = mol * mw  # g/Wpeak
    n_cm3 = g / density_stp  # cm³/Wpeak
    l = n_cm3 / 1000  # L/Wpeak
    l_pressurized = l * 1.01325 / STORAGE_PRESSURE  # L/Wpeak (if stored under pressure)
    calorific_value = mol * mw * heating_value * 0.27777777777778  # Wh/Wpeak
    revenue = cost / 1000 * g  # USD/Wpeak
    revenue_after_tax = revenue - (carbon_tax * mol * MW_CO2 * c_atoms)  # USD/Wpeak
    loss_tax = 100 - (100 * revenue_after_tax / revenue)  # Percentage revenue loss due to tax

    # Round results for cleaner output
    l, l_pressurized = round(l, 4), round(l_pressurized, 4)
    g, calorific_value, loss_tax = round(g, 2), round(calorific_value, 2), round(loss_tax, 2)

    return [g, l, l_pressurized, calorific_value, revenue, revenue_after_tax, loss_tax]

def display_results():
    """
    Displays results for different fuel derivatives, including hydrogen, oxygen, ammonia, methane, methanol, and kerosene.
    """
    volume_reduction = round(100 - (100 * 1.01325 / STORAGE_PRESSURE), 1)

    # Calculate H2 mass for the chosen city and supply mode
    H2_kg_value = get_h2_kg(CITY, 'constant')

    # Calculate fuel derivatives for different fuels
    fuels_data = {
        'H2': fuel_derivatives(H2_kg_value, 119.96, 2.01568, 8.988e-5, 0, 1, 7.17, get_carbon_tax(CITY)),
        'O2': fuel_derivatives(H2_kg_value, 0, 15.999, 1.429e-3, 0, 0.5, 0.2185, get_carbon_tax(CITY)),
        'NH3': fuel_derivatives(H2_kg_value, 18.646, 17.03022, 0.769e-3, 0, 0.6666667, 0.7288, get_carbon_tax(CITY)),
        'CH4': fuel_derivatives(H2_kg_value, 50.00, 16.04236, 0.717e-3, 1, 0.5, 1.9170, get_carbon_tax(CITY)),
        'CH3OH': fuel_derivatives(H2_kg_value, 19.930, 32.04, 0.792, 1, 0.5, 0.6202, get_carbon_tax(CITY)),
        'C12H26−C15H32': fuel_derivatives(H2_kg_value, 43.00, 170, 0.82, 13.5, 0.0689655172, 1.1950, get_carbon_tax(CITY)),
    }

    # Extracting results
    gs = [data[0] for data in fuels_data.values()]
    ls = [data[1] for data in fuels_data.values()]
    ls_pressurized = [data[2] for data in fuels_data.values()]
    calorific_values = [data[3] for data in fuels_data.values()]
    revenues_usd = [data[4] for data in fuels_data.values()]
    revenues_after_tax_usd = [data[5] for data in fuels_data.values()]
    losses_tax = [data[6] for data in fuels_data.values()]

    # Convert revenues to EUR
    conversion_rate = 0.950365  # EUR/USD as of November 2024
    revenues_eur = [round(rev * conversion_rate, 4) for rev in revenues_usd]
    revenues_after_tax_eur = [round(rev * conversion_rate, 2) for rev in revenues_after_tax_usd]

    # Print results
    print('Hydrogen fuel derivatives\n')
    print('Chemicals: hydrogen, oxygen, ammonia, methane, methanol, kerosene (kerosene fuel)')
    print('      -->     H2       O2      NH3      CH4     CH3OH      C12H26−C15H32\n')
    print(f'Mass of product (g/Wpeak):\n{gs}\n')
    print(f'Volume of product at 25ºC and 1 atm (L/Wpeak):\n{ls}\n')
    print(f'Volume of product at 25ºC and {STORAGE_PRESSURE} bar (L/Wpeak):\n{ls_pressurized}\n')
    print(f'Volume reduction: {volume_reduction}%\n')
    print(f'Calorific values (Wh/Wpeak):\n{calorific_values}\n')
    print(f'Revenue (€/Wpeak):\n{revenues_eur}\n')
    print(f'Revenue + carbon tax (€/Wpeak):\n{revenues_after_tax_eur}\n')
    print(f'Revenue loss due to carbon tax (%):\n{losses_tax}\n')
    
    return gs, calorific_values, revenues_after_tax_eur

if __name__ == '__main__':
    gs, calorific_values, revenues_after_tax_eur = display_results()
    
carbon_tax = get_carbon_tax(CITY)
                         
   
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
ax1.set_ylim(0,2700)
ax1.set_xticks([])
ax1.set_yticks([])
    
ax2.set_title('Mass production (g/W$_{{p}}$$^{{PV}}$/year)',fontsize=size_title)
palette = sns.color_palette(palette, len(gs))
for i, (value, label, color) in enumerate(zip(gs, bars, palette), start=1):
    ax2.bar(i, value, label=label, width=bar_width, color=color)
    ax2.text(i, value, f'{round(value)}', ha='center', va='bottom', fontsize=number_size)
ax2.set_ylim(0,310)
ax2.set_xticks([])
ax2.set_yticks([])

ax3.set_title('Revenue (€/W$_{{p}}$$^{{PV}}$/year)',fontsize=size_title)
palette = sns.color_palette(palette, len(revenues_after_tax_eur))
for i, (value, label, color) in enumerate(zip(revenues_after_tax_eur, bars, palette), start=1):
    ax3.bar(i, value, label=label, width=bar_width, color=color)
    ax3.text(i, value, f'{value}', ha='center', va='bottom', fontsize=number_size,color='black')
ax3.set_xticks(range(1, len(bars) + 1))
ax3.set_xticklabels(bars, rotation=45, ha='right', fontsize=label_font)
ax3.text(2.7,0.280,f'Global price, Nov 2024\nCarbon tax = {round(carbon_tax*0.91*10e6,1)} €/ton CO$_2$', fontsize=7.5)
ax3.set_ylim(0,0.33)
ax3.set_yticks([])

plt.tight_layout()
plt.subplots_adjust(wspace=0)
plt.show()

