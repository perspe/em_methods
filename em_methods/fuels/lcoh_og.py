' Calculates the levelized cost of hydrogen (LCOH) for each location'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys


city = 'Sines'
scenario = 1

# Installed PV, prices, energy flows and water consumption to produce 1 kg of H2 (results of module 4.3 and 4.4)

water_consumption = 9.95e-3   # m3
EC_supply = 52180             # Wh
oxygen_price = 0.21           # €/kg
oxygen_mass = (15.999*(1000/2.01568)/2)/1000  # kg
oxygen_revenue = oxygen_price * oxygen_mass   # €/year for 1 kg of H2 produced, assuming 100% efficiency


if city == 'Sines':
    electricity_price = 0.20e-3 # €/Wh
    water_price = 1.20          # €/m3
    if scenario == 1:
        installed_PV = 36.728   # W
        EC_capacity = 5.957     # W
        battery_capacity = 11.018 # W
        grid_imports = 28630    # Wh
        grid_exports = 34434    # Wh     
    elif scenario == 2:
        installed_PV = 41.922   # Wh
        EC_capacity = 20.961    # W
        battery_capacity = 0    # W
        grid_imports = 0        # Wh
        grid_exports = 13997    # Wh
    else:
        print('scenario must be 1 or 2')
        
elif city == 'Edmonton':
    electricity_price = 0.11e-3 # €/Wh
    water_price = 1.12          # €/m3
    if scenario == 1:
        installed_PV = 51.198   # W
        EC_capacity = 5.957     # W
        battery_capacity = 15.359 # W
        grid_imports = 29623    # Wh
        grid_exports = 35426    # Wh      
    elif scenario == 2:
        installed_PV = 56.320   # Wh
        EC_capacity = 28.160    # W
        battery_capacity = 0    # W
        grid_imports = 0        # Wh
        grid_exports = 11598    # Wh
    else:
        print('scenario must be 1 or 2')
        
elif city == 'Crystal Brook':
    electricity_price = 0.24e-3 # €/Wh
    water_price = 1.98          # €/m3
    if scenario == 1:
        installed_PV = 39.925   # W
        EC_capacity = 5.957     # W
        battery_capacity = 11.978 # W
        grid_imports = 27614    # Wh
        grid_exports = 33418    # Wh     
    elif scenario == 2:
        installed_PV = 44.936   # Wh
        EC_capacity = 22.468    # W
        battery_capacity = 0    # W
        grid_imports = 0        # Wh
        grid_exports = 13076    # Wh
    else:
        print('scenario must be 1 or 2')
else:
    print('Invalid city')  

# PV, EC and battery costs

discount_rate = 7.5    # %
lifetime = 25          # years 

PV_capex = 0.9 * installed_PV             # €/year for 1 kg of H2 produced
PV_opex = 0.017 * installed_PV            # €/year for 1 kg of H2 produced
EC_capex = 1.666 * EC_capacity            # €/year for 1 kg of H2 produced
EC_opex = (0.033+0.25) * EC_capacity      # €/year for 1 kg of H2 produced
battery_capex = 1.74 * battery_capacity   # €/year for 1 kg of H2 produced
battery_opex = 0.04 * battery_capacity    # €/year for 1 kg of H2 produced
capex = PV_capex + EC_capex + battery_capex
opex = PV_opex + EC_opex + battery_opex


# Create lists (values per year) for LCOH calculation

capex_list = [capex, PV_capex, EC_capex, battery_capex]
capex, PV_capex, EC_capex, battery_capex = [[value] + [0] * (lifetime-1) for value in capex_list]  # €, exists only in year 1, zero after

opex_list = [opex, PV_opex, EC_opex, battery_opex]
opex, PV_opex, EC_opex, battery_opex = [[value] * lifetime for value in opex_list]         # €/year
                                
expenses = [water_price * water_consumption] * lifetime                                    # €/year
revenue = [(grid_exports - grid_imports) * electricity_price + oxygen_revenue] * lifetime  # €/year
hydgrogen_mass = [1] * lifetime                                                            # kg/year
empty_list = [0] * lifetime

 
# LCOH calculation (does not consider degradation)

def calculate_LCOH(n, I, O_M, E, R, H, r):
    '''
    n: Lifetime of the project (in years)
    I: List or array of capital expenditures for each year
    O_M: List or array of O&M costs for each year
    E: List or array of grid imports and water expenses for each year
    R: List or array of revenues (grid exports and oxygen) for each year
    H: List or array of hydrogen production for each year, in kg
    r: Discount rate (as a decimal, e.g., 0.07 for 7%)'''
    
    numerator = np.sum([(I[i] + O_M[i] + E[i] - R[i]) / (1 + r)**i for i in range(n)])  # Numerator: Summation of discounted costs minus revenues
    denominator = np.sum([H[i] / (1 + r)**i for i in range(n)])  # Denominator: Summation of discounted hydrogen production

    LCOH = numerator / denominator if denominator != 0 else np.inf  # Calculate LCOH
    
    return LCOH

LCOH = calculate_LCOH(lifetime, capex, opex, expenses, revenue, hydgrogen_mass, discount_rate/100)

LCOH_PV = calculate_LCOH(lifetime, PV_capex, PV_opex, empty_list, empty_list, hydgrogen_mass, discount_rate/100)
LCOH_EC = calculate_LCOH(lifetime, EC_capex, EC_opex, empty_list, empty_list, hydgrogen_mass, discount_rate/100)
LCOH_battery = calculate_LCOH(lifetime, battery_capex, battery_opex, empty_list, empty_list, hydgrogen_mass, discount_rate/100)
LCOH_revenues = calculate_LCOH(lifetime, empty_list, empty_list, empty_list, revenue, hydgrogen_mass, discount_rate/100)
LCOH_expenses = calculate_LCOH(lifetime, empty_list, empty_list, expenses, empty_list, hydgrogen_mass, discount_rate/100)
LCOH_total = LCOH_PV + LCOH_EC + LCOH_battery + LCOH_revenues + LCOH_expenses


LCOH_list = [LCOH, LCOH_PV, LCOH_EC, LCOH_battery, LCOH_revenues, LCOH_expenses, LCOH_total]
LCOH, LCOH_PV, LCOH_EC, LCOH_battery, LCOH_revenues, LCOH_expenses, LCOH_total = [round(val,2) for val in LCOH_list]

print('LCOH =',LCOH,' €/kg H2')
print('')
print('PV =',LCOH_PV)
print('EC =',LCOH_EC)
print('battery =',LCOH_battery)
print('revenues =',LCOH_revenues)
print('expenses =',LCOH_expenses)
print('')
print('LCOH total =',LCOH_total,' €/kg H2')


# PLOT

components_pos = ['PV', 'EC', 'Battery', 'Water']
values_pos = [LCOH_PV, LCOH_EC, LCOH_battery, LCOH_expenses]
components_neg = ['Revenues*']
values_neg = [-LCOH_revenues]
total_pos = np.sum(values_pos)
total_neg = -values_neg[0]

fig, ax = plt.subplots(figsize=(1.2,2.8))
plt.rcParams['figure.dpi']=300
colors = sns.color_palette('mako_r', len(values_pos+values_neg))
number_size = 9.5
LCOH_size = 10
bar_width = 0.95

base = 0
for i in range(len(values_pos)):
    ax.bar(0, values_pos[i], width=bar_width, bottom=base, color=colors[i], label=components_pos[i])
    if values_pos[i] > 0:
        ax.text(0, base + values_pos[i]/2, f'{values_pos[i]:.2f}', ha='center', va='center', fontsize=number_size)
    base += values_pos[i]

ax.bar(1, total_neg, width=bar_width, bottom=base, color=colors[-1], label=components_neg[0])
ax.text(1, base + total_neg/2, f'{-total_neg:.2f}', ha='center', va='center', color='white', fontsize=number_size)
ax.plot([-0.5, 1.5], [total_pos+total_neg, total_pos+total_neg], color='black', lw=1)
ax.text(1, total_pos+total_neg-1.5, f'{LCOH}', ha='center', va='center', fontsize=LCOH_size, weight='bold')
ax.set_ylabel('LCOH (€/kg H2)')
ax.set_title(f'{city}, scenario {scenario}', fontsize=9)
#ax.text(1.05, 0.55, '*Revenues = oxygen +\nPV electricity sales', fontsize=6.5, transform=ax.transAxes, ha='left')
ax.set_ylim(0, 18)
ax.set_yticks([0, 3, 6, 9, 12, 15, 18]),
ax.set_xticks([]), ax.set_xticklabels([])
#ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
#plt.savefig(f'C:\\Users\\Cristina\\OneDrive - FCT NOVA\\Artigo 1 - GW-scale Solar-to-H2\\Figures\\3rd version\\LCOH_{city}_{scenario}.svg')

