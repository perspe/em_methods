'''
SOLAR FUELS DERIVED FROM HYDROGEN

Fuels to study: H2 (hydrogen), NH3 (ammonia), MeOH (CH3OH, methanol), CH4 (methane), jet (kerosene: C12H26−C15H32.)
https://www.engineeringtoolbox.com/fuels-higher-calorific-values-d_169.html
https://ec.europa.eu/eurostat/documents/38154/16135593/Hydrogen+-+Reporting+instructions.pdf/
https://macro.lsu.edu/howto/solvents/methanol.htm
https://en.wikipedia.org/wiki/Heat_of_combustion
https://businessanalytiq.com/index/

'''

import matplotlib.pyplot as plt

city = 'Sines'

def H2_kg(city, EC_supply):
    
    '''
    Gives the mass of H2 produced in kg/Wpeak, in a certain city --> Results of script 4.3 and 4.4
    
    city: 'Sines', 'Edmonton' or 'Crystal Brook'
    EC_supply: 'constant' or 'variable'
    
    '''
    values = {'Sines': {'constant': 27.34758880209588,'variable': 23.8878},
    'Edmonton': {'constant': 19.64191460924014,'variable': 17.7885},
    'Crystal Brook': {'constant': 25.081214039491254,'variable': 22.2923}}

    if city in values and EC_supply in values[city]:
        return values[city][EC_supply]
    else:
        raise ValueError("Invalid city or EC_supply value")

H2_kg = H2_kg(city, 'constant')


# Constants

R = 8.3144598      # J/(K.mol)
T = 298.15         # K   (25ºC)
P = 101325         # PA (1 atm)
MW_CO2 = 44.01     # g/mol 
MW_H2 = 2.01568    # g/mol

carbon_tax = 54.58e-6   # $/g CO2 emissions, April 2024 EU
 

def fuel_derivatives(heating_value, MW, density_25C, H_atoms, C_atoms, stoichiometry, cost_EU, cost_USA, carbon_tax):
    
    '''
    Variable Units and Meaning:
        heating_values: MJ/kg, lower heating value LHV (Net calorific value)
        MW: g/mol, molecular weight
        density_STP: g/cm3, density standard temperature and pressure (at 25ºC and 1 atm)
        H_atoms: number of H atoms in one fuel molecule
        C_atoms: number of C atoms in one fuel molecule
        stoichiometry: for each H2 molecule we will generate x molecules of derivative
        costs_EU: $/kg, most update price of fuel in EU
        costs_USA: $/kg, most update price of fuel in USA
        carbon_tax: $/g CO2 emissions, carbon tax        
    '''
    
    mol = H2_kg/MW_H2 * stoichiometry                                  # mol/Wpeak
    g = mol * MW                                                       # g/Wpeak
    Ncm3 = g/density_25C                                               # cm3/Wpeak
    L = Ncm3/1000                                                      # L/Wpeak
    L_pressurized = L*1.01325/250                                      # L/Wpeak if stored at 250 bar 
    calorific_value = mol*MW*heating_value*0.27777777777778            # Wh/Wpeak
    revenue_EU = cost_EU/1000 * g                                      # $/Wpeak
    revenue_USA = cost_USA/1000 * g                                    # $/Wpeak
    revenue_after_tax = revenue_EU - (carbon_tax*mol*MW_CO2*C_atoms)   # $/Wpeak
    loss_tax = 100 - (100*revenue_after_tax/revenue_EU)                # in % of revenue loss

    # Round values

    L, L_pressurized = [round(val,4) for val in [L, L_pressurized]]
    g, calorific_value, loss_tax = [round(val,2) for val in [g, calorific_value, loss_tax]]
    
    return [g, L, L_pressurized, calorific_value, revenue_EU, revenue_after_tax, loss_tax, revenue_USA]

volume_reduction = round(100 - (100*1.01325/250),1)

H2_data = fuel_derivatives(119.96, 2.01568, 8.23890e-5, 2, 0, 1, 5.23, 5.23, carbon_tax)         # prices of August 2024
NH3_data = fuel_derivatives(18.646, 17.03022, 6.960942e-4, 3, 0, 0.6666667, 0.46, 0.49, carbon_tax)
CH4_data = fuel_derivatives(50.00, 16.04236, 6.557164e-4, 4, 1, 0.5, 1.49, 0.8, carbon_tax)
CH3OH_data = fuel_derivatives(19.930, 32.04, 0.7866, 4, 1, 0.5, 0.38, 0.61, carbon_tax)
jet_data = fuel_derivatives(43.00, 170, 0.8201, 29, 13.5, 0.0689655172, 1.21, 0.74, carbon_tax)


gs = [H2_data[0], NH3_data[0], CH4_data[0], CH3OH_data[0], jet_data[0]]
Ls = [H2_data[1], NH3_data[1], CH4_data[1], CH3OH_data[1], jet_data[1]]
Ls_pressurized = [H2_data[2], NH3_data[2], CH4_data[2], CH3OH_data[2], jet_data[2]]
calorific_values = [H2_data[3], NH3_data[3], CH4_data[3], CH3OH_data[3], jet_data[3]]
revenues_EU = [H2_data[4], NH3_data[4], CH4_data[4], CH3OH_data[4], jet_data[4]]
revenues_after_tax = [H2_data[5], NH3_data[5], CH4_data[5], CH3OH_data[5], jet_data[5]]
losses_tax = [H2_data[6], NH3_data[6], CH4_data[6], CH3OH_data[6], jet_data[6]]
revenues_USA = [H2_data[7], NH3_data[7], CH4_data[7], CH3OH_data[7], jet_data[7]]


def dolar_to_eur(*dollar_lists):
    return [[round(revenue * 0.90,5) for revenue in dollar_list] for dollar_list in [*dollar_lists]]    # conversion rate of August 2024

revenues_EU, revenues_after_tax, revenues_USA = dolar_to_eur(*[revenues_EU, revenues_after_tax, revenues_USA])


print('Hydrogen fuel derivatives\n')
print('Fuels: hydrogen, ammonia, methane, methanol, kerosene (jet fuel)')
print('      --> H2       NH3      CH4     CH3OH      C12H26−C15H32')
print('')
print(f'mass of product (g/Wpeak):\n{gs}\n')
print(f'volume of product at 25C and 1 atm (L/Wpeak):\n{Ls}\n')
print(f'volume of product at 25C and 200 bar (L/Wpeak):\n{Ls_pressurized}\n')
print(f'volume reduction: {volume_reduction}%\n')
print(f'calorific values (Wh/Wpeak):\n{calorific_values}\n')
print(f'revenue EU (€/Wpeak):\n{revenues_EU}\n')
print(f'revenue EU + carbon tax (€/Wpeak):\n{revenues_after_tax}\n')
print(f'revenue loss due to carbon tax (%):\n{losses_tax}\n')
print(f'revenue USA (€/Wpeak):\n{revenues_USA}\n')

      
# Plot


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 3.25))
plt.rcParams['figure.dpi'] = 200

bars = ['hydrogen', 'ammonia', 'methane', 'methanol', 'kerosene']
alpha_values = [0.2, 0.4, 0.6, 0.8, 1]
alpha_values2 = [0.6, 0.8, 1]
bar_width = 1
number_size = 9
label_font = 10
size_title = 10

ax1.set_title('Calorific value\n(Wh/W$_{{peak}}$/year)',fontsize=size_title)
for i, (value, label, alpha) in enumerate(zip(calorific_values, bars, alpha_values), start=1):
    ax1.bar(i, value, label=label, color='teal', width=bar_width, alpha=alpha)
    ax1.text(i, value, f'{round(value)}', ha='center', va='bottom', fontsize=number_size)
ax1.set_xticks(range(1, len(bars) + 1))
ax1.set_xticklabels(bars, rotation=45, ha='right', fontsize=label_font)
ax1.set_ylim(0,2100)  
ax1.set_xticks([0, 1, 2, 3, 4, 5, 6])
ax1.set_yticks([]), ax1.set_yticklabels([]), ax1.set_ylabel('')
    
ax2.set_title('Mass production\n(g/W$_{{peak}}$/year)',fontsize=size_title)
for i, (value, label, alpha) in enumerate(zip(gs, bars, alpha_values), start=1):
    ax2.bar(i, value, label=label, color='goldenrod', width=bar_width, alpha=alpha)
    ax2.text(i, value, f'{round(value)}', ha='center', va='bottom', fontsize=number_size)
ax2.set_xticks(range(1, len(bars) + 1))
ax2.set_xticklabels(bars, rotation=45, ha='right', fontsize=label_font) 
ax2.set_ylim(0,240)  
ax2.set_xticks([0, 1, 2, 3, 4, 5, 6]),
ax2.set_yticks([]), ax2.set_yticklabels([]), ax2.set_ylabel('')

ax3.set_title('Volume production\n(L/W$_{{peak}}$/year)',fontsize=size_title)
for i, (value, label, alpha) in enumerate(zip(Ls, bars, alpha_values), start=1):
    ax3.bar(i, value, label=label, color='darkgreen', width=bar_width, alpha=alpha)
    ax3.text(i, value, f'{round(value,2)}*', ha='center', va='bottom', fontsize=number_size)
for i, (value, label, alpha) in enumerate(zip(Ls_pressurized[:-2], bars, alpha_values), start=1):
    ax3.bar(i, value, label=label, color='darkgreen', width=bar_width, alpha=alpha)
    ax3.text(i, value, f'{round(value,2)}$^♦$', ha='center', va='bottom', fontsize=number_size)
ax3.set_xticks(range(1, len(bars) + 1))
ax3.set_xticklabels(bars, rotation=45, ha='right', fontsize=label_font)
ax3.text(5.8,310, '*at 1.01325 bar\n$^♦$at 250 bar', fontsize =9, horizontalalignment='right')
ax3.set_ylim(0,370)  
ax3.set_xticks([0, 1, 2, 3, 4, 5, 6])
ax3.set_yticks([]), ax3.set_yticklabels([]), ax3.set_ylabel('')

ax4.set_title('Revenue\n(€/W$_{{peak}}$/year)',fontsize=size_title)
for i, (value, label, alpha) in enumerate(zip(revenues_after_tax, bars, alpha_values), start=1):
    ax4.bar(i, value, label=label, color='dimgray', width=bar_width, alpha=alpha)
    ax4.text(i, value, f'{value}', ha='center', va='bottom', fontsize=number_size,color='black')
for i, (value, label, alpha) in enumerate(zip(revenues_USA, bars, alpha_values), start=1):
    ax4.bar(i, value, label=label, color='firebrick', width=bar_width, alpha=alpha)
    ax4.text(i, value, f'{value}', ha='center', va='bottom', fontsize=number_size, color='darkred')
ax4.set_xticks(range(1, len(bars) + 1))
ax4.set_xticklabels(bars, rotation=45, ha='right', fontsize=label_font)
ax4.set_ylim(0,0.165)  
ax4.set_xticks([0, 1, 2, 3, 4, 5, 6])
ax4.set_yticks([]), ax4.set_yticklabels([]), ax4.set_ylabel('')

plt.tight_layout()
plt.subplots_adjust(wspace=0)
#plt.savefig(f'C:\\Users\\Cristina\\OneDrive - FCT NOVA\\Artigo - GW-scale Solar-to-H2\\Figures\\3rd version\\{city}_figures10.svg')
plt.show()



