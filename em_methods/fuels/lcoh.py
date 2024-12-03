" Calculates the levelized cost of hydrogen (LCOH) for each location"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Default simulation variables
CITY = "Sines"
SCENARIO = 1
EC_LIFETIME = 25

# Installed PV, prices, energy flows and water consumption to produce 1 kg of H2 (results of module 4.3 and 4.4)
WATER_CONSUMPTION = 9.95e-3  # m3
EC_SUPPLY = 52180  # Wh
OX_PRICE = 0.21  # €/kg
OX_MASS = (15.999 * (1000 / 2.01568) / 2) / 1000  # kg

if CITY == "Sines":
    electricity_price = 0.20e-3  # €/Wh
    water_price = 1.20  # €/m3
    if SCENARIO == 1:
        installed_PV = 30.289  # W
        ec_capacity = 5.458  # W
        battery_capacity = 9.087  # W
        grid_imports = 27314  # Wh
        grid_exports = 26310  # Wh
    elif SCENARIO == 2:
        installed_PV = 36.916  # W
        ec_capacity = 18.458  # W
        battery_capacity = 0  # W
        grid_imports = 0  # Wh
        grid_exports = 10462  # Wh
    else:
        raise Exception("Invalid Scenario (choose either 1 or 2)")
elif CITY == "Edmonton":
    electricity_price = 0.11e-3  # €/Wh
    water_price = 1.12  # €/m3
    if SCENARIO == 1:
        installed_PV = 42.223  # W
        ec_capacity = 5.458  # W
        battery_capacity = 12.667  # W
        grid_imports = 28528  # Wh
        grid_exports = 27195  # Wh
    elif SCENARIO == 2:
        installed_PV = 49.1431  # W
        ec_capacity = 24.572  # W
        battery_capacity = 0  # W
        grid_imports = 0  # Wh
        grid_exports = 7837  # Wh
    else:
        raise Exception("Invalid Scenario (choose either 1 or 2)")
elif CITY == "Crystal Brook":
    electricity_price = 0.24e-3  # €/Wh
    water_price = 1.98  # €/m3
    if SCENARIO == 1:
        installed_PV = 32.928  # W
        ec_capacity = 5.458  # W
        battery_capacity = 9.878  # W
        grid_imports = 26583  # Wh
        grid_exports = 25430  # Wh
    elif SCENARIO == 2:
        installed_PV = 39.316  # W
        ec_capacity = 19.658  # W
        battery_capacity = 0  # W
        grid_imports = 0  # Wh
        grid_exports = 9280  # Wh
    else:
        raise Exception("Invalid Scenario (choose either 1 or 2)")
else:
    raise Exception("Invalid City")


# PV, EC and battery CAPEX and OPEX, in €/year for 1 kg of H2 produced
DISCOUNT_RATE = 7.5  # %
LIFETIME = 25  # years

pv_capex = 0.9 * installed_PV
pv_opex = 0.017 * installed_PV
ec_capex = 1.666 * ec_capacity
ec_opex = 0.02 * ec_capex + (0.075 * LIFETIME / EC_LIFETIME) * ec_capex
battery_capex = 1.84260942 * battery_capacity
battery_opex = 0.04206468 * battery_capacity

capex = pv_capex + ec_capex + battery_capex
opex = pv_opex + ec_opex + battery_opex

# Other OPEX, in €/year for 1 kg of H2 produced

water = water_price * WATER_CONSUMPTION  # €/m3 * m3/1kg H2
electricity = np.abs([(grid_exports - grid_imports) * electricity_price])  # Wh*eur/Wh
oxygen = oxygen_revenue = OX_PRICE * OX_MASS  #  €/kg*kg


# Create lists (values per year) for LCOH calculation

capex_list = [capex, pv_capex, ec_capex, battery_capex]
capex, pv_capex, ec_capex, battery_capex = [
    [value] + [0] * (LIFETIME - 1) for value in capex_list
]  # €, exists only in year 1, zero after

opex_list = [opex, pv_opex, ec_opex, battery_opex, water, electricity, oxygen]
opex, pv_opex, ec_opex, battery_opex, water, electricity, oxygen = [
    [value] * LIFETIME for value in opex_list
]  # €/year for 1 kg of H2 produced


hydgrogen_mass = [1] * LIFETIME
empty_list = [0] * LIFETIME

if grid_exports > grid_imports:
    expenses = water
    revenues = oxygen + electricity
else:
    expenses = water + electricity
    revenues = oxygen


# LCOH calculation (does not consider degradation)


def lcoh(
    lifetime: int,
    capital_expenditures,
    om_cost,
    expenses,
    revenues,
    hydro_prod,
    discount_rate: float,
):
    """
    lifetime: Lifetime of the project (in years)
    capital_expenditures: List or array of capital expenditures for each year - CAPEX
    om_cost: List or array of O&M (opeartion and maintenance) costs for each year - OPEX
    expenses: List or array of expenses for each year  (water and electricity if exports < imports)
    revenues: List or array of revenues for each year (oxygen and electricity if exports > imports)
    hydro_prod: List or array of hydrogen production for each year, in kg
    discount_rate: Discount rate (as a decimal, e.g., 0.07 for 7%)"""
    numerator = np.sum(
        [
            (capital_expenditures[i] + om_cost[i] + expenses[i] - revenues[i])
            / (1 + discount_rate) ** i
            for i in range(lifetime)
        ]
    )
    denominator = np.sum(
        [hydro_prod[i] / (1 + discount_rate) ** i for i in range(lifetime)]
    )
    lcoh = numerator / denominator if denominator != 0 else np.inf
    return lcoh

if __name__ == "__main__":
    lcoh_og = lcoh(
        LIFETIME, capex, opex, expenses, revenues, hydgrogen_mass, DISCOUNT_RATE / 100
    )
    lcoh_pv = lcoh(
        LIFETIME,
        pv_capex,
        pv_opex,
        empty_list,
        empty_list,
        hydgrogen_mass,
        DISCOUNT_RATE / 100,
    )
    lcoh_ec = lcoh(
        LIFETIME,
        ec_capex,
        ec_opex,
        empty_list,
        empty_list,
        hydgrogen_mass,
        DISCOUNT_RATE / 100,
    )
    lcoh_battery = lcoh(
        LIFETIME,
        battery_capex,
        battery_opex,
        empty_list,
        empty_list,
        hydgrogen_mass,
        DISCOUNT_RATE / 100,
    )
    lcoh_water = lcoh(
        LIFETIME,
        empty_list,
        empty_list,
        water,
        empty_list,
        hydgrogen_mass,
        DISCOUNT_RATE / 100,
    )
    lcoh_oxygen = lcoh(
        LIFETIME,
        empty_list,
        empty_list,
        empty_list,
        oxygen,
        hydgrogen_mass,
        DISCOUNT_RATE / 100,
    )

    if grid_exports > grid_imports:
        lcoh_electricity = lcoh(
            LIFETIME,
            empty_list,
            empty_list,
            empty_list,
            electricity,
            hydgrogen_mass,
            DISCOUNT_RATE / 100,
        )
    else:
        lcoh_electricity = lcoh(
            LIFETIME,
            empty_list,
            empty_list,
            electricity,
            empty_list,
            hydgrogen_mass,
            DISCOUNT_RATE / 100,
        )

    lcoh_total = (
        lcoh_pv + lcoh_ec + lcoh_battery + lcoh_water + lcoh_electricity + lcoh_oxygen
    )

    lcoh_list = [
        lcoh_og,
        lcoh_pv,
        lcoh_ec,
        lcoh_battery,
        lcoh_water,
        lcoh_oxygen,
        lcoh_electricity,
        lcoh_total,
    ]
    (
        lcoh_og,
        lcoh_pv,
        lcoh_ec,
        lcoh_battery,
        lcoh_water,
        lcoh_oxygen,
        lcoh_electricity,
        lcoh_total,
    ) = [round(val, 2) for val in lcoh_list]

    print("LCOH =", lcoh_og, " €/kg H2")
    print("")
    print("PV =", lcoh_pv)
    print("EC =", lcoh_ec)
    print("battery =", lcoh_battery)
    print("water =", lcoh_water)
    print("oxygen =", lcoh_oxygen)
    print("electricity =", lcoh_electricity)
    print("")
    print("LCOH total =", lcoh_total, " €/kg H2")
# PLOT
    if grid_exports > grid_imports:
        components_pos = ["PV", "EC", "battery", "water"]
        values_pos = [lcoh_pv, lcoh_ec, lcoh_battery, lcoh_water]
        components_neg = ["oxygen", "electricity"]
        values_neg = [-lcoh_oxygen, -lcoh_electricity]
    else:
        components_pos = ["PV", "EC", "battery", "water", "electricity"]
        values_pos = [lcoh_pv, lcoh_ec, lcoh_battery, lcoh_water, lcoh_electricity]
        components_neg = ["oxygen"]
        values_neg = [-lcoh_oxygen]

    total_pos = np.sum(values_pos)
    total_neg = -np.sum(values_neg)
    fig, ax = plt.subplots(figsize=(1.2, 2.8))
    plt.rcParams["figure.dpi"] = 300
    colors = sns.color_palette("mako_r", len(values_pos + values_neg))
    number_size = 9
    lcoh_size = 10
    bar_width = 0.95
    base_pos = 0
    for i in range(len(values_pos)):
        ax.bar(
            0,
            values_pos[i],
            width=bar_width,
            bottom=base_pos,
            color=colors[i],
            label=components_pos[i],
        )
        if values_pos[i] > 0:
            ax.text(
                0,
                base_pos + values_pos[i] / 2,
                f"{values_pos[i]:.2f}",
                ha="center",
                va="center",
                fontsize=number_size,
            )
        base_pos += values_pos[i]
    # Plot negative values
    base_neg = base_pos + total_neg
    for i in range(len(values_neg)):
        ax.bar(
            1,
            values_neg[i],
            width=bar_width,
            bottom=base_neg,
            color=colors[len(values_pos) + i],
            label=components_neg[i],
        )
        if values_neg[i] > 0:
            ax.text(
                1,
                base_neg + values_neg[i] / 2,
                f"{values_neg[i]:.2f}",
                ha="center",
                va="center",
                fontsize=number_size,
                color="white",
            )
        base_neg += values_neg[i]


    ax.plot(
        [-0.5, 1.5], [total_pos + total_neg, total_pos + total_neg], color="black", lw=1
    )
    ax.text(
        1,
        total_pos + total_neg - 1.5,
        f"{lcoh_og}",
        ha="center",
        va="center",
        fontsize=lcoh_size,
        weight="bold",
    )
    ax.set_ylabel("LCOH (€/kg H2)")
    ax.set_title(f"{CITY}, scenario {SCENARIO}", fontsize=9)
    ax.set_ylim(0, 16)
    ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16]),
    ax.set_xticks([]), ax.set_xticklabels([])
    # ax.set_yticks([]), ax.set_yticklabels([])
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    # plt.savefig(f'C:\\Users\\Cristina\\OneDrive - FCT NOVA\\Artigo 1 - GW-scale Solar-to-H2\\Figures\\4th version\\7, 8 and S10 results\\Figure 7_{city}_{scenario}.svg')
    plt.show()
