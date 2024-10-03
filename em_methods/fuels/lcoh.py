import numpy as np


def calculate_LCOH(
    lifetime, expenditures, om, imports, revenue, hydrogen, discount_rate
):
    """
    lifetime: Lifetime of the project (in years)
    expenditures: Capital expenditures each year
    om: O&M costs for each year
    imports: grid imports and water expenses for each year
    revenue: revenues (grid exports and oxygen) for each year
    hydrogen: hydrogen production for each year, in kg
    discount_rate: Discount rate (as a decimal, e.g., 0.07 for 7%)
    """
    # Numerator: Summation of discounted costs minus revenues
    numerator = np.sum(
        (expenditures + om + imports - revenue) / (1 + discount_rate) ** lifetime
    )
    # Denominator: Summation of discounted hydrogen production
    denominator = np.sum(
        hydrogen / (1 + discount_rate) ** lifetime
    )
    lcoh = numerator / denominator if denominator != 0 else np.inf
    return lcoh
