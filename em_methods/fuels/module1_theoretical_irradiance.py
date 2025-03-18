'''

For a given location and given hour (local time) of a defined day of the year, it calculates:
    Solar times and angles 
    Sunrise, sunset and hours of sun
    AM, irradiance on the horizontal plane (GHI) and PV panel surface (GTI).

Can use an yearly fixed tilt, or find the ideal tilt (maximizes irradiance on PV panel).

Plots Equation of Time (EoT) and AM using 3 different equations that describes it.

'''


from datetime import datetime
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# User inputs

LTh, LTm = 13, 30  # Local time (hours, minutes)
day, month = 21, 9  # Day and month of the year
city = 'Sines'  # City for the data
file_path = 'C:\\Users\\Cristina\\OneDrive - FCT NOVA\\M-ECO2\\Scripts\\PV'  # File path to data (update as needed)
define_tilt = 'fixed'  # 'fixed' or 'ideal': fixed uses latitude, ideal determines optimal tilt


def import_data_from_excel(file_path, city):
    """
    Imports city-specific solar and meteorological data from an Excel file.

    Parameters:
        file_path (str): Base file path to the Excel data file.
        city (str): Name of the city to extract data for.

    Returns:
        dict: A dictionary containing metadata and time-series data.
    """
    data = pd.read_excel(
        f'{file_path}\\GHI {city}\\dataGHI_{city}.xlsx', sheet_name='data_LT'
    )
    meta = {
        key: data.iat[i, 1] for key, i in zip(
            ['longitude', 'latitude', 'timezone', 'height', 'tilt'], range(1, 6)
        )
    }
    tmy_data = {
        col: data[col].tolist()
        for col in ['Month', 'hours', 'days', 'GHI_DS1', 'DNI_DS1', 'DHI_DS1', 'temperature', 'wind']
    }
    return {**meta, **tmy_data}


# Load data

data = import_data_from_excel(file_path, city)
(
    longitude, latitude, timezone, height, tilt,
    L_hours, L_days, L_months, L1_GHI, L1_DNI, L1_DHI, L_Tair, L_wind
) = (
    data['longitude'], data['latitude'], data['timezone'], data['height'], data['tilt'],
    data['hours'], data['days'], data['Month'],
    data['GHI_DS1'], data['DNI_DS1'], data['DHI_DS1'], data['temperature'], data['wind']
)

     
     
def days_since_start_of_year(month, day):
    """
    Calculates the number of days since the start of the year.

    Parameters:
        month (int): Month of the year.
        day (int): Day of the month.

    Returns:
        int: Days since the start of the year.
    """
    input_date = datetime(datetime.now().year, month, day)
    start_of_year = datetime(datetime.now().year, 1, 1)
    return (input_date - start_of_year).days


# Solar angles, irradiances (GHI and DNI), AM, sunrise and sunset calculations

def solar_parameters(LTh, LTm, month, day, timezone, longitude, latitude, height):
    """
    Calculates solar times, angles, and irradiances for a given location and time.

    Parameters:
        LTh (int): Local time (hours).
        LTm (int): Local time (minutes).
        month (int): Month of the year.
        day (int): Day of the month.
        timezone (int): Timezone offset (hours).
        longitude (float): Longitude of the location (degrees).
        latitude (float): Latitude of the location (degrees).
        height (float): Height above sea level (kilometers).

    Returns:
        tuple: Solar parameters including angles, irradiances, and times.
    """
    LT = LTh + LTm / 60
    d = days_since_start_of_year(month, day)
    LSTM = 15 * timezone
    B = 360 * (d - 81) / 365
    EoT = (
        9.87 * math.sin(math.radians(2 * B))
        - 7.53 * math.cos(math.radians(B))
        - 1.5 * math.sin(math.radians(B))
    )
    TC = 4 * (longitude - LSTM) + EoT
    LST = LT + TC / 60
    LST_min = int(60 * (LST - int(LST)))
    HRA = 15 * (LST - 12)

    declination = 23.45 * math.sin(math.radians(360 * (d - 81) / 365))

    declination_r, latitude_r, HRA_r = (
        math.radians(declination),
        math.radians(latitude),
        math.radians(HRA),
    )

    zenith_r = math.acos(
        math.sin(latitude_r) * math.sin(declination_r)
        + math.cos(latitude_r) * math.cos(declination_r) * math.cos(HRA_r)
    )
    elevation_r = math.radians(90) - zenith_r

    azimuth_r = math.acos(
        (
            math.sin(declination_r) * math.cos(latitude_r)
            - math.cos(declination_r) * math.sin(latitude_r) * math.cos(HRA_r)
        )
        / math.cos(elevation_r)
    )
    if LST > 12:
        azimuth_r = 2 * math.pi - azimuth_r
    
    if latitude > 0:
        azimuth_panel_r = math.radians(180)
    else:
        0
    
    sunrise = (
        12 - (1 / 15) * math.degrees(
            math.acos(-math.tan(latitude_r) * math.tan(declination_r))
        )
        - TC / 60
    )
    sunset = (
        12 + (1 / 15) * math.degrees(
            math.acos(-math.tan(latitude_r) * math.tan(declination_r))
        )
        - TC / 60
    )
    sunlight = sunset - sunrise
    sunlight_min = int(60 * (sunlight-int(sunlight)))   
    AM = 1 / (
        math.cos(zenith_r) + 0.50572 * (96.07995 - math.degrees(zenith_r)) ** -1.6364
    )
    DNI = 1.353 * ((1 - 0.14 * height) * (0.7 ** (AM ** 0.678)) + 0.14 * height)
    GHI = 1.1 * DNI * math.sin(elevation_r)

    return (
        d, latitude_r, HRA_r, declination_r, elevation_r, zenith_r, azimuth_r, azimuth_panel_r,
        LSTM, EoT, TC, LST, LST_min, AM, GHI, DNI, sunrise, sunset, sunlight, sunlight_min
    )


# Global Tilted Irradiance calculation

def calculate_gti(define_tilt, tilt, latitude_r, declination_r, HRA_r, DNI, elevation_r, azimuth_r, azimuth_panel_r, GHI):
    """
    Calculates the Global Tilted Irradiance (GTI), total irradiance falling on a PV panel's surface.

    Parameters:
        define_tilt (str): Either 'fixed' for a predefined tilt angle or 'ideal' for optimal tilt calculation.
        tilt (float): Fixed tilt angle in degrees (used if define_tilt is 'fixed').
        latitude_r (float): Latitude in radians.
        declination_r (float): Solar declination angle in radians.
        HRA_r (float): Hour angle in radians.
        DNI (float): Direct Normal Irradiance (kW/m2).
        elevation_r (float): Solar elevation angle in radians.
        azimuth_panel_r (float): Panel azimuth angle in radians.
        azimuth_r (float): Solar azimuth angle in radians.
        GHI (float): Global Horizontal Irradiance (kW/m2).

    Returns:
        tuple: (incidence, GTI, gain, tilt_ideal, incidence_ideal)
            incidence (list or float): Solar incidence angles (°).
            GTI (list or float): Global Tilted Irradiance values (kW/m2).
            gain (list or float): Irradiance gain percentages (%).
            tilt_ideal (float or None): Optimal tilt angle (°) if define_tilt is 'ideal', else None.
            incidence_ideal (float or None): Incidence angle (°) at optimal tilt, else None.
    """
    if define_tilt == 'fixed':
        tilt_r = math.radians(tilt)
        if latitude_r > 0:  # Northern Hemisphere
            incidence_r = math.acos(
                math.sin(latitude_r - tilt_r) * math.sin(declination_r) +
                math.cos(latitude_r - tilt_r) * math.cos(declination_r) * math.cos(HRA_r)
            )
        else:  # Southern Hemisphere
            incidence_r = math.acos(
                math.sin(latitude_r + tilt_r) * math.sin(declination_r) +
                math.cos(latitude_r + tilt_r) * math.cos(declination_r) * math.cos(HRA_r)
            )
        incidence = round(math.degrees(incidence_r), 1)
        GTI = 1.1 * DNI * (
            math.cos(elevation_r) * math.sin(tilt_r) * math.cos(azimuth_panel_r - azimuth_r) +
            math.sin(elevation_r) * math.cos(tilt_r)
        )
        gain = (GTI * 100 / GHI) - 100
        tilt_ideal, incidence_ideal = None, None

    elif define_tilt == 'ideal':
        incidence, GTI, gain = [], [], []
        tilt_values = np.linspace(0, 90, 100)

        for tilt_value in tilt_values:
            tilt_r = math.radians(tilt_value)
            if latitude_r > 0:  # Northern Hemisphere
                incidence_r = math.acos(
                    math.sin(latitude_r - tilt_r) * math.sin(declination_r) +
                    math.cos(latitude_r - tilt_r) * math.cos(declination_r) * math.cos(HRA_r)
                )
            else:  # Southern Hemisphere
                incidence_r = math.acos(
                    math.sin(latitude_r + tilt_r) * math.sin(declination_r) +
                    math.cos(latitude_r + tilt_r) * math.cos(declination_r) * math.cos(HRA_r)
                )

            incidence_value = round(math.degrees(incidence_r), 1)
            GTI_value = 1.1 * DNI * (
                math.cos(elevation_r) * math.sin(tilt_r) * math.cos(azimuth_panel_r - azimuth_r) +
                math.sin(elevation_r) * math.cos(tilt_r)
            )
            gain_value = (GTI_value * 100 / GHI) - 100

            incidence.append(incidence_value)
            GTI.append(round(GTI_value, 3))
            gain.append(round(gain_value, 3))

        max_gti_index = np.argmax(GTI)
        tilt_ideal = round(tilt_values[max_gti_index], 2)
        incidence_ideal = round(incidence[max_gti_index], 1)
    
    else:
        raise ValueError("Invalid value for 'define_tilt'. Must be 'fixed' or 'ideal'.")

    return incidence, GTI, gain, tilt_ideal, incidence_ideal



# Display all results

def format_and_round_variables(variables, is_angle=False, decimal_places=3):
    """
    Rounds a list of variables to the specified decimal places. Optionally converts radians to degrees.

    Parameters:
        variables (list): List of numeric variables.
        is_angle (bool): Whether to convert radians to degrees before rounding. Default is False.
        decimal_places (int): Number of decimal places to round to. Default is 3.

    Returns:
        list: Rounded variables.
    """
    if is_angle:
        variables = [math.degrees(var) for var in variables]
    return [round(var, decimal_places) for var in variables]


def display_solar_results(city, d, LSTM, EoT, TC, LST, LST_min, latitude, HRA, declination, elevation, zenith, azimuth, sunrise, sunset, sunlight, AM, GHI, GTI, tilt, incidence, gain):
    """
    Formats and displays solar calculation results in a professional manner.

    Parameters:
        city (str): Name of the city.
        d (int): Day of the year.
        LSTM (float): Local Standard Time Meridian (°).
        EoT (float): Equation of Time (min).
        TC (float): Time Correction (min).
        LST (float): Local Solar Time (hours).
        LST_min (int): Minutes part of Local Solar Time.
        latitude (float): Latitude (°).
        HRA (float): Hour Angle (°).
        declination (float): Solar declination (°).
        elevation (float): Solar elevation angle (°).
        zenith (float): Zenith angle (°).
        azimuth (float): Azimuth angle (°).
        sunrise (float): Sunrise time (hours).
        sunset (float): Sunset time (hours).
        sunlight (float): Hours of sunlight.
        AM (float): Air Mass.
        GHI (float): Global Horizontal Irradiance (kW/m²).
        GTI (float): Global Tilted Irradiance (kW/m²).
        tilt (float): Tilt angle (°).
        incidence (float): Incidence angle (°).
        gain (float): Gain (%).
    """
    print(f"\nSolar Analysis for {city}")
    print("-------------------------------------------")
    print("Solar Time Parameters:")
    print(f"  Day of the Year: {d}")
    print(f"  Local Standard Time Meridian (LSTM): {LSTM:.2f}°")
    print(f"  Equation of Time (EoT): {EoT:.2f} minutes")
    print(f"  Time Correction (TC): {TC:.2f} minutes")
    print(f"  Local Solar Time (LST): {int(LST)}:{LST_min:02d}")

    print("\nSolar Angles:")
    print(f"  Latitude (φ): {latitude:.2f}°")
    print(f"  Hour Angle (HRA): {HRA:.2f}°")
    print(f"  Declination (δ): {declination:.2f}°")
    print(f"  Elevation (α): {elevation:.2f}°")
    print(f"  Zenith (θ): {zenith:.2f}°")
    print(f"  Azimuth (γ): {azimuth:.2f}°")

    print("\nSunlight Information:")
    print(f"  Sunrise: {int(sunrise)}:{int((sunrise - int(sunrise)) * 60):02d}")
    print(f"  Sunset: {int(sunset)}:{int((sunset - int(sunset)) * 60):02d}")
    print(f"  Total Sunlight Duration: {int(sunlight)}:{int((sunlight - int(sunlight)) * 60):02d} hours")

    print("\nIrradiance:")
    print(f"  Air Mass (AM): {AM:.3f}")
    print(f"  Global Horizontal Irradiance (GHI): {GHI:.3f} kW/m²")
    print(f"  Global Tilted Irradiance (GTI): {GTI:.3f} kW/m²")
    print(f"  Tilt Angle: {tilt:.2f}°")
    print(f"  Incidence Angle: {incidence:.2f}°")
    print(f"  Gain: {gain:.2f}%")


# Example usage

d, latitude_r, HRA_r, declination_r, elevation_r, zenith_r, azimuth_r, azimuth_panel_r, LSTM, EoT, TC, LST, LST_min, AM, GHI, DNI, sunrise, sunset, sunlight, sunlight_min = solar_parameters(LTh, LTm, month, day, timezone, longitude, latitude, height)

incidence, GTI, gain, tilt_ideal, incidence_ideal = calculate_gti(
     define_tilt, tilt, latitude_r, declination_r, HRA_r, DNI, elevation_r, azimuth_r, azimuth_panel_r, GHI
 )

angular_vars = [latitude_r, HRA_r, declination_r, elevation_r, zenith_r, azimuth_r]
latitude, HRA, declination, elevation, zenith, azimuth = format_and_round_variables(angular_vars, is_angle=True, decimal_places=1)

non_angular_vars = [LSTM, EoT, TC, AM, GHI]
LSTM, EoT, TC, AM, GHI_int = format_and_round_variables(non_angular_vars, decimal_places=3)

display_solar_results(city, d, LSTM, EoT, TC, LST, LST_min, latitude, HRA, declination, elevation, zenith, azimuth, sunrise, sunset, sunlight, AM, GHI, GTI, tilt, incidence, gain)



# Plot EoT and AM

days = np.linspace(0, 365, 365)
EoT_values, declination = [], []
for d in days:
    B = 360 * (d-81) / 365
    EoT = 9.87*math.sin(math.radians(2*B)) - 7.53*math.cos(math.radians(B)) - 1.5*math.sin(math.radians(B))
    decl = 23.45 * math.sin(math.radians((d - 81) * 360 / 365))
    EoT_values.append(EoT)
    declination.append(decl)

zenith_range = np.linspace(0,90,180)
AM_range, AM_Kasten_range, AM_Rozenberg_range = [], [], []
for zenith in zenith_range:
    AM = 1/math.cos(math.radians(zenith)) if zenith < 89 else None
    AM_Rozenberg = (math.cos(math.radians(zenith))+0.025*math.exp(-11*math.cos(math.radians(zenith))))**-1
    AM_Kasten = 1/(math.cos(math.radians(zenith))+0.50572*(96.07995-zenith)**(-1.6364))
    AM_range.append(AM)
    AM_Rozenberg_range.append(AM_Rozenberg)
    AM_Kasten_range.append(AM_Kasten)

plt.rcParams['figure.dpi']=200
plt.figure(figsize=(10, 3.5))

plt.subplot(1, 2, 1)
plt.plot(days, EoT_values, color='tab:blue',label='EoT')
plt.xlabel('number of days since the start of the year')
plt.ylabel('Equation of Time (minutes)', color='tab:blue')
plt.tick_params(axis='y', labelcolor='tab:blue')
ax2 = plt.twinx()
ax2.set_ylabel('Declination angle (°)', color='tab:orange')
ax2.plot(days, declination, color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(zenith_range, AM_range, color='tab:green', label = "Without considering earth's curvature")
plt.plot(zenith_range, AM_Rozenberg_range, color='tab:blue', label = 'Rozenberg')
plt.plot(zenith_range, AM_Kasten_range, color='tab:orange',linestyle = '--', label = 'Kasten and Young')
plt.xlabel('zenith angle (°)')
plt.xlim(0,90)
plt.ylim(0,40)
plt.ylabel('Air mass')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()