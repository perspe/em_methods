'''
PURPOSE

Considers a PV-EC grid-connected and battery-assisted system. For 1 location and 1 Typical Meteorologial Year (TMY) dataset (DS1), calculates Irradiance (on earth's surface and on PV panel),
PV energy yield (with and without considering temperature), EC supply, and grid imports and exports, throughout a set time period (e.g. year)

Time frame = few months to a full year

Considers discontinuous EC operation: EC works only when solar energy is avalaible, in day-night cycles.
EC power (percentage of utilization) follows PV energy yield. EC capacity = 50% of PV capacity.  There's no batery or grid imports.

Units of energy: can chose between W/Wpeak or W. 

Plot data and obtain max, min, average, standard deviation, and total values. 


DESCRIPTION

1.	Set PV data (area, efficiency, power, temperature coefficient, mounting type), system losses (14% as default) and EC data (H2 production, power and water consumption, system losses  (14% as default))
    EC supply is the PV average annual yield --> obrained from Module 4.3
        
2.	Select city (Sines, Edmonton or Crystal Brook), time period (year as default), import empiric data from a TMY: and 
    i.	DNI: Direct Normal Irradiance on a plane always normal to sun rays (W/m2)
    ii. DHI: Diffuse Horizontal Irradiance - on Earth's surface (W/m2)
    ii. GHI: Global Horizontal Irradiance - on Earth's surface (W/m2)
    ii.	Air temperature (degrees Celsius) and wind speed (m/s) at 10m height
    iii.Latitude, Longitude, height, timezone, ideal solar panel tilt 
    *select values of lists according to the set period
        --> def import_data_from_excel_1dataset

3.	Use the solar angles/times equations and impiric data to create new lists:
    i.   theoretical and empiric (DS1) GHI and GTI (GTI considers shallow angle reflection losses)
    ii.  temperature on the PV panel;
    iii. theoretical and empiric (DS1) PV energy yield, with and without considering temperature losses.
        --> def days_since_start_of_year   &   def solar_parameters_1dataset
        
4.	Convert data to relative values - divide per peak power density (W/Wpeak), or absolute - assume a PV capacity (MW)

5.  Calculate total values per day; max and min temperature and wind speed; and statistics of GHI, GTI and PV yield for theoretical and empiric (DS1) date. Calculate temperature and reflection losses.
        --> def lists_per_day   &   def max_temp_wind   &   def calculate_statistics
    
6.	EC calculations: EC supply, H2 productions by mass, volume and calorific value, water comsumption, EC efficiency
        --> def EC_calculations_2   &   def hydrogen_volume_to_mass        

7.  Determine flow of energy between PV, EC and grid, per hour:
        --> def calculate_energy_usage_2
        
8.  Determine EC values statistics, create lists per hour and day, and round values  
    
9.  VERIFICATION : check if the net energy flow is zero: energy in (PV) equals energy out (EC + exports)        
        
10.  Plots energy flow per hour: PV energy yield, (with and without considering losses: refletcion, tmeperature and PV system), grid imports and exports, battery storage, EC supply and H2 calorific value 


PV energy losses: https://joint-research-centre.ec.europa.eu/photovoltaic-geogralatitudecal-information-system-pvgis/getting-started-pvgis/pvgis-data-sources-calculation-methods_en
Temperature depedence: https://doi.org/10.1016/j.solmat.2008.05.016Get rights and content

Solar panel details: c-Si (LONGI LR7-72HGD-620M), size 1.1 x 2.3 m2, https://www.longi.com/en/products/modules/hi-mo-7/
Electrolyzer details: DQ1000, Alkaline Electrolyser | 1000 Nm³/h, https://hydrogen.johncockerill.com/wp-content/uploads/sites/3/2023/04/dq-1000-def-2-hd-en.pdf
    
NREL TMY database (used mainly for America and Australia): https://nsrdb.nrel.gov/data-viewer


GLOSSARY

    L = list, V = value
    t = theoretical, 1 = DS1, 2 = DS2
    T = considering temperature
    gen = generated power
    To = total
    pd = interval is 1 day, pm = interval is 1 month, d = f day d
    i = index of a list, d = of day d, m = of month m 
    ter = term of an equation

'''


from datetime import datetime
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import sys


# User inputs

city = 'Sines'
year,day,month = 2025, [1,31],[1,12]    # define period of time: some months too entire year
PVutilization = 1                       # 1 = average anual power generated
file_path = 'C:\\Users\\Cristina\\OneDrive - FCT NOVA\\M-ECO2\\Scripts\\PV'
 
units = 'relative'                  # chose between 'relative' or 'absolute, and defined the values below:

if units == 'relative':
    EC_capacity = 0.5               # Wh/Wpeak, max EC capacity, corresponds to half of max PV production 
elif units =='absolute':
    PV_capacity = 10e6              # Wh
    EC_capacity = 0.5*PV_capacity   # Wh, max EC capacity, corresponds to half of max PV production 
else:
    print('Units should be "relative" or "absolute"')
    sys.exit()
        
'''
    Relative yield lists
           --> represents the amount of energy produced per unit of installed PV capacity over a relative period (W/Wpeak)
           --> PV capacity is not set, calculations are relative to the 1 Wpeak of installed PV

     Absolute yield lists
           --> represents the amount of energy produced considering the defined PV capacity over a relative period (W)
           --> a PV capacity of 10MW produces 10MW at peak power under standard test conditions (STC: 1000 W/m2 irradiance, 25ºC and AM 1.5)
'''


# PV parameters

V_MPP = 44.55                  # V, voltage at maximum power point at STD (1000 W/m2 irradiance and 25ºC)
I_MPP = 13.92                  # A, current at maximum power point at STD 
PV_peak_power = V_MPP * I_MPP  # W, power at maximum power point at STD  = peak power
PV_efficiency = 23.0           # %, PV panel efficiency at STD 
Tcoef_power = 0.0028           # ºC^-1
Tref = 25                      # °C, STC reference temperature
w = 1                          # mounting type: '1' for PV park, '1.2' for flat roof, '1.8' for sloped roof, '2.4' for façade integrated 
PV_system_losses = 14          # %
ar = 0.169                     # angular loss coefficient
PV_area = PV_peak_power / (1000 * PV_efficiency/100)    # m2, active area of PV panel

# EC parameters
   
EC_production_ph = 1000                                         # Nm3/hour, full power
EC_consumption = 4300                                           # Wh/Nm3 H2  (energy need for one hour to produce 1 Nm3/h)
EC_system_losses = 10                                           # %
EC_water_Nm3 = 0.82                                             # L per Nm3 H2
H2_heating_value = 119.96                                       # MJ/kg, Energy density of hydrogen gas , LHV 

EC_consumption_ph = EC_consumption * EC_production_ph           # Wh/h  (energy need for one hour to produce 1000 Nm3 H2)
EC_water = EC_water_Nm3 * EC_production_ph / EC_consumption_ph  # L water / Wh 
    


def import_data_from_excel_1dataset(file_path, city):  # Import data from excel file
    
    '''
    Import city's data in Local Time from an organized excel file
    Variables and units:
        longitude = degrees
        latitude = degrees
        timezone = hours
        height (height above sea level) = kilometers 
        tilt (tilt of PV panel) = degrees 
        temperature (on air) = degrees Celsius 
        wind speed = m/s
        GHI (Global Horizontal Irradiance, on earth's surface) = W/m2 
        DNI (Direct Normal Irradiance, on a plane always normal to sun rays) = W/m2
        DHI (Diffuse Horizontal Irradiance, which has been scattered or diffused by the atmosphere) = W/m2
    '''
    
    data = pd.read_excel(f'{file_path}\\GHI {city}\\dataGHI_{city}.xlsx', sheet_name='data_LT')   # Construct the full file path and read the data
    meta = {key: data.iat[i, 1] for key, i in zip(['longitude', 'latitude', 'timezone', 'height', 'tilt'], range(1, 6))}
    tmy_data = {col: data[col].tolist() for col in ['Month', 'hours', 'days', 'GHI_DS1', 'DNI_DS1', 'DHI_DS1', 'temperature', 'wind']}
    return {**meta, **tmy_data}

data = import_data_from_excel_1dataset(file_path, city)
(longitude, latitude, timezone, height, tilt, L_hours, L_days, L_months, L1_GHI, L1_DNI, L1_DHI, L_Tair, L_wind) = (
    data['longitude'], data['latitude'], data['timezone'], data['height'], data['tilt'], data['hours'], data['days'], data['Month'],
    data['GHI_DS1'], data['DNI_DS1'], data['DHI_DS1'], data['temperature'], data['wind'])



# Data for calculations

def days_since_start_of_year(year, month, day):
    date_object = datetime(year, month, day)
    start_of_year = datetime(year, 1, 1)
    day = (date_object - start_of_year).days + 1
    return day

d_start, d_end = days_since_start_of_year(year, month[0], day[0]), days_since_start_of_year(year, month[1], day[1])
n_days = int(d_end - d_start)
days = np.linspace(d_start, d_end, n_days+1)
months = np.linspace(month[0],month[1],month[1]-month[0]+1)
L_days_hours = [L_days[i] + L_hours[i]/24 for i in range(len(L_days))]     # for ploting GTI and power throughout the set period
L_days_months = [int((value * (365/12))-(365/12/2)) for value in months] 

# Select values of lists according to the set period

i_start, i_end = L_days_hours.index(d_start), L_days_hours.index(d_end)+24
L_range = [L_hours, L_days, L_months,  L1_GHI, L1_DNI, L1_DHI, L_Tair, L_wind, L_days_hours]
L_filtered = [L[i_start:i_end + 1] for L in L_range]
L_hours, L_days, L_months, L1_GHI, L1_DNI, L1_DHI, L_Tair, L_wind, L_days_hours = L_filtered
   


def solar_parameters_1dataset(L_days_hours, tilt, latitude, longitude, timezone, height, ar, L1_DNI, L1_DHI, L_Tair, L_wind, PV_efficiency, PV_system_losses, Tcoef_power, Tref):
    
    '''
    For 1 dataset and 1 location, calculates:
        sunrise and sunset times
        GHI and GTI with considering refletcion losses
        Temperature on PV panel
        PV energy yield with and without considering temperature
    
    Variables and units:
        tilt (tilt of PV panel) = degrees 
        longitude and latitude = degrees
        timezone = hours
        height (height above sea level) = kilometers
        ar (angular loss coefficient) = -
        GHI (Global Horizontal Irradiance, on earth's surface) = W/m2 
        DNI (Direct Normal Irradiance, on a plane always normal to sun rays) = W/m2
        DHI (Diffuse Horizontal Irradiance, which has been scattered or diffused by the atmosphere) = W/m2
        Temperature on air and PV panel = degrees Celsius
        PV energy yield = W/m2
        wind speed = m/s
        panel efficiency = %
        system loss = %
        Tcoef_power (efficiency correction coefficient for temperature= = degrees Celsius^(-1) 
        Tref (reference temeprature), ususaly 25 degrees Celsius
    '''
    
    Lt_GHI, Lt_GTI, Lt_norefl_GTI, L1_GTI, L1_norefl_GTI = [], [], [], [], []
    Lt_Tpv, L1_Tpv, Lt_gen, L1_gen, LtT_gen, L1T_gen = [], [], [], [], [], []
    
    for i in range(0, len(L_days_hours)):

        # Angles (radians)
        tilt_r, latitude_r = math.radians(tilt), math.radians(latitude)
        azimuth_panel_r = math.pi if latitude >= 0 else 0  # measured from the from North to East, solar panel faces south/north in the north/south hemisphere
        declination_r = math.radians(23.45 * math.sin(math.radians(360 * (L_days[i] - 81) / 365)))
        B = 360 * (L_days[i] - 81) / 365
        EoT = 9.87 * math.sin(math.radians(2 * B)) - 7.53 * math.cos(math.radians(B)) - 1.5 * math.sin(math.radians(B))
        TC = 4 * (longitude - 15 * timezone) + EoT
        LST = L_hours[i] + TC / 60
        HRA_r = math.radians(15 * (LST - 12))
        
        if latitude > 0:
            incidence_r = math.acos(math.sin(latitude_r - tilt_r) * math.sin(declination_r) + math.cos(latitude_r - tilt_r) * math.cos(declination_r) * math.cos(HRA_r))
        else:
            incidence_r = math.acos(math.sin(latitude_r + tilt_r) * math.sin(declination_r) + math.cos(latitude_r + tilt_r) * math.cos(declination_r) * math.cos(HRA_r))
        
        zenith_r = math.acos(math.sin(latitude_r) * math.sin(declination_r) + math.cos(latitude_r) * math.cos(declination_r) * math.cos(HRA_r))
        elevation_r = math.radians(90) - zenith_r
        n_azimuth = math.sin(declination_r) * math.cos(latitude_r) - math.cos(declination_r) * math.sin(latitude_r) * math.cos(HRA_r)
        d_azimuth = math.cos(elevation_r)
        azimuth_r = math.acos(n_azimuth / d_azimuth) if LST < 12 else math.radians(360) - math.acos(n_azimuth / d_azimuth)
        
        # Theoretical Global Horizontal Irradiance (GHI) and Global Tilted Irradiance (GTI) in W/m2
        AL_i = 1 - ((1 - math.exp(-math.cos(incidence_r) / ar)) / (1 - math.exp(-1 / ar)))  # Shallow-angle reflection losses
        AM = 1 / (math.cos(zenith_r) + 0.50572 * (96.07995 - math.degrees(zenith_r)) ** -1.6364) if 0 <= zenith_r < math.radians(90) else 1000  # air mass, Kasten and Young Formula
        Lt_DNI_i = 1353 * ((1 - 0.14 * height) * (0.7 ** (AM ** 0.678)) + 0.14 * height)
        tilt_factor = (math.cos(elevation_r) * math.sin(tilt_r) * math.cos(azimuth_panel_r - azimuth_r) + math.sin(elevation_r) * math.cos(tilt_r))
        
        Lt_GHI_i = 1.1 * Lt_DNI_i * math.sin(elevation_r) if 0 <= zenith_r < math.radians(90) else 0
        Lt_GTI_i = 1.1 * Lt_DNI_i * tilt_factor * (1 - AL_i) if 0 <= incidence_r < math.radians(90) else 0
        Lt_norefl_GTI_i = 1.1 * Lt_DNI_i * tilt_factor if 0 <= incidence_r < math.radians(90) else 0  # without considering reflection
        
        Lt_GHI.append(Lt_GHI_i if Lt_GHI_i > 0 else 0)
        Lt_GTI.append(Lt_GTI_i if Lt_GTI_i > 0 else 0)
        Lt_norefl_GTI.append(Lt_norefl_GTI_i if Lt_norefl_GTI_i > 0 else 0)
        
        # GTI for DS1 and DS2 TMY datasets in W/m2
        new_azimuth_panel_r = (azimuth_panel_r - math.pi) % (2 * math.pi)  # orientation measured from South to West
        DNI_correction_1 = math.sin(declination_r) * math.sin(latitude_r) * math.cos(tilt_r) - math.sin(declination_r) * math.cos(latitude_r) * math.sin(tilt_r) * math.cos(new_azimuth_panel_r)
        DNI_correction_2 = math.cos(declination_r) * math.cos(latitude_r) * math.cos(tilt_r) * math.cos(HRA_r) + math.cos(declination_r) * math.sin(new_azimuth_panel_r) * math.sin(HRA_r) * math.sin(tilt_r)
        DNI_correction_3 = math.cos(declination_r) * math.sin(latitude_r) * math.sin(tilt_r) * math.cos(new_azimuth_panel_r) * math.cos(HRA_r)
        DNI_correction = DNI_correction_1 + DNI_correction_2 + DNI_correction_3  # DNI_tilted = DNI from the TMY * DNI_correction
        DHI_correction = (math.radians(180) - tilt_r) / math.radians(180)  # DHI_tilted = DHI from the TMY * DHI_correction
        L1_GTI_i = max(0, L1_DNI[i] * DNI_correction + L1_DHI[i] * DHI_correction * (1 - AL_i))  # GTI = DNI_tilted + DHI_tilted
        L1_norefl_GTI_i = max(0, L1_DNI[i] * DNI_correction + L1_DHI[i] * DHI_correction)
        L1_GTI.append(L1_GTI_i)
        L1_norefl_GTI.append(L1_norefl_GTI_i)
        
        # PV Temperature
        termT = w * 0.32/(8.91+2.0*L_wind[i]/0.67)
        Lt_Tpv_i = L_Tair[i] + termT*Lt_GTI_i
        L1_Tpv_i = L_Tair[i] + termT*L1_GTI_i
        Lt_Tpv.append(Lt_Tpv_i)
        L1_Tpv.append(L1_Tpv_i) 
        
        
        # Generated power (W/m2) with and without considering temperature losses
        termG = PV_efficiency * (100 - PV_system_losses) / 10000
        Lt_gen_i = termG * Lt_GTI_i
        L1_gen_i = termG * L1_GTI_i
        LtT_gen_i = Lt_gen_i * (1 - Tcoef_power * (Lt_Tpv_i - Tref))
        L1T_gen_i = L1_gen_i * (1 - Tcoef_power * (L1_Tpv_i - Tref))
        
        Lt_gen.append(Lt_gen_i)
        L1_gen.append(L1_gen_i)
        LtT_gen.append(LtT_gen_i if Lt_Tpv_i > Tref else Lt_gen_i)
        L1T_gen.append(L1T_gen_i if L1_Tpv_i > Tref else L1_gen_i)
    
    return Lt_GHI, Lt_GTI, Lt_norefl_GTI, Lt_gen, LtT_gen, L1_GTI, L1_norefl_GTI, L1_gen, L1T_gen, Lt_Tpv, L1_Tpv

results = solar_parameters_1dataset(L_days_hours, tilt, latitude, longitude, timezone, height, ar, L1_DNI, L1_DHI, L_Tair, L_wind, PV_efficiency, PV_system_losses, Tcoef_power, Tref)
Lt_GHI, Lt_GTI, Lt_norefl_GTI, Lt_gen, LtT_gen, L1_GTI, L1_norefl_GTI, L1_gen, L1T_gen, Lt_Tpv, L1_Tpv = results      



#  Select relative or absolute yield:

if units == 'relative':
    #  relative yield lists --> represents the amount of energy produced per unit of installed PV capacity over a relative period
    PV_peak_power_density = PV_peak_power / PV_area         # Wpeak/m2, this is equal to PV_efficiency * 1000 W/m2
    Lt_gen, LtT_gen, L1_gen, L1T_gen = [[val_PV / PV_peak_power_density for val_PV in lst_PV] for lst_PV in [Lt_gen, LtT_gen, L1_gen, L1T_gen]]  # W/m2 / Wpeak/m2 = W/Wpeak (always <1)

elif units == 'absolute':
    # absolute yield lists --> represents the amount of energy produced considering the defined PV capacity over a relative period
    PV_area_park = PV_capacity * PV_area / (PV_peak_power)   # m2, active area of PV system
    Lt_gen, LtT_gen, L1_gen, L1T_gen = [[val_PV * PV_area_park for val_PV in lst_PV] for lst_PV in [Lt_gen, LtT_gen, L1_gen, L1T_gen]]  # W (values below are W instead of W/Wpeak)
else:
    print('Units should be "relative" or "absolute"')
    sys.exit()



Lists_PV = Lt_GHI, Lt_GTI, Lt_gen, LtT_gen, L1_GHI, L1_GTI, L1_gen, L1T_gen  # irradiance in W/m2, PV yield in W/Wpeak or W

def lists_per_day(days, L_days, *args):
    '''
    Total daily values per throughout the set period (Wh/Wpeak or Wh per day --> Wh per day = suming the 24 values of W during a day
    '''
    L_lists = list(zip(*args))
    L_lists_pd = [[] for _ in range(len(args))]
    for d in days:
        V_lists_pd = [0] * len(args)
        for i_row in range(len(L_days)):
            if L_days[i_row] == d:
                for i_col in range(len(V_lists_pd)):
                    V_lists_pd[i_col] += float(L_lists[i_row][i_col])
        for i_col in range(len(V_lists_pd)):
            L_lists_pd[i_col].append(V_lists_pd[i_col])
    return L_lists_pd

Lt_GHI_pd, Lt_GTI_pd, Lt_gen_pd, LtT_gen_pd, L1_GHI_pd, L1_GTI_pd, L1_gen_pd, L1T_gen_pd = lists_per_day(days, L_days, *Lists_PV)
           

def lists_per_month(months, L_months, *args):
    '''
    Total monthly values throughout the set period (Wh/Wpeak per month)
    '''
    L_lists = list(zip(*args))
    L_lists_pm = [[] for _ in range(len(args))]
    for m in months:
        V_lists_pm = [0] * len(args)
        for i_row in range(len(L_months)):
            if L_months[i_row] == m:
                for i_col in range(len(V_lists_pm)):
                    V_lists_pm[i_col] += float(L_lists[i_row][i_col])
        for i_col in range(len(V_lists_pm)):
            L_lists_pm[i_col].append(V_lists_pm[i_col])
    return L_lists_pm

Lt_GHI_pm, Lt_GTI_pm, Lt_gen_pm, LtT_gen_pm, L1_GHI_pm, L1_GTI_pm, L1_gen_pm, L1T_gen_pm = lists_per_month(months, L_months, *Lists_PV)


def max_temp_wind(days, L_days, *args):
    '''
    Max temperature and wind per day
    '''
    L_climate = list(zip(*args))
    L_climate_pd = [[] for _ in range(len(args))]
    for d in days:
        V_climate_pd = [-50, -50, -50, -50]
        for i_row in range(len(L_days)):
            if L_days[i_row] == d:
                for i_col in range(0, len(V_climate_pd)):
                    V_climate_pd[i_col] = max(V_climate_pd[i_col], L_climate[i_row][i_col]) # selects maximum value 
        for i_col in range(0, len(V_climate_pd)):
            L_climate_pd[i_col].append(V_climate_pd[i_col])
    return L_climate_pd

L_Tair_pd, Lt_Tpv_pd, L1_Tpv_pd, L_wind_pd = max_temp_wind(days, L_days, L_Tair, Lt_Tpv, L1_Tpv, L_wind)



def calculate_statistics(*args):
    '''
    Calculate maximum, average, and minimum values of each list and round values in W/Wpeak (Wh/Wpeak per set period for total value)
    '''
    To = [round(np.trapz(L, dx=1), 3) for L in args]
    Max = [round(max(L), 3) for L in args]
    Min = [round(min(L), 3) for L in args]
    Avr = [round(statistics.mean(L), 3) for L in args]
    Std = [round(statistics.stdev(L),3) for L in args]
    return To, Max, Min, Avr, Std

#                0       1       2        3       4       5           6          7       8       9       10         11          12      13
L_ranges_ph = [L_Tair, Lt_Tpv, L1_Tpv, L_wind, Lt_GHI, Lt_GTI, Lt_norefl_GTI, Lt_gen, LtT_gen, L1_GHI, L1_GTI, L1_norefl_GTI, L1_gen, L1T_gen]
To, Max, Min, Avr, Std = calculate_statistics(*L_ranges_ph)
#                  0          1          2           3          4           5
L_ranges_pd = [Lt_GHI_pd, Lt_GTI_pd, LtT_gen_pd, L1_GHI_pd, L1_GTI_pd, L1T_gen_pd]
To_pd, Max_pd, Min_pd, Avr_pd, Std_pd = calculate_statistics(*L_ranges_pd)


temperature_losst, temperature_loss1 = round(100-(To[8]*100/To[7]),2), round(100-(To[13]*100/To[12]),2)  # Temperature losses
reflection_losst, reflection_loss1 = round(100-(To[5]*100/To[6]),2), round(100-(To[10]*100/To[11]),2)    # Reflection losses



# HYDROGEN PRODUCTION


def hydrogen_volume_to_mass(volume):        # m3, convert volume to mass (PV = nRT) -> Nm3 is volume at 1 atm and 25ºC
    R, T, P = 8.314, 298.15, 101325         # J/(mol·K), K, Pa
    molar_mass_h2 = 0.002016                # kg/mol
    n = (P * volume) / (R * T)              # Calculate moles (n) and mass for 298 K (25ºC) and 101325 Pa(1 atm)
    return n * molar_mass_h2                # Calculate mass of H2 in kg


def EC_calculations_2(EC_capacity, EC_production_ph, EC_consumption_ph, EC_water, EC_system_losses, H2_heating_value, L1T_gen):
    
    '''
    Calculates EC number of units and create new lists per hour for:
        EC percentage of usage, EC energy supply, H2 volume production and water comsumption 
    Varaibles and units
        EC_Capacity = W or W/Wpeak
        EC_production_ph (volume of H2 produced per hour) = Nm3/h
        EC_consumption_ph (energy consumed by the EC per hour) = Wh/h
        EC_water (water consumed per Wh of energy consumed) = L/Wh
        EC_system_losses (EC ssytem losses in the electronic parts) = %
        H2_heating_value (lower heating value of hydrogen, aka energy denisty) = MJ/kg
        L1T_gen (PV energy yield considering temperature, reflection and system losses per hour) = W/Wpeak or W
    '''

    EC_consumption_losses_ph = EC_consumption_ph * (1+EC_system_losses/100)                      # Wh/h, energy consumption of one EC unit considering system losses
    EC_units = EC_capacity / EC_consumption_ph                                                   # EC units. Can be units/Wpeak (relative) or units (absolute)
    EC_supply_cons = [V1T_gen / (EC_units * EC_consumption_losses_ph) for V1T_gen in L1T_gen]    # ratio between energy supplied and required
    L_EC_usage = np.clip(EC_supply_cons, 0, 1)                                                   # sets EC usage between 0 and 1: it means that above 1 there is excess of soalr energy (that will be exported)
    L_EC_supply_ph = EC_units * EC_consumption_ph * L_EC_usage                                   # Wh/Wpeak/h or Wh/h, Energy required by EC units
    L_H2_volume_ph = EC_units * EC_production_ph * L_EC_usage                                    # Nm3/Wpeak/h or Nm3/h H2 production
    L_EC_water_ph = L_EC_supply_ph * EC_water *1000                                              # mL/Wpeak/h or mL/h, water comsumption
    
    L_H2_mass_ph = [hydrogen_volume_to_mass(V_H2_volume_ph)*1000 for V_H2_volume_ph in L_H2_volume_ph]      # g/Wpeak/h or g/h
    L_H2_calorific_power_ph = [H2_mass_ph/1000 * H2_heating_value / 0.0036 for H2_mass_ph in L_H2_mass_ph]  # W/Wpeak/h or W/h, energy of hydrogen, (1 Wh = 0.0036 MJ) 
    L_EC_usage, L_EC_supply_ph, L_H2_volume_ph, L_EC_water_ph = L_EC_usage.tolist(), L_EC_supply_ph.tolist(), L_H2_volume_ph.tolist(), L_EC_water_ph.tolist()

    return EC_units, L_EC_usage, L_EC_supply_ph, L_H2_volume_ph, L_H2_mass_ph, L_H2_calorific_power_ph, L_EC_water_ph                       

EC_units, L_EC_usage, L_EC_supply_ph, L_H2_volume_ph, L_H2_mass_ph, L_H2_calorific_power_ph, L_EC_water_ph = EC_calculations_2(EC_capacity, EC_production_ph, EC_consumption_ph, EC_water, EC_system_losses, H2_heating_value, L1T_gen)



def calculate_energy_usage_2(supplied_energy, required_energy):
    '''
    Determine flow of energy between PV, EC and grid, per hour:
        •	When PV power generation exceeds the EC energy requirements, excess energy is exported to the grid.
    '''
    hours = len(supplied_energy)
    grid_exports = np.zeros(hours)
    
    for i in range(hours):
        energy_difference = supplied_energy[i] - required_energy[i]       
        grid_exports[i] = energy_difference  if energy_difference > 0 else 0
    return grid_exports.tolist()

L_grid_exports_ph = calculate_energy_usage_2(L1T_gen, L_EC_supply_ph)  # Calculate energy stored in battery, energy imported from the grid, and energy exported to the grid


# Create H2 lists

#                    0               1                2              3                  4                   5   
Lists_H2_ph = [L_grid_exports_ph, L_EC_supply_ph, L_H2_volume_ph, L_H2_mass_ph, L_H2_calorific_power_ph, L_EC_water_ph]
To_H2, Max_H2, Min_H2, Avr_H2, Std_H2 = calculate_statistics(*Lists_H2_ph)                                                                                 # Total (Wh/Wpeak or Wh per set period), max, average and standard deviation (W/Wpeak)
L_grid_exports_pd, L_EC_supply_pd, L_H2_volume_pd, L_H2_mass_pd, L_H2_calorific_power_pd, L_EC_water_pd = lists_per_day(days, L_days, *Lists_H2_ph)        # Total daily values throughout the set period (Wh/Wpeak or Wh per day)
L_grid_exports_pm, L_EC_supply_pm, L_H2_volume_pm, L_H2_mass_pm, L_H2_calorific_power_pm, L_EC_water_pm = lists_per_month(months, L_months, *Lists_H2_ph)  # Total monthly values throughout the set period (Wh/Wpeak or Wh per month)         
#                      0                1                2              3                  4                   5   
Lists_H2_pd = [L_grid_exports_pd, L_EC_supply_pd, L_H2_volume_pd, L_H2_mass_pd, L_H2_calorific_power_pd, L_EC_water_pd]
To_H2_pd, Max_H2_pd, Min_H2_pd, Avr_H2_pd, Std_H2_pd = calculate_statistics(*Lists_H2_pd)

EC_efficiency = round(To_H2[4]*100/To_H2[1],2)  # %

# Round and convert values 

L_irr = [L1_GHI_pm, L1_GTI_pm, Lt_GHI_pd, Lt_GTI_pd ,L1_GHI_pd, L1_GTI_pd]    # W/m2 to kW/m2
L1_GHI_pm, L1_GTI_pm, Lt_GHI_pd, Lt_GTI_pd ,L1_GHI_pd, L1_GTI_pd= [[round(value/1000,2) for value in L] for L in L_irr] 

L_pm = [L1T_gen_pm, L_grid_exports_pm, L_EC_supply_pm, L_H2_calorific_power_pm, L_H2_volume_pm, L_H2_mass_pm]  # Wh or g / Wpeak
L1T_gen_pm, L_grid_exports_pm, L_EC_supply_pm, L_H2_calorific_power_pm, L_H2_volume_pm, L_H2_mass_pm = [[round(value,2) for value in L] for L in L_pm]  #Wh/Wpeak or Wh, for plot 4 and print 

# Net flow

net_flow_pm = [round(PV - EC - exp,2) for PV, EC, exp in zip (L1T_gen_pm, L_EC_supply_pm, L_grid_exports_pm)]    



# Plots 


lines_width = 1.2

# Temperature and wind speed

plt.figure(figsize=(12,2.3))
plt.rcParams['figure.dpi']=200
plt.title(f'{city}',fontsize=11)
plt.ylabel('Max temperature (°C)',fontsize=12)
plt.plot(days, Lt_Tpv_pd, color='black', linewidth=lines_width,label='PV panel (T)')
plt.plot(days, L1_Tpv_pd, color='teal', linewidth=lines_width,label='PV panel (DS1)')
plt.plot(days, L_Tair_pd, color='grey', linewidth=lines_width,label='On air')
plt.xlim(0,365)
x_ticks = np.arange(0, 365, 365/12)
x_labels_pos = x_ticks + (365/12)/2
plt.xticks(x_ticks, labels=['' for _ in x_ticks])
plt.gca().set_xticks(x_labels_pos, minor=True)
plt.gca().set_xticklabels(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'], minor=True)
plt.xlabel('month')
plt.ylim(-25,75), plt.yticks([-25,0,25,50,75,])
plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1)) 
plt.grid(True)
plt.tight_layout()
#plt.savefig(f'C:\\Users\\Cristina\\OneDrive - FCT NOVA\\Artigo - GW-scale Solar-to-H2\\Figures\\2nd version\\{city}_plot1.svg')
plt.show()


# GHI and GTI 

plt.figure(figsize=(12,2.3))
plt.rcParams['figure.dpi']=200
plt.title(f'{city}',fontsize=11)
plt.ylabel('Irradiation\n(kWh/m$^2$/day)')
plt.plot(days, Lt_GHI_pd,color='black', linewidth=lines_width,label='GHI (T)')
plt.plot(days, Lt_GTI_pd, linestyle = '--',color='black', linewidth=lines_width,label='GTI (T)')
plt.plot(days, L1_GHI_pd, color='teal', linewidth=lines_width,label='GHI (DS1)')
plt.plot(days, L1_GTI_pd, linestyle = '--',color='teal', linewidth=lines_width, label='GTI (DS1)')
plt.ylim(0,10), plt.yticks([0,2,4,6,8,10])
plt.xlim(0,365)
x_ticks = np.arange(0, 365, 365/12)
x_labels_pos = x_ticks + (365/12)/2
plt.xticks(x_ticks, labels=['' for _ in x_ticks])
plt.gca().set_xticks(x_labels_pos, minor=True)
plt.gca().set_xticklabels(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'], minor=True)
plt.xlabel('month')
plt.grid(True)
plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1)) 
plt.tight_layout()
#plt.savefig(f'C:\\Users\\Cristina\\OneDrive - FCT NOVA\\Artigo - GW-scale Solar-to-H2\\Figures\\2nd version\\{city}_plot2.svg')
plt.show()



# PV energy yield

plt.figure(figsize=(12,2))
plt.rcParams['figure.dpi']=200
plt.title(f'{city}',fontsize=11)
ylabel = 'PV energy yield\n(Wh/W$_{{peak}}$/day)' if units == 'relative' else 'PV energy yield\n(Wh/day)'
plt.ylabel(ylabel)
plt.plot(days, LtT_gen_pd, label = 'Theoretical',color='brown', linewidth=lines_width)
plt.plot(days, L1T_gen_pd, label = 'DS1',color='goldenrod', linewidth=lines_width)
plt.plot(days, L1_gen_pd, label = 'DS1 without\nconsidering temperature',color='goldenrod', linewidth=lines_width, linestyle ='--')
plt.fill_between(days, L1T_gen_pd, color='goldenrod', alpha=0.25)
Vmax = 8 if units == 'relative' else 1.2*max(LtT_gen_pd)
plt.ylim(0,Vmax), plt.yticks([0,0.25*Vmax, 0.5*Vmax, 0.75*Vmax, Vmax])
plt.xlim(0,365)
x_ticks = np.arange(0, 365, 365/12)
x_labels_pos = x_ticks + (365/12)/2
plt.xticks(x_ticks, labels=['' for _ in x_ticks])
plt.gca().set_xticks(x_labels_pos, minor=True)
plt.gca().set_xticklabels(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'], minor=True)
plt.xlabel('month')
plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1)) 
plt.grid(True)
plt.tight_layout()
#plt.savefig(f'C:\\Users\\Cristina\\OneDrive - FCT NOVA\\Artigo - GW-scale Solar-to-H2\\Figures\\2nd version\\{city}_plot3.svg')
plt.show()


# Energy flow between PV, EC, battery and grid

plt.figure(figsize=(12,2.5))
plt.rcParams['figure.dpi']=200
plt.title(f'{city}, discontinuous EC operation, day-night cycles',fontsize=11)
ylabel = 'Energy Flow (Wh/W$_{{peak}}$/day)' if units == 'relative' else 'Energy Flow (Wh/day)'
plt.ylabel(ylabel)
plt.plot(days, LtT_gen_pd, label = 'PV yield (T)',color='black', linewidth=lines_width)
plt.plot(days, L1T_gen_pd, label = 'PV yield (DS1)',color='goldenrod', linewidth=lines_width)
plt.plot(days, L_grid_exports_pd,  label = 'Grid export', color='firebrick', linewidth=lines_width)
plt.plot(days, L_EC_supply_pd, label = 'EC supply', color = 'teal', linewidth=lines_width, linestyle = '--')
plt.plot(days, L_H2_calorific_power_pd, label = ' H$_2$ energy', color='teal', linewidth=lines_width)
plt.fill_between(days, L_H2_calorific_power_pd, color='black', alpha=0.08)
Vmax = 8 if units == 'relative' else 1.2*max(LtT_gen_pd)
plt.ylim(0,Vmax), plt.yticks([0, 0.25*Vmax, 0.5*Vmax, 0.75*Vmax, Vmax])
plt.xlim(0,365)
x_ticks = np.arange(0, 365, 365/12)
x_labels_pos = x_ticks + (365/12)/2
plt.xticks(x_ticks, labels=['' for _ in x_ticks])
plt.gca().set_xticks(x_labels_pos, minor=True)
plt.gca().set_xticklabels(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'], minor=True)
plt.xlabel('month')
plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1)) 
plt.grid(True)
plt.tight_layout()

#plt.savefig(f'C:\\Users\\Cristina\\OneDrive - FCT NOVA\\Artigo - GW-scale Solar-to-H2\\Figures\\2nd version\\{city}_plot4.svg')
plt.show()


# Bar chart total

if city == 'Sines':               # Results from Module 4.3, Wh/Wpeak
    grid_exports_To_2 = 937.545
    grid_imports_To_2 = 779.524
    H2_calorific_power_To_2 = 907.273
if city == 'Edmonton':
    grid_exports_To_2 = 691.947
    grid_imports_To_2 = 578.588 
    H2_calorific_power_To_2 = 650.852
if city == 'Crystal Brook':
    grid_exports_To_2 = 837.01
    grid_imports_To_2 = 691.643
    H2_calorific_power_To_2 = 834.626

bar_width = 0.9
alpha = 0.4
plt.figure(figsize=(6,3))
plt.rcParams['figure.dpi']=300

plt.ylabel('Total energy (Wh/W$_{{peak}}$/year)')
bars = [
    (1, To[13], 'goldenrod', 'PV energy yield',1),
    (4.1, grid_imports_To_2, 'darkgreen', 'Grid imports (s1)',1),
    (2.1, grid_exports_To_2, 'firebrick', 'Grid exports (s2)',1),
    (6.1, H2_calorific_power_To_2, 'teal', 'H2 calorific value (s1)',1),
    (4.9, 0, 'darkgreen', 'Grid imports (s2)',alpha),
    (2.9, To_H2[0], 'firebrick', 'Grid exports (s1)',alpha),
    (6.9, To_H2[4], 'teal', 'H2 calorific value (s2)',alpha)]
     
for bar in bars:
    plt.bar(bar[0], bar[1], label=bar[3], color=bar[2], alpha=bar[4],width=bar_width)
    plt.text(bar[0], bar[1] + 5, f'{round(bar[1])}', ha='center', va='bottom', fontsize=9)

plt.gca().axes.get_xaxis().set_visible(False)
plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
plt.ylim(0,130+To[13])
plt.text(4,To[13],f'{city}',fontsize=9)
plt.tight_layout()

#plt.savefig(f'C:\\Users\\Cristina\\OneDrive - FCT NOVA\\Artigo - GW-scale Solar-to-H2\\Figures\\2nd version\\{city}_plot5.svg')
plt.show()



# Monthly bar charts

bar_width = 7
plt.figure(figsize=(11.5,2.3))
plt.suptitle(f'{city}, discontinuous operation (day-night cycles)',fontsize=12)
plt.rcParams['figure.dpi']=300

L_days_months_0 = [L-7 for L in L_days_months]
L_days_months_1 = [L for L in L_days_months]
L_days_months_2 = [L+7 for L in L_days_months]

ylabel = 'Total monthly energy\n(Wh/W$_{{peak}}$/month)' if units == 'relative' else 'Total monthly energy\n(Wh/month)'
plt.ylabel(ylabel)
plt.bar(L_days_months_0, L1T_gen_pm, label = 'PV energy yield', color='goldenrod', width=bar_width)
plt.bar(L_days_months_1, L_grid_exports_pm, label = 'Grid exports', color='firebrick', width=bar_width)
plt.bar(L_days_months_2, L_H2_calorific_power_pm, label = 'H2 calorific value', color='teal', width=bar_width)
plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1)) 
plt.xlim(0,365)
x_ticks = np.arange(0, 365, 365/12)
x_labels_pos = x_ticks + (365/12)/2
plt.xticks(x_ticks, labels=['' for _ in x_ticks])
plt.gca().set_xticks(x_labels_pos, minor=True)
plt.gca().set_xticklabels(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'], minor=True)
plt.xlabel('month')
plt.grid(True)
plt.tight_layout()

#plt.savefig(f'C:\\Users\\Cristina\\OneDrive - FCT NOVA\\Artigo - GW-scale Solar-to-H2\\Figures\\2nd version\\{city}_plot6.svg')
plt.show()



# Print

print('')
print('From day',d_start,'to',d_end,'of',year,f'in {city} (',n_days+1,'days), β =',tilt,'°')
print('')
print('Panel efficiency =',round(PV_efficiency,2),'%')
print('EC efficiency =',EC_efficiency,'%')
print('EC capacity =',round(EC_capacity,2),'Wh/Wpeak')
print('average EC usage =',round(statistics.mean(L_EC_usage)*100),'%')
print('')
print('Change in output power due to temperature | reflection:')
print('The:',temperature_losst,'% | ',reflection_losst,'%')
print('DS1:',temperature_loss1,'% | ',reflection_loss1,'%')
print('')
print('TEMPERATURE AND WIND SPEED')
print('')
print('minimum | maximum (ºC or m/s)')
print('Air temperature = ',Min[0],'|',Max[0])
print('Panel temp. The = ',Min[1],'|',Max[1])
print('Panel temp. DS1 = ',Min[2],'|',Max[2])
print('Wind speed      = ',Min[3],'|',Max[3])
print('')
print('')
print('IRRADIATION')
print('')
print('-      Total      | average per day | std deviation per day  | maximum ')
print('kWh/m2/set period |    W/m2/day     |     W/Wpeak/day        |  W/m2')
print('')
print('Irradiation on horizontal plane')
print('The:',round(To[4]/1000,3),'|',Avr_pd[0],'|',Std_pd[0],'|',Max[4])
print('DS1:',round(To[9]/1000,3),'|',Avr_pd[3],'|',Std_pd[3],'|',Max[9])
print('')
print('Irradiation on PV panel, β =',tilt,'°')
print('The:',round(To[5]/1000,3),'|',Avr_pd[1],'|',Std_pd[1],'|',Max[5])
print('DS1:',round(To[10]/1000,3),'|',Avr_pd[4],'|',Std_pd[4],'|',Max[10])
print('')
print('ENERGY EXCHANGES and EC OPERATION')
print('')
print('-      Total        | average per day | std deviation per day | maximum')
print('Wh/Wpeak/set period |   W/Wpeak/day   |      W/Wpeak/day      | W/Wpeak')
print('')
print('Power generation considering T, β =',tilt,'°')
print('The:',round(To[8],3),'|',Avr_pd[2],'|',Std_pd[2],'|',Max[8])
print('DS1:',round(To[13],3),'|',Avr_pd[5],'|',Std_pd[5],'|',Max[13])
print('')
print('Grid exports')
print('DS1:',round(To_H2[0],3),'|',Avr_H2_pd[0],'|',Std_H2_pd[0],'|',Max_H2[0])
print('')
print('EC supply')
print('DS1:',round(To_H2[1],3),'|',Avr_H2_pd[1],'|',Std_H2_pd[1],'|',Max_H2[1])
print('')
print('H2 production at 1 atm, 0ºC and 0% relative humidity (L instead of Wh)')
print('DS1:',round(To_H2[2],3),'|',Avr_H2_pd[2],'|',Std_H2_pd[2],'|',Max_H2[2])
print('')
print('H2 mass (g instead of Wh)')
print('DS1:',round(To_H2[3],3),'|',Avr_H2_pd[3],'|',Std_H2_pd[3],'|',Max_H2[3])
print('')
print('H2 calorific value')
print('DS1:',round(To_H2[4],3),'|',Avr_H2_pd[4],'|',Std_H2_pd[4],'|',Max_H2[4])
print('')
print('Water consumption (mL instead of)')
print('DS1:',round(To_H2[5],3),'|',Avr_H2_pd[5],'|',Std_H2_pd[5],'|',Max_H2[5])
print('')
print('')
print('LISTS PER MONTH DS1')
print('')
print('- Irradiation on horizontal plane (kWh/m2/month)')
print(L1_GHI_pm)
print('')
print('- Irradiation on PV panel (Wh/Wpeak/month)')
print(L1_GTI_pm)
print('')
print('- Power generated considering Temperature (Wh/Wpeak/month)')
print(L1T_gen_pm)
print('')
print('- Energy exported to grid (Wh/Wpeak/month)')
print(L_grid_exports_pm)
print('')
print('- Energy supplied to the EC (Wh/Wpeak/month)')
print(L_EC_supply_pm)
print('')
print('- H2 calorific value produced (Wh/Wpeak/month)')
print(L_H2_calorific_power_pm)
print('')
print('H2 volume production (L/Wpeak/month, at 1 atm, 0ºC and 0% relative humidity)')
print(L_H2_volume_pm)
print('')
print('- H2 mass production (g/Wpeak/month)')
print(L_H2_mass_pm)
print('')
print('- Net flow: PV = EC + exp (should be 0)')
print(net_flow_pm)
