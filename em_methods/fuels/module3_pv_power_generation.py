'''
PURPOSE

Calculates irradiance on surface (GHI), and on tilted surface (GTI, considering reflection losses) and generated PV power (with and without considering temperature)
throughout a set time period (e.g. year) in a city (e.g. Sines) – use theoretical values and compare with 2 databases:
    DS1 : NREL 1998-2022
    DS2: EU PVGIS 2005-2020
    (In case of Edmonton, DS2 is NREL 2005-2015, because EU PVGIS data doesn't cover American continent)

UNITS: kW/m2

Plot data and obtain max, average, min, and total values. 

DESCRIPTION

1.	Select city (Sines, Edmonton or Crystal Brook), time period (year as default) and import data: 
    i.	DNI: Direct Normal Irradiance on a plane always normal to sun rays (W/m2)
    ii. DHI: Diffuse Horizontal Irradiance - on Earth's surface (W/m2)
    ii. GHI: Global Horizontal Irradiance - on Earth's surface (W/m2)
    ii.	Air temperature and wind speed at 10m height from DS1
    iii.Latitude, Longitude, height, timezone, ideal solar panel tilt, 

2.	Set PV data (efficiency, temperature coefficient, mounting type), system losses (14% as default) and spectral variation losses (see maps)

3.	Use the solar angles/times equations and imported data to create new lists:
    i.   theoretical, DS1 and DSB2 GHI and GTI (Global Tilt Irradiance, considering shallow angle reflection losses)
    ii.  temperature on the panel;
    iii. generated power of DS1 and DS2 with and without considering temperature losses.

4.	Convert data to kW/m2 and select values of lists according to the set period

5.  Calculate total per daily, monthly and yearly/set period (kWh/m2/day or month or year/setperiod) theoretical, DS1 and DS2 GHI, GTI and power generation

6.	Calculate maximum, average and minimun values of each list.

7.	Plot PV and air temperature, wind speed, GTI and power generation vs number of days since the start of the year, and print output values.


PV energy losses: https://joint-research-centre.ec.europa.eu/photovoltaic-geogralatitudecal-information-system-pvgis/getting-started-pvgis/pvgis-data-sources-calculation-methods_en
Temperature depedence: https://doi.org/10.1016/j.solmat.2008.05.016Get rights and content
Solar panel details: c-Si (LONGI LR7-72HGD-620M), size 1.1 x 2.3 m2, https://www.longi.com/en/products/modules/hi-mo-7/

EU database (used mainly for EU, Africa and Asia): https://re.jrc.ec.europa.eu/pvg_tools/en/#TMY
NREL database (used mainly for America and Australia): https://nsrdb.nrel.gov/data-viewer


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
import statistics
import matplotlib.pyplot as plt



# User inputs

city = 'Sines'
year,day,month = 2025, [18,24],[7,7] # define period of time
file_path = 'C:\\Users\\(...)'  # edit path

# PV parameters

V_MPP = 44.55                 # V, voltage at maximum power point at STD (1000 W/m2 irradiance and 25ºC)
I_MPP = 13.92                 # A, current at maximum power point at STD 
peak_power = V_MPP * I_MPP    # W, power at maximum power point at STD  = peak power
PV_efficiency = 23.0          # %, PV panel efficiency at STD 
Tcoef_power = 0.0028          # ºC^-1
Tref = 25                     # °C, STC reference temperature
w = 1                         # mounting type: '1' for PV park, '1.2' for flat roof, '1.8' for sloped roof, '2.4' for façade integrated 
system_losses = 14            # %
ar = 0.169                    # angular loss coefficient


def import_data_from_excel_2datasets(file_path, city):
    
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
    tmy_data = {col: data[col].tolist() for col in ['hours', 'days', 'GHI_DS1', 'DNI_DS1', 'DHI_DS1', 
                                                     'GHI_DS2', 'DNI_DS2', 'DHI_DS2', 'temperature', 'wind']}
    return {**meta, **tmy_data}

data = import_data_from_excel_2datasets(file_path, city)
(longitude, latitude, timezone, height, tilt, L_hours, L_days, L1_GHI, L1_DNI, L1_DHI, L2_GHI, L2_DNI, L2_DHI, L_Tair, L_wind) = (
    data['longitude'], data['latitude'], data['timezone'], data['height'], data['tilt'], data['hours'], data['days'],
    data['GHI_DS1'], data['DNI_DS1'], data['DHI_DS1'],
    data['GHI_DS2'], data['DNI_DS2'], data['DHI_DS2'],
    data['temperature'], data['wind'])



# Data for calculations

def days_since_start_of_year(year, month, day):
    date_object = datetime(year, month, day)
    start_of_year = datetime(year, 1, 1)
    day = (date_object - start_of_year).days + 1
    return day

d_start, d_end = days_since_start_of_year(year, month[0], day[0]), days_since_start_of_year(year, month[1], day[1])
n_days = int(d_end - d_start)
days = np.linspace(d_start, d_end, n_days+1)
L_days_hours = [L_days[i] + L_hours[i]/24 for i in range(len(L_days))]     # for ploting GTI and power throughout the year 


# Select values of lists according to the set period

i_start, i_end = L_days_hours.index(d_start), L_days_hours.index(d_end)+24
L_range = [L_hours, L_days,  L1_GHI, L1_DNI, L1_DHI, L2_GHI, L2_DNI, L2_DHI, L_Tair, L_wind, L_days_hours]
L_filtered = [L[i_start:i_end + 1] for L in L_range]
L_hours, L_days, L1_GHI, L1_DNI, L1_DHI, L2_GHI, L2_DNI, L2_DHI, L_Tair, L_wind, L_days_hours = L_filtered
   


def solar_parameters_2datasets(L_days_hours, tilt, latitude, longitude, timezone, height, ar, L1_DNI, L1_DHI, L2_DNI, L2_DHI, L_Tair, L_wind, PV_efficiency, system_losses, Tcoef_power, Tref):
    
    '''
    Calculate solar times and angles, AM, GHI, GTI (considering refeltcion losses), Temperature on PV panel, PV energy yield with and without considering temperature
    
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
    
    Lt_GHI, Lt_GTI, Lt_norefl_GTI = [], [], []
    L1_GTI, L2_GTI, L1_norefl_GTI, L2_norefl_GTI = [], [], [], []
    Lt_Tpv, L1_Tpv, L2_Tpv = [], [], []
    Lt_gen, L1_gen, L2_gen, LtT_gen, L1T_gen, L2T_gen = [], [], [], [], [], []
    
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
        L2_GTI_i = max(0, L2_DNI[i] * DNI_correction + L2_DHI[i] * DHI_correction * (1 - AL_i))
        L1_norefl_GTI_i = max(0, L1_DNI[i] * DNI_correction + L1_DHI[i] * DHI_correction)  # without considering reflection
        L2_norefl_GTI_i = max(0, L2_DNI[i] * DNI_correction + L2_DHI[i] * DHI_correction)
        
        L1_GTI.append(L1_GTI_i)
        L2_GTI.append(L2_GTI_i)
        L1_norefl_GTI.append(L1_norefl_GTI_i)
        L2_norefl_GTI.append(L2_norefl_GTI_i)
        
        # PV Temperature
        termT = w * 0.32/(8.91+2.0*L_wind[i]/0.67)
        Lt_Tpv_i = L_Tair[i] + termT*Lt_GTI_i
        L1_Tpv_i = L_Tair[i] + termT*L1_GTI_i
        L2_Tpv_i = L_Tair[i] + termT*L2_GTI_i
        Lt_Tpv.append(Lt_Tpv_i)
        L1_Tpv.append(L1_Tpv_i)
        L2_Tpv.append(L2_Tpv_i)  
        
        
        # Generated power (W/m2) with and without considering temperature losses
        termG = PV_efficiency * (100 - system_losses) / 10000
        Lt_gen_i = termG * Lt_GTI_i
        L1_gen_i = termG * L1_GTI_i
        L2_gen_i = termG * L2_GTI_i
        LtT_gen_i = Lt_gen_i * (1 - Tcoef_power * (Lt_Tpv_i - Tref))
        L1T_gen_i = L1_gen_i * (1 - Tcoef_power * (L1_Tpv_i - Tref))
        L2T_gen_i = L2_gen_i * (1 - Tcoef_power * (L2_Tpv_i - Tref))
        
        Lt_gen.append(Lt_gen_i)
        L1_gen.append(L1_gen_i)
        L2_gen.append(L2_gen_i)
        LtT_gen.append(LtT_gen_i if Lt_Tpv_i > Tref else Lt_gen_i)
        L1T_gen.append(L1T_gen_i if L1_Tpv_i > Tref else L1_gen_i)
        L2T_gen.append(L2T_gen_i if L2_Tpv_i > Tref else L2_gen_i)
    
    return Lt_GHI, Lt_GTI, Lt_norefl_GTI, L1_GTI, L2_GTI, L1_norefl_GTI, L2_norefl_GTI, Lt_Tpv, L1_Tpv, L2_Tpv, Lt_gen, L1_gen, L2_gen, LtT_gen, L1T_gen, L2T_gen

results = solar_parameters_2datasets(L_days_hours, tilt, latitude, longitude, timezone, height, ar, L1_DNI, L1_DHI, L2_DNI, L2_DHI, L_Tair, L_wind, PV_efficiency, system_losses, Tcoef_power, Tref)
Lt_GHI, Lt_GTI, Lt_norefl_GTI, L1_GTI, L2_GTI, L1_norefl_GTI, L2_norefl_GTI, Lt_Tpv, L1_Tpv, L2_Tpv, Lt_gen, L1_gen, L2_gen, LtT_gen, L1T_gen, L2T_gen = results      



def convert_to_kw(*args):    # Convert to kW
    return [[value / 1000 for value in L] for L in args]
kwvalues = convert_to_kw(Lt_GHI, Lt_GTI, Lt_gen, LtT_gen, Lt_norefl_GTI,  L1_GHI, L1_GTI, L1_gen, L1T_gen, L1_norefl_GTI,  L2_GHI, L2_GTI, L2_gen, L2T_gen, L2_norefl_GTI)
Lt_GHI, Lt_GTI, Lt_gen, LtT_gen, Lt_norefl_GTI,  L1_GHI, L1_GTI, L1_gen, L1T_gen, L1_norefl_GTI,  L2_GHI, L2_GTI, L2_gen, L2T_gen, L2_norefl_GTI = kwvalues


     
def calculate_totals(*args):    # Total theoretical, DS1 and DS2 GHI, GTI and power generation 
    return [(np.trapz(L[1:], dx=1)) for L in args]
V_list_To = calculate_totals(Lt_GHI, Lt_GTI, Lt_gen, LtT_gen, Lt_norefl_GTI, L1_GHI, L1_GTI, L1_gen, L1T_gen, L1_norefl_GTI, L2_GHI, L2_GTI, L2_gen, L2T_gen, L2_norefl_GTI)
Vt_GHI_To, Vt_GTI_To, Vt_gen_To, VtT_gen_To, Vt_norefl_GTI_To, V1_GHI_To, V1_GTI_To, V1_gen_To, V1T_gen_To, V1_norefl_GTI_To, V2_GHI_To, V2_GTI_To, V2_gen_To, V2T_gen_To, V2_norefl_GTI_To = V_list_To



def reflection_losses(*args):   # Shallow angle reflection losses
    Vt_reflection_losses = round(100-(Vt_GTI_To*100/Vt_norefl_GTI_To),3)
    V1_reflection_losses = round(100-(V1_GTI_To*100/V1_norefl_GTI_To),3)
    V2_reflection_losses = round(100-(V2_GTI_To*100/V2_norefl_GTI_To),3)
    return Vt_reflection_losses, V1_reflection_losses, V2_reflection_losses
Vt_reflection_losses, V1_reflection_losses, V2_reflection_losses = reflection_losses(Vt_GTI_To, V1_GTI_To, V2_GTI_To,Vt_norefl_GTI_To, V1_norefl_GTI_To, V2_norefl_GTI_To)



def temperature_losses(*args):    # High-temperature induced losses
    temperature_losst = round(100 - (VtT_gen_To * 100 / Vt_gen_To), 2)
    temperature_loss1 = round(100 - (V1T_gen_To * 100 / V1_gen_To), 2)
    temperature_loss2 = round(100 - (V2T_gen_To * 100 / V2_gen_To), 2)
    return temperature_losst, temperature_loss1, temperature_loss2
temperature_losst, temperature_loss1, temperature_loss2 = temperature_losses(VtT_gen_To, Vt_gen_To, V1T_gen_To, V1_gen_To, V2T_gen_To, V2_gen_To)



def lists_per_day(days, L_days, *args):   # Total daily values per throughout the set period (kWh/m2/day)
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
Lists_pd = lists_per_day(days, L_days,Lt_GHI, Lt_GTI, Lt_gen, LtT_gen, L1_GHI, L1_GTI, L1_gen, L1T_gen, L2_GHI, L2_GTI, L2_gen, L2T_gen)
Lt_GHI_pd, Lt_GTI_pd, Lt_gen_pd, LtT_gen_pd, L1_GHI_pd, L1_GTI_pd, L1_gen_pd, L1T_gen_pd, L2_GHI_pd, L2_GTI_pd, L2_gen_pd, L2T_gen_pd = Lists_pd
        


def calculate_statistics(L_ranges):   # Calculate maximum, average and minimun values of each list and round values
    Max = [round(max(L), 3) for L in L_ranges]
    Min = [round(min(L), 3) for L in L_ranges]
    Avr = [round(statistics.mean(L),3) for L in L_ranges]
    return Max, Min, Avr
#             0       1       2        3       4        5      6       7       8      9       10       11       12      13
L_ph = [L_Tair, Lt_Tpv, L1_Tpv, L2_Tpv, L_wind, Lt_GHI, Lt_GTI, L1_GHI, L1_GTI, L2_GHI, L2_GTI, LtT_gen, L1T_gen, L2T_gen]
Max, Min, Avr = calculate_statistics(L_ph)
#                  0          1          2          3          4         5           6           7          8
L_pd = [Lt_GHI_pd, Lt_GTI_pd, L1_GHI_pd, L1_GTI_pd, L2_GHI_pd, L2_GTI_pd, LtT_gen_pd, L1T_gen_pd, L2T_gen_pd]
Max_pd, Min_pd, Avr_pd = calculate_statistics(L_pd)


L_total = [Vt_GHI_To, Vt_GTI_To, V1_GHI_To, V1_GTI_To, V2_GHI_To, V2_GTI_To, VtT_gen_To, V1T_gen_To, V2T_gen_To]    # Round total values
Vt_GHI_To, Vt_GTI_To, V1_GHI_To, V1_GTI_To, V2_GHI_To, V2_GTI_To, VtT_gen_To, V1T_gen_To, V2T_gen_To = [round(To,2) for To in L_total]


# Plots

lines_width = 0.9
L_days_hours = [L - d_start + day[0] for L in L_days_hours] 
plt.rcParams['figure.dpi']=200
plt.figure(figsize=(13,4.5))
plt.suptitle(f'{city}',fontsize=16)

plt.subplot(3,1,1)
plt.ylabel('Temperature\n(°C)')
plt.plot(L_days_hours, Lt_Tpv, color='black', linewidth=lines_width,label='T PV panel (T)')
plt.plot(L_days_hours, L1_Tpv, color='goldenrod', linewidth=lines_width,label='DS1 PV panel')
plt.plot(L_days_hours, L2_Tpv, color='saddlebrown', linewidth=lines_width,label='DS2 PV panel')
plt.plot(L_days_hours, L_Tair, color='grey', linewidth=lines_width,label='On air')
plt.legend(fontsize=8, loc='upper right')
plt.tick_params(axis='x',labelcolor='white')
plt.ylim(0,75)
plt.yticks([0,25,50,75])
plt.xlim(day[0],day[1]+1)
plt.xticks(range(day[0],day[1]+2, 1))
plt.grid(True)

ax2 = plt.twinx()
ax2.plot(L_days_hours, L_wind, color='firebrick', linewidth=lines_width,alpha=0.4)
ax2.set_ylabel('Wind speed (m·s$^{-1}$)', color='firebrick')
ax2.set_ylim(0, 10)
plt.yticks([0,3,6,9], color='firebrick')
ax2.tick_params(axis='x', labelcolor='white') 

plt.subplot(3,1,2)
plt.ylabel('Irradiance\n(kW·m$^{-2}$)')
plt.plot(L_days_hours, Lt_GHI, label = 'T GHI',color='black', linewidth=lines_width)
plt.plot(L_days_hours, Lt_GTI, label = 'T GTI',linestyle = '--',color='black', linewidth=lines_width)
plt.plot(L_days_hours, L1_GHI, label = 'DS1 GHI',color='goldenrod', linewidth=lines_width)
plt.plot(L_days_hours, L1_GTI, label = 'DS1 GTI',linestyle = '--',color='goldenrod', linewidth=lines_width)
plt.plot(L_days_hours, L2_GHI, label = 'DS2 GHI',color='saddlebrown', linewidth=lines_width)
plt.plot(L_days_hours, L2_GTI, label = 'DS2 GTI',linestyle = '--',color='saddlebrown', linewidth=lines_width)
plt.legend(fontsize=8, loc='upper right')
plt.tick_params(axis='x', labelcolor='white')
plt.ylim(0,1.1)
plt.yticks([0,0.3,0.6,0.9])
plt.xticks(range(day[0],day[1]+2, 1))
plt.xlim(day[0],day[1]+1)
plt.grid(True)

plt.subplot(3,1,3)
plt.ylabel('PV Power gen.\n(kW·m$^{-2}$)')
plt.plot(L_days_hours, LtT_gen, label = 'T',color='black', linewidth=lines_width)
plt.plot(L_days_hours, Lt_gen, label = 'T*',color='black', linestyle = '--', linewidth=lines_width)
plt.plot(L_days_hours, L1T_gen, label = 'DS1',color='goldenrod', linewidth=lines_width)
plt.plot(L_days_hours, L1_gen, label = 'DS1*',color='goldenrod', linestyle = '--', linewidth=lines_width)
plt.plot(L_days_hours, L2T_gen, label = 'DS2',color='saddlebrown', linewidth=lines_width)
plt.plot(L_days_hours, L2_gen, label = 'DS2*',color='saddlebrown', linestyle = '--', linewidth=lines_width)
plt.text(day[1]-0.8,0.19,'*without considering temperature',fontsize=9)
plt.xlabel('day')
plt.legend(fontsize=8, loc='upper right')
plt.ylim(0,0.22)
plt.yticks([0,0.06,0.12,0.18])
plt.xlim(day[0],day[1]+1)
plt.xticks(range(day[0],day[1]+2, 1))
plt.grid(True)

plt.tight_layout()
plt.subplots_adjust(top=0.9, hspace=0, wspace=0)
plt.show()


# Print

print('')
print('From day',d_start,'to',d_end,'of',year,f'in {city} (',n_days+1,'days), β =',tilt,'°')
print('')
print('Change in output power due to temperature | reflection:')
print('The:',temperature_losst,'% | ',Vt_reflection_losses,'%')
print('DS1:',temperature_loss1,'% | ',V1_reflection_losses,'%')
print('DS2:',temperature_loss2,'% | ',V2_reflection_losses,'%')
print('')
print('TEMPERATURE AND WIND SPEED')
print('')
print('                minimum | maximum')
print('Air temperature = ',Min[0],'|',Max[0],'°C')
print('Panel temp The =  ',Min[1],'|',Max[1],'°C')
print('Panel temp DS1 =  ',Min[2],'|',Max[2],'°C')
print('Panel temp DS2 =  ',Min[3],'|',Max[3],'°C')
print('Wind speed =       ',Min[4],'|',Max[4],'m/s')
print('')
print('IRRADIATION: GHI | GTI ')
print('')
print('- Total per year (kWh/m2/year)')
print('The:',Vt_GHI_To,'|',Vt_GTI_To)
print('DS1:',V1_GHI_To,'|',V1_GTI_To)
print('DS2:',V2_GHI_To,'|',V2_GTI_To)
print('')
print('- Average per day (kWh/m2/day)')
print('The:',Avr_pd[0],'|',Avr_pd[1])
print('DS1:',Avr_pd[2],'|',Avr_pd[3])
print('DS2:',Avr_pd[4],'|',Avr_pd[5])
print('')
print('- Maximum Irradiation (kWh/m2)')
print('The:',Max[5],'|',Max[6])
print('DS1:',Max[7],'|',Max[8])
print('DS2:',Max[9],'|',Max[10])
print('')
print('PV ENERGY YIELD, considering T')
print('')
print('-  Total    |   average  | maximum')
print('kWh/m2/year | kWh/m2/day | kWh/m2')
print('The: ',VtT_gen_To,' |   ',Avr_pd[6],'  |',Max[11])
print('DS1: ',V1T_gen_To,' |   ',Avr_pd[7],'  |',Max[12])
print('DS2: ',V2T_gen_To,' |   ',Avr_pd[8],'  |',Max[13])
print('')
