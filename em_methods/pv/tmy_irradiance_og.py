'''
PURPOSE

--> For 3 locations and 3 separate days, plots the GHI and GTI using theoretical solar equations and two separate databases.
    Also prints max and total GTI and GHI for the 3 cities and 3 datasets (T, DS1 and DS2).

DESCRITPION

1. Select 3 days of the year (default summer solstice, winter solstice, and autmunal equinox)
    
2. For 3 locations (default: Sines, Edmonton and Crystal Brook), prepare excel files with the Typical Meteorological Year (TMY) datasets (at least from 2 databases, here NREL and EU) and location's information:
        - location's information:
            - longitude, latitude, timezone, height, PV panel tilt
        - TMY datasets information (in Local Time):
            - wind speed (m/s), air temperature (ºC), GHI, DNI, DHI*

    *Definitions:
        Global Horizontal Irradiance (GHI) - total irradiance falling on the earth's surface, aka, on the horizontal plane
        Direct Normal Irradiance (DNI) - irradiance received by a surface that is always held perpendicular (or normal) to the solar rays that come in a straight line.
        Diffuse Horizontal Irradiance (DHI) - irradiance received by a horizontal surface which has been scattered or diffused by the atmosphere.
               For theoretical irradiances, assume that DHI is 10% of of DNI
        
3. Import TMY and location's data

4. Calculate the Solar times and angles (angles finished with "_r" are in radians) https://www.pveducation.org/pvcdrom/properties-of-sunlight/solar-time

5. Calculate AM and GHI (for theoretical values), and GTI** (for theoretical and TMY-based)  throughout the day 
    - theoretical: https://www.pveducation.org/pvcdrom/properties-of-sunlight/air-mass
    - TMY-based: https://www.pveducation.org/pvcdrom/properties-of-sunlight/making-use-of-tmy-data
   
    **Global Tilted Irradiance (GTI) - total irradiance falling on the PV panel's surface, aka, on a tiltes surface
           Consider irradiance losses due to shallow angle reflection (Martin, N. and J.M. Ruiz, 2013)

6. Calculates max and total values of GHI and GTI for the 3 citiesand datasets (T, DS1 ands DS2)

7. Plots the GHI and GTI for the 3 cities and 3 datasets (T, DS1 ands DS2) throughout the selected 3 days.


Databases:
    - EU (used mainly for EU, Africa and Asia): https://re.jrc.ec.europa.eu/pvg_tools/en/#TMY
    - NREL (used mainly for America and Australia): https://nsrdb.nrel.gov/data-viewer
    
'''


import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


# User inputs

dates = [(21, 6), (21, 9), (21, 12)]   # selet three different days (day, month)
year = 2025

cities_data = {
    'Sines': {'file_path': 'C:\\Users\\Cristina\\OneDrive - FCT NOVA\\M-ECO2\\Scripts\\PV\\GHI Sines\\dataGHI_Sines.xlsx'},
    'Edmonton': {'file_path': 'C:\\Users\\Cristina\\OneDrive - FCT NOVA\\M-ECO2\\Scripts\\PV\\GHI Edmonton\\dataGHI_Edmonton.xlsx'},
    'Crystal Brook': {'file_path': 'C:\\Users\\Cristina\\OneDrive - FCT NOVA\\M-ECO2\\Scripts\\PV\\GHI Crystal Brook\\dataGHI_Crystal Brook.xlsx'}}



def days_since_start_of_year(year, month, day):
    date_object = datetime(year, month, day)
    start_of_year = datetime(year, 1, 1)
    return (date_object - start_of_year).days + 1



def get_data_for_day(city_data, day, month, year):
    
    '''
    Calculates theoretical, and TMY-based (DS1 and DS2) AM, GHI and GTI (with and without considering losses due to shalow angle reflection) = W/m2
    '''
    
    longitude, latitude, timezone, height, tilt = city_data['location_info']
    days_all = city_data['days_all']
    hours_all = city_data['hours_all']
    DS1_GHI_all = city_data['DS1_GHI_all']
    DS1_DNI_all = city_data['DS1_DNI_all']
    DS1_DHI_all = city_data['DS1_DHI_all']
    DS2_GHI_all = city_data['DS2_GHI_all']
    DS2_DNI_all = city_data['DS2_DNI_all']
    DS2_DHI_all = city_data['DS2_DHI_all']


    d = days_since_start_of_year(year, month, day)
    hours, DS1_GHI, DS1_DNI, DS1_DHI, DS2_GHI, DS2_DNI, DS2_DHI = [], [], [], [], [], [], []
    
    for i in range(len(days_all)):
        if days_all[i] == d:
            hours.append(hours_all[i])
            DS1_GHI.append(DS1_GHI_all[i])
            DS1_DNI.append(DS1_DNI_all[i])
            DS1_DHI.append(DS1_DHI_all[i])
            DS2_GHI.append(DS2_GHI_all[i])
            DS2_DNI.append(DS2_DNI_all[i])
            DS2_DHI.append(DS2_DHI_all[i])

    # Solar angles and times 
    
    declination = 23.45 * math.sin(math.radians(360 * (d-81) / 365))
    B = 360 * (d-81) / 365
    EoT = 9.87 * math.sin(math.radians(2*B)) - 7.53 * math.cos(math.radians(B)) - 1.5 * math.sin(math.radians(B))
    TC = 4 * (longitude - 15 * timezone) + EoT

    variables = [declination, latitude, tilt]
    variables = [math.radians(var) for var in variables]
    declination_r, latitude_r, tilt_r = variables

    # AM and irradiances (GHI and GTI) for DS1 and DS2
    
    AM, Th_GHI, Th_GTI, DS1_GTI, DS2_GTI = [], [], [], [], []
    Th_GTI_eff, DS1_GTI_eff, DS2_GTI_eff = [], [], []
    ar = 0.169
    
    for h in hours:
        LST = h + TC / 60
        HRA_r = math.radians(15 * (LST - 12))
        
        if latitude > 0:
            incidence_r = math.acos(math.sin(latitude_r-tilt_r)*math.sin(declination_r) + math.cos(latitude_r-tilt_r)*math.cos(declination_r)*math.cos(HRA_r))
        else:
            incidence_r = math.acos(math.sin(latitude_r+tilt_r)*math.sin(declination_r) + math.cos(latitude_r+tilt_r)*math.cos(declination_r)*math.cos(HRA_r))
        
        zenith_r = math.acos(math.sin(latitude_r)*math.sin(declination_r) + math.cos(latitude_r)*math.cos(declination_r)*math.cos(HRA_r))
        elevation_r = math.radians(90) - zenith_r
        n_azimuth = math.sin(declination_r) * math.cos(latitude_r) - math.cos(declination_r) * math.sin(latitude_r) * math.cos(HRA_r)
        d_azimuth = math.cos(elevation_r)
        azimuth_r = math.acos(n_azimuth/d_azimuth) if LST < 12 else math.radians(360) - math.acos(n_azimuth/d_azimuth)
        azimuth_panel_r = math.radians(180) if latitude >=0 else 0     # measured from the from North to East, solar pannel faces south/north in the north/south hemisphere
        
        tilt_factor = (math.cos(elevation_r)*math.sin(tilt_r)*math.cos(azimuth_panel_r-azimuth_r)+math.sin(elevation_r)*math.cos(tilt_r))
        AL_value = 1 - ((1-math.exp(-math.cos(incidence_r)/ar))/(1-math.exp(-1/ar)))      # Shallow-angle reflection losses in %
        
       
        if 0 <= zenith_r <= math.radians(90):
            AM_value = 1/(math.cos(zenith_r)+0.50572*(96.07995-math.degrees(zenith_r))**(-1.6364))   # air mass, Kasten and Young Formula
            DNI_value = 1353*((1-0.14*height)*(0.7**(AM_value**0.678))+0.14*height)                  # Direct Normal Irradiance, W/m2
            GHI_value = 1.1 * DNI_value * math.sin(elevation_r)                                      # Global Horizontal Irradiance, W/m2  
            GTI_value = 1.1 * DNI_value * tilt_factor                                                # Global Tilted Irradiance, W/m2
            GTI_eff_value = GTI_value * (1 - AL_value)                                               # effective GTI, W/m2, considering shallow angle reflection losses
        else: 
            AM_value, GHI_value, GTI_value, GTI_eff_value = 1000, 0, 0, 0
        AM.append(AM_value)
        Th_GHI.append(GHI_value if GHI_value>0 else 0)
        Th_GTI.append(GTI_value if GTI_value>0 else 0)
        Th_GTI_eff.append(GTI_eff_value if GTI_value>0 else 0)
        
        # GTI for the 3 TMY datasets in W/m2
        
        new_azimuth_panel_r = (azimuth_panel_r - math.pi) % (2 * math.pi)               # orientation measured from South to West
        DNI_correction_1 = math.sin(declination_r)*math.sin(latitude_r)*math.cos(tilt_r) - math.sin(declination_r)*math.cos(latitude_r)*math.sin(tilt_r)*math.cos(new_azimuth_panel_r)
        DNI_correction_2 = math.cos(declination_r)*math.cos(latitude_r)*math.cos(tilt_r)*math.cos(HRA_r) + math.cos(declination_r)*math.sin(new_azimuth_panel_r)*math.sin(HRA_r)*math.sin(tilt_r)
        DNI_correction_3 = math.cos(declination_r)*math.sin(latitude_r)*math.sin(tilt_r)*math.cos(new_azimuth_panel_r)*math.cos(HRA_r)
        DNI_correction = DNI_correction_1 + DNI_correction_2 + DNI_correction_3         # DNI_tilted = DNI from the TMY * DNI_correction
        DHI_correction = (math.radians(180)-tilt_r)/math.radians(180)                   # DHI_tilted = DHI from the TMY * DHI_correction
        
        i = hours.index(h)
        DS1_GTI_value = max(0,DS1_DNI[i] * DNI_correction + DS1_DHI[i] * DHI_correction)    # GTI = DNI_tilted + DHI_tilted
        DS2_GTI_value = max(0,DS2_DNI[i] * DNI_correction + DS2_DHI[i] * DHI_correction)
 
        DS1_GTI.append(DS1_GTI_value)               
        DS2_GTI.append(DS2_GTI_value)
        DS1_GTI_eff.append(max(0,DS1_GTI_value * (1 - AL_value)))
        DS2_GTI_eff.append(max(0,DS2_GTI_value * (1 - AL_value)))
    
    Th_reflection = round(100 - (100*np.trapz(Th_GTI_eff, dx=1)/np.trapz(Th_GTI, dx=1)),2)    
    DS1_reflection = round(100 - (100*np.trapz(DS1_GTI_eff, dx=1)/np.trapz(DS1_GTI, dx=1)),2)  
    DS2_reflection = round(100 - (100*np.trapz(DS2_GTI_eff, dx=1)/np.trapz(DS2_GTI, dx=1)),2)
    
    return hours, Th_GHI, Th_GTI_eff, DS1_GHI, DS1_GTI_eff, DS2_GHI, DS2_GTI_eff, Th_reflection, DS1_reflection, DS2_reflection



def load_city_data(file_path):
    
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
    
    data = pd.read_excel(file_path, sheet_name='data_LT')
    longitude = data.iat[1, 1]         # degrees, longitude
    latitude = data.iat[2, 1]          # degrees, latitude
    timezone = data.iat[3, 1]          # hours
    height = data.iat[4, 1]            # kilometers, height above sea level
    tilt = data.iat[5, 1]              # degrees, panel tilt angle

    city_data = {
        'location_info': (longitude, latitude, timezone, height, tilt),
        'days_all': data['days'].values,
        'hours_all': data['hours'].values,
        'DS1_GHI_all': data['GHI_DS1'].values,
        'DS1_DNI_all': data['DNI_DS1'].values,
        'DS1_DHI_all': data['DHI_DS1'].values,
        'DS2_GHI_all': data['GHI_DS2'].values,
        'DS2_DNI_all': data['DNI_DS2'].values,
        'DS2_DHI_all': data['DHI_DS2'].values,}
    return city_data



# Plots

for city_name, city_info in cities_data.items():
    city_info['data'] = load_city_data(city_info['file_path'])

fig, axs = plt.subplots(3, 3, figsize=(9, 7), sharey=True)
linewidth=1.2     
def get_month_name(month):
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if 1 <= month <= 12:
        return month_names[month - 1]
    return "Invalid month"

def suffix(day):
    if 10 <= day % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: '$^s$$^t$', 2: '$^n$$^d$', 3: '$^r$$^d$'}.get(day % 10, 'th')
    return f"{suffix}"

for i, (city_name, city_info) in enumerate(cities_data.items()):
    for j, (day, month) in enumerate(dates):
        hours, Th_GHI, Th_GTI_eff, DS1_GHI, DS1_GTI_eff, DS2_GHI, DS2_GTI_eff, Th_reflection, DS1_reflection, DS2_reflection = get_data_for_day(city_info['data'], day, month, year)
        print(f'{city_name}, {get_month_name(month)} {day}')
        print('Theoretical | DS1 | DS2')
        print(f'GHI max (W/m2) / total (Wh/m2) = {round(max(Th_GHI))}/{round(np.trapz(Th_GHI, dx=1))} | {round(max(DS1_GHI))}/{round(np.trapz(DS1_GHI, dx=1))} | {round(max(DS2_GHI))}/{round(np.trapz(DS2_GHI, dx=1))} ')
        print(f'GTI max (W/m2) / total (Wh/m2) = {round(max(Th_GTI_eff))}/{round(np.trapz(Th_GTI_eff, dx=1))} | {round(max(DS1_GTI_eff))}/{round(np.trapz(DS1_GTI_eff, dx=1))} | {round(max(DS2_GTI_eff))}/{round(np.trapz(DS2_GTI_eff, dx=1))} ')
        print(f'reflection losses (%) = {Th_reflection} | {DS1_reflection} | {DS2_reflection}\n')
        ax = axs[i, j]
        ax.plot(hours, Th_GHI, color='black', linewidth=linewidth, label='T GHI')
        ax.plot(hours, Th_GTI_eff, color='black', linestyle='--',linewidth=linewidth, label='T GTI')
        ax.plot(hours, DS1_GHI, color='goldenrod', linewidth=linewidth, label='DS1 GHI')
        ax.plot(hours, DS1_GTI_eff, color='goldenrod', linestyle='--',linewidth=linewidth,  label='DS1 GTI')
        ax.plot(hours, DS2_GHI, color='saddlebrown', linewidth=linewidth, label='DS2 GHI')
        ax.plot(hours, DS2_GTI_eff, color='saddlebrown', linewidth=linewidth, linestyle='--', label='DS2 GTI')
        ax.grid(True,alpha=0.5)
        ax.text(5, 1000, f'{day}{suffix(day)} {get_month_name(month)}')
        ax.set_ylim(0,1100)
        ax.set_xlim(4,22)
        
        if j == 0:
            ax.set_yticks([0,200,400,600,800,1000])
            ax.set_ylabel(f'Solar Irradiance (W·m$^-$$^2$)\nin {city_name}')
            
        if i == 2:
            ax.set_xticks(np.arange(5, 22, 4))
            ax.set_xlabel('Local Time (hours)')
            

        if i == 0 and j == 2:
            ax.legend(loc='upper right', bbox_to_anchor=(1, 1))

plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0, wspace=0)
plt.show()