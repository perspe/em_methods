'''

For a given location and given hour (local time) of a defined day of the year, it calculates:
    Solar times and angles 
    Sunrise, sunset and hours of sun
    AM, irradiance on the horizontal plane (GHI) and PV panel surface (GTI).

Can use an yearly fixed tilt, or find the ideal tilt (maximizes irradiance on PV panel).

Plots Equation of Time (EoT) and AM using 3 different equations that describes it

'''

from datetime import datetime
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# User inputs

LTh, LTm = 13, 30            # hours, minutes, of Local time
day, month = 21, 9 
city = 'Sines'
file_path = 'C:\\Users\\Cristina\\OneDrive - FCT NOVA\\M-ECO2\\Scripts\\PV'
define_tilt = 'ideal'   # chose between 'fixed' or 'ideal'. Fixed = an yearly fixed tilt corresponding to the latitude. Ideal = Script determines the ideal, which maximized irradiance collection



def import_data_from_excel(file_path, city):  # Import data from excel file

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

data = import_data_from_excel(file_path, city)
(longitude, latitude, timezone, height, tilt, L_hours, L_days, L_months, L1_GHI, L1_DNI, L1_DHI, L_Tair, L_wind) = (
    data['longitude'], data['latitude'], data['timezone'], data['height'], data['tilt'], data['hours'], data['days'], data['Month'],
    data['GHI_DS1'], data['DNI_DS1'], data['DHI_DS1'], data['temperature'], data['wind'])

 

def days_since_start_of_year(month, day):
    input_date = datetime(datetime.now().year, month, day)
    start_of_year = datetime(datetime.now().year, 1, 1)
    days = (input_date - start_of_year).days
    return days



def solar_parameters_for_a_set_time (LTh, LTm, month, day, timezone, longitude, latitude, height):
    
    '''
    Calculate solar times, solar angles, sunrise time, sunset time, AM, and irradiances
    
    Variables and units:
        Solar times:
            LT (Local Time) = hours
            LSTM (Local Standard Time Meridian) = hours
            EoT (Equation of Time, correction factor used to reconcile differences between solar time and clock time) = minutes
            TC (Time Correction) = minutes
            sunsrise and sunset = in Local Time
        Solar angles: B (Solar angle), HRA (Hour angle), declination, zenith, azimuth, incidence, elevation, tilt.
            -> when ended with "_r", untis are radians. Otherwise, degrees.
        Irradiance:
            AM (Air mass) = -
            DNI (Direct Normal Irradiance) = kW/m2
            GHI (Global Horizontal Irradiance) = kW/m2
    '''
        
    # Solar times
    
    LT = LTh+LTm/60
    d = days_since_start_of_year(month, day)
    LSTM = 15 * timezone
    B = 360 * (d-81) / 365
    EoT = 9.87*math.sin(math.radians(2*B)) - 7.53*math.cos(math.radians(B)) - 1.5*math.sin(math.radians(B))
    TC = 4 * (longitude - LSTM) + EoT
    LST = LT + TC / 60
    LST_min = int(60 * (LST-int(LST)))
    HRA = 15 * (LST - 12)

    # Solar angles   

    declination = 23.45 * math.sin(math.radians(360 * (d-81) / 365)) 

    variables = [declination, latitude, HRA]
    variables = [math.radians(var) for var in variables]  
    declination_r, latitude_r, HRA_r = variables             
    zenith_r = math.acos(math.sin(latitude_r)*math.sin(declination_r) + math.cos(latitude_r)*math.cos(declination_r)*math.cos(HRA_r))
    elevation_r = math.radians(90) - zenith_r
    n_azimuth_r = math.sin(declination_r) * math.cos(latitude_r) - math.cos(declination_r) * math.sin(latitude_r) * math.cos(HRA_r)
    d_azimuth_r = math.cos(elevation_r)
    azimuth_r = math.acos(n_azimuth_r/d_azimuth_r) if LST < 12 else math.radians(360) - math.acos(n_azimuth_r/d_azimuth_r)
    azimuth_panel_r = math.radians(180) if latitude > 0 else 0   # solar pannel faces south/north in the north/south hemisphere

    # Sunrise and sunset times
    
    sunrise = 12 - (1/15)*math.degrees(math.acos(-math.tan(latitude_r)*math.tan(declination_r))) - TC/60   # in hours
    sunset = 12 + (1/15)*math.degrees(math.acos(-math.tan(latitude_r)*math.tan(declination_r))) - TC/60    # in hours
    sunlight = sunset - sunrise
    sunlight_min = int(60 * (sunlight-int(sunlight)))   

    # Irradiance

    AM = 1/(math.cos(zenith_r)+0.50572*(96.07995-math.degrees(zenith_r))**(-1.6364))  # Kasten and Young Formula
    DNI = 1.353*((1-0.14*height)*(0.7**(AM**0.678))+0.14*height)                      
    GHI = 1.1 * DNI * math.sin(elevation_r)   # assumes that DHI is 10% of of DNI                                          

    return d, latitude_r, HRA_r, declination_r, elevation_r, zenith_r, azimuth_r, azimuth_panel_r, LSTM, EoT, TC, LST, LST_min, AM, GHI, DNI, sunrise, sunset, sunlight, sunlight_min

d, latitude_r, HRA_r, declination_r, elevation_r, zenith_r, azimuth_r, azimuth_panel_r, LSTM, EoT, TC, LST, LST_min, AM, GHI, DNI, sunrise, sunset, sunlight, sunlight_min = solar_parameters_for_a_set_time (LTh, LTm, month, day, timezone, longitude, latitude, height)



# Print solar parameters results

variables = [latitude_r, HRA_r, declination_r, elevation_r, zenith_r, azimuth_r]
variables = [round(math.degrees(var),1) for var in variables]  
latitude, HRA, declination, elevation, zenith, azimuth = variables

variables = [LSTM, EoT, TC, AM, GHI]
variables = [round(var,3) for var in variables]  
LSTM, EoT, TC, AM, GHI_int = variables

print(f'{city}\n')
print(f'Solar Time:\nday of the year = {d}\nLSTM = {LSTM}°\nEoT = {EoT} min\nTC = {TC} min\nLST = {int(LST)}:{LST_min}')
print(f'\nSolar Angles:\nlatitude, phi = {latitude}°\nHRA = {HRA}°\ndeclination, delta = {declination}°\nnelevation, alpha = {elevation}°\nzenith, teta = {zenith}°\nazimuth, gama = {azimuth}°')
print(f'\nsunrise at {int(sunrise)}:{int((sunrise-int(sunrise))*60)}')
print(f'sunset at {int(sunset)}:{int((sunset-int(sunset))*60)}')
print(f'hours of sunlight = {int(sunlight)}:{sunlight_min}')



def GTI_for_a_Set_time(define_tilt, tilt, latitude_r, declination_r, HRA_r, DNI, elevation_r, azimuth_panel_r, azimuth_r, GHI):
    
    '''
    Calculates the Global Tilted Irradiance, total irradiance falling on the PV panel's surface = kW/m2
    '''
    
    if define_tilt == 'fixed':
        
        tilt_r = math.radians(tilt)
        if latitude > 0:    #north hemisphere
            incidence_r = math.acos(math.sin(latitude_r-tilt_r)*math.sin(declination_r) + math.cos(latitude_r-tilt_r)*math.cos(declination_r)*math.cos(HRA_r)) 
        else:               # south hemisphere
            incidence_r = math.acos(math.sin(latitude_r+tilt_r)*math.sin(declination_r) + math.cos(latitude_r+tilt_r)*math.cos(declination_r)*math.cos(HRA_r)) 
        incidence = round(math.degrees(incidence_r),1)   
            
        GTI = 1.1*DNI * (math.cos(elevation_r)*math.sin(tilt_r)*math.cos(azimuth_panel_r-azimuth_r)+math.sin(elevation_r)*math.cos(tilt_r))  
        gain = GTI*100/GHI - 100        # percentage of gain in irradiance by tilting solar panel
        tilt_ideal, incidence_ideal = None, None
    
    else:        
        incidence, GTI, gain = [], [], []
        tilt = np.linspace (0,90,100)
        for tilt_value in tilt:
            tilt_r = math.radians(tilt_value)
            if latitude > 0:    #north hemisphere
                incidence_r = math.acos(math.sin(latitude_r-tilt_r)*math.sin(declination_r) + math.cos(latitude_r-tilt_r)*math.cos(declination_r)*math.cos(HRA_r)) 
            else:               # south hemisphere
                incidence_r = math.acos(math.sin(latitude_r+tilt_r)*math.sin(declination_r) + math.cos(latitude_r+tilt_r)*math.cos(declination_r)*math.cos(HRA_r)) 
            incidence_value = round(math.degrees(incidence_r),1)  
            GTI_value = 1.1*DNI * (math.cos(elevation_r)*math.sin(tilt_r)*math.cos(azimuth_panel_r-azimuth_r)+math.sin(elevation_r)*math.cos(tilt_r))  
            gain_value = GTI_value*100/GHI - 100        # percentage of gain in irradiance by tilting solar panel
            incidence.append(incidence_value)
            GTI.append(round(GTI_value,3))
            gain.append(round(gain_value,3))

        tilt_ideal = round(tilt[GTI.index(max(GTI))],2)
        incidence_ideal = round(incidence[GTI.index(max(GTI))],1)        
        
    return incidence, GTI, gain, tilt_ideal, incidence_ideal
    
incidence, GTI, gain, tilt_ideal, incidence_ideal = GTI_for_a_Set_time(define_tilt, tilt, latitude_r, declination_r, HRA_r, DNI, elevation_r, azimuth_panel_r, azimuth_r, GHI)



# Print GTI results 

print(f'\nIrradiance:\nAM = {AM}\nGlobal Horizontal irradiance (GHI) = {GHI_int} kW/m2') 
if define_tilt == 'fixed': 
    print(f'Global Tilted Irradiance (GTI) = {round(GTI,3)} kW/m2\nTilt = {tilt}°\nIncidence = {incidence}°\nGain = {round(gain,1)} %')
elif define_tilt == 'ideal':
    print(f'Global Tilted Irradiance (GTI) = GTI = {round(max(GTI),3)} kW/m2\nTilt = {tilt_ideal}°\nIncidence = {incidence_ideal}°\nGain = {round(max(gain),2)} %')   
else:
    print("The input variable 'define_tilt' should be 'fixed' or 'ideal'")
    
    
    
# Plots

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
