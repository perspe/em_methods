import datetime
import logging
import math
from typing import Union

import numpy as np
import numpy.typing as npt
from pytz import timezone
import scipy.constants as scc

# Get module logger
logger = logging.getLogger("sim")


def solar_declination(
    days_start_year: Union[npt.NDArray[np.integer], int]
) -> Union[int, npt.NDArray[np.floating]]:
    """
    Calculate solar declination
    """
    gamma = 2 * scc.pi * (days_start_year - 81) / 365
    declination = 23.45 * np.sin(gamma)
    return declination


def eot_main(
    days_start_year: Union[npt.NDArray[np.integer], int]
) -> Union[float, npt.NDArray[np.floating]]:
    gamma = 2 * scc.pi * (days_start_year - 81) / 365
    eot = 9.87 * np.sin(2 * gamma) - 7.53 * np.cos(gamma) - 1.5 * np.sin(gamma)
    return eot


def eot1(dt: datetime.datetime, days_start_year: int) -> Union[float, npt.NDArray]:
    """
    Old formula to calculate EoT
    """
    gamma = 2 * scc.pi / 365 * (days_start_year - 1 + (dt.hour - 12) / 24)
    eot = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    return eot


def am_kasten(zenith: Union[float, npt.NDArray]) -> Union[float, npt.NDArray]:
    """
    Calculate solar power from Kasten formula
    """
    # Calculate Power (Kasten and Young)
    zenith_rad = np.radians(zenith)
    air_mass = 1 / (math.cos(zenith_rad) + 0.50572 * (96.07995 - zenith) ** (-1.6364))
    return air_mass


def am_rozenberg(zenith: Union[float, npt.NDArray]) -> Union[float, npt.NDArray]:
    """
    Calculate solar power from the Rozenberg formula
    """
    zenith_rad = np.radians(zenith)
    air_mass = (
        np.cos(zenith_rad) + 0.025 * np.exp(-11 * np.cos(math.radians(zenith)))
    ) ** -1
    return air_mass


def solar_angle(
    longitude: float, latitude: float, dt: datetime.datetime, default_tz=timezone("GMT"), eot_function=eot_main
):
    """
    Determine the solar angle from coordinate and time information
    Args:
        longitude/latitude: Spacial information of the wanted location
        dt: date and time for the calculations (using datetime.date)
        default_tz: timezone to consider as default in the datetime.date argument
    Return:
        zenith, azimuth: Solar Angles (in degree)
        sunrise, sunset: Time of sunrise and sunset
    """
    if dt.tzinfo is None:
        logger.debug("Updating timezone info to default")
        dt = dt.replace(tzinfo=default_tz)
    # Determine the solar time (hour angle) in the wanted solar time
    days_start_year: int = (dt.date() - datetime.date(dt.date().year, 1, 1)).days + 1
    time_diff = dt.strftime("%z")
    time_diff = int(time_diff[1:2]) + int(time_diff[2:]) / 60
    lstm = 15 * time_diff
    eot = eot_function(days_start_year)
    declination = solar_declination(days_start_year)
    if not isinstance(eot, float) or not isinstance(declination, float):
        raise TypeError("EoT or declination have wrong type")
    tc = 4 * (longitude - lstm) + eot
    lst = dt + datetime.timedelta(minutes=tc)
    lst_offset_h = lst + datetime.timedelta(hours=-12)
    lst_offset_h = lst_offset_h.hour + lst_offset_h.minute / 60
    hra = 15 * lst_offset_h
    logger.debug(
        f"""
Initial date: {dt}
Day of the year: {days_start_year}
Time difference (Timezone): {time_diff}
LSTM: {lstm}
EoT: {eot}
TC: {tc}
LST_offset: {lst_offset_h}
LST: {lst}
Hour angle in local solar time: {hra}
"""
    )
    # Convert delta, hra and latitude to radians
    declination, hra, latitude = tuple(
        np.radians(var) for var in [declination, hra, latitude]
    )
    solar_angle = math.degrees(math.acos(-math.tan(latitude) * math.tan(declination)))
    sunrise = datetime.timedelta(hours=12, minutes=-solar_angle * 4 - tc)
    sunset = datetime.timedelta(hours=12, minutes=+solar_angle * 4 - tc)
    zenith = math.acos(
        math.sin(latitude) * math.sin(declination)
        + math.cos(latitude) * math.cos(declination) * math.cos(hra)
    )
    elevation = scc.pi / 2 - zenith
    n_azimuth = math.sin(declination) * math.cos(latitude) - math.cos(
        declination
    ) * math.sin(latitude) * math.cos(hra)
    d_azimuth = math.cos(elevation)
    if hra < 0:
        azimuth = math.degrees(math.acos(n_azimuth / d_azimuth))
    else:
        azimuth = 360 - math.degrees(math.acos(n_azimuth / d_azimuth))
    logger.debug(
        f"""
Solar Angles:
Sunrise in h: {sunrise}
Sunset in h: {sunset}
Delta: {math.degrees(declination)}
Alpha angle: {math.degrees(elevation)}
Zenith angle: {math.degrees(zenith)}
Azimuth Angle: {azimuth}
"""
    )
    return math.degrees(zenith), azimuth, sunrise, sunset


def solar_power(
    longitude: float,
    latitude: float,
    height: float,
    tilt: float,
    dt: datetime.datetime,
    default_tz=timezone("GMT"),
    am_function=am_kasten
):
    """
    Determine the solar power incident on a module, from its location on
    the globe and the datetime
    Args:
        longitude/latitude: Spacial information of the wanted location
        height: height above sea level for the module
        beta: inclination angle of the module
        dt: date and time for the calculations (using datetime.date)
        default_tz: timezone to consider as default in the datetime.date argument
    Return:
        ghi: Global Horizontal Irradiance
        gti: GLobal Tilted Irradiance
        gain: Irradiance gain by titlting the solar panel
    """
    zenith, azimuth, *_ = solar_angle(longitude, latitude, dt, default_tz)
    zenith_rad, azimuth_rad = math.radians(zenith), math.radians(azimuth)
    air_mass = am_function(zenith)
    tilt_rad = math.radians(tilt)
    elevation = scc.pi / 2 - zenith_rad
    azimuth_pannel = scc.pi if latitude > 0 else 0
    # Direct Normal irradiance
    dni = 1.353 * ((1 - 0.14 * height) * (0.7 ** (air_mass**0.678)) + 0.14 * height)
    # Global horizontal irradiance
    ghi = 1.1 * dni * math.sin(elevation)
    # Global Tilted irradiance
    gti = (
        1.1
        * dni
        * (
            math.cos(elevation)
            * math.sin(tilt_rad)
            * math.cos(azimuth_pannel - azimuth_rad)
            + math.sin(elevation) * math.cos(tilt_rad)
        )
    )
    # Direct and total power (considering a diffuse power of 10% pdirect)
    gain = (
        gti * 100 / ghi - 100
    )  # percentage of gain in irradiance by tilting solar panel
    logger.debug(
        f"""
Irradiance values:
AM:{air_mass}
Global Horizontal Irradiance (GHI) (kW/m2): {ghi}
Global Tilted Irradiance (W/m2): {gti}
Gain: {gain}
"""
    )
    return ghi, gti, gain
