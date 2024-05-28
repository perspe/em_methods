import datetime
import logging
import math

import numpy as np
from pytz import timezone
import scipy.constants as scc

# Get module logger
logger = logging.getLogger("sim")


def solar_angle(
    longitude: float, latitude: float, dt: datetime.datetime, default_tz=timezone("GMT")
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
        dt = dt.astimezone(default_tz)
    # Determine the solar time (hour angle) in the wanted solar time
    days_start_year: int = (dt.date() - datetime.date(dt.date().year, 1, 1)).days + 1
    time_diff = dt.strftime("%z")
    time_diff = int(time_diff[1:2]) + int(time_diff[2:]) / 60
    lstm = 15 * time_diff
    gamma = 2 * scc.pi / 365 * (days_start_year - 1 + (dt.hour - 12) / 24)
    eq_time = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    tc = 4 * (longitude - lstm) + eq_time
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
EoT: {eq_time}
TC: {tc}
LST_offset: {lst_offset_h}
LST: {lst}
Hour angle in local solar time: {hra}
"""
    )
    # Determine the solar angles and the sunrise/sunset times
    delta = 23.45 * math.sin(math.radians(360 * (days_start_year - 81) / 365))
    # Convert delta, hra and latitude to radians
    delta, hra, latitude = {np.radians(var) for var in [delta, hra, latitude]}
    solar_angle = math.degrees(math.acos(-math.tan(latitude) * math.tan(delta)))
    sunrise = datetime.timedelta(hours=12, minutes=-solar_angle * 4 - tc)
    sunset = datetime.timedelta(hours=12, minutes=+solar_angle * 4 - tc)
    zenith = math.acos(
        math.sin(latitude) * math.sin(delta)
        + math.cos(latitude) * math.cos(delta) * math.cos(hra)
    )
    alpha = scc.pi / 2 - zenith
    n_azimuth = math.sin(delta) * math.cos(latitude) - math.cos(delta) * math.sin(
        latitude
    ) * math.cos(hra)
    d_azimuth = math.cos(alpha)
    if hra < 0:
        azimuth = math.degrees(math.acos(n_azimuth / d_azimuth))
    else:
        azimuth = 360 - math.degrees(math.acos(n_azimuth / d_azimuth))
    logger.debug(
        f"""
Solar Angles:
Sunrise in h: {sunrise}
Sunset in h: {sunset}
Delta: {math.degrees(delta)}
Alpha angle: {math.degrees(alpha)}
Zenith angle: {math.degrees(zenith)}
Azimuth Angle: {azimuth}
"""
    )
    return math.degrees(zenith), azimuth, sunrise, sunset


def solar_power(
    longitude,
    latitude,
    dt,
    height,
    beta,
    hemisphere="north",
    default_tz=timezone("GMT"),
):
    zenith, azimuth, *_ = solar_angle(longitude, latitude, dt, default_tz)
    alpha = scc.pi/2 - zenith
    # Calculate Power
    air_mass = 1 / (
        math.cos(math.radians(zenith)) + 0.50572 * (96.07995 - zenith) ** (-1.6364)
    )
    # Direct and total power (considering a diffuse power of 10% pdirect)
    pdirect = (
        1.353
        * ((1 - 0.14 * height) * (0.7 ** (air_mass**0.678)) + 0.14 * height)
        * 1000
    )
    pin = 1.1 * pdirect
    azimuth_pannel = (
        180 if hemisphere == "north" else 0
    )
    pmodule = pin * (
        math.cos(alpha) * math.sin(beta) * math.cos(azimuth_pannel - azimuth)
        + math.sin(alpha) * math.cos(beta)
    )
    return air_mass, pdirect, pin, pmodule
