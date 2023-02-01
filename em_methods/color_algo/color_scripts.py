
import numpy as np
from colour_system import cs_hdtv
cs = cs_hdtv

def rgb2hex(r,g,b):
    """Return color as #rrggbb for the given color values."""
    return '#%02x%02x%02x' % (r,g,b)
import time

def hex2rgb(hex_string):  # FUNÇÃO DE CONVERSÃO DO HEXCODE
    if type(hex_string) == str:
        hex_string = hex_string.upper()
        r_hex = hex_string[1:3]
        g_hex = hex_string[3:5]
        b_hex = hex_string[5:7]
        return int(r_hex, 16), int(g_hex, 16), int(b_hex, 16)
    else:
        return hex_string

def timeelapsed(start, end): #TAKES IN SECONDS
    timeelapsed = end - start
    hours = int(round(timeelapsed/3600, 0))
    minutes = int(round(timeelapsed % 3600 / 60, 0))
    seconds = int(round(timeelapsed % 60, 0))
    ctime = time.ctime(end)[4:-5]
    print(f'\n \n \t Simulations Finished on {ctime} \n \n' +
          '\t Duration: ' +
          f'{hours} hours, {minutes} minutes and {seconds} seconds.')

def satcolorgen(T): #TAKES IN A LIST FOR X=[380:781:5]
    cs = cs_hdtv
    spec = np.array(T)
    satcolor = cs.spec_to_rgb(spec, out_fmt='html')
    satcolor = hex2rgb(satcolor)
    return satcolor


def colordist(current, desired): #TAKES IN CURRENT AS RGB, DESIRED AS HEX

    desired_rgb = hex2rgb(desired)
    R, G, B = [*desired_rgb]
    r, g, b = [*current]
    colordistance = ((R-r)**2+(G-g)**2+(B-b)**2)**(1/2)
    return colordistance
