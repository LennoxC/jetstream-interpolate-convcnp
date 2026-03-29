from math import cos
import math
from jetstream_interpolate_convcnp.utils.constants import R

def metres_to_degrees(metres, latitude):
    """
    Convert a distance in metres to degrees of latitude and longitude at a given latitude.
    """
    # Convert latitude to radians
    lat_rad = latitude * (math.pi / 180)
    
    # Calculate the length of one degree of latitude and longitude in metres
    lat_length = (math.pi * R) / 180
    lon_length = (math.pi * R * cos(lat_rad)) / 180
    
    # Convert metres to degrees
    lat_degrees = metres / lat_length
    lon_degrees = metres / lon_length
    
    return lat_degrees, lon_degrees

def degrees_to_metres(degrees, latitude):
    """
    Convert a distance in degrees of latitude and longitude to metres at a given latitude.
    """
    # Convert latitude to radians
    lat_rad = latitude * (math.pi / 180)
    
    # Calculate the length of one degree of latitude and longitude in metres
    lat_length = (math.pi * R) / 180
    lon_length = (math.pi * R * cos(lat_rad)) / 180
    
    # Convert degrees to metres
    lat_metres = degrees * lat_length
    lon_metres = degrees * lon_length
    
    return lat_metres, lon_metres