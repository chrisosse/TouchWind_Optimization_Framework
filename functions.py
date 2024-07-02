import numpy as np
import pandas as pd
import scipy as sc
from scipy import optimize
import math


# Sine
def sind(angle):
    return np.sin(np.radians(angle))


# Cosine
def cosd(angle):
    return np.cos(np.radians(angle))


# Tangent
def tand(angle):
    return np.tan(np.radians(angle))


# Arcsine
def arcsind(angle):
    return np.degrees(np.arcsin(angle))


# Arccosine
def arccosd(angle):
    return np.degrees(np.arccos(angle))


# Acttangent in plane
def arctan2d(y, x):
    return np.degrees(np.arctan2(y, x))


# Find nearest number in list
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    value = array[idx]
    return idx, value


# Create farm layout
def create_layout(
    shape: str = 'rectangular',
    n_x: int = 1,
    n_y: int = 1,
    spacing_x: float = 5,
    spacing_y: float = 5,
    D_rotor: float = 126,
):
    if shape == 'hexagonal':
        # Calculate the vertical and horizontal spacing between hexagon centers
        vertical_spacing = 0.5 * D_rotor * spacing_x
        horizontal_spacing = np.sqrt(3) * D_rotor * spacing_y

        # Lists to store the x and y coordinates of the grid points
        x_i = np.zeros([n_x, n_y])
        y_i = np.zeros([n_x, n_y])

        # Generate the coordinates of the hexagonal grid
        for row in range(n_y):
            for col in range(n_x):
                x = col * horizontal_spacing
                y = row * vertical_spacing

                # Shift every other row horizontally by half the spacing
                if row % 2 == 1:
                    x += horizontal_spacing / 2
                
                x_i[col, row] = x
                y_i[col, row] = y
    else:
        x_i, y_i = np.meshgrid(
            D_rotor * spacing_x * np.arange(0, n_x, 1),
            D_rotor * spacing_y * np.arange(0, n_y, 1),
        )

    return x_i, y_i


# Create uniform set of yaw and tilt angles
def create_uniform_angles(
        yaw: float, 
        tilt: float, 
        n_x: int, 
        n_y: int,
    ):
    yaw_i = yaw * np.ones([n_x, n_y])
    tilt_i = tilt * np.ones([n_x, n_y])

    return yaw_i, tilt_i


# Get misalignment angle
def get_misalignment_angle(
    yaw,
    tilt,
): 
    misalignment_angle = arccosd(cosd(yaw) * cosd(tilt))

    return misalignment_angle


# Get power correction factor for misalignment angle
def get_correction_factor_misalignment(
    yaw,
    tilt,
    a=0.9886719416000512,
    b=2.3649834828818133,
):
    misalignment_angle = get_misalignment_angle(
        yaw,
        tilt
    )

    correction_factor = a * np.abs(cosd(misalignment_angle))**b

    return correction_factor


# Get streamwise velocity profile
def U_profile(
    z, 
    z_ref, 
    U_ref, 
    alpha,
):
    z[z==0] = 1e-6
    U = U_ref * (z / z_ref)**alpha
    
    return U


# Get spanwise velocity profile
def V_profile(
    z, 
    a, 
    b, 
    c, 
    d
):
    z[z==0] = 1e-6
    V = a**(b * z + c) + d
    
    return V


def get_downwind_masks(
    wind_directions,
    layout,
    exclude_range: float = 90.,
    use_coordinates: bool = True,
):
    # NOTE: This function only works for farm layouts that are
    # created with the func.create_farm_layout function. 
    # TODO: Create function that automatically gets right mask 
    # for all kinds of layouts.
    
    # Get number of rows and columns
    x_i = layout['x_i']
    y_i = layout['y_i']
    n_x = layout['n_x']
    n_y = layout['n_y']
    n_turbines = n_x * n_y

    # If wind direction is single value, make list
    try:
        len(wind_directions)
    except:
        wind_directions = [wind_directions]

    # Create mask to exclude downwind turbines
    downwind_masks = np.zeros((len(wind_directions), n_turbines), dtype=bool)

    for idw, wd in enumerate(wind_directions):
        # For bottom column (wd = 0°)
        if wd % 360 >= 360 - exclude_range / 2 or wd % 360 <= 0 + exclude_range / 2:
            if use_coordinates:
                downwind_masks[idw] = (downwind_masks[idw]) | (y_i == np.min(y_i))
            else:
                downwind_masks[idw, :n_x] = True
        
        # For left row (wd = 90°)
        if wd % 360 >= 90 - exclude_range / 2 and wd % 360 <= 90 + exclude_range / 2:
            if use_coordinates:
                downwind_masks[idw] = (downwind_masks[idw]) | (x_i == np.min(x_i))
            else:
                for turb in range(n_turbines):
                    if turb % n_x == 0:
                        downwind_masks[idw, turb] = True

        # For top column (wd = 180°)
        if wd % 360 >= 180 - exclude_range / 2 and wd % 360 <= 180 + exclude_range / 2:
            if use_coordinates:
                downwind_masks[idw] = (downwind_masks[idw]) | (y_i == np.max(y_i))
            else:
                downwind_masks[idw, -n_x:] = True

        # For right row (wd = 270°)
        if wd % 360 >= 270 - exclude_range / 2 and wd % 360 <= 270 + exclude_range / 2:
            if use_coordinates:
                downwind_masks[idw] = (downwind_masks[idw]) | (x_i == np.max(x_i))
            else:
                for turb in range(n_turbines):
                    if turb % n_x == n_x - 1:
                        downwind_masks[idw, turb] = True

    return downwind_masks