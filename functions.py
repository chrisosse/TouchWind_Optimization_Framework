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
def create_farm_layout(
    D_rotor: float = 126,
    n_x: int = 1,
    n_y: int = 1,
    spacing_x: float = 5,
    spacing_y: float = 5,
    hexagonal: bool = False,
    spacing_hex: float = 5,
):
    if hexagonal:
        # Calculate the vertical and horizontal spacing between hexagon centers
        vertical_spacing = 0.5 * D_rotor * spacing_hex
        horizontal_spacing = np.sqrt(3) * D_rotor * spacing_hex

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


# # Create farm configuration
# def create_farm_config(
#     case: dict,
#     extra_layers: int = 0,
# ):
#     # Get number of turbines
#     n_x = int(case['n_x'])
#     n_y = int(case['n_y'])
#     n_turbines = n_x * n_y

#     # Initialize farm layout
#     x_i = np.zeros((n_x, n_y + extra_layers))
#     y_i = np.zeros((n_x, n_y + extra_layers))
#     yaw_i = np.zeros((n_x, n_y + extra_layers))
#     tilt_i = np.zeros((n_x, n_y + extra_layers))

#     # Get x and y positions of turbines
#     if case['equal']:
#         spacing_x = case['spacing_x'] * case['D_rotor']
#         spacing_y = case['spacing_y'] * case['D_rotor']
        
#         if case['hexagonal']:
#             x_i[0, 0] = case['x_0']
#             y_i[0, 0] = case['y_0']

#             x_i[1, 0] = x_i[0, 0] + cosd(30) * spacing_x
#             y_i[1, 0] = y_i[0, 0] + sind(30) * spacing_y

#             for i in range(2, n_x):
#                 x_i[i, 0] = x_i[i-2, 0] + cosd(30) * spacing_x * 2
#                 y_i[i, 0] = y_i[i-2, 0]

#             for i in range(1, n_y + extra_layers):
#                 x_i[:, i] = x_i[:, 0]
#                 y_i[:, i] = y_i[:, i-1] + sind(30) * spacing_y * 2

#         else:
#             for i in range(n_x):
#                 for j in range(0, n_y + extra_layers):
#                     x_i[i, j] = case['x_0'] + spacing_x * i
#                     y_i[i, j] = case['y_0'] + spacing_y * j

#         for i in range(n_x):
#             for j in range(0, n_y + extra_layers):
#                 yaw_i[i, j] = case['yaw_0']
#                 tilt_i[i, j] = case['tilt_0']
#     else:
#         for i in range(n_turbines):
#             x_i[0, i] = case['x_' + str(i)]
#             y_i[0, i] = case['y_' + str(i)]
#             yaw_i[0, i] = case['yaw_' + str(i)]
#             tilt_i[0, i] = case['tilt_' + str(i)]

#     # Create farm layout dictionary
#     farm_config = {
#         'wind_directions': np.array([case['wd']]),
#         'wind_speeds': np.array([case['U_ref']]),
#         'x_i': x_i,
#         'y_i': y_i,
#         'yaw_i': yaw_i,
#         'tilt_i': tilt_i,
#         'D_rotor': case['D_rotor'],
#         'n_x': n_x,
#         'n_y': n_y,
#         'n_turbines': n_turbines,
#     }

#     return farm_config


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