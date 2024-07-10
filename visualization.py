import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions as func
import math

from matplotlib.animation import FuncAnimation
from import_model import WakeModeling


# Colors (TouchWind orange, darkblue, blue, lightblue)
colors = ['#F79123', '#014043', '#059589', '#19F7E2']
colorgrad = [
    '#014043',
    '#034C4E',
    '#045858',
    '#066563',
    '#07716D',
    '#097D78',
    '#0B8983',
    '#0C958D',
    '#0EA298',
    '#0FAEA2',
    '#11BAAD',
    '#13C6B8',
    '#14D2C2',
    '#16DFCD',
    '#17EBD7',
    '#19F7E2',
]


# Get color schemes
def get_color_schemes():
    return colors, colorgrad


# Plot farm layout 
def plot_farm_layout(x, y):
    fig = plt.figure()
    fig.set_size_inches(6, 4)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot()
    ax.scatter(x / 1000, y / 1000, s=200, marker='2', color=colors[1])
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_axisbelow(True)
    ax.grid()
    ax.set_aspect('equal')
    spacing = 500
    ax.set_xlim(((np.min(x)-spacing)/1000, (np.max(x)+spacing)/1000))
    ax.set_ylim(((np.min(y)-spacing)/1000, (np.max(y)+spacing)/1000))
    plt.show()

def plot_turbine_powers(
    turbine_powers,
    figsize=(12, 4)
):
    n_turbines = len(turbine_powers)
    
    fig = plt.figure()
    fig.set_size_inches(figsize)
    ax = fig.add_subplot(1, 1, 1)
    x_locs = np.arange(n_turbines)
    width = 0.7
    bars = ax.bar(x_locs, turbine_powers/1e6, width=width, color=colors[0])
    for turb, bar in enumerate(bars):
        yval = bar.get_height() - 0.5
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, np.round(turbine_powers[turb]/1e6, 2), ha='center', va='bottom', rotation=90)
    ax.set_xticks(x_locs)
    ax.set_xticklabels(["T{0}".format(i) for i in range(n_turbines)])
    ax.set_ylabel("Power [MW]")
    ax.set_axisbelow(True)
    ax.grid(axis='y')
    ax.set_ylim(0, 5.5)
    plt.show()


# PLot velocity field of X, Y, or Z plane at certain distances
def plot_velocity_field(
    wakemodeling: WakeModeling,
    config: dict,
    component: str = 'U',
    plane: str = 'X',
    distance: float = 500.,
    idw: int = 0,
    ids: int = 0,
    x_coords: np.ndarray = None,
    y_coords: np.ndarray = None,
    z_coords: np.ndarray = None,
    x_resolution: int = 100,
    y_resolution: int = 100,
    z_resolution: int = 100,
    U_bounds: tuple = (4, 10),
    offset: float = 500.,
    fig_size: tuple = (6, 4),
    shrink: float = 0.5,
    levels: int = 200,
):    
    # Make distance a float
    distance = float(distance)

    # Get coordinates
    coordinates = {}

    if x_coords == None:
        coordinates['X'] = np.linspace(np.min(config['x_ij']) - offset, np.max(config['x_ij']) + offset, x_resolution)
    elif len(x_coords) == 2:
        coordinates['X'] = np.linspace(x_coords[0], x_coords[1], x_resolution)
    else:
        coordinates['X'] = x_coords

    if y_coords == None:
        coordinates['Y'] = np.linspace(np.min(config['y_ij']) - offset, np.max(config['y_ij']) + offset, y_resolution)
    elif len(x_coords) == 2:
        coordinates['Y'] = np.linspace(y_coords[0], y_coords[1], y_resolution)
    else:
        coordinates['Y'] = y_coords

    if z_coords == None:
        coordinates['Z'] = np.linspace(0, 300, z_resolution)
    elif len(x_coords) == 2:
        coordinates['Z'] = np.linspace(z_coords[0], z_coords[1], z_resolution)
    else:
        coordinates['Z'] = z_coords

    # Set direction of plane only to distance of plane
    coordinates[plane] = np.array([distance])

    # Only calculate velocity field of given wind conditions
    config_copy = config.copy()

    config_copy['wind_directions'] = [config_copy['wind_directions'][idw]]
    config_copy['wind_speeds'] = [config_copy['wind_speeds'][ids]]

    # Calculate velocity field
    velocity_field = wakemodeling.get_velocity_field(
        config_copy,
        coordinates,
    )

    # Ensure the right plane and data is plotted
    if plane == 'X':
        x_axis, y_axis = 'Y', 'Z'
        data = velocity_field[component][0, :, :].T
    if plane == 'Y':
        x_axis, y_axis = 'X', 'Z'
        data = velocity_field[component][:, 0, :].T
    if plane == 'Z':
        x_axis, y_axis = 'X', 'Y'
        data = velocity_field[component][:, :, 0].T
    
    # Create plot
    fig, ax = plt.subplots()
    fig.set_size_inches(fig_size)
    ax.contourf(coordinates[x_axis], coordinates[y_axis], 
                data, 
                vmin=U_bounds[0], vmax=U_bounds[1], levels=levels, cmap='jet')
    
    # Create colorbar
    cbar_U = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=U_bounds[0], vmax=U_bounds[1]))
    cbar_U.set_array([])
    cbar_U = plt.colorbar(cbar_U, ax=ax, shrink=shrink, location='right')
    cbar_U.set_label(f'{component} [m/s]')

    # Plot settings, titles and labels
    ax.set_aspect('equal')
    ax.set_xlabel(f'{x_axis.lower()} [m]')
    ax.set_ylabel(f'{y_axis.lower()} [m]')
    # ax.set_title(f'Wind direction: { np.round(farm_config['wind_direction']) }°')
    plt.show()