import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from cases import Case

# Some of the nicest colorschemes you've ever seen
#   (TouchWind orange, darkblue, blue, lightblue):
colors = ['#F79123', '#014043', '#059589', '#19F7E2']
#   (TouchWind darkblue to lightblue)
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


def get_color_schemes():
    '''
    Get some of the most beautiful colorschemes you've
    ever seen.

    Returns
    -------
    _type_
        _description_
    '''
    return colors, colorgrad


def plot_farm_layout(
    x: np.ndarray, 
    y: np.ndarray,
    spacing: float = 500.,
    figsize: tuple = (6, 4),
):
    '''
    Get a plot of the wind farm layout.

    Parameters
    ----------
    x : np.ndarray
        1D or 2D array containing the x-coordinates [m] of the turbines
    y : np.ndarray
        1D or 2D array containing the y-coordinates [m] of the turbines
    spacing : float, optional
        Spacing around all turbines so turbines are not at edge of plot, 
        by default 500.
    figsize : tuple, optional
        Figure size where first value is width ["] and second is height 
        ["], by default (6, 4)

    Returns
    -------
    fig : plt.figure.Figure
        Plot of the wind farm layout
    '''
    fig = plt.figure()
    fig.set_size_inches(figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot()
    ax.scatter(x / 1000, y / 1000, s=200, marker='2', color=colors[1])
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_axisbelow(True)
    ax.grid()
    ax.set_aspect('equal')
    ax.set_xlim(((np.min(x)-spacing)/1000, (np.max(x)+spacing)/1000))
    ax.set_ylim(((np.min(y)-spacing)/1000, (np.max(y)+spacing)/1000))
    
    return fig


def plot_turbine_powers(
    turbine_powers: np.ndarray,
    width: float = 0.7,
    show_values: bool = True,
    figsize: tuple = (6, 4),
):
    '''
    Get a bar plot of all turbine plots.

    Parameters
    ----------
    turbine_powers : np.ndarray
        1D array of turbine powers [W]
    case : Case, optional
        The case containing values
    width : float, optional
        The width of the bars, by default 0.7
    show_values : bool, optional
        Indicates whether to show values above bars, by default True
    figsize : tuple, optional
        Figure size where first value is width ["] and second is height 
        ["], by default (12, 4)

    Returns
    -------
    fig : plt.figure.Figure
        Barplot of the turbine powers
    '''
    # Get number of turbines
    n_turbines = len(turbine_powers)
    
    fig = plt.figure()
    fig.set_size_inches(figsize)
    ax = fig.add_subplot(1, 1, 1)
    x_locs = np.arange(n_turbines)
    bars = ax.bar(x_locs, turbine_powers/1e6, width=width, color=colors[0])
    
    # Show turbine power values above bars
    if show_values: 
        for turb, bar in enumerate(bars):
            yval = bar.get_height() - 0.5
            ax.text(bar.get_x() + bar.get_width()/2, yval + 1, 
                    np.round(turbine_powers[turb]/1e6, 2), 
                    ha='center', va='bottom', rotation=90)
            
    ax.set_xticks(x_locs)
    ax.set_xticklabels(['T{0}'.format(i) for i in range(n_turbines)])
    ax.set_ylabel('Power [MW]')
    ax.set_axisbelow(True)
    ax.grid(axis='y')
    ax.set_ylim(0, 5.5)
    
    return fig


def plot_velocity_field(
    model,
    case: Case,
    component: str = 'U',
    plane: str = 'X',
    distance: float = None,
    idw: int = 0,
    ids: int = 0,
    x_coords: np.ndarray = None,
    y_coords: np.ndarray = None,
    z_coords: np.ndarray = None,
    x_resolution: int = 100,
    y_resolution: int = 100,
    z_resolution: int = 100,
    bounds: tuple = None,
    offset: float = None,
    shrink: float = 0.5,
    levels: int = 200,
    title: bool = False,
    fig_size: tuple = (6, 4),
):  
    '''
    _summary_

    Parameters
    ----------
    model : instance of model class
        An instance of a model class
    case : Case
        A case containing wind farm specifications
    component : str, optional
        The wind velocity component to show, by default 'U'
    plane : str, optional
        The plane in which to show the velocity, by default 'X'
    distance : float, optional
        Distance from the origin to show the plane set 
        automatically when None, by default None.
    idw : int, optional
        The ID of the wind direction to show, by default 0
    ids : int, optional
        The ID of the wind speed to show, by default 0
    x_coords : np.ndarray, optional
        The x-coordinates [m] of the grid, indicating where to 
        calculate the wind velocity. When None automatically set 
        to include all turbines, when two values set to that range 
        with given resolution, when list set to that list. 
        By default None
    y_coords : np.ndarray, optional
        The y-coordinates [m] of the grid. Furthermore equal to 
        x_coords, by default None
    z_coords : np.ndarray, optional
        The z-coordinates [m] of the grid. Furthermore equal to 
        x_coords, by default None
    x_resolution : int, optional
        Resolution in the x-direction, by default 100
    y_resolution : int, optional
        Resolution in the y-direction, by default 100
    z_resolution : int, optional
        Resolution in the z-direction, by default 100
    bounds : tuple, optional
        Boundaries of velocity in plot, if None set automatically 
        based on component. By default None
    offset : float, optional
        Offset to show larger range than only turbines, if None 
        set automatically based on plane. By default None
    shrink : float, optional
        Size of velocity colorbar, by default 0.5
    levels : int, optional
        Number of velocity levels in plot, by default 200
    title : bool, optional
        Indicator to set a title including the wind direction, 
        by default False
    fig_size : tuple, optional
        Figure size where first value is width ["] and second is height 
        ["], by default (6, 4)

    Returns
    -------
    fig : plt.figure.Figure
        Contourplot of a plane in the wind farm domain

    Raises
    ------
    ValueError
        Raises error when wrong type of component is given
    ValueError
        Raises error when wrong type of plane is given
    '''
    # State valid components and planes
    valid_components = ['U', 'V', 'W']
    valid_planes = ['X', 'Y', 'Z']

    # Raise error when no valid component or plane
    if component not in valid_components:
        raise ValueError(
            f'Invalid component value: {component}. Valid component are {valid_components}'
        )
    if plane not in valid_planes:
        raise ValueError(
            f'Invalid plane value: {plane}. Valid planes are {valid_planes}'
        )

    # Set distance and offset according to plane
    if plane == 'X':
        distance = 500
        offset = 500
    elif plane == 'Y':
        distance = 0
        offset = 500
    else:
        distance = 90
        offset = 500   

    # Set bounds if no bounds set
    if bounds == None:
        if component == 'U':
            bounds = (4, 10)
        else:
            bounds = (-1.5, 1.5)
    
    # Make distance a float
    distance = float(distance)

    # Get x and y coordinates of turbines
    x_i = case.layout['x_i']
    y_i = case.layout['y_i']

    # Get coordinates
    coordinates = {}

    if type(x_coords) == type(None):
        coordinates['X'] = np.linspace(np.min(x_i) - offset, np.max(x_i) + offset, x_resolution)
    elif len(x_coords) == 2:
        coordinates['X'] = np.linspace(x_coords[0], x_coords[1], x_resolution)
    else:
        coordinates['X'] = x_coords

    if type(y_coords) == type(None):
        coordinates['Y'] = np.linspace(np.min(y_i) - offset, np.max(y_i) + offset, y_resolution)
    elif len(y_coords) == 2:
        coordinates['Y'] = np.linspace(y_coords[0], y_coords[1], y_resolution)
    else:
        coordinates['Y'] = y_coords

    if type(z_coords) == type(None):
        coordinates['Z'] = np.linspace(0, 300, z_resolution)
    elif len(z_coords) == 2:
        coordinates['Z'] = np.linspace(z_coords[0], z_coords[1], z_resolution)
    else:
        coordinates['Z'] = z_coords

    # Set direction of plane only to distance of plane
    coordinates[plane] = np.array([distance])

    # Only calculate velocity field of given wind conditions
    case_copy = copy.deepcopy(case)

    case_copy.conditions['directions'] = [case_copy.conditions['directions'][idw]]
    case_copy.conditions['speeds'] = [case_copy.conditions['speeds'][ids]]

    # Calculate velocity field
    velocity_field = model.get_velocity_field(
        case_copy,
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
                vmin=bounds[0], vmax=bounds[1], levels=levels, cmap='jet')
    
    # Create colorbar
    cbar_U = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=bounds[0], vmax=bounds[1]))
    cbar_U.set_array([])
    cbar_U = plt.colorbar(cbar_U, ax=ax, shrink=shrink, location='right')
    cbar_U.set_label(f'{component} [m/s]')

    # Invert x axis in plane is X
    if plane == 'X':
        plt.gca().invert_xaxis()

    # Plot settings, titles and labels
    ax.set_aspect('equal')
    ax.set_xlabel(f'{x_axis.lower()} [m]')
    ax.set_ylabel(f'{y_axis.lower()} [m]')
    if title:
        direction = np.round(case.conditions['directions'][idw], 2)
        ax.set_title(f'Wind direction: {direction}°')
    
    return fig