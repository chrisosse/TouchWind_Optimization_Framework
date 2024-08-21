import numpy as np
import pandas as pd
import scipy as sc
from scipy import optimize
import math

def sind(angle: float,):
    '''
    Get sine with input angle in degrees

    Parameters
    ----------
    angle : float
        The angle [degrees]

    Returns
    -------
    sine(angle) : float
        The sine value
    '''
    return np.sin(np.radians(angle))


def cosd(angle: float,):
    '''
    Get cosine with input angle in degrees

    Parameters
    ----------
    angle : float
        The angle [degrees]

    Returns
    -------
    cosine(angle) : float
        The cosine value
    '''
    return np.cos(np.radians(angle))


def tand(angle: float,):
    '''
    Get tangent with input angle in degrees

    Parameters
    ----------
    angle : float
        The angle [degrees]

    Returns
    -------
    tangent(angle) : float
        The tangent value
    '''
    return np.tan(np.radians(angle))


def arcsind(value: float,):
    '''
    Get the angle in degrees of the inverse sine

    Parameters
    ----------
    value : float
        A value

    Returns
    -------
    arcsine(value) : float
        An angle [degrees]
    '''
    return np.degrees(np.arcsin(value))


def arccosd(angle: float,):
    '''
    Get the angle in degrees of the inverse cosine

    Parameters
    ----------
    value : float
        A value

    Returns
    -------
    arccosine(value) : float
        An angle [degrees]
    '''
    return np.degrees(np.arccos(angle))


def arctan2d(
    y: float, 
    x: float,
):
    '''
    Get the angle in degrees of the inverse tangent in 2D plane

    Parameters
    ----------
    y : float
        Value in y-direction
    x : float
        Value in y-direction

    Returns
    -------
    arctan2(y, x) : float
        An angle [degrees]
    '''
    return np.degrees(np.arctan2(y, x))


def find_nearest(
    array: np.ndarray, 
    value: float,
):
    '''
    Find the nearest ID and value in a list to a given value

    Parameters
    ----------
    array : np.ndarray
        array containing all values
    value : float
        value to find the closest to in the list

    Returns
    -------
    idx : int
        The ID of the closest value in the list
    value:
        The actual value in the list which is closest
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    value = array[idx]
    return idx, value


def create_layout(
    shape: str = 'rectangular',
    n_x: int = 1,
    n_y: int = 1,
    spacing_x: float = 5.,
    spacing_y: float = 5.,
    D_rotor: float = 126.,
):
    '''
    Create a farm layout which can either be rectangular or hexagonal.

    Parameters
    ----------
    shape : str, optional
        The shape the farm, either 'rectangular' or 'hexagonal', 
        by default 'rectangular'
    n_x : int, optional
        The number of turbines in the x-direction, by default 1
    n_y : int, optional
        The number of turbines in the y-direction, by default 1
    spacing_x : float, optional
        The turbine spacing [D] in the x-direction, by default 5.
    spacing_y : float, optional
        The turbine spacing [D] in the x-direction, by default 5.
    D_rotor : float, optional
        The rotor diameter [m], by default 126.

    Returns
    -------
    x_i : np.ndarray
        An array containing the x-coordinates [m] of the turbines
    y_i : np.ndarray
        An array containing the y-coordinates [m] of the turbines
    '''
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


def create_uniform_angles(
    yaw: float = 0., 
    tilt: float = 0., 
    n_x: int = 1, 
    n_y: int = 1,
):
    '''
    Create uniform angles for all turbines

    Parameters
    ----------
    yaw : float
        The yaw angle [degrees] of all turbines, by default 0.
    tilt : float
        The tilt angle [degrees] of all turbines, by default 0.
    n_x : int
        The number of turbines in the x-direction, by default 1
    n_y : int
        The number of turbines in the y-direction, by default 1

    Returns
    -------
    yaw_i : np.ndarray
        An array containing the yaw angles [degrees] of the turbines
    tilt_i : np.ndarray
        An array containing the tilt angles [degrees] of the turbines
    '''
    yaw_i = yaw * np.ones([n_x, n_y])
    tilt_i = tilt * np.ones([n_x, n_y])

    return yaw_i, tilt_i


def get_misalignment_angle(
    yaw: float | np.ndarray,
    tilt: float | np.ndarray,
): 
    '''
    Get the misalignment angle based on the yaw and tilt angle

    Parameters
    ----------
    yaw : float | np.ndarray
        Yaw angle [degrees] value(s)
    tilt : float | np.ndarray
        Tilt angle [degrees] value(s)

    Returns
    -------
    misalignment_angle : float | np.ndarray,
        Misalignment angle [degrees] value(s)
    '''
    misalignment_angle = arccosd(cosd(yaw) * cosd(tilt))

    return misalignment_angle


def get_correction_factor_misalignment(
    yaw: float | np.ndarray,
    tilt: float | np.ndarray,
    a: float = 0.9886719416000512,
    b: float = 2.3649834828818133,
):
    '''
    Return correction factor for rotor misalignment, 
    to be used to correct the turbine powers

    Parameters
    ----------
    yaw : float | np.ndarray
        Yaw angle [degrees] value(s)
    tilt : float | np.ndarray
        Tilt angle [degrees] value(s)
    a : float, optional
        Fitted value, by default 0.9886719416000512
    b : float, optional
        Fitted value, by default 2.3649834828818133

    Returns
    -------
   correction_factor : float | np.ndarray
        The correction factor(s) [-]
    '''
    misalignment_angle = get_misalignment_angle(
        yaw,
        tilt
    )

    correction_factor = a * np.abs(cosd(misalignment_angle))**b

    return correction_factor


def U_profile(
    z: float | np.ndarray, 
    z_ref: float, 
    U_ref: float, 
    alpha: float, 
):
    '''
    Get the streamwise velocity profile

    Parameters
    ----------
    z : float | np.ndarray
        Value(s) in the height [m]
    z_ref : float
        Reference height [m]
    U_ref : float
        Reference velocity at reference height [m/s]
    alpha : float
        Value [-]

    Returns
    -------
    U : float | np.ndarray
        Velocity(s) in the height [m]
    '''
    z[z==0] = 1e-6
    U = U_ref * (z / z_ref)**alpha
    
    return U


def V_profile(
    z: float | np.ndarray, 
    a: float,  
    b: float,  
    c: float,  
    d: float, 
):
    '''
    Get the spanwise velocity profile

    Parameters
    ----------
    z : float | np.ndarray
        Value(s) in the height [m]
    a : float
        Value [-]
    b : float
        Value [-]
    c : float
        Value [-]
    d : float
        Value [-]

    Returns
    -------
    V : float | np.ndarray
        Velocity(s) in the height [m]
    '''
    z[z==0] = 1e-6
    V = a**(b * z + c) + d
    
    return V


def get_coordinate_grid(
    layout: dict,
    resolution: int = 100,
    offset: float = 150.,
):
    '''
    _summary_

    Parameters
    ----------
    case : Case
        Instance of a case
    resolution : int, optional
        Resolution in all directions, by default 100
    offset : float, optional
        Offset from side turbines in x- and y-direction, by default 150.

    Returns
    -------
    coordinates : dict
        Dictionary containing coordinates in x-, y- and z-directions
    '''
    coordinates = {
        'X': np.linspace(np.min(layout['x_i']) - offset, np.max(layout['x_i']) + offset, resolution),
        'Y': np.linspace(np.min(layout['x_i']) - offset, np.max(layout['x_i']) + offset, resolution),
        'Z': np.linspace(1e-6, 300, round(300/15))
    }

    return coordinates


def get_downwind_masks(
    wind_directions: float | np.ndarray,
    layout: dict,
    exclude_range: float = 90.,
    use_coordinates: bool = True,
):
    '''
    Create masks that indicate which turbines are located downwind 
    for all given wind directions. This function only works if:
        
    - The layout is rectangular and similar to ones made with the create_layout
    function. This means turbine rows (in x-direction) are placed after eachother.
    - The layout is rectangular and rows and columns share identical coordinates.

    Parameters
    ----------
    wind_directions : float | np.ndarray
        Wind direction(s) [degrees]
    layout : dict
        The dictionary containing the layout of the wind farm
    exclude_range : float, optional
        The wind direction range in which turbines are deemed
        downwind, by default 90.
    use_coordinates : bool, optional
        Indicator whether to use coordinates instead of standard way
        of construction of farm layout (i.e. difference between point 
        1 and 2 in function description), by default True

    Returns
    -------
    downwind_masks : 2D np.ndarray
        Mask(s) for the given wind direction(s) indicating whether 
        a turbine is located downwind. True indicates the turbine 
        being downwind. Shape: (idw, turb)
    '''
    # Get number of rows and columns
    x_i = layout['x_i'].flatten()
    y_i = layout['y_i'].flatten()
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
        if (wd % 360 >= 360 - exclude_range / 2 or wd % 360 <= 0 + exclude_range / 2) and n_y > 1:
            if use_coordinates:
                downwind_masks[idw] = (downwind_masks[idw]) | (y_i == np.min(y_i))
            else:
                downwind_masks[idw, :n_x] = True
        
        # For left row (wd = 90°)
        if (wd % 360 >= 90 - exclude_range / 2 and wd % 360 <= 90 + exclude_range / 2) and n_x > 1:
            if use_coordinates:
                downwind_masks[idw] = (downwind_masks[idw]) | (x_i == np.min(x_i))
            else:
                for turb in range(n_turbines):
                    if turb % n_x == 0:
                        downwind_masks[idw, turb] = True

        # For top column (wd = 180°)
        if (wd % 360 >= 180 - exclude_range / 2 and wd % 360 <= 180 + exclude_range / 2) and n_y > 1:
            if use_coordinates:
                downwind_masks[idw] = (downwind_masks[idw]) | (y_i == np.max(y_i))
            else:
                downwind_masks[idw, -n_x:] = True

        # For right row (wd = 270°)
        if (wd % 360 >= 270 - exclude_range / 2 and wd % 360 <= 270 + exclude_range / 2) and n_x > 1:
            if use_coordinates:
                downwind_masks[idw] = (downwind_masks[idw]) | (x_i == np.max(x_i))
            else:
                for turb in range(n_turbines):
                    if turb % n_x == n_x - 1:
                        downwind_masks[idw, turb] = True

    return downwind_masks