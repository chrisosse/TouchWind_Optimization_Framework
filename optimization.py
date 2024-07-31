import numpy as np
import pandas as pd
import time
from IPython.display import clear_output
from cases import (
    Case,
    CaseManager,
)

class Optimization:
    '''
    Class used for optimization, which included an instance of the case
    to optimize, and an instance of the model used to run the case.
    '''
    def __init__(
        self,
        model,
        case: Case,
        wd: float,
        vars: np.ndarray = np.array(['yaw', 'tilt']),
        mask:  np.ndarray = None,
        method: str = 'SLSQP',
        penalty: bool = False,
        total_power_conv: float = 0.,
        downwind_values: dict = {'yaw': 0, 'tilt': 5},
        n_particles: int = 1,
        n_discretized_wds: int = 1,
        sigma_wd: float = 2.5,
    ):
        '''
        _summary_

        Parameters
        ----------
        model : Instance of model
            An instance of the model to be used to run case
        case : Case
            Instance of a case
        wd : float
            Wind direction [degrees]
        vars : np.ndarray, optional
            Array containing variable names to be optimized, 
            by default np.array(['yaw', 'tilt'])
        mask : np.ndarray, optional
            Array with the mask of turbines to exclude from optimization, 
            by default None
        method : str, optional
            The optimization method to be used, by default 'SLSQP'
            (Sequential Least SQuares Programming)
        penalty : bool, optional
            Indicater for whether to apply a penalty if total power is lower
            than total conventional power, by default False
        total_power_conv : float, optional
            Total conventional power [W], by default 0.
        downwind_values : _type_, optional
            Dictionary containing values for downwind or excluded turbines,
            by default {'yaw': 0, 'tilt': 5}
        n_particles : int, optional
            Number of particles [-], used in some optimization algorithms, 
            by default 1
        n_discretized_wds : int, optional
            Number of wind directions to take into account in probabilistic method
            for accounting for wind direction uncertainty, by default 1
        sigma_wd : float, optional
            Standard deviation [degrees] for wind direction uncertainty, 
            by default 2.5
        '''
        # Initialize variables
        self.model = model
        self.case = case
        self.wd = wd
        self.vars = vars
        self.mask = mask
        self.method = method
        self.penalty = penalty
        self.total_power_conv = total_power_conv
        self.downwind_values = downwind_values
        self.n_particles = n_particles
        self.n_discretized_wds = n_discretized_wds
        self.sigma_wd = sigma_wd
        self.function_values = []

        # Set discretized wind directions for solution robustness
        self.set_wind_directions()

        # Set case wind conditions to right wind conditions
        self.case.conditions['directions'] = self.wind_directions
        
        # Reinitialize wind farm
        self.model.reinitialize_farm(
            self.case.conditions,
            self.case.layout,
        )


    def set_wind_directions(
        self,      
    ):
        '''
        Set wind directions based on probabilistic method and number of
        particles.
        '''
        # Get discretized wind directions to take into account
        # TODO: Now, only 1 degree is in between directions. This should
        # be changable or good value should be found.
        self.discretized_wds = np.linspace(
            self.wd - ((self.n_discretized_wds - 1) / 2), 
            self.wd + ((self.n_discretized_wds - 1) / 2), 
            self.n_discretized_wds,
        )

        # Get the probabilities of each wind direction based on Gaussian distribution
        p_wds = 1 / (self.sigma_wd * np.sqrt(2 * np.pi)) * \
            np.exp(-(self.discretized_wds - self.wd)**2 / (2 * self.sigma_wd**2))
        self.p_wds = 1 / np.sum(p_wds) * p_wds

        # Get the list of wind directions for all particles
        wind_directions = []
        for wd in self.discretized_wds:
            wind_directions += [wd] * self.n_particles

        self.wind_directions = np.array(wind_directions)
    

    def get_all_variables(
        self,
        input: np.ndarray,
    ):
        '''
        Get the values for all wind turbines, and not only the 
        optimized turbines

        Parameters
        ----------
        input : np.ndarray
            Array containing the input values from optimization algorithm

        Returns
        -------
        variables : dict
            Dictionary containing arrays with values for each variable 
            which is optimized
        '''
        # Get number of turbines
        n_turbines = self.case.layout['n_turbines']

        # If no downwind mask
        if type(self.mask) == type(None):
            self.mask = np.zeros((n_turbines), dtype=bool)

        # Get number of turbines to optimize
        n_turbines_opt = len(self.mask) - sum(self.mask)

        # Initialize variables
        variables = {}

        # Loop over all optimizable parameters
        for idv, var in enumerate(self.vars):
            variables[var] = np.ones((self.n_particles, 1, n_turbines)) * self.downwind_values[var]

            # Initialize counter
            counter = 0

            # Fill with input values if not excluded by mask
            for turb in range(n_turbines):
                if not self.mask[turb]:
                    variables[var][:, :, turb] = input[idv * n_turbines_opt + counter]
                    counter += 1

        return variables
    

    def get_variables_for_wds(
        self,
        variables: dict,
    ):
        '''
        Get the arrays of variables for all wind directions taken into 
        account for the probabilistic wind direction uncertainty method

        Parameters
        ----------
        variables : dict
            Dictionary containing arrays with values for each variable 
            which is optimized

        Returns
        -------
        variables : dict
            Dictionary containing arrays with values for each variable
            and all wind directions which is optimized
        '''
        # Stack as many arrays as there are wind directions taken into account
        for var in self.vars:
            variables[var] = np.concatenate([variables[var]] * self.n_discretized_wds)
        
        # Correct yaw for rotated wind
        # TODO: fix that yaw also accounted for when not optimizing yaw
        if 'yaw' in self.vars:
            for idw, wd in enumerate(self.wind_directions):
                start = idw * self.n_particles
                end = (idw + 1) * self.n_particles
                variables['yaw'][start:end] = variables['yaw'][start:end] - (wd - self.wd)

        return variables
    

    def cost_function(
        self,
        input: np.ndarray,
    ):
        '''
        The cost function which should be minimized by the 
        algorithm. In this case, we minimize the negative value
        of the total power. A penalty can be given when total power 
        is lower than conventional total power.

        Parameters
        ----------
        input : np.ndarray
            Array containing the input values from optimization algorithm

        Returns
        -------
        -total power
            Negative total power of the wind farm
        '''
        # Get variables for all wind turbines
        variables = self.get_all_variables(input)

        # Get variables for all wind directions
        variables = self.get_variables_for_wds(variables)

        # Add variables to turbines in case
        for var in self.vars:
            self.case.turbines[f'{var}_i'] = variables[var]
        
        # Get turbine powers
        turbine_powers = self.model.get_turbine_powers(
            self.case.turbines,
        )

        # Get total power per wind direction 
        # (NOTE: Only 1 wind speed should be taken into account)
        total_powers = np.sum(turbine_powers, axis=(1, 2))

        # Initialize weighted average total powers
        weighted_avg_total_powers = np.zeros(self.n_particles)
        
        # Get weighted average total power per particle based on probability distribution
        for idw in range(self.n_discretized_wds):
            start = idw * self.n_particles
            end = (idw + 1) * self.n_particles

            weighted_avg_total_powers += total_powers[start:end] * self.p_wds[idw]

            # If penalty, get the power differences with conventional power
            if self.penalty:
                if idw == round(self.n_discretized_wds / 2):
                    power_differences = self.total_power_conv - total_powers[start:end]

        # If penalty, penalize if power is lower than conventional power
        if self.penalty:
            for p, power_difference in enumerate(power_differences):
                if power_difference > 0:
                    weighted_avg_total_powers[p] -= power_difference
        
        # Add function values, turbine and total powers to self
        self.function_values.append(weighted_avg_total_powers)
        self.turbine_powers = turbine_powers[int(np.floor(self.n_discretized_wds/2)), 0]
        self.total_power = np.sum(self.turbine_powers)

        # Return negative value because minimizing
        return -weighted_avg_total_powers


def get_optimization_dataframe(
    location: str,
    filename: str,
    optimization_parameters: np.ndarray,
    layout: dict,
):
    '''
    Get the initialized optimization results dataframe,
    containing columns for optimization data and optimized
    turbine settings

    Parameters
    ----------
    location : str
        Directory of optimization dataframe .csv file
    filename : str
        File name of optimization dataframe .csv file
    optimization_parameters : np.ndarray
        List of parameters which are optimized
    layout : dict
        Dictionary containing the layout of the wind farm

    Returns
    -------
    df_optimization : pd.DataFrame
        Optimization dataframe
    '''
    # Get number of turbines
    n_turbines = layout['n_turbines']
    
    # Try to load already existing file
    try:
        df_optimization = pd.read_csv(location + filename)
        
        print('Old data loaded')
    
    # otherwise create new dataframe
    except:
        data_turbines = ['power'] + optimization_parameters
        
        # Add optimization columns
        columns_wind_farm = [
            'wind_direction',
            'total_power',
            'total_power_conv',
            'time',
            'success',
            'message',
            'n_iterations',
            'function_values',
            'result',
        ]

        # Add turbine columns
        for col in data_turbines:
            columns_wind_farm = columns_wind_farm + \
                [f'{col}_{turb}' for turb in range(n_turbines)]
        
        # Create dataframe
        df_optimization = pd.DataFrame(columns=columns_wind_farm)

        # Set certain columns to objects to allow lists in single cells
        df_optimization['total_power'] = df_optimization['total_power'].astype('object')
        df_optimization['function_values'] = df_optimization['function_values'].astype('object')
        df_optimization['result'] = df_optimization['result'].astype('object')
        
        print('No old data loaded')

    return df_optimization


def get_result_bounds_guess_dicts(
    optimization_parameters: np.ndarray,
    downwind_masks: np.ndarray,
    bounds: dict,
    guesses: dict,
):
    '''
    Get a list of dictionaries containing the boundaries and guesses
    of all optimizable parameters per wind turbine for all wind directions

    Parameters
    ----------
    optimization_parameters : np.ndarray
        Array containing optimizable parameter names
    downwind_masks : np.ndarray
        Array containing all downwind turbine masks
    bounds : dict
        Dictionary with the parameter boundaries
    guesses : dict
        Dictionary with the parameter guesses

    Returns
    -------
    result_bounds_guess_dicts : list
        List of dictionaries with boundaries and guesses
    '''
    # If only one mask, make list of masks
    if len(np.shape(downwind_masks)) == 1:
        downwind_masks = [downwind_masks]

    # Initialize parameter bounds lists
    result_bounds_guess_dicts = []

    # Go over all masks
    for mask in downwind_masks:
        bounds_dict = {}

        # Go over all optimizable parameters
        for p in optimization_parameters:
            for turb, m in enumerate(mask):
                if not m:
                    bounds_dict[p + f'_{turb}'] = [(bounds[p][0], bounds[p][1]), guesses[p]]
        
        # Append new dictionary to list
        result_bounds_guess_dicts.append(bounds_dict)

    return result_bounds_guess_dicts


def get_turbine_powers_conv(
    model, 
    case: Case,
    wind_directions: np.ndarray,
    values_conv: dict,
):
    '''
    Get the conventional total powers for all wind directions

    Parameters
    ----------
    model : Instance of model
        Instance of the model to run the cases with
    case : Case
        Instance of a case
    wind_directions : np.ndarray
        Array of all wind directions to take into account
    values_conv : dict
        Dictionary containing the conventional turbine settings

    Returns
    -------
    total_powers_conv : np.ndarray
        Array containing the conventional powers for all wind directions
    '''
    # Set case parameters
    case.conditions['directions'] = wind_directions
    case.turbines['yaw_i'] = np.ones_like(case.turbines['yaw_i']) * values_conv['yaw']
    case.turbines['tilt_i'] = np.ones_like(case.turbines['tilt_i']) * values_conv['tilt']
    
    # Run model
    total_powers_conv = model.run(case)
    
    return total_powers_conv


def get_all_parameters(
    n_turbines: int,
    parameter_names: np.ndarray,
    parameters_optimized: np.ndarray,
    optimization_parameters: np.ndarray,
    downwind_values: dict,
):  
    '''
    Get a dictionary containing the values for all turbines,
    without the downwind turbines excluded

    Parameters
    ----------
    n_turbines : int
        Number of turbines
    parameter_names : np.ndarray
        Array containing all optimized parameter names
    parameters_optimized : np.ndarray
        Array containing the optimized parameter values
    optimization_parameters : np.ndarray
        Array containing optimizable parameter names
    downwind_values : dict
        Dictionary containing values for downwind or excluded turbines,

    Returns
    -------
    all_parameters : dict
        Dictionary containing optimizable parameters for all turbines
    '''
    # Initialize dictionary for all parameters
    all_parameters = {}

    # Loop over all optimizable parameters
    for p in optimization_parameters:
        # Loop over all turbines
        for turb in range(n_turbines):
            # Set all parameters to downwind values
            all_parameters[f'{p}_{turb}'] = downwind_values[p]

    # Loop over all optimized parameter names
    for idp, p in enumerate(parameter_names):
        # Set respective turbine to optimized value
        all_parameters[p] = parameters_optimized[idp]

    return all_parameters


def print_progress(
    start_time: float,
    idw: int,
    wind_directions: np.ndarray,
):  
    '''
    Function to print the progress of optimization

    Parameters
    ----------
    start_time : float
        The start time of optimization
    idw : int
        The current ID of the wind direction
    wind_directions : np.ndarray
        Array containing all wind directions
    '''
    # Clear output, but wait till new output is ready
    clear_output(wait=True)

    # Get end time
    end_time = time.time()

    # Get elapsed time and estimation of remaining time
    elapsed_time = end_time - start_time
    estimated_time_remaining = (elapsed_time / (idw + 1)) * (len(wind_directions) - (idw + 1))

    # Print some cool stuff
    print(f'Progress: {round(((idw + 1) / len(wind_directions)) * 100, 1)} %')
    print(f'    Mean time per optimization run: {round(elapsed_time / ((idw + 1) * 60), 1)} minutes')
    print(f'    Estimation: {round(estimated_time_remaining / 60)} minutes remaining')
    print(f'    Currently at wind direction {round(wind_directions[idw], 1)} ({idw+1} of {len(wind_directions)})')