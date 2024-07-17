import numpy as np
import pandas as pd
import time
from IPython.display import clear_output
from cases import (
    Case,
    CaseManager,
)

class Optimization:
    def __init__(
        self,
        model,
        case: Case,
        wd: float,
        vars: list = ['yaw', 'tilt'],
        mask: list = None,
        method: str = 'SLSQP',
        penalty: bool = False,
        total_power_conv: float = 0.,
        downwind_values: dict = {'yaw': 0, 'tilt': 5},
        n_particles: int = 1,
        n_discretized_wds: int = 1,
        sigma_wd: float = 2.5,
    ):
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

        # Ste discretized wind directions for solution robustness
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
        input,
    ):
        # Get number of turbines
        n_turbines = self.case.layout['n_turbines']

        # If no downwind mask
        if type(self.mask) == type(None):
            self.mask = np.zeros((n_turbines), dtype=bool)

        # Get number of turbines to optimize
        n_turbines_opt = len(self.mask) - sum(self.mask)

        # Initialize variables
        variables = {}

        for idv, var in enumerate(self.vars):
            variables[var] = np.ones((self.n_particles, 1, n_turbines)) * self.downwind_values[var]

            counter = 0

            # Fill with input values
            for turb in range(n_turbines):
                if not self.mask[turb]:
                    variables[var][:, :, turb] = input[idv * n_turbines_opt + counter]
                    counter += 1

        return variables
    

    def get_variables_for_wds(
        self,
        variables,
    ):
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
        input,
    ):
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

        # Get averaged over wds and wss total power
        total_power = np.sum(np.mean(turbine_powers, axis=(0, 1)))
        
        # Add function values, turbine and total powers to self
        self.function_values.append(total_power)
        self.turbine_powers = turbine_powers[int(np.floor(self.n_discretized_wds/2)), 0]
        self.total_power = np.sum(self.turbine_powers)

        # Return negative value because minimizing
        return -total_power


def get_optimization_dataframe(
    location,
    filename,
    optimization_parameters,
    layout,
):
    n_turbines = layout['n_turbines']
    
    try:
        df_optimization = pd.read_csv(location + filename)
        print('Old data loaded')
    except:
        data_turbines = ['power'] + optimization_parameters
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
        for col in data_turbines:
            columns_wind_farm = columns_wind_farm + \
                [f'{col}_{turb}' for turb in range(n_turbines)]
        df_optimization = pd.DataFrame(columns=columns_wind_farm)
        df_optimization['function_values'] = df_optimization['function_values'].astype('object')
        df_optimization['result'] = df_optimization['result'].astype('object')
        print('No old data loaded')

    return df_optimization


def get_result_bounds_guess_dicts(
    optimization_parameters,
    downwind_masks,
    bounds,
    guesses,
):
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
        
        result_bounds_guess_dicts.append(bounds_dict)

    return result_bounds_guess_dicts


def get_turbine_powers_conv(
    model, 
    case: Case,
    wind_directions: list,
    values_conv: dict,
):
    case.conditions['directions'] = wind_directions
    case.turbines['yaw_i'] = np.ones_like(case.turbines['yaw_i']) * values_conv['yaw']
    case.turbines['tilt_i'] = np.ones_like(case.turbines['tilt_i']) * values_conv['tilt']
    
    return model.run(case)


def get_all_parameters(
    n_turbines,
    parameter_names,
    parameters_optimized,
    optimization_parameters,
    downwind_values,
):  
    all_parameters = {}

    for p in optimization_parameters:
        for turb in range(n_turbines):
            all_parameters[f'{p}_{turb}'] = downwind_values[p]

    for idp, p in enumerate(parameter_names):
        all_parameters[p] = parameters_optimized[idp]

    return all_parameters


def print_progress(
    start_time,
    idw,
    wind_directions,
):  
    clear_output(wait=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    estimated_time_remaining = (elapsed_time / (idw + 1)) * (len(wind_directions) - (idw + 1))
    print(f'Progress: {round(((idw + 1) / len(wind_directions)) * 100, 1)} %')
    print(f'    Mean time per optimization run: {round(elapsed_time / ((idw + 1) * 60), 1)} minutes')
    print(f'    Estimation: {round(estimated_time_remaining / 60)} minutes remaining')
    print(f'    Currently at wind direction {round(wind_directions[idw], 1)} ({idw+1} of {len(wind_directions)})')