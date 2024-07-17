import numpy as np
import functions as func
from floris.tools import FlorisInterface as floris

from cases import Case

class CCM:
    '''
    If one wants to reinstall FLORIS, type the following in the terminal:
    'pip install -e floris_tilt'
    '''
    def __init__(
        self,
        model_params: dict = None,
        input_file: str = 'model_files/CCM/case_initial.yaml',
    ):
        self.input_file = input_file

        if model_params == None:
            model_params = {
                'ad': 0,
                'bd': -0.0018192983887298023,
                'cd': 1.0803331806986867,
                'dd': -0.09040629347972164,
                'alpha': 0.58,
                'beta': 0.077,
                'dm': 1.0,
                'c_s1': 0.0563691592,
                'c_s2': 0.1376631233159683,
                'a_s': 0.3253111149080571,
                'b_s': 0.012031554853652504,
                'a_f': 3.11,
                'b_f': -0.68,
                'c_f': 2.223295807654856,
                'wr_gain': 0.5392489436318193,
                'ma_gain': 1.7431079762733077,
                'wr_decay_gain': 3.207532818500954,
                'ma_decay_gain': 1.7832719494462048,
            }

        self.model_params = model_params

        self.farm = floris(self.input_file)

        self.set_model_params(
            self.model_params
        )
        

    def set_model_params(
        self,
        model_params: dict,
    ):
        '''
        Set model parameters to the CCM model.

        Args:
            model_params (dict): dictionary containing the model parameters
        '''
        # Loop over all model parameters
        for key in model_params.keys():
            found = False

            # Check if in 'gaussm' and if so, apply change
            if key in self.farm.floris.wake.wake_deflection_parameters['gaussm'].keys():
                self.farm.floris.wake.wake_deflection_parameters['gaussm'][key] = model_params[key]
                found = True

            # Check if in 'ccm' and if so, apply change
            if key in self.farm.floris.wake.wake_velocity_parameters['ccm'].keys():
                self.farm.floris.wake.wake_velocity_parameters['ccm'][key] = model_params[key]
                found = True
            
            # Print message if parameter is not found
            if not found:
                print(f'Key named "{key}" not found in either model')  


    def reinitialize_farm(
        self,
        conditions: dict,
        layout: dict,
        model_params: dict = None,
    ):
        # Set model parameters to given or standard parameters
        if model_params is None:
            model_params = self.model_params
        
        # Run twice, since for some reason sometimes things didn't update
        # after running is once (TODO?)
        for _ in range(2):
            # Update farm layout, wind direction and speed
            self.farm.reinitialize(
                layout_x=layout['x_i'].flatten(), 
                layout_y=layout['y_i'].flatten(), 
                wind_directions=conditions['directions'],
                wind_speeds=conditions['speeds'],
            )

            # Set model parameters
            self.set_model_params(model_params)
            
            # Set ABL parameters
            if conditions['ABL_params'] is not None:
                U_params = conditions['ABL_params']['U_params']
                V_params = conditions['ABL_params']['V_params']

                # Set parameters of streamwise velocity profile
                self.farm.floris.flow_field.reference_wind_height = U_params[0]
                self.farm.floris.flow_field.wind_speeds = [U_params[1]]
                self.farm.floris.flow_field.wind_shear = U_params[2]
                
                # Set parameters of spanwise velocity profile
                self.farm.floris.flow_field.a = V_params[0]
                self.farm.floris.flow_field.b = V_params[1]
                self.farm.floris.flow_field.c = V_params[2]
                self.farm.floris.flow_field.d = V_params[3]    


    def get_turbine_powers(
        self,
        turbines: dict,
    ):  
        # Get yaw and tilt angles flattened and adjust for number of wind conditions
        yaw_angles = turbines['yaw_i']#.flatten()[None, None]
        tilt_angles = turbines['tilt_i']#.flatten()[None, None]
        thrust_coefs = turbines['thrustcoef_i']

        # Ensure yaw and tilt have right dimensions
        if len(np.shape(yaw_angles)) == 1:
            yaw_angles = yaw_angles[None, None]
        if len(np.shape(tilt_angles)) == 1:
            tilt_angles = tilt_angles[None, None]
        if len(np.shape(thrust_coefs)) == 1:
            thrust_coefs = thrust_coefs[None, None]

        # Create copy of farm so initial farm is not messed up
        farm_copy = self.farm.copy()

        # Calculate wakes
        farm_copy.calculate_wake(
            yaw_angles=yaw_angles,
            tilt_angles=tilt_angles,
            thrust_coefs=thrust_coefs,
        )

        # Get misalignment correction factors. This is not done anymore 
        # in FLORIS itself (TODO?)
        correction_factors = func.get_correction_factor_misalignment(
            yaw_angles,
            tilt_angles,
        )

        # Get total power (need to account for air density and correction factor)
        turbine_powers = farm_copy.get_turbine_powers() * \
            farm_copy.floris.flow_field.air_density * \
            correction_factors

        return turbine_powers
    

    def run(
        self,
        case: Case,
        model_params: dict = None,
    ):
        # Reinitialize farm
        self.reinitialize_farm(
            case.conditions,
            case.layout,
            model_params,
        )

        # Get turbine powers
        turbine_powers = self.get_turbine_powers(
            case.turbines,
        )

        return turbine_powers
        

    def get_velocity_field(
        self,
        case: Case,
        coordinates: dict,
        model_params: dict = None,
    ):
        # Reinitialize farm
        self.reinitialize_farm(
            case.conditions,
            case.layout,
            model_params,
        )
        
        # Get yaw and tilt angles and thrust coefficient of turbines
        yaw_i = case.turbines['yaw_i'].flatten()
        tilt_i = case.turbines['tilt_i'].flatten()
        thrustcoef_i = case.turbines['thrustcoef_i'].flatten()

        # Create copy of farm so initial farm is not messed up
        farm_copy = self.farm.copy()

        # Calculate wakes
        _, flowfield, _ = farm_copy.calculate_full_domain(
            x_bounds=coordinates['X'],
            y_bounds=coordinates['Y'],
            z_bounds=coordinates['Z'],
            yaw_angles=yaw_i[None, None],
            tilt_angles=tilt_i[None, None],
            thrust_coefs=thrustcoef_i[None, None],
        )

        # Save velocities in velocity field
        velocity_field = {
            'U': flowfield.u_sorted[0, 0],
            'V': flowfield.v_sorted[0, 0],
            'W': flowfield.w_sorted[0, 0],
        }

        return velocity_field