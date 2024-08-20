import numpy as np
import functions as func
from floris.tools import FlorisInterface as floris

from cases import Case

class CCM:
    '''
    Class which can be used to run cases with the Cumulative Curl
    Misalignment (CCM) model, as is included in FLORIS_tilt (FLORIS
    version 3.4.1, github.com/chrisosse/floris_tilt) by Chris Osse 
    (contact: ossechris@gmail.com).

    If one wants to reinstall FLORIS_tilt, open the python environment
    in the terminal, go to the directory where FLORIS_tilt is located, 
    and run the following command: "pip install -e floris_tilt".
    '''
    def __init__(
        self,
        model_params: dict = None,
        input_file: str = 'model_files/CCM/case_initial.yaml',
    ):
        '''
        Initializer of the CCM class, which initialized FLORIS to
        use the CCM model with the right model parameters.

        Parameters
        ----------
        model_params : dict, optional
            Dictionary containing all model parameters. When None,
            standard calibrated parameters are used. By default None
        input_file : str, optional
            Directory to the yaml file to use as input for FLORIS, 
            by default 'model_files/CCM/case_initial.yaml'
        '''
        # If no model parameters given, set calibrated ones. 
        # NOTE: Add own calibrated values if required.
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

        # Set farm and model parameters
        self.farm = floris(input_file)
        self.model_params = model_params
        self.set_model_params(
            self.model_params
        )
        

    def set_model_params(
        self,
        model_params: dict,
    ):
        '''
        Set the model parameters in FLORIS CCM model

        Parameters
        ----------
        model_params : dict
            Dictionary containing the model parameters
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
        '''
        Reinitialize farm with wind conditions and farm layout

        Parameters
        ----------
        conditions : dict
            Dictionary containing wind conditions
        layout : dict
            Dictionary containing farm layout
        model_params : dict, optional
            Dictionary containing model parameters, by default None
        '''
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
        '''
        Get the turbine powers of all turbines for all wind 
        directions

        Parameters
        ----------
        turbines : dict
            Dictionary containing turbine settings

        Returns
        -------
        turbine_powers : np.ndarray
            3D array containing the turbine powers [W]. Shape:
            (n_directions, n_speeds, n_turbines)
        '''
        # Get yaw and tilt angles
        yaw_angles = turbines['yaw_i']
        tilt_angles = turbines['tilt_i']
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
        '''
        Run a case with the CCM model.

        Parameters
        ----------
        case : Case
            Instance of a case
        model_params : dict, optional
            Dictionary containing the model parameters, by default None

        Returns
        -------
        turbine_powers : np.ndarray
            3D array containing the turbine powers [W]. Shape:
            (idw, ids, turb)
        '''
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

        # Set turbine powers to case
        case.turbine_powers = turbine_powers

        return turbine_powers
        

    def get_velocity_field(
        self,
        case: Case,
        coordinates: dict,
        model_params: dict = None,
    ):
        '''
        Get the velocity field of a case in a 3D grid

        Parameters
        ----------
        case : Case
            Instance of a case
        coordinates : dict
            Dictionary containing 'X', 'Y', and 'Z' grid coordinates. 
        model_params : dict, optional
            Dictionary containing the model parameters, by default None

        Returns
        -------
        velocity_field : dict
            Dictionary containing arrays with velocity components 'U', 'V', and 'W'.
            Shape of arrays: (idx, idy, idz)
        '''
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