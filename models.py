import numpy as np
import pandas as pd
import scipy as sc
import functions as func
from floris.tools import FlorisInterface as floris

class LES:
    '''
    This class provides tools for handling data obtained with Large Eddy Simulations.
    The output of functions in this class is standardized, so data structures are
    equal between different datasources.
    '''
    def __init__(
        self,
    ):
        pass


    # Get flowfield dataframe
    def get_df_flowfield(
        case_name: str = None,
        location: str = '../LES/',
    ):
        '''
        Get the flowfield dataframe of a LES case.

        Args:
            case_name (str): the name of the case
            location (str): the path to the LES data

        Returns:
            DataFrame: dataframe containing the flowfield data
        '''
        df_flowfield = pd.read_csv(location + case_name + '/' + case_name + '.csv')

        return df_flowfield
    

    def get_gridpoints(
        df_flowfield,    
    ):
        '''
        Get the gridpoints of the LES data

        Args:
            df_flowfield (DataFrame): dataframe containing LES flowfield

        Returns:
            tuple: respectivily the ndarrays of X, Y and Z gridpoints
        '''
        X = np.array(sorted(df_flowfield['Points:0'].unique()))
        Y = np.array(sorted(df_flowfield['Points:1'].unique()))
        Z = np.array(sorted(df_flowfield['Points:2'].unique()))
        
        return X, Y, Z
    

    def get_ABL_params(
        self,
        case_name: str = '1TURB_wd270_ws10_1x_y0_t5',
        location: str = '../LES/',
        z_ref_guess: float = 100,
        U_ref_guess: float = 10,
        alpha_guess: float = 0.12,
    ):  
        '''
        Get the fitted parameters that describe the Atmospheric Boundary
        Layer (ABL) as close as possible below 600m.

        Args:
            case_name (str): the name of the case
            location (str): the path to the LES data
            z_ref_guess (float): initial guess for reference height
            U_ref_guess (float): initial guess for reference velocity
            alpha_guess (float): initial guess for alpha

        Returns:
            dict: containing two arrays with fitted values describing the ABL
        '''
        # Get flowfield
        df_flowfield = self.get_df_flowfield(
            case_name,
            location,
        )

        # Get LES grid points
        X, Y, Z = self.get_gridpoints(df_flowfield)

        # Get U and V profiles of LES simulation at inflow boundary (X = 0)
        U_LES = df_flowfield[df_flowfield['Points:0'] == 0].groupby(['Points:2'])['UAvg:0'].mean().to_numpy()
        V_LES =  df_flowfield[df_flowfield['Points:0'] == 0].groupby(['Points:2'])['UAvg:1'].mean().to_numpy()

        # Get id of points at Z = 600, right before inversion layer starts
        idz, z_value = func.find_nearest(Z, 600)

        # Get U and V profiles from just above Z = 0 to Z = 600, before inversion layer
        cut_start = 2
        Z_cut = Z[cut_start:idz+1]
        U_LES_cut = U_LES[cut_start:idz+1]
        V_LES_cut = V_LES[cut_start:idz+1]

        # Fit streamwise velocity profile parameters
        U_params, _ = sc.optimize.curve_fit(
            func.U_profile, 
            Z_cut, 
            U_LES_cut, 
            p0=[z_ref_guess, U_ref_guess, alpha_guess], 
            bounds=([1, 1, 1e-6], [1e6, 1e6, 1])
        )

        # Fit spanwise velocity profile parameters
        V_params, _ = sc.optimize.curve_fit(
            func.V_profile, 
            Z_cut, 
            V_LES_cut
        )

        return {
            'U_params': U_params,
            'V_params': V_params,
        }


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
    
    def get_velocity_field(
        self,
        case,
        coordinates,
    ):
        # Get x and y coordinates of turbines
        x_i = case.layout['x_i'].flatten()
        y_i = case.layout['y_i'].flatten()

        # Create copy of farm so initial farm is not messed up
        farm_copy = self.farm.copy()

        # Calculate wakes
        _, flowfield, _ = farm_copy.calculate_full_domain(
            x_bounds=coordinates['X'],
            y_bounds=coordinates['Y'],
            z_bounds=coordinates['Z'],
            yaw_angles=x_i[None, None],
            tilt_angles=y_i[None, None],
        )

        # Save velocities in velocity field
        velocity_field = {
            'U': flowfield.u_sorted[0, 0],
            'V': flowfield.v_sorted[0, 0],
            'W': flowfield.w_sorted[0, 0],
        }

        return velocity_field
    

class TestModel:
    def __init__(
        self,
        model_params: dict = None,
    ):
        # Set model parameters to standard parameter values
        if model_params == None:
            model_params = {
                # Add parameter name and value
            }

        self.model_params = model_params

        self.set_model_params(
            self.model_params
        )

    def set_model_params(
        self,
        model_params,
    ):
        # Enter code here to set model parameters to the model
        pass
        
    def reinitialize_farm(
        self,
        conditions: dict,
        layout: dict,
        model_params: dict = None,
    ):  
        # Set model parameters to given or standard parameters
        if model_params is None:
            model_params = self.model_params
        
        # Set model parameters
            self.set_model_params(model_params)

        self.conditions = conditions

        # Add code to reinitialize farm with wind conditions and farm layout
        
    def get_turbine_powers(
        self,
        turbines: dict,
    ): 
        # Change this code to get the power of all turbines for all conditions
        # shape of turbine powers is (n_directions, n_speeds, n_turbines)
        turbine_powers = np.zeros((
            len(self.conditions['directions']), 
            len(self.conditions['speeds']),
            9,
        ))

        turbine_powers[:, :] = np.random.random(size=9)
        
        return turbine_powers