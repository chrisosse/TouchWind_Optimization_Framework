import numpy as np
import pandas as pd
import scipy as sc
import functions as func

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
        self,
        case_name: str = None,
        location: str = '../LES/',
    ):
        '''
        Get the flowfield dataframe of a LES case.

        Parameters
        ----------
        case_name : str, optional
            the name of the case, by default None
        location : str, optional
            the path to the LES data, by default '../LES/'

        Returns
        -------
        df_flowfield : pd.DataFrame
            dataframe containing the flowfield data
        '''
        df_flowfield = pd.read_csv(location + case_name + '/' + case_name + '.csv')

        return df_flowfield
    

    def get_gridpoints(
        self,
        df_flowfield: pd.DataFrame,    
    ):
        '''
        Get the gridpoints of the LES data

        Parameters
        ----------
        df_flowfield : pd.DataFrame
            dataframe containing LES flowfield

        Returns
        -------
        tuple
            respectivily the ndarrays of X, Y and Z gridpoints
        '''
        X = np.array(sorted(df_flowfield['Points:0'].unique()))
        Y = np.array(sorted(df_flowfield['Points:1'].unique()))
        Z = np.array(sorted(df_flowfield['Points:2'].unique()))
        
        return X, Y, Z
    

    def get_coordinates(
        self,
        df_flowfield: pd.DataFrame,
    ):
        '''
        Get the coordinates of the LES data

        Parameters
        ----------
        df_flowfield : pd.DataFrame
            dataframe containing LES flowfield

        Returns
        -------
        coordinates : dictionary
            dictionary containing the coordinates of the LES data
        '''
        X, Y, Z = self.get_gridpoints(df_flowfield)

        coordinates = {
            'X': X,
            'Y': Y,
            'Z': Z,
        }

        return coordinates
    

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

        Parameters
        ----------
        case_name : str, optional
            the name of the case, by default '1TURB_wd270_ws10_1x_y0_t5'
        location : str, optional
            the path to the LES data, by default '../LES/'
        z_ref_guess : float, optional
            initial guess for reference height, by default 100
        U_ref_guess : float, optional
            initial guess for reference velocity, by default 10
        alpha_guess : float, optional
            initial guess for alpha, by default 0.12

        Returns
        -------
        dictionary
            containing two arrays with fitted values describing the ABL
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
    

    def get_velocity_field(
        self,
        df_flowfield,
    ):
        # Get gridpoints
        X, Y, Z = self.get_gridpoints(df_flowfield)

        # Get shape of velocity field
        shape = (len(X), len(Y), len(Z))

        # Initialize velocity field
        velocity_field = {
            'U': np.zeros(shape),
            'V': np.zeros(shape),
            'W': np.zeros(shape),
        }

        # Get number of x planes
        n_planes = len(X)

        # Loop over all planes
        for p in range(n_planes):
            # Get current plane
            X_mask = df_flowfield['Points:0'] == X[p]
            plane = df_flowfield[X_mask]

            # Get dataframe for each velocity component
            df_U = plane.pivot(index='Points:1', columns='Points:2', values='UAvg:0')
            df_V = plane.pivot(index='Points:1', columns='Points:2', values='UAvg:1')
            df_W = plane.pivot(index='Points:1', columns='Points:2', values='UAvg:2')

            # Change dataframes to numpy arrays
            velocity_field['U'][p] = df_U.to_numpy()
            velocity_field['V'][p] = df_V.to_numpy()
            velocity_field['W'][p] = df_W.to_numpy()

        return velocity_field
        