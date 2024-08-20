import numpy as np
import pandas as pd
import functions as func
from models.LES import LES

class Case:
    '''
    Class to create a case, which contains all information
    about the farm conditions and turbine settings.
    '''
    def __init__(
        self,
        name: str = 'test', 
    ):  
        '''
        Initialized a case

        Parameters
        ----------
        name : str, optional
            Name of the case, by default 'test'
        '''
        # Set name
        self.name = name
        
        
    def set_layout(
        self,
        shape: str,
        n_x: int = 1,
        n_y: int = 1,
        spacing_x: int = 5,
        spacing_y: int = 5,
        D_rotor: float | np.ndarray = 126,
        x_i: np.ndarray = None,
        y_i: np.ndarray = None,
        z_i: np.ndarray = None,
    ):
        '''
        Set the layout of the wind farm.

        Parameters
        ----------
        shape : str
            The shape of the wind farm layout. Can either be
            "rectangular", "hexagonal", or "custom"
        n_x : int, optional
            Number of turbines in the x-direction, by default 1
        n_y : int, optional
            Number of turbines in the y-direction, by default 1
        spacing_x : int, optional
            Turbine spacing in the x-direction, by default 5
        spacing_y : int, optional
            Turbine spacing in the y-direction, by default 5
        D_rotor : float | np.ndarray, optional
            Rotor diameter(s) [m], by default 126
        x_i : np.ndarray, optional
            Array containing the turbine x-coordinates, by default None
        y_i : np.ndarray, optional
            Array containing the turbine y-coordinates, by default None
        '''
        # Set layout to predefined layout
        if x_i is not None and y_i is not None:
            n_turbines = len(x_i)
        # Set layout according to specifications
        else: 
            x_i, y_i = func.create_layout(
                shape,
                n_x,
                n_y,
                spacing_x,
                spacing_y,
                D_rotor,
            )
            n_turbines = n_x * n_y

        # Set hub height to 90 if None
        if z_i is None:
            z_i = np.array([90] * n_turbines)

        # Set layout
        self.layout = {
            'shape': shape,
            'n_x': n_x,
            'n_y': n_y,
            'x_i': x_i,
            'y_i': y_i,
            'z_i': z_i,
            'n_turbines': n_turbines,
        }


    def set_conditions(
        self,
        directions: np.ndarray = np.array([270.]),
        speeds: np.ndarray = np.array([10.]),
        TI: np.ndarray = np.array([0.06]),
        ABL_params: dict = None,
    ):
        '''
        Set the wind conditions of the case

        Parameters
        ----------
        directions : np.ndarray, optional
            Array containing all wind directions [degrees], 
            by default np.array([270.])
        speeds : np.ndarray, optional
            Array containing all wind speeds [m/s], by default np.array([10.])
        TI : np.ndarray, optional
            Array containing all Turbulent Intensities [-], by default np.array([0.06])
        ABL_params : dict, optional
            Dictionary containing ABL parameters for the streamwise
            and spanwise direction, by default None
        '''
        # Convert directions and speeds to np.ndarray if single value
        if type(directions) == int or type(directions) == float:
            directions = np.array([directions])
        if type(speeds) == int or type(speeds) == float:
            speeds = np.array([speeds])

        # Set conditions
        self.conditions = {
            'directions': directions,
            'speeds': speeds,
            'TI': TI,
            'ABL_params': ABL_params,
        }


    def set_turbines(
        self,
        yaw_i: np.ndarray,
        tilt_i: np.ndarray,
        thrustcoef_i: np.ndarray = None,
        D_rotor_i: np.ndarray = None,       
    ):  
        '''
        Set turbine settings

        Parameters
        ----------
        yaw_i : np.ndarray
            Array containing yaw angles [degrees]
        tilt_i : np.ndarray
            Array containing tilt angles [degrees]
        thrustcoef_i : np.ndarray, optional
            Array containing thrust coefficients [-], by default None
        D_rotor_i : np.ndarray, optional
            Array containing rotor diameters [m], by default None
        '''
        # TODO: Add functionality to set CT and D for individual turbs
        if D_rotor_i is None:
            D_rotor_i = np.array(self.layout['n_turbines'] * [126])

        # Set turbines
        self.turbines = {
            'yaw_i': yaw_i,
            'tilt_i': tilt_i,
            'thrustcoef_i': thrustcoef_i,
            'D_rotor_i': D_rotor_i,
        }


class CaseManager:
    '''
    Class to create a case manager, containing several cases. 
    These cases can either be added manually or by importing them
    from a .csv file.
    '''
    def __init__(
        self,
        name: str = 'Case Manager',
        ref_model = LES(),
        ref_data_location: str = '../LES/',
        ref_standard_case: str = '1TURB_wd270_ws10_1x_y0_t5',
    ):
        '''
        Initializes the casemanager.

        Parameters
        ----------
        name : str, optional
            Name of the case manager instance, by default 'Case Manager'
        ref_model : Instance of model class, optional
            Model used for the reference data, by default LES()
        ref_data_location : str, optional
            Directory to the reference data, by default '../LES/'
        ref_standard_case : str, optional
            Standard reference case, by default '1TURB_wd270_ws10_1x_y0_t5'
        '''
        # Set case manager name
        self.name = name

        # Set reference model, data location and case
        self.ref_model = ref_model
        self.ref_data_location = ref_data_location
        self.standard_ref_case = ref_standard_case

        # Initialize case
        self.cases = {}

        # Get Atmospheric Boundary Layer parameters
        self.standard_ref_ABL_params = ref_model.get_ABL_params(
            ref_standard_case,
            ref_data_location,
        )

        # Set case names
        self.set_case_names()


    def set_case_names(
        self,
    ):
        '''
        Set case names to casemanager of all cases in cases dictionary
        '''
        self.case_names = list(self.cases.keys())


    def get_cases(
        self,
    ):
        '''
        Get all cases in a list

        Returns
        -------
        cases : list
            List of all cases outside of a dictionary
        '''
        return list(self.cases.values())


    def load_csv_cases(
        self,
        location: str = '../TouchWind_Optimization_Framework/',
        file_name: str = 'test_cases.csv',
        masks: dict = {},
    ):
        '''
        Load cases from a .scv file

        Parameters
        ----------
        location : str, optional
            Directory to the .csv file, by default '../TouchWind_Optimization_Framework/'
        file_name : str, optional
            Name of the .csv file, by default 'test_cases.csv'
        masks : dict, optional
            Masks to exclude certain cases from .csv file, by default {}
        '''
        # Read the .csv file
        df_cases = pd.read_csv(location + file_name)

        # Select right cases by applying masks
        for mask in masks:
            df_cases = df_cases[df_cases[mask] == masks[mask]].reset_index(drop=True)

        # Loop over all cases in .csv file
        for name in df_cases['case_name']:
            case_dict = df_cases[df_cases['case_name'] == name].iloc[0]

            # Create instance of a case
            case = Case(
                name,
            )

            # If all turbines are equal
            if case_dict['equal']:
                x_i, y_i = func.create_layout(
                    case_dict['shape'],
                    case_dict['n_x'],
                    case_dict['n_y'],
                    case_dict['spacing_x'],
                    case_dict['spacing_y'],
                    case_dict['D_rotor'],
                )
                x_i = x_i.flatten() + case_dict[f'x_0']
                y_i = y_i.flatten() + case_dict[f'y_0']
                yaw_i = np.ones(len(x_i)) * case_dict[f'yaw_0']
                tilt_i = np.ones(len(x_i)) * case_dict[f'tilt_0']
            # if turbines are not equal
            else:
                n_turbines = case_dict['n_x'] * case_dict['n_y']
                x_i = np.ones(n_turbines)
                y_i = np.ones(n_turbines)
                yaw_i = np.ones(n_turbines)
                tilt_i = np.ones(n_turbines)

                for turb in range(n_turbines):
                    x_i[turb] = case_dict[f'x_{turb}']
                    y_i[turb] = case_dict[f'y_{turb}']
                    yaw_i[turb] = case_dict[f'yaw_{turb}']
                    tilt_i[turb] = case_dict[f'tilt_{turb}']
            
            # Set Atmospheric Boundary Layer parameters
            if case_dict['load_ABL'] == 'standard':
                ABL_params = self.standard_ref_ABL_params
            elif case_dict['load_ABL'] == 'ref':
                ABL_params = self.ref_model.get_ABL_params(
                    name,
                    self.ref_data_location,
                )
            else:
                ABL_params = None

            # Set case layout
            case.set_layout(
                shape=case_dict['shape'],
                n_x=case_dict['n_x'],
                n_y=case_dict['n_y'],
                spacing_x=case_dict['spacing_x'],
                spacing_y=case_dict['spacing_y'],
                D_rotor=case_dict['D_rotor'],
                x_i = x_i,
                y_i = y_i,
            )

            # Set case wind conditions
            case.set_conditions(
                directions=[case_dict['wd']],
                speeds=[case_dict['U_ref']],
                ABL_params=ABL_params,
            )

            # Set turbine settings
            case.set_turbines(
                yaw_i=yaw_i,
                tilt_i=tilt_i,
                D_rotor_i=case_dict['D_rotor'],
            )

            # Set case name
            self.cases[name] = case

        # Reset case names
        self.set_case_names()


    def add_case(
        self,
        case: Case,
    ):
        '''
        Add a new case to the case manager.

        Parameters
        ----------
        case : Case
            Instance of a case
        '''
        # set case
        self.cases[case.name] = case

        # Reset case names
        self.set_case_names()

    def remove_case(
        self,
        name: str,
    ):
        '''
        Remove case from case manager

        Parameters
        ----------
        name : str
            Name of a case
        '''
        # Remove case from dictionary
        self.cases.pop(name)

        # Reset case names
        self.set_case_names()
