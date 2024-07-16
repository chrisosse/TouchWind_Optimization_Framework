import numpy as np
import pandas as pd
import functions as func
from models import (
    LES,
    CCM,
    TestModel,
)

class Case:
    def __init__(
        self,
        name: str = 'test', 
        model: CCM = CCM(),
        predef_case: dict = {}
    ):  
        # Set name
        self.name = name
        self.model = model
        

    def set_layout(
        self,
        shape: str,
        n_x: int = 1,
        n_y: int = 1,
        spacing_x: int = 5,
        spacing_y: int = 5,
        D_rotor: float = 126,
        x_i: list = None,
        y_i: list = None,
    ):
        # Set layout to predefined layout
        if x_i is not None and y_i is not None:
            print('x_i and y_i set to predefined values')
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

        self.layout = {
            'shape': shape,
            'n_x': n_x,
            'n_y': n_y,
            'x_i': x_i,
            'y_i': y_i,
            'n_turbines': n_turbines,
        }


    def set_conditions(
        self,
        directions: list = np.array([270.]),
        speeds: list = np.array([10.]),
        TI: list = np.array([0.06]),
        ABL_params: dict = None,
    ):
        if type(directions) == int or type(directions) == float:
            directions = np.array([directions])

        if type(speeds) == int or type(speeds) == float:
            speeds = np.array([speeds])

        self.conditions = {
            'directions': directions,
            'speeds': speeds,
            'TI': TI,
            'ABL_params': ABL_params,
        }


    def set_turbines(
        self,
        yaw_i: list,
        tilt_i: list,
        thrustcoef_i: list = None,
        D_rotor_i: list = None,       
    ):  
        # TODO: Add functionality to set CT and D for individual turbs
        
        if D_rotor_i is None:
            D_rotor_i = np.array(self.layout['n_turbines'] * [126])
        # if len(D_rotor_i) == 1:
        #     D_rotor_i = np.array(self.layout['n_turbines'] * [D_rotor_i[0]])
        # if len(yaw_i) == 1:
        #     yaw_i = np.array(self.layout['n_turbines'] * [yaw_i[0]])
        # if len(tilt_i) == 1:
        #     tilt_i = np.array(self.layout['n_turbines'] * [tilt_i[0]])

        self.turbines = {
            'yaw_i': yaw_i,
            'tilt_i': tilt_i,
            'thrustcoef_i': thrustcoef_i,
            'D_rotor_i': D_rotor_i,
        }
    
    def run(
        self,
    ):
        self.model.reinitialize_farm(
            conditions=self.conditions,
            layout=self.layout,
        )
        
        self.turbine_powers = self.model.get_turbine_powers(
            turbines=self.turbines,
        )

        return self.turbine_powers


class CaseManager:
    def __init__(
        self,
        name: str = 'Case Manager',
        ref_model = LES,
        ref_data_location: str = '../LES/',
        ref_standard_case: str = '1TURB_wd270_ws10_1x_y0_t5',
    ):
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
            ref_model,
            ref_standard_case,
            ref_data_location,
        )

        self.set_case_names()


    def set_case_names(
        self,
    ):
        self.case_names = list(self.cases.keys())


    def get_cases(
        self,
    ):
        return list(self.cases.values())


    def load_csv_cases(
        self,
        location: str = '../TouchWind_Optimization_Framework/',
        file_name: str = 'test_cases.csv',
        masks: dict = {},
        model = CCM(),
    ):
        df_cases = pd.read_csv(location + file_name)

        # Select right cases by applying masks
        for mask in masks:
            df_cases = df_cases[df_cases[mask] == masks[mask]].reset_index(drop=True)

        # Create cases
        for name in df_cases['case_name']:
            case_dict = df_cases[df_cases['case_name'] == name].iloc[0]

            case = Case(
                name,
                model,
            )

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
                    self.ref_model,
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

            case.set_turbines(
                yaw_i=yaw_i,
                tilt_i=tilt_i,
                D_rotor_i=case_dict['D_rotor'],
            )

            self.cases[name] = case

        self.set_case_names()