import numpy as np
from cases import Case

from models.FVW_model import FVW_model

class FVW:
    '''
    TODO: Documentation!
    '''
    def __init__(
        self,
        model_params: dict = None,
    ):
        '''
        Initializer of the testmodel.

        Parameters
        ----------
        model_params : dict, optional
            Dictionary containing all model parameters. When None,
            standard calibrated parameters are used. By default None
        '''
        # The standard model parameter set
        standard_model_params = {
            'n_r': 40,
            'n_e': 16,
            'n_p': 16,
        }

        # Add input model params to standard model params
        for key, value in zip(model_params.keys(), model_params.values()):
            standard_model_params[key] = value

        # Set updated model params as model params
        self.model_params = standard_model_params

        # Create instance of FVW model
        self.farm = FVW_model(self.model_params)


    def set_model_params(
        self,
        model_params: dict,
    ):
        '''
        Set the model parameters in FVW model

        Parameters
        ----------
        model_params : dict
            Dictionary containing the model parameters
        '''
        # Run initialize model again
        self.farm.__init__(model_params)
        

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
        
        # Set model parameters
        self.set_model_params(model_params)

        # Set conditions
        self.farm.set_wind_conditions(
            conditions['directions'][0],
            conditions['speeds'][0],
            conditions['ABL_params'],
        )

        # Get turbine positions
        turbine_positions = np.zeros((layout['n_turbines'], 3))
        turbine_positions[:, 0] = layout['x_i']
        turbine_positions[:, 1] = layout['y_i']
        turbine_positions[:, 2] = layout['z_i']
        turbine_positions = turbine_positions.tolist()

        # Set layout
        self.farm.set_wind_farm_layout(
            turbine_positions,
        )
        

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
            (n wind directions, n speeds, n turbines)
        '''
        # TODO Calculate axial induction from thrust coefficient
        axial_inductions = np.ones((len(turbines['thrustcoef_i']))) * 0.27
        
        # Set turbine properties
        self.farm.set_turbine_properties(
            turbines['yaw_i'],
            turbines['tilt_i'],
            axial_inductions,
            turbines['D_rotor_i'],
        )
        
        # Run model
        self.farm.run_model()

        # Get turbine powers
        turbine_powers = np.zeros((1, 1, len(turbines['yaw_i'])))
        turbine_powers[0, 0] = self.farm.get_turbine_powers()
        
        return turbine_powers
    

    def run(
        self,
        case: Case,
        model_params: dict = None,
    ):
        '''
        Run a case with the ... model.

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
        # Create your own code below
        self.reinitialize_farm(
            case.conditions,
            case.layout,
            model_params,
        )

        turbine_powers = self.get_turbine_powers(
            case.turbines
        )

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
        # Create your own code below
        shape = (
            len(coordinates['X']),
            len(coordinates['Y']),
            len(coordinates['Z'])
        )

        velocity_field = {
            'U': np.zeros(shape),
            'V': np.zeros(shape),
            'W': np.zeros(shape),
        }

        return velocity_field