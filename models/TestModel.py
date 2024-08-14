import numpy as np

from cases import Case

class TestModel:
    '''
    Class which can be used to implement a new model. Copy
    this file and change everything needed to make it work for 
    the new model. The mandatory functions and their in and 
    outputs are already stated in this file.
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
        '''
        Set the model parameters in FLORIS CCM model

        Parameters
        ----------
        model_params : dict
            Dictionary containing the model parameters
        '''
        # Enter code here to set model parameters to the model
        pass
        
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

        self.conditions = conditions

        # Add code to reinitialize farm with wind conditions and farm layout
        

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
            (idw, ids, turb)
        '''
        # Change this code to get the power of all turbines for all conditions
        # shape of turbine powers is (n_directions, n_speeds, n_turbines)
        n_turbines = len(turbines['yaw_i'])
        
        turbine_powers = np.zeros((
            len(self.conditions['directions']), 
            len(self.conditions['speeds']),
            n_turbines,
        ))

        turbine_powers[:, :] = np.random.random(size=n_turbines)
        
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