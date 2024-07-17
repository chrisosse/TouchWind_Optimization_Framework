import numpy as np

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