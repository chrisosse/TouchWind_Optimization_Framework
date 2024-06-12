# Import models
from floris.tools import FlorisInterface as floris

# Add names of newly added models in this list
model_names = [
    'CCM',
]

class CCM:
    '''
    If you want to update changes made in FLORIS, first move towards
    the right folder in the terminal with right environment by (for example):
    cd Documents\Technische Universiteit Eindhoven\Graduation Project
    And afterwards reinstall FLORIS as follows:
    pip install -e floris_tilt
    '''
    def __init__(
        self,
        input_file: str = 'model_files/CCM/case_initial.yaml',
    ):
        self.input_file = input_file
        
    def return_model(self):
        return floris(self.input_file)


# Create class to get model
class WakeModeling:
    def __init__(
        self,
        model_name: str,
    ): 
        if model_name not in model_names:
            raise ValueError('"model_name" is not in list of valid model names')
        
        self.model_name = model_name

        # Add new models right here
        if model_name == 'CCM':
            self.model_class = CCM()
            self.model = self.model_class.return_model()

    def get_model(self):
        return self.model
    
    def get_model_class(self):
        return self.model_class
        
    