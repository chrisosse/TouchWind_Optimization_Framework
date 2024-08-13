# README
This readme guides you through the process of using the Optimization Framework build for TouchWind.

## Installation
To be able to use and edit this repository, you can clone it on your local device. You can make a new branch of your own or fork the project, where you can freely work and make commitments. It is recommended to create a new environment, where all required packages can be installed separately from the main Python environment.

Required Python packages that have to be installed are:
- floris_tilt: https://github.com/chrisosse/floris_tilt
  Installation guide:
  1. Open the Python environment in the terminal, and go the desired path
  2. Run the following command: `git clone https://github.com/chrisosse/floris_tilt.git` (or your own forked project)
  3. Run the following command: `pip install -e floris_tilt`
  4. If one has made changes to FLORIS, one has to repeat step 3 to implement the changes
- Packages stated in the requirements file
  Installation guide:
  1. Open the Python environment in the terminal, and go to the path of this project
  2. Run the following command: `pip install -r requirements.txt`

## How to use
The Python notebook `framework_test.ipynb` shows examples of how to use different framework functions. All these functions are elaborated in the notebook, next to the documentation of the function itself. Below, the capabilities of the framework are given:

1. Choose a model to run cases with, and the type of reference data.
2. Load cases which are specified in a .csv file.
3. Create custom cases.
4. Run cases (get turbine powers and velocity fields).
5. Optimization of wind farms.
6. Analysis of the optimization process.
7. Obtain turbine powers and visualization of reference data.
