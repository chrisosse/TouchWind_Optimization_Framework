## This script is meant to hold all the functions used in plotting the data
## =============== Packages =============== 
import os
import numpy as np


def load_wakelengthoutputfiles_into_dataframe(pathtofolders: str, mainfolders: list, parameters: list, pathtofiles:str='wakelengthOutput\\', folder_data: list = [], RANS=False):
    """----------
    Description
    ----------
    Load Paraview CSV data of planeplots and lineplots into python data
    
    ... (rest of the docstring remains the same) ...
    
    ----------            
    Returns
    ----------
    folder_data: list
            Contains the data as loaded in, sorted based on folder, parameter, and wakedata
    """

    for folder in mainfolders:
        parameter_data = []

        for parameter in parameters:
            file_path = os.path.join(pathtofolders, folder, pathtofiles, parameter+".csv")
            if not (os.path.exists(file_path) and os.path.isfile(file_path)):
                continue  # Skip to the next parameter if the file doesn't exist
            else:
                file_data_matrix = np.loadtxt(file_path, delimiter=",", skiprows=1)
            
                wake_data = []
                
                if RANS:
                    correction=1
                else:
                    correction=0
                    
                try:
                    PointID = file_data_matrix[:,0]
                except:
                    PointID = None
                try:
                    X = file_data_matrix[:,28+correction]
                except:
                    X = None
                try:
                    Y = file_data_matrix[:,29+correction]
                except:
                    Y = None
                try:
                    Z = file_data_matrix[:,30+correction]
                except:
                    Z = None
                try:
                    UAvg_X = file_data_matrix[:,74+correction]
                except:
                    UAvg_X = None
                try:
                    UAvg_Y = file_data_matrix[:,75+correction]
                except:
                    UAvg_Y = None
                try:
                    UAvg_M = file_data_matrix[:,77+correction]
                except:
                    UAvg_M = None
                try:
                    TAvg = file_data_matrix[:,62+correction]
                except:
                    TAvg = None
                try:
                    VKEF = file_data_matrix[:,112]
                except:
                    VKEF = None
                try:
                    VKEF_max = file_data_matrix[:,114] #75 percentile
                except:
                    VKEF_max = None
                try:
                    VKEF_min = file_data_matrix[:,113]
                except:
                    VKEF_min = None
                try:
                    TI = file_data_matrix[:,109]
                except:
                    TI = None        
                try:
                    TI_max = file_data_matrix[:,111]
                except:
                    TI_max = None    
                try:
                    TI_min = file_data_matrix[:,110]
                except:
                    TI_min = None    
                try:
                    kSGS = file_data_matrix[:,14]
                except:
                    kSGS = None    
                #try:
                #    TI_maxvalue = file_data_matrix[:,115]
                #except:
                #    TI_maxvalue = None    
                try:
                    kResolved = file_data_matrix[:,12]
                except:
                    kResolved = None  
                try:
                    U_75 = file_data_matrix[:,114]
                except:
                    U_75 = None  
                try:
                    U_25 = file_data_matrix[:,115]
                except:
                    U_25 = None  

                wake_data.append({
                        'PointID': PointID,
                        'X': X,
                        'Y': Y, 
                        'Z': Z,
                        'UAvg_X': UAvg_X,
                        'UAvg_Y': UAvg_Y,
                        'UAvg_M': UAvg_M, 
                        'TAvg': TAvg, 
                        'kResolved': kResolved, 
                        'VKEF': VKEF,
                        'VKEF_max': VKEF_max, 
                        'VKEF_min': VKEF_min,
                        'TI': TI, 
                        'TI_max': TI_max, 
                        'TI_min': TI_min, 
                        'UAvg75_M': U_75, 
                        'UAvg25_M': U_25, 
                        'kSGS':kSGS
                        #'TI_maxvalue': TI_maxvalue, 
                        })
                    
                parameter_data.append({
                    'parameter': parameter,
                    'wake_data': wake_data
                    })
        
        folder_data.append({
            'folder': folder,
            'parameter_data': parameter_data
        })

    return folder_data


def searchandretrievewakedata(data:list, target_folder:str, target_parameter:str,xrange:list=[-1], datadesired:str='UAvg_M', axisdesired:str='X', everything=False):
    """----------
    Description
    ----------
    Search and returns numpy matrix of specific data from CSV loaded dataset

    ----------
    Parameters
    ----------
    data: list
            Contains the SOWFA data sorted based on folder, paramter
    target_folder: str
            string of specific folder to load
    target_parameter: str
            string of specific parameter to load
    xrange: list
            list of int inputs, with the x range to load, at default it is -1 which results in loading all x length
    datadesired: str
            string of specfics to load, e.g. values for the complete azimuth values as in file, or sum or average of these arrays, or even the timeindex from this array, at default the complete azimuth values. 
            options: "PointID", "X", "UAvg_X", "UAvg_M"
       
    ----------
    Returns
    ----------
    data: numpy matrix
            Contains 2 dimensional numpy matrix with axis of (x, value)    
    """

    folder_dict = next((entry for entry in data if entry['folder'] == target_folder), None)
    assert folder_dict is not None, f'The folder {target_folder} was not found in the data'
    returndata = None
    
    
    parameter_data_for_folder = folder_dict['parameter_data']
    for param_entry in parameter_data_for_folder:
        parameter = param_entry['parameter']
        if parameter == target_parameter:                       
            for wakedata_entry in param_entry['wake_data']:
                
                
                wake_x_array = wakedata_entry[axisdesired]
                
                
                if xrange == [-1]:
                    if np.sum(returndata ==None) != 0: returndata = np.empty([2, len(wake_x_array)])
                    returndata[:] = [wake_x_array, wakedata_entry[datadesired]]
                
                else:
                    boleantimerange = np.logical_and(wake_x_array >= xrange[0], wake_x_array <= xrange[1])                          
                    
                    if np.sum(returndata ==None) != 0: 
                        returndata = None
                        returndata = np.empty([2, sum(boleantimerange)])
                    returndata[:] = [wake_x_array[boleantimerange], wakedata_entry[datadesired][boleantimerange]]
            
    return returndata


def find_true_starts(arr):
    """----------
    Description
    ----------
    Find starts of continious sections of True in an array

    ----------
    Parameters
    ----------
    arr: bolean array
         an Array of bolean inputs

    ----------            
    Returns
    ----------
    true_starts: list
        list of indx where true sections start
    """
    true_starts = []
    in_true_segment = False
    
    for idx, value in enumerate(arr):
        if value and not in_true_segment:
            in_true_segment = True
            true_starts.append(idx)
        elif not value and in_true_segment:
            in_true_segment = False
    
    return true_starts


def set_size(width=455.24411, fraction=1, ratioxy = 1):
    """----------
    Description
    ----------
    Set figure dimensions to avoid scaling in LaTeX.

    ----------
    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    ----------
    Returns
    ----------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (ratioxy*fig_width_in, fig_height_in)

    return fig_dim


def find_nearest_indx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def load_turbineoutputfiles_into_dataframe(pathtofolders: str, mainfolders: list, parameters: list, pathtofiles:str='turbineOutput\\20000\\', folder_data=[]):
    """----------
    Description
    ----------
    Load SOWFA poweroutput data from the multiple folders into data
    
    ----------
    Parameters
    ----------
    pathtofolders: str
            system path to directiory where multiple folders are located in which the poweroutput data needs to be loaded from
    mainfolders: list
            list of str inputs, with the name of the folders to load
    parameters: list
            list of str inputs, with the name of the parameters to load
    pathtofiles: str
            When in the main folder, direct path to specific folder structure in which the SOWFA parameter files are located
      
    ----------            
    Returns
    ----------
    folder_data: list
            Contains the data as loaded in but only sorted based on folder, paramter, and turbine
    """
    
    for folder in mainfolders:
        parameter_data = []

        for parameter in parameters:
            file_path = os.path.join(pathtofolders, folder, pathtofiles, parameter)
            assert ( os.path.exists(file_path) and os.path.isfile(file_path) ), f'The parameter {parameter} was not found in folder {folder}'

            file_data_matrix = np.loadtxt(file_path)
            
            header = None
            with open(file_path, 'r') as fileID:
                firstLine = fileID.readline().strip()
                header = firstLine.split('    ')

            matrixindx_turbine = header.index('#Turbine')
            matrixindx_time = header.index('Time(s)')

            turbines = np.unique(file_data_matrix[:, matrixindx_turbine])
            turbine_data = []

            for turbine in turbines:
                turbine_subset = file_data_matrix[file_data_matrix[:, matrixindx_turbine] == turbine]
                turbine_values = turbine_subset[:, len(header)-1:]
                turbine_sum = np.sum(turbine_values, axis=1)
                turbine_avg = np.mean(turbine_values, axis=1)
                turbine_time = turbine_subset[:, matrixindx_time]

                turbine_data.append({
                    'turbine_index': turbine,
                    'time_array': turbine_time,
                    'values': turbine_values,
                    'sum': turbine_sum,
                    'avg': turbine_avg
                })

            parameter_data.append({
                'parameter': parameter,
                'turbine_data': turbine_data
            })

        folder_data.append({
            'folder': folder,
            'parameter_data': parameter_data
        })

    return folder_data


def searchandretrievedata(data:list, target_folder:str, target_parameter:str, target_turbine: list=[-1], timerange:list=[-1], datadesired:str='values'):
    """----------
    Description
    ----------
    Search and returns numpy matrix of specific data from SOWFA loaded dataset

    ----------    
    Parameters
    ----------
    data: list
            Contains the SOWFA data sorted based on folder, paramter, and turbine, designed for output of load_turbineoutputfiles_into_dataframe
    target_folder: str
            string of specific folder to load
    target_parameter: str
            string of specific parameter to load
    target_turbine: list
            list of int inputs, with the turbine number to load, at default it is -1 which results in loading all turbines
    timerange: list
            list of [min max] time to load, at default it is -1 which results in loading complete timerange
    datadesired: str
            string of specfics to load, e.g. values for the complete azimuth values as in file, or sum or average of these arrays, or even the timeindex from this array, at default the complete azimuth values. 
            options: "turbine_index", "time_array", "values", "sum", "avg"
         
    ----------            
    Returns
    ----------
    data: numpy matrix
            Contains 3 dimensional numpy matrix with axis of (turbines, time, azimuth)
    """
    
    folder_dict = next((entry for entry in data if entry['folder'] == target_folder), None)
    assert folder_dict is not None, f'The folder {target_folder} was not found in the data'
    returndata = None
    
    parameter_data_for_folder = folder_dict['parameter_data']
    for param_entry in parameter_data_for_folder:
        parameter = param_entry['parameter']
        if parameter == target_parameter:                       
            indx = 0
            for turbine_entry in param_entry['turbine_data']:
                turbine_index = int(turbine_entry['turbine_index'])
                if target_turbine == [-1] or turbine_index in target_turbine:
                    
                    turbine_time_array = turbine_entry['time_array']
                    nrturbinestosave = len(param_entry['turbine_data']) if target_turbine == [-1] else len(target_turbine)
                    try:
                        nrazimuthangles = turbine_entry[datadesired].shape[1]
                    except:
                        turbine_entry[datadesired] = turbine_entry[datadesired].reshape(turbine_entry[datadesired].shape[0],1)
                        nrazimuthangles = turbine_entry[datadesired].shape[1]
                    
                    if timerange == [-1]:
                        if np.sum(returndata ==None) != 0: returndata = np.empty([nrturbinestosave, len(turbine_time_array), nrazimuthangles])
                        returndata[indx,:, :] = turbine_entry[datadesired]
                    else:
                        boleantimerange = np.logical_and(turbine_time_array >= timerange[0], turbine_time_array <= timerange[1])                          
                        if np.sum(returndata ==None) != 0: returndata = np.empty([nrturbinestosave, np.sum(boleantimerange), nrazimuthangles])
                        returndata[indx, :, :] = turbine_entry[datadesired][boleantimerange].reshape(np.sum(boleantimerange),1)
                    indx+=1
    return returndata


def calculate_FU(data:list):
    """----------
    Description
    ----------
    Calculates F * U for every azimuth angle and put the result back into the data

    ----------    
    Parameters
    ----------
    data: list
            Contains the SOWFA data sorted based on folder, paramter, and turbine
    
    ----------
    Returns
    ----------
    data: list
            Contains the SOWFA data sorted based on folder, paramter, and turbine,  
    """ 
    for folder_entry in data:        
        # obtain data and calculate
        axialforce = searchandretrievedata(data, folder_entry['folder'], 'axialForce', [-1], [-1], 'values')
        axialforce_time = searchandretrievedata(data, folder_entry['folder'], 'axialForce', [-1], [-1], 'time_array')
        velocity = searchandretrievedata(data, folder_entry['folder'], 'Vaxial', [-1], [-1], 'values')
        velocity_time = searchandretrievedata(data, folder_entry['folder'], 'Vaxial', [-1], [-1], 'time_array')
        FU = axialforce*velocity
        
        # put dataa back in same dataset
        turbine_data = []
        for turbine in range(FU.shape[0]):
            turbine_values = FU[turbine,:,:]
            turbine_sum = np.sum(turbine_values, axis=1)
            turbine_avg = np.mean(turbine_values, axis=1)
            
            assert all(velocity_time[turbine] == axialforce_time[turbine]), 'The timearrays for FU multiplications are not equal'
            turbine_time = velocity_time[turbine]

            turbine_data.append({
                'turbine_index': turbine,
                'time_array': turbine_time,
                'values': turbine_values,
                'sum': turbine_sum,
                'avg': turbine_avg
            })
        folder_entry['parameter_data'].append({
            'parameter': 'FU',
            'turbine_data': turbine_data
        })
    return data


def calculate_TUAvg(data:list):
    """----------
    Description
    ----------
    Calculates T * Uavg for every azimuth angle and put the result back into the data

    ----------    
    Parameters
    ----------
            Contains the SOWFA data sorted based on folder, paramter, and turbine
    
    ----------
    Returns
    ----------
    data: list
            Contains the SOWFA data sorted based on folder, paramter, and turbine,         
    """
    for folder_entry in data:        
        # obtain data and calculate
        thrust = searchandretrievedata(data, folder_entry['folder'], 'thrust', [-1], [-1], 'values')
        thrust_time = searchandretrievedata(data, folder_entry['folder'], 'thrust', [-1], [-1], 'time_array')
        velocity = searchandretrievedata(data, folder_entry['folder'], 'Vaxial', [-1], [-1], 'avg')
        velocity_time = searchandretrievedata(data, folder_entry['folder'], 'Vaxial', [-1], [-1], 'time_array')
        TUavg = thrust*velocity
        
        # put dataa back in same dataset
        turbine_data = []
        for turbine in range(TUavg.shape[0]):
            turbine_values = TUavg[turbine,:,:]
            turbine_sum = np.sum(turbine_values, axis=1)
            turbine_avg = np.mean(turbine_values, axis=1)
            
            assert all(velocity_time[turbine] == thrust_time[turbine]), 'The timearrays for TUAvg multiplications are not equal'
            turbine_time = velocity_time[turbine]

            turbine_data.append({
                'turbine_index': turbine,
                'time_array': turbine_time,
                'values': turbine_values,
                'sum': turbine_sum,
                'avg': turbine_avg
            })
        folder_entry['parameter_data'].append({
            'parameter': 'TU',
            'turbine_data': turbine_data
        })
    return data
