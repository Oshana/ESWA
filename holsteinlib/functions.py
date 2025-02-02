import numpy as np
import pandas as pd
import math
import warnings
import glob
import os
from itertools import combinations
from collections import Counter

# the location of the Dataset folder should be kept constant
def return_holstein_dataset_path(parent_directory_name):
    
    current_directory = os.getcwd()

    parent_directory_path = os.path.abspath(current_directory)
    while os.path.basename(parent_directory_path) != parent_directory_name:
        parent_directory_path = os.path.dirname(parent_directory_path)
        if parent_directory_path == '/':
            raise FileNotFoundError(f"Parent directory '{parent_directory_name}' not found in the path")

    # Construct the absolute path to the dataset directory
    dataset_directory = os.path.join(parent_directory_path, "Datasets", "RawDataset")

    csv_files = glob.glob(os.path.join(dataset_directory, "*.csv"))

    if csv_files:
        highest_version_file = max(csv_files, key=lambda x: int(x.split("_v")[-1].split(".")[0]), default=None)
        if highest_version_file:
            return os.path.abspath(highest_version_file)
        else:
            print("No Dataset files found.")
    else:
        print("No CSV Dataset found.")

# find the data amounts per subject
def get_data_amounter_per_subject(data):
    # Determine all unique behavior names
    all_behaviors = set()
    for behaviors in data.values():
        all_behaviors.update(behaviors.keys())

    # Flatten the dictionary structure
    flattened_data = []
    for subject_id, behaviors in data.items():
        row = {'subject_id': subject_id}
        for behavior in all_behaviors:
            row[behavior] = len(behaviors.get(behavior, []))
        flattened_data.append(row)

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(flattened_data)

    # Replace missing values with 0
    df.fillna(0, inplace=True)

    # Ensure all values are integers
    df = df.astype({behavior: 'int' for behavior in all_behaviors})

    return df

# calculate magnitude using X,Y,Z axes readings
def calculate_magnitude(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)


# calculate VeDBA using X,Y,Z axes readings
def calculate_VeDBA(dinamic_x, dinamic_y, dinamic_z):
    return np.sqrt(dinamic_x**2 + dinamic_y**2 + dinamic_z**2)


# calculate ODBA using X,,Y,Z axes readings
def calculate_ODBA(dinamic_x, dinamic_y, dinamic_z):
    return abs(dinamic_x) + abs(dinamic_y) + abs(dinamic_z)


'''(see illustration https://howthingsfly.si.edu/flight-dynamics/roll-pitch-and-yaw)
    X = Side to Side - Z (our case)
    Y = Front to back - Y (our case)
    Z = Top to Bottom - X (our case)
    
    A similar implementation can be found at:
    https://engineering.stackexchange.com/questions/3348/calculating-pitch-yaw-and-roll-from-mag-acc-and-gyro-data
    '''

# calculate the pitch of a movement based on X,Y,Z axes readings
def calculate_pitch(static_x, static_y, static_z):
    return (
        180
        * math.atan2(static_z, np.sqrt(static_y * static_y + static_x * static_x))
        / math.pi
    )

# calculate the roll of a movement based on X,Y,Z axes readings
def calculate_roll(static_x, static_y, static_z):
    return (
        180
        * math.atan2(static_y, np.sqrt(static_x * static_x + static_z * static_z))
        / math.pi
    )


# return the raw data belonging to a set of labels as data, labels
def combine_class_data(data_dict, keys = ['accX', 'accY', 'accZ', 'adjMag' ,'ODBA', 'VeDBA', 'pitch', 'roll'], trim=0):
    
    def trim_data(data):
        return [data[key][:trim] if trim > 0 else data[key] for key in keys]
    
    X = [trim_data(data) for _, data_set in data_dict.items() for data in data_set]
    y = [label for label, data_set in data_dict.items() for _ in data_set]
 
    return X, y


# return the feature data belonging to a set of labels as data, labels
def combine_feature_data(data_dict):
    X = [data for _, data_set in data_dict.items() for data in data_set]
    y = [label for label, data_set in data_dict.items() for _ in data_set]
 
    X_array = np.array(X, dtype=object)  # dtype=object for mixed-length sequences
    y_array = np.array(y)
    
    return X, y 

# identifying the calves with data for all the considered labels
def get_test_calves(data_amounts_df, CONSIDERED_LABELS):
    test_calves = (
        data_amounts_df[data_amounts_df > 0]  
        .groupby('calf_id')                   
        .filter(lambda x: (x > 0).all().all()) # Filtering groups where all values are > 0
        ['calf_id']                           
        .unique()                             
    )
            
    return test_calves

# Generate all possible combinations of calves
def generate_calf_sets(all_calves, num_to_select):
    calf_combinations = list(combinations(all_calves, num_to_select))
    return calf_combinations


# train_test_ratio = train/test = if split size = 0.2, 100- 80/20 = 4
def find_optimal_calf_combinations_for_split(all_calves, num_to_select, data_amounts_df, train_test_ratio, 
                                             is_test_set=True, cv=10):
    
    all_calf_combinations = generate_calf_sets(all_calves, num_to_select)
    
    deviations = {}
    
    for combination in all_calf_combinations:
        test_counts = data_amounts_df[data_amounts_df.calf_id.isin(combination)].sum().values[1:]
        train_counts = data_amounts_df[~data_amounts_df.calf_id.isin(combination)].sum().values[1:]
        
        # identifying labels with zeor data points in the test set
        mask = np.array(test_counts) != 0

        # filtering out the labels with zeor label counts in the test set
        filtered_train_counts = np.array(train_counts)[mask]

         # filtering out the labels with zeor label counts
        filtered_test_counts = np.array(test_counts)[mask]

        train_test_label_ratios = np.array(filtered_train_counts) / np.array(filtered_test_counts)

        deviation = np.abs(train_test_label_ratios - train_test_ratio)

        mean_deviation = np.mean(deviation)
        
        deviations[mean_deviation] = combination
        
    if is_test_set:
        min_deviation = min(deviations.keys())
        return deviations[min_deviation]
    else:
        sorted_deviations = sorted(deviations.items())[:cv]
        return [value for key, value in sorted_deviations]

# check for duplicats in an array of array
def check_for_duplicates(array_of_arrays):
    seen = set()
    
    for sub_arr in array_of_arrays:
        sorted_arr = tuple(sorted(sub_arr))
        if sorted_arr in seen:
            return 'Error: Duplicated found'
        seen.add(sorted_arr)
    
    return 'Success: No Duplicates found'


# checks for anything other than numbers including None, nan and inf
def contains_non_numeric(arr):
    arr = np.array([np.nan if x is None else x for x in arr])
    return np.isnan(arr).any()


def set_rcParams(x,y,font_size=12):
    plt.rcParams['figure.figsize'] = (x,y)
    plt.rcParams['font.size'] = font_size