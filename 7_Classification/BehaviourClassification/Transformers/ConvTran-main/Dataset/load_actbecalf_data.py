import sys
sys.path.append('../../../')

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

dataset_path = '../../../../Datasets/SixBehaviourClassification/six_label_window_dataset_v1.pickle'
calf_set_info_path = '../../../../Datasets/SixBehaviourClassification/six_label_calf_split_info.pkl'

max_length = 75 # 3*25

def preprocess_data(data_dict, max_length, scaler=None, fit_scaler=False):
    X = []
    y = []
    
    for subject_id, labels_dict in data_dict.items():
        for label, df_list in labels_dict.items():
            for df in df_list:
                features = df[['accX', 'accY', 'accZ', 'adjMag', 'ODBA', 'VeDBA', 'pitch', 'roll']].values
                
                X.append(features)
                y.append(label)
    
    # Pad sequences to ensure they have the same length
    X = pad_sequences(X, maxlen=max_length, dtype='float32', padding='post', truncating='post')
    
    y = np.array(y)
    
    # Standardize the feature data
    num_features = X.shape[2]
    # Reshape X to 2D array for standardization (ignoring padding)
    X_reshaped = X.reshape(-1, num_features)
    
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reshaped)
    else:
        X_scaled = scaler.transform(X_reshaped)
    
    X = X_scaled.reshape(-1, max_length, num_features)
    
    return X, y, scaler


def load(config):
    # dataset
    with open(dataset_path, 'rb') as f:
        window_dataset = pickle.load(f)

    # calf train:valid:test set information
    with open(calf_set_info_path, 'rb') as f:
        calf_set_info = pickle.load(f)
        
    all_calf_ids = calf_set_info['all_calves']
    test_calf_ids = calf_set_info['test_calves']
    valid_calf_ids = calf_set_info['valid_calf_id_sets'][0]
    
    train_dict = {}
    test_dict = {}
    val_dict = {}

    for calf in window_dataset.keys():
        if calf in test_calf_ids:
            test_dict[calf] = window_dataset[calf]
        elif calf in valid_calf_ids:
            val_dict[calf] = window_dataset[calf]
        else:
            train_dict[calf] = window_dataset[calf]
            
    X_train, y_train, scaler = preprocess_data(train_dict, max_length, fit_scaler=True)
    X_val, y_val, _ = preprocess_data(val_dict, max_length, scaler=scaler, fit_scaler=False)
    X_test, y_test, _ = preprocess_data(test_dict, max_length, scaler=scaler, fit_scaler=False)
    
    # Encode string labels to integers
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)
    
    Data = {}
    Data['train_data'] = X_train
    Data['train_label'] = y_train_encoded
    Data['val_data'] = X_val
    Data['val_label'] = y_val_encoded
    Data['test_data'] = X_test
    Data['test_label'] = y_test_encoded
            
    return Data