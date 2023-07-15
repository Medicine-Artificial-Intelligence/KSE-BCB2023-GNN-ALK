import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
class data_preprocessing:
    def __init__(self, data, raw_dir, smiles_col='Canomicalsmiles', ID_col='ID', target_col='pChEMBL', thresh=7):
        self.data = data[[ID_col,smiles_col,target_col]]
        self.ID_col = ID_col
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.thresh = thresh
        self.raw_dir = raw_dir
        
    def target_bin(self, data, thresh, target_col):
        t1 = data[target_col] < thresh 
        data.loc[t1, target_col] = 0
        t2 = data[target_col] >= thresh 
        data.loc[t2, target_col] = 1
        data[target_col] = data[target_col].astype('int64')
        return data
    
    def split_data(self, data,ID_col, target_col):
        
        data_train, data_test = train_test_split(data, test_size = 0.2,
                                                                random_state =42, stratify=data[target_col])
        
        data_train, data_valid = train_test_split(data, test_size = 0.2,
                                                                random_state =42, stratify=data[target_col])
        data_train.reset_index(drop = True, inplace = True)
        data_test.reset_index(drop = True, inplace = True)
        data_valid.reset_index(drop = True, inplace = True)
        
        return data_train, data_test, data_valid
        
    def fit(self):
        data = self.target_bin(data=self.data, target_col = self.target_col, thresh = self.thresh)
        self.data_train, self.data_test, self.data_valid = self.split_data(data=data, ID_col = self.ID_col, target_col = self.target_col)
        self.data_train.to_csv(self.raw_dir+'/train.csv')
        self.data_test.to_csv(self.raw_dir+'/test.csv')
        self.data_valid.to_csv(self.raw_dir+'/valid.csv')
        