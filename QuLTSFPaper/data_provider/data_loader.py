import os
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_infos, time_features
import warnings

warnings.filterwarnings('ignore')

class DatasetStandard(Dataset):
    """Class to preprocess train/val/test datasets."""
    
    def __init__(self, flag, size, forecasting_task, data_path, 
                 target, split_ratio=0.9,freq='h', scale=True, timeenc=0):
        
        assert flag in ['train', 'test']
        type_map = {'train': 0, 'test': 1}
        self.set_type = type_map[flag]
        
        # size [seq_len,  pred_len]
        self.seq_len = size[0]
        self.pred_len = size[1]
        self.forecasting_task = forecasting_task
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.data_path = data_path
        self.split_ratio = split_ratio
        self.__read_data__()

    def __read_data__(self):
        #### read raw data
        df_raw = pd.read_csv(self.data_path)
        df_raw.columns = df_raw.columns.str.lower()
        
        #### to put the target at the end
        cols = list(df_raw.columns)
        cols.remove(self.target.lower())
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target.lower()]]
        
        #### split into train/val/test
        num_train = int(len(df_raw) * self.split_ratio)
        
        #### define the limits of the splits (take into account history context for val/test)
        border1s = [0, num_train - self.seq_len]
        border2s = [num_train, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        #### specify data to use (whether to use other features for pred or not)
        if self.forecasting_task == 'M' or self.forecasting_task == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.forecasting_task == 'S':
            df_data = df_raw[[self.target.lower()]]
            
        #### scale data based on the distribution of trainset
        self.scaler_train = StandardScaler()
        self.scaler_train.fit(df_data[border1s[0]:border2s[0]].values)  # Fit on train data
        if self.scale:
            data = self.scaler_train.transform(df_data.values)  # Scale using training scaler
        else:
            data = df_data.values
        # Now we have a numpy.ndarray
        
        #### get data_stamp which is an array with additional columns: month, day, weekday, hour
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            data_stamp = time_infos(df_stamp, freq=self.freq) # add month, day, week ... info
        elif self.timeenc == 1: # Extract time info + apply preprocessing to make time features between -0.5 and 0.5
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        #### we get the data and the time stamps we will use
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index # start index for the seq input
        s_end = s_begin + self.seq_len # end endex of the seq input (seq_len ahead)
        r_begin = s_end # start index for the target sequence 
        r_end = r_begin + self.pred_len # end index for the target sequence
        
        #### specify data for prediction
        seq_x = self.data_x[s_begin:s_end] # input seq
        seq_y = self.data_y[r_begin:r_end] # target seq
        seq_x_mark = self.data_stamp[s_begin:s_end] # timestamps of input seq
        seq_y_mark = self.data_stamp[r_begin:r_end] # timestamps of target seq

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler_train.inverse_transform(data)  
    

class DatasetWithOverlap(Dataset):
    """Class to preprocess train/val/test datasets."""
    
    def __init__(self, flag, size, forecasting_task, data_path, 
                 target, freq='h', overlap=2, scale=True, timeenc=0):
        
        assert flag in ['train', 'test']
        type_map = {'train': 0, 'test': 1}
        self.set_type = type_map[flag]
        
        # size [seq_len,  pred_len]
        self.seq_len = size[0]
        self.pred_len = size[1]
        self.forecasting_task = forecasting_task
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.overlap = overlap
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        #### read raw data
        df_raw = pd.read_csv(self.data_path)
        df_raw.columns = df_raw.columns.str.lower()
        
        #### to put the target at the end
        cols = list(df_raw.columns)
        cols.remove(self.target.lower())
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target.lower()]]
        
        #### split into train/val/test
        num_train = int(len(df_raw) * 0.9)
        
        #### define the limits of the splits (take into account history context for val/test)
        border1s = [0, num_train - self.seq_len]
        border2s = [num_train, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        #### specify data to use (whether to use other features for pred or not)
        if self.forecasting_task == 'M' or self.forecasting_task == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.forecasting_task == 'S':
            df_data = df_raw[[self.target.lower()]]
            
        #### scale data based on the distribution of trainset
        self.scaler_train = StandardScaler()
        self.scaler_train.fit(df_data[border1s[0]:border2s[0]].values)  # Fit on train data
        if self.scale:
            data = self.scaler_train.transform(df_data.values)  # Scale using training scaler
        else:
            data = df_data.values
        # Now we have a numpy.ndarray
        
        #### get data_stamp which is an array with additional columns: month, day, weekday, hour
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            data_stamp = time_infos(df_stamp, freq=self.freq) # add month, day, week ... info
        elif self.timeenc == 1: # Extract time info + apply preprocessing to make time features between -0.5 and 0.5
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        #### we get the data and the time stamps we will use
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index # start index for the seq input
        s_end = s_begin + self.seq_len # end endex of the seq input (seq_len ahead)
        r_begin = s_end - self.overlap # start index for the target sequence 
        r_end = s_end + self.pred_len - self.overlap # end index for the target sequence
        
        #### specify data for prediction
        seq_x = self.data_x[s_begin:s_end] # input seq
        seq_y = self.data_y[r_begin:r_end] # target seq
        seq_x_mark = self.data_stamp[s_begin:s_end] # timestamps of input seq
        seq_y_mark = self.data_stamp[r_begin:r_end] # timestamps of target seq

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler_train.inverse_transform(data)  