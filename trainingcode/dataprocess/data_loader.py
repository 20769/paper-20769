import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from timefeatures import time_features
import warnings
import numpy as np
from einops import rearrange
import torch
import torch.nn.functional as F



warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', max_n=20):

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_step = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_step = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.len = self.__len__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp[:,0]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_step

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]
        
        index_list = np.arange(index,index+self.seq_len+self.pred_step,1)
        
        seq_x = seq_x.astype('float32')
        seq_y = seq_y.astype('float32')
        seq_x_mark = seq_x_mark.astype('float32')
        ## normalize the index
        # norm_index = index_list / self.len
        return seq_x, seq_x_mark, seq_y # , seq_y_mark,norm_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_step + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', max_n=20):
        # size [seq_len, label_len, pred_step]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_step = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_step = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.flag = flag
        self.len = self.__len__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp[:,0]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_step

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]
        
        index_list = np.arange(index,index+self.seq_len+self.pred_step,1)
        
        seq_x = seq_x.astype('float32')
        seq_y = seq_y.astype('float32')
        seq_x_mark = seq_x_mark.astype('float32')
        ## normalize the index
        # norm_index = index_list / self.len
        return seq_x, seq_x_mark, seq_y# , seq_y_mark,norm_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_step + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', max_n=20):
        # size [seq_len, label_len, pred_step]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_step = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_step = size[2]
        # init
        assert flag in ['train', 'test','val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.max_n = max_n
        self.__read_data__()
        self.len = self.__len__()
        


    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            self.times = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            self.times = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            # transpose to [time,]
            self.times = self.times.transpose(1, 0)
            # Convert 2D array to 1D float array
            self.times = self.times.dot(10**np.arange(self.times.shape[-1])[::-1])

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        # create time features into float vector
        input_size = self.data_x.shape[1]
        if input_size > self.max_n:
            self.data_x = self.data_x[:, :self.max_n]
            self.data_y = self.data_y[:, :self.max_n]
    
    def fit(self, data):
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        return mean, std

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        if self.pred_step == 1:
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[s_end]
            times = self.times[s_begin:s_end]
            # turn into float32 
            seq_x = seq_x.astype('float32')
            seq_y = seq_y.astype('float32')
            times = times.astype('float32')
        
            return seq_x, times, seq_y
        else:
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_step

            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            times = self.times[s_begin:s_end]
            seq_x = seq_x.astype('float32')
            seq_y = seq_y.astype('float32')
            times = times.astype('float32')
            return seq_x, times, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_step + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def get_CUTS_dataset(self, data, bs, sample_period=1):
        
        pred_step = self.pred_step
        n_nodes = data.shape[1]
        input_step = self.seq_len
        t, n, d = data.shape    
        first_sample_t = input_step
        random_t_list = np.arange(first_sample_t, t, sample_period).tolist()
        np.random.shuffle(random_t_list)
        observ_mask = torch.ones_like(data)

        for batch_i in range(len(random_t_list) // bs):
            x = torch.zeros([bs, n_nodes, n_nodes, input_step, d]).to(data.device)
            y = torch.zeros([bs, n_nodes, pred_step, d]).to(data.device)
            t = torch.zeros([bs]).to(data.device).long()
            mask = torch.zeros([bs, n_nodes, pred_step, d]).to(data.device)
            for data_i in range(bs):
                data_t = random_t_list.pop()
                x_data = rearrange(data[data_t-input_step : data_t, :], "t n d -> 1 n t d")
                x[data_i, :, :, :] = x_data.expand(n_nodes, -1, -1, -1)
                y_i = data[data_t : data_t+pred_step*sample_period : sample_period, :]
                mask_i = observ_mask[data_t:data_t+pred_step, :]
                if y_i.shape[0] < pred_step:
                    padding_size = pred_step - y_i.shape[0]
                    y_i = F.pad(y_i, (0, 0, 0, 0, 0, padding_size))
                    mask_i = F.pad(mask_i, (0, 0, 0, 0, 0, padding_size))
                    break
                y[data_i, :, :, :] = rearrange(y_i, "t n d -> n t d")
                t[data_i] = data_t
                mask[data_i, :, :, :] = rearrange(observ_mask[data_t:data_t+pred_step, :], "t n d -> n t d")

                yield x, y, t, mask
            
    def get_data(self):
        data = torch.tensor(self.data_x).unsqueeze(-1)
        return data

class Dataset_sin(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='sin.csv',
                 target='y', scale=True, timeenc=0, freq='h'):

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_step = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_step = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        print(cols)
        cols.remove(self.target)
        cols.remove('x')
        df_raw = df_raw[['x'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values


        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_step

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = np.zeros_like(seq_x)
        seq_y_mark = np.zeros_like(seq_y)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_step + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)