import torch
import os
import numpy as np

import sys
sys.path.append('./')
from dataprocess.preprocess import ECGchunk, getStats, tensorize_normalize_ECG, time_series_split
from dataprocess.preprocess import load_data_as_dataloader


class EpiDataset(torch.utils.data.Dataset):
    def __init__(self, X, times, y, seq_len = 96, augment_negative = None):
        self.X = X # Shape: (T, N, d) or (T,,d)
        self.times = times # Shape: (T)
        self.y = y # Shape: (N,)
        self.seq_len = seq_len
        self.d_inp = self.X.shape[-1]

        if augment_negative is not None:
            mu, std = X.mean(dim=1), X.std(dim=1, unbiased = True)
            num = int(self.X.shape[1] * augment_negative)
            Xnull = torch.stack([mu + torch.randn_like(std) * std for _ in range(num)], dim=1).to(self.X.get_device())

            self.X = torch.cat([self.X, Xnull], dim=1)
            extra_times = torch.arange(self.X.shape[0]).to(self.X.get_device())
            self.times = torch.cat([self.times, extra_times.unsqueeze(1).repeat(1, num)], dim = -1)
            self.y = torch.cat([self.y, (torch.ones(num).to(self.X.get_device()).long() * 2)], dim = 0)

    def __len__(self):
        return self.X.shape[1]
    
    def __getitem__(self, idx):
        x = self.X[-self.seq_len:,idx,:]
        T = self.times[-self.seq_len: ,idx]
        y = self.y[idx]
        if len(y.shape) == 1:
            y = y.unsqueeze(0)
        return x, T, y 

def process_Epilepsy(split_no = 1, device = None, base_path = './dataset/Epilepsy/'):


    split_path = 'split_{}.npy'.format(split_no)
    idx_train, idx_val, idx_test = np.load(os.path.join(base_path, split_path), allow_pickle = True)

    X, y = torch.load(os.path.join(base_path, 'all_epilepsy.pt'))

    Ptrain, ytrain = X[idx_train], y[idx_train]
    Pval, yval = X[idx_val], y[idx_val]
    Ptest, ytest = X[idx_test], y[idx_test]

    T, F = Ptrain[0].shape
    D = 1

    Ptrain_static_tensor = np.zeros((len(Ptrain), D))

    mf, stdf = getStats(Ptrain)
    #print('Before tensor_normalize_other', Ptrain.shape)
    Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize_ECG(Ptrain, ytrain, mf, stdf)
    Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize_ECG(Pval, yval, mf, stdf)
    Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize_ECG(Ptest, ytest, mf, stdf)
    #print('After tensor_normalize (X)', Ptrain_tensor.shape)

    Ptrain_tensor = Ptrain_tensor.permute(2, 0, 1)
    Pval_tensor = Pval_tensor.permute(2, 0, 1)
    Ptest_tensor = Ptest_tensor.permute(2, 0, 1)

    #print('Before s-permute', Ptrain_time_tensor.shape)
    Ptrain_time_tensor = Ptrain_time_tensor.squeeze(2).permute(1, 0)
    Pval_time_tensor = Pval_time_tensor.squeeze(2).permute(1, 0)
    Ptest_time_tensor = Ptest_time_tensor.squeeze(2).permute(1, 0)

    train_chunk = ECGchunk(Ptrain_tensor, None, Ptrain_time_tensor, ytrain_tensor, device = device)
    val_chunk = ECGchunk(Pval_tensor, None, Pval_time_tensor, yval_tensor, device = device)
    test_chunk = ECGchunk(Ptest_tensor, None, Ptest_time_tensor, ytest_tensor, device = device)

    return train_chunk, val_chunk, test_chunk

def load_Epilepsy(batch_size, split_no=[1, 2, 3, 4, 5, 6], device=None, base_path='./dataset/Epilepsy/', print_shape=True, num_workers=16, seq_len=96):
    def process_splits(split_no):
        X_list, time_list, y_list = [], [], []
        for i in split_no:
            chunk = process_Epilepsy(split_no=i, device=device, base_path=base_path)
            for i in range(len(chunk)):
                X_list.append(chunk[i].X)
                time_list.append(chunk[i].time)
                y_list.append(chunk[i].y)
        return torch.cat(X_list, dim=1), torch.cat(time_list, dim=1), torch.cat(y_list, dim=-1)
    
    train_X, train_time, train_y = process_splits(split_no)
    val_X, val_time, val_y = process_splits(split_no)
    test_X, test_time, test_y = process_splits(split_no)
    
    train_dataset = EpiDataset(train_X, train_time, train_y, seq_len=seq_len)
    val_dataset = EpiDataset(val_X, val_time, val_y, seq_len=seq_len)
    test_dataset = EpiDataset(test_X, test_time, test_y, seq_len=seq_len)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    if print_shape:
        print(f'dataset length: {len(train_loader.dataset)}')
        for batch in train_loader:
            print(f'X:{batch[0].shape}')
            print(f'time:{batch[1].shape}')
            print(f'y:{batch[2].shape}')
            d_inp = batch[0].shape[2]
            max_len = batch[0].shape[1]
            total_len = batch[0].shape[1]
            n_classes = 2 if len(batch[2].shape) == 1 else batch[2].shape[1]
            break
    else:
        d_inp = train_loader.dataset.d_inp
        max_len = train_loader.dataset.max_len
        total_len = train_loader.dataset.X.shape[0]
        n_classes = 2 if len(train_loader.dataset.y.shape) == 1 else train_loader.dataset.y.shape[1]
    
    return train_loader, val_loader, test_loader, d_inp, max_len, total_len, n_classes
