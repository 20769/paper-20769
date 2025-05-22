import numpy as np
import random
import torch
import os

def save_data_from_dataloader(dataloader, save_path):
    all_data = []
    all_labels = []
    all_times = []

    for x, T, y in dataloader:
        all_data.append(x.cpu().numpy())
        all_labels.append(y.cpu().numpy())
        all_times.append(T.cpu().numpy())
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_times = np.concatenate(all_times, axis=0)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path + '_data.npy', all_data)
    np.save(save_path + '_labels.npy', all_labels)
    np.save(save_path + '_times.npy', all_times)

from torch.utils.data import Dataset, DataLoader

class CacheDataset(Dataset):
    def __init__(self, data_path, labels_path, time_path):
        self.data = np.load(data_path)
        self.labels = np.load(labels_path)
        self.time = np.load(time_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        time = torch.tensor(self.time[idx], dtype=torch.float32)
        return data, time, label

def load_data_as_dataloader(data_path, labels_path, time_path, batch_size=32, num_workers=16, shuffle=True):
    dataset = CacheDataset(data_path, labels_path, time_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

class ECGchunk:
    '''
    Class to hold chunks of ECG data
    '''
    def __init__(self, train_tensor, static, time, y, device = None):
        self.X = train_tensor.to(device)
        self.static = None if static is None else static.to(device)
        self.time = time.to(device)
        self.y = y.to(device)

    def choose_random(self):
        n_samp = self.X.shape[1]           
        idx = random.choice(np.arange(n_samp))

        static_idx = None if self.static is None else self.static[idx]
        #print('In chunk', self.time.shape)
        return self.X[idx,:,:].unsqueeze(dim=1), \
            self.time[:,idx].unsqueeze(dim=-1), \
            self.y[idx].unsqueeze(dim=0), \
            static_idx

    def get_all(self):
        static_idx = None # Doesn't support non-None 
        return self.X, self.time, self.y, static_idx

    def __getitem__(self, idx): 
        static_idx = None if self.static is None else self.static[idx]
        return self.X[:,idx,:], \
            self.time[:,idx], \
            self.y[idx].unsqueeze(dim=0)
            #static_idx

def getStats(P_tensor):
    if len(P_tensor.shape) == 2:
        P_tensor = P_tensor.unsqueeze(0)
    N, T, F = P_tensor.shape
    if isinstance(P_tensor, np.ndarray):
        Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    else:
        Pf = P_tensor.permute(2, 0, 1).reshape(F, -1).detach().clone().cpu().numpy()
    mf = np.zeros((F, 1))
    stdf = np.ones((F, 1)).squeeze()
    eps = 1e-7
    for f in range(F):
        vals_f = Pf[f, :]
        vals_f = vals_f[vals_f > 0]
        mf[f] = np.mean(vals_f)
        stdf[f] = np.std(vals_f)
        stdf[f] = np.max([stdf[f], eps])
    return mf, stdf

def tensorize_normalize_ECG(P, y, mf, stdf):
    F, T = P[0].shape

    P_time = np.zeros((len(P), T, 1))
    for i in range(len(P)):
        tim = torch.linspace(0, T, T).reshape(-1, 1)
        P_time[i] = tim
    P_tensor = mask_normalize_ECG(P, mf, stdf)
    P_tensor = torch.Tensor(P_tensor)

    P_time = torch.Tensor(P_time) / 60.0

    y_tensor = y
    y_tensor = y_tensor.type(torch.LongTensor)
    return P_tensor, None, P_time, y_tensor

def tensorize_normalize_Math(P, y, mf, stdf):
    N, F = P.shape

    # Normalize P tensor
    P_tensor = mask_normalize_Math(P, mf, stdf)
    P_tensor = torch.Tensor(P_tensor)

    # Convert y to tensor
    y_tensor = torch.Tensor(y).type(torch.LongTensor)

    # check nan
    where_are_NaNs = torch.isnan(P_tensor)
    P_tensor[where_are_NaNs] = 0


    return P_tensor, None, y_tensor

def mask_normalize_ECG(P_tensor, mf, stdf):
    """ Normalize time series variables. Missing ones are set to zero after normalization. """
    P_tensor = P_tensor.numpy()
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2,0,1)).reshape(F,-1)
    M = 1*(P_tensor>0) + 0*(P_tensor<=0)
    M_3D = M.transpose((2, 0, 1)).reshape(F, -1)
    for f in range(F):
        Pf[f] = (Pf[f]-mf[f])/(stdf[f]+1e-18)
    Pf = Pf * M_3D
    Pnorm_tensor = Pf.reshape((F,N,T)).transpose((1,2,0))
    return Pnorm_tensor

def mask_normalize_Math(P_tensor, mf, stdf):
    """ Normalize time series variables for Math dataset. Missing ones are set to zero after normalization. """
    P_tensor = P_tensor.numpy()
    N, F = P_tensor.shape
    M = 1 * (P_tensor > 0) + 0 * (P_tensor <= 0)
    
    # Normalize each feature
    for f in range(F):
        P_tensor[:, f] = (P_tensor[:, f] - mf[f]) / (stdf[f] + 1e-18)
    
    # Apply mask
    P_tensor = P_tensor * M
    
    return P_tensor

def mask_normalize(P_tensor, mf, stdf):
    """ Normalize time series variables. Missing ones are set to zero after normalization. """
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2,0,1)).reshape(F,-1)
    M = 1*(P_tensor>0) + 0*(P_tensor<=0)
    M_3D = M.transpose((2, 0, 1)).reshape(F, -1)
    for f in range(F):
        Pf[f] = (Pf[f]-mf[f])/(stdf[f]+1e-18)
    Pf = Pf * M_3D
    Pnorm_tensor = Pf.reshape((F,N,T)).transpose((1,2,0))
    Pfinal_tensor = np.concatenate([Pnorm_tensor, M], axis=2)
    return Pfinal_tensor

import torch

def time_series_split(src, time, max_len, step_len=1, device=None):
    # Initialize lists to hold output sequences
    src_sequences = []
    time_sequences = []

    # Check if src has a batch dimension
    if len(src.shape) == 2:
        # No batch dimension, add one for consistency
        src = src.unsqueeze(0)
        time = time.unsqueeze(0)
        b = 1
    else: 
        b = src.shape[0]
    src_sequences = [[] for i in range(b)]
    time_sequences = [[] for i in range(b)]
    # Iterate over each batch
    for b in range(src.shape[0]):
        # Iterate over the source data with a stride of step_len
        for i in range(0, src.shape[1] - max_len, step_len):
            # Extract a sequence from the source data
            src_seq = src[b, i:i + max_len]
            # Extract the corresponding times
            time_seq = time[b, i:i + max_len]

            # Append the sequences to the output lists
            src_sequences[b].append(src_seq)
            time_sequences[b].append(time_seq)

    # Convert the lists to tensors
    src_sequences = torch.stack([torch.stack(batch) for batch in src_sequences])
    time_sequences = torch.stack([torch.stack(batch) for batch in time_sequences])
    if b == 0:
        src_sequences = src_sequences.squeeze(0)
        time_sequences = time_sequences.squeeze(0)

    # If a device is specified, move the tensors to that device
    if device is not None:
        src_sequences = src_sequences.to(device)
        time_sequences = time_sequences.to(device)

    return src_sequences, time_sequences #shape (batch, seq_len, max_len, features) (batch, seq_len, max_len)

def restore_time_series(src_sequences, time_sequences, step_len=1):
    """
    Restore the original time series from the split sequences.

    Args:
        src_sequences (torch.Tensor): Split source sequences of shape (batch, seq_len, max_len, features).
        time_sequences (torch.Tensor): Split time sequences of shape (batch, seq_len, max_len).
        step_len (int): Step length used during the splitting.

    Returns:
        torch.Tensor: Restored source sequences of shape (batch, original_len, features).
        torch.Tensor: Restored time sequences of shape (batch, original_len).
    """
    batch_size, seq_len, max_len, features = src_sequences.shape
    original_len = (seq_len - 1) * step_len + max_len

    restored_src = torch.zeros((batch_size, original_len, features))
    restored_time = torch.zeros((batch_size, original_len))

    for b in range(batch_size):
        for i in range(seq_len):
            start_idx = i * step_len
            end_idx = start_idx + max_len
            restored_src[b, start_idx:end_idx] = src_sequences[b, i]
            restored_time[b, start_idx:end_idx] = time_sequences[b, i]

    return restored_src, restored_time

def restore_mask(mask_sequences, step_len=1):
    """
    Restore the original mask sequences from the split sequences.

    Args:
        mask_sequences (torch.Tensor): Split mask sequences of shape (batch, seq_len, max_len, features).
        step_len (int): Step length used during the splitting.

    Returns:
        torch.Tensor: Restored mask sequences of shape (batch, original_len, features).
    """
    batch_size, seq_len, max_len, features = mask_sequences.shape
    original_len = (seq_len - 1) * step_len + max_len

    restored_mask = torch.zeros((batch_size, original_len, features))

    for b in range(batch_size):
        for i in range(seq_len):
            start_idx = i * step_len
            end_idx = start_idx + max_len
            restored_mask[b, start_idx:end_idx] = mask_sequences[b, i]

    return restored_mask