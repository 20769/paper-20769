import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import sys
sys.path.append('./')

class BoneAgeDataset(Dataset):
    def __init__(self, data, seq_len, data_type='Distal', max_len=10, step_len=1):
        data = torch.tensor(data, dtype=torch.float32)
        normal_data_name = ['UWaveGestureLibraryAll', 'MixedShapesSmallTrain']

        self.X = data[:, 1:].unsqueeze(-1).permute(1, 0, 2)
        self.times = torch.linspace(0, 2 * np.pi, self.X.shape[0]).unsqueeze(1).repeat(1, self.X.shape[1])
        self.y = data[:, 0]

        if data_type in normal_data_name:
            self.y -= 1
        elif data_type == 'Wafer':
            self.y = (self.y > 0).float()

        self.seq_len = seq_len
        self.step_len = step_len
        self.max_len = max_len
        self.d_inp = self.X.shape[-1]
    
    def __len__(self):
        return self.X.shape[1]

    def __getitem__(self, idx):
        x = self.X[-self.seq_len:, idx, :]
        T = self.times[-self.seq_len:, idx]
        y = self.y[idx].long()

        if len(y.shape) == 1:
            y = y.unsqueeze(0)
        x = self.data_augmentation(x)

        return x, T, y

    def data_augmentation(self, x):
        noise = torch.randn_like(x) * 0.1 

        if np.random.rand() > 0.5:
            shift_amount = np.random.randint(-2, 3) 
            if shift_amount > 0:
                shift_amount = min(shift_amount, x.shape[0])  # Ensure shift_amount does not exceed sequence length
                x = torch.cat((x[shift_amount:], torch.zeros(shift_amount, x.shape[1])), dim=0)
            elif shift_amount < 0:
                shift_amount = max(shift_amount, -x.shape[0])  # Ensure shift_amount does not exceed sequence length
                x = torch.cat((torch.zeros(-shift_amount, x.shape[1]), x[:shift_amount]), dim=0)

        if np.random.rand() > 0.5:
            scale_factor = np.random.uniform(0.8, 1.2)
            x *= scale_factor

        return x

def process_bone_age(data_path, batch_size, seq_len, data_type='Distal', test_size=0.2, scaler_type='minmax', threshold=0.25, n_max=100):
    if data_type == 'Wafer':
        train_data_path = data_path + 'Wafer_TRAIN.tsv'
        test_data_path = data_path + 'Wafer_TEST.tsv'
        train_data = pd.read_csv(train_data_path, sep='\t', header=None).values
        test_data = pd.read_csv(test_data_path, sep='\t', header=None).values
        data_np = np.concatenate((train_data, test_data), axis=0)
    elif data_type == 'Distal':
        train_data_path = data_path + 'DistalPhalanxOutlineCorrect_TRAIN.tsv'
        test_data_path = data_path + 'DistalPhalanxOutlineCorrect_TEST.tsv'
        train_data = pd.read_csv(train_data_path, sep='\t', header=None).values
        test_data = pd.read_csv(test_data_path, sep='\t', header=None).values
        data_np = np.concatenate((train_data, test_data), axis=0)
    elif data_type == 'MixedShapesSmallTrain':
        train_data_path = data_path + 'MixedShapesSmallTrain_TRAIN.tsv'
        test_data_path = data_path + 'MixedShapesSmallTrain_TEST.tsv'
        train_data = pd.read_csv(train_data_path, sep='\t', header=None).values
        test_data = pd.read_csv(test_data_path, sep='\t', header=None).values
        data_np = np.concatenate((train_data, test_data), axis=0)
    elif data_type == 'UWaveGestureLibraryAll':
        train_data_path = data_path + 'UWaveGestureLibraryAll_TRAIN.tsv'
        test_data_path = data_path + 'UWaveGestureLibraryAll_TEST.tsv'
        train_data = pd.read_csv(train_data_path, sep='\t', header=None).values
        test_data = pd.read_csv(test_data_path, sep='\t', header=None).values
        data_np = np.concatenate((train_data, test_data), axis=0)
    else:
        raise ValueError('data type error!')

    dataset = BoneAgeDataset(data_np, seq_len=seq_len, data_type=data_type)

    train_size = int((1 - test_size) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    val_size = int(0.3 * len(test_dataset))
    test_size = len(test_dataset) - val_size
    val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [val_size, test_size])

    return train_dataset, val_dataset, test_dataset

def load_bone_age(batch_size, seq_len, data_type='Distal', test_size=0.2, scaler_type='minmax', threshold=0.25, n_max=100, print_shape=True, device='cuda', reload=False, save_path='./dataset/DistalPhalanxOutlineCorrect/', num_workers=0):
    data_paths = {
        'Wafer': './dataset/Wafer/',
        'Distal': './dataset/DistalPhalanxOutlineCorrect/',
        'MixedShapesSmallTrain': './dataset/MixedShapesSmallTrain/',
        'UWaveGestureLibraryAll': './dataset/UWaveGestureLibraryAll/'
    }
    
    if data_type not in data_paths:
        raise ValueError('data type error!')
    
    data_path = data_paths[data_type]
    train_dataset, val_dataset, test_dataset = process_bone_age(data_path, batch_size, seq_len, data_type, test_size, scaler_type, threshold, n_max)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    
    if print_shape:
        print(f'dataset length: {len(train_loader.dataset)}')
        for batch in train_loader:
            print(f'X:{batch[0].shape}')
            print(f'time:{batch[1].shape}')
            print(f'y:{batch[2].shape}')
            d_inp = batch[0].shape[-1]
            max_len = batch[0].shape[-2]
            total_len = batch[0].shape[-2]
            if len(batch[2].shape) == 1:
                unique_classes = set()
                for i, batch in enumerate(train_loader):
                    unique_classes.update(batch[2].tolist())
                    if i >= 49:
                        break
                n_classes = len(unique_classes)
            else:
                n_classes = batch[2].shape[1]
            break
    else:
        d_inp = train_loader.dataset.d_inp
        max_len = train_loader.dataset.max_len
        total_len = train_loader.dataset.X.shape[0]
        if len(train_loader.dataset.y.shape) == 1:
            unique_classes = set()
            for i, batch in enumerate(train_loader):
                unique_classes.update(batch[2].tolist())
                if i >= 49:
                    break
            n_classes = len(unique_classes)
        else:
            n_classes = train_loader.dataset.y.shape[1]
    
    return train_loader, val_loader, test_loader, d_inp, max_len, total_len, n_classes
