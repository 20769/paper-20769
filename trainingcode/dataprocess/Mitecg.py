import torch
import os
import numpy as np
import sys
sys.path.append('./')


class MitecgDataset(torch.utils.data.Dataset):
    def __init__(self, X, times, y, seq_len=30, augment_negative=None):
        self.X = X  # Shape: (T, N, d) or (T,,d)
        self.times = times  # Shape: (T)
        self.y = y  # Shape: (N,)
        self.seq_len = seq_len
        self.d_inp = self.X.shape[-1]

        if augment_negative is not None:
            mu, std = X.mean(dim=1), X.std(dim=1, unbiased=True)
            num = int(self.X.shape[1] * augment_negative)
            Xnull = torch.stack([mu + torch.randn_like(std) * std for _ in range(num)], dim=1).to(self.X.device)
            self.X = torch.cat([self.X, Xnull], dim=1)
            extra_times = torch.arange(self.X.shape[0]).to(self.X.device)
            self.times = torch.cat([self.times, extra_times.unsqueeze(1).repeat(1, num)], dim=-1)
            self.y = torch.cat([self.y, torch.ones(num, device=self.X.device).long() * 2], dim=0)

    def __len__(self):
        return self.X.shape[1]

    def __getitem__(self, idx):
        x = self.X[-self.seq_len:, idx, :]
        T = self.times[-self.seq_len:, idx]
        y = self.y[idx]
        if len(y.shape) == 1:
            y = y.unsqueeze(0)
        return x, T, y

mitecg_base_path = './dataset/MITECG'

def process_Mitecg(split_no=1, device=None, hard_split=True, normalize=False, exclude_pac_pvc=False, balance_classes=False, div_time=False, need_binarize=False, base_path=mitecg_base_path):
    split_path = f'split={split_no}.pt'
    idx_train, idx_val, idx_test = torch.load(os.path.join(base_path, split_path))
    if hard_split:
        X = torch.load(os.path.join(base_path, 'all_data/X.pt'))
        y = torch.load(os.path.join(base_path, 'all_data/y.pt')).squeeze()
        times = (torch.arange(360) / 60.0).repeat(X.shape[1], 1).T
        saliency = torch.load(os.path.join(base_path, 'all_data/saliency.pt'))
    else:
        X, times, y = torch.load(os.path.join(base_path, 'all_data.pt'))

    Ptrain, time_train, ytrain = X[:, idx_train, :].float(), times[:, idx_train], y[idx_train].long()
    Pval, time_val, yval = X[:, idx_val, :].float(), times[:, idx_val], y[idx_val].long()
    Ptest, time_test, ytest = X[:, idx_test, :].float(), times[:, idx_test], y[idx_test].long()

    if normalize:
        mu, std = Ptrain.mean(), Ptrain.std()
        Ptrain, Pval, Ptest = (Ptrain - mu) / std, (Pval - mu) / std, (Ptest - mu) / std

    if div_time:
        time_train, time_val, time_test = time_train / 60.0, time_val / 60.0, time_test / 60.0

    if exclude_pac_pvc:
        train_mask_in, val_mask_in, test_mask_in = ytrain < 3, yval < 3, ytest < 3
        Ptrain, time_train, ytrain = Ptrain[:, train_mask_in, :], time_train[:, train_mask_in], ytrain[train_mask_in]
        Pval, time_val, yval = Pval[:, val_mask_in, :], time_val[:, val_mask_in], yval[val_mask_in]
        Ptest, time_test, ytest = Ptest[:, test_mask_in, :], time_test[:, test_mask_in], ytest[test_mask_in]

    if need_binarize:
        ytrain, yval, ytest = (ytrain > 0).long(), (yval > 0).long(), (ytest > 0).long()

    if balance_classes:
        def balance_data(P, time, y):
            diff_to_mask = (y == 0).sum() - (y == 1).sum()
            mask_out = (y == 0).nonzero(as_tuple=True)[0][:diff_to_mask]
            to_mask_in = torch.tensor([i not in mask_out for i in range(P.shape[1])])
            return P[:, to_mask_in, :], time[:, to_mask_in], y[to_mask_in]

        Ptrain, time_train, ytrain = balance_data(Ptrain, time_train, ytrain)
        Pval, time_val, yval = balance_data(Pval, time_val, yval)
        Ptest, time_test, ytest = balance_data(Ptest, time_test, ytest)

    train_chunk = ECGchunk(Ptrain, None, time_train, ytrain, device=device)
    val_chunk = ECGchunk(Pval, None, time_val, yval, device=device)
    test_chunk = ECGchunk(Ptest, None, time_test, ytest, device=device)

    gt_exps = saliency.transpose(0, 1).unsqueeze(-1)[:, idx_test, :]
    if exclude_pac_pvc:
        gt_exps = gt_exps[:, test_mask_in, :]
    return train_chunk, val_chunk, test_chunk

def load_Mitecg(batch_size, seq_len=100, split_no=[1, 2, 3, 4, 5, 6], device=None, base_path=mitecg_base_path, print_shape=True, num_workers=16):
    train_X_list, train_time_list, train_y_list = [], [], []
    val_X_list, val_time_list, val_y_list = [], [], []
    test_X_list, test_time_list, test_y_list = [], [], []

    for i in split_no:
        train, val, test = process_Mitecg(split_no=i, device=device, base_path=base_path)
        train_X_list.append(train.X)
        train_time_list.append(train.time)
        train_y_list.append(train.y)
        val_X_list.append(val.X)
        val_time_list.append(val.time)
        val_y_list.append(val.y)
        test_X_list.append(test.X)
        test_time_list.append(test.time)
        test_y_list.append(test.y)

    train_X, train_time, train_y = torch.cat(train_X_list, dim=1), torch.cat(train_time_list, dim=1), torch.cat(train_y_list, dim=-1)
    val_X, val_time, val_y = torch.cat(val_X_list, dim=1), torch.cat(val_time_list, dim=1), torch.cat(val_y_list, dim=-1)
    test_X, test_time, test_y = torch.cat(test_X_list, dim=1), torch.cat(test_time_list, dim=1), torch.cat(test_y_list, dim=-1)

    train_dataset = MitecgDataset(train_X, train_time, train_y, seq_len=seq_len)
    val_dataset = MitecgDataset(val_X, val_time, val_y, seq_len=seq_len)
    test_dataset = MitecgDataset(test_X, test_time, test_y, seq_len=seq_len)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    if print_shape:
        print(f'dataset length: {len(train_loader.dataset)}')
        for batch in train_loader:
            d_inp = batch[0].shape[-1]
            max_len = batch[0].shape[-2]
            total_len = batch[0].shape[-2]
            n_classes = max(batch[2].unique()) + 1 if len(batch[2].shape) == 1 else batch[2].shape[1].item()
        n_classes = 5
    else:
        d_inp = train_loader.dataset.d_inp
        max_len = train_loader.dataset.seq_len
        total_len = train_loader.dataset.X.shape[0]
        n_classes = 2 if len(train_loader.dataset.y.shape) == 1 else train_loader.dataset.y.shape[1].item()

    return train_loader, val_loader, test_loader, d_inp, max_len, total_len, n_classes



