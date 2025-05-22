import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from torch.utils.data import DataLoader
from torch.utils.data import random_split


data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = flag != 'test'
    drop_last = True
    batch_size = args.batch_size
    freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, 0, args.pred_step],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        max_n=args.max_n,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader

class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)

    def __getattr__(self, attr):
        try:
            value = self[attr]
            if isinstance(value, dict):
                value = DotDict(value)
            return value
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{attr}'")
    
    def __setattr__(self, attr, value):
        if isinstance(value, dict):
            value = DotDict(value)
        self[attr] = value
    
    def __delattr__(self, attr):
        try:
            del self[attr]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{attr}'")

default_args = {
    'data': 'custom',
    'root_path': 'dataset/all_six_datasets/electricity',
    'data_path': 'electricity.csv',
    'features': 'M',
    'seq_len': 96,
    'label_len': 96,
    'pred_step': 1,
    'batch_size': 32,
    'num_workers': 0,
    'embed': 'timeF',
    'freq': 'h',
    'target': 'OT',
    'max_n': 25,
}

base_path = './dataset/all_six_datasets/'
def load_forcast(batch_size, data_name, base_path=base_path, max_n=25, print_shape=True, num_workers=0, seq_len=100, pred_step=1):
    args = default_args
    if 'ETT' in data_name:
        args['data'] = data_name
        args['root_path'] = base_path + 'ETT-small'
    else:
        args['data'] = 'custom'
        args['root_path'] = base_path + data_name
    args['seq_len'] = seq_len
    args['batch_size'] = batch_size
    args['num_workers'] = num_workers
    args['data_path'] = data_name + '.csv'
    args['max_n'] = max_n
    args['pred_step'] = pred_step

    args = DotDict(args)
    train_set, _ = data_provider(args, "train")
    vali_data, _ = data_provider(args, flag='val')
    test_data, _ = data_provider(args, flag='test')

    # 合并数据集
    full_dataset = train_set + vali_data + test_data

    # 数据集划分比例
    total_len = len(full_dataset)
    train_len = int(total_len * 0.7)  # 70% 用于训练
    val_len = int(total_len * 0.2)   # 20% 用于验证
    test_len = total_len - train_len - val_len  # 剩余 10% 用于测试

    # 随机划分数据集
    train_set, vali_set, test_set = random_split(full_dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    vali_loader = DataLoader(vali_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for batch in train_loader:
        d_inp = batch[0].shape[2]
        max_len = batch[0].shape[1]
        total_len = batch[0].shape[1]
        if print_shape:
            print(f'X:{batch[0].shape}')
            print(f'time:{batch[1].shape}')
            print(f'y:{batch[2].shape}')
            print(f'dataset length: {len(train_loader.dataset)}')
    return train_loader, vali_loader, test_loader, d_inp, max_len, total_len, d_inp
