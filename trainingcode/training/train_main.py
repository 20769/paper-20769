import os, sys
import time # debug
from os.path import join as opj
from os.path import dirname as opd

sys.path.append(opd(opd(__file__)))
from utils.logger import MyLogger, reproduc
from utils.opt_type import MultiCADopt
from training.train import train
from training.test import test
from model.MultiModel import MultiModel 
import random
import torch
from datetime import datetime
from omegaconf import OmegaConf
import argparse
from torch.utils.tensorboard import SummaryWriter
import numpy as np



def prepare_data(opt, device="cuda"):
    start_time = time.time()
    data_path = opt.data_path
    batch_size = opt.batch_size
    num_workers = opt.num_workers
    if opt.name == "Epilepsy":
        from dataprocess.Epilepsy import load_Epilepsy
        split_no = opt.split_no
        train_loader, val_loader, test_loader, d_inp, max_len, total_len, n_classes = load_Epilepsy(
            seq_len=opt.seq_len,
            batch_size=batch_size, split_no=split_no, base_path=data_path, device=device,
              num_workers=num_workers)
    elif opt.name == "Mitecg":
        from dataprocess.Mitecg import load_Mitecg
        split_no = opt.split_no
        train_loader, val_loader, test_loader, d_inp, max_len, total_len, n_classes = load_Mitecg(
            seq_len=opt.seq_len,
            batch_size=batch_size, split_no=split_no, base_path=data_path, device=device, num_workers=num_workers)
    elif opt.name == "UCR":
        from dataprocess.Boneage import load_bone_age
        seq_len = opt.seq_len
        data_type = opt.data_name
        train_loader, val_loader, test_loader, d_inp, max_len, total_len, n_classes = load_bone_age(
            batch_size=batch_size, seq_len=seq_len, data_type=data_type, test_size=0.2, scaler_type='minmax', threshold=0.25, n_max=100, print_shape=True, device=device, save_path=data_path, num_workers=num_workers)
    elif opt.name == "forcast":
        from dataprocess.forcasting import load_forcast
        train_loader, val_loader, test_loader, d_inp, max_len, total_len, n_classes = load_forcast(
            seq_len=opt.seq_len,data_name=opt.data_name,
            batch_size=batch_size, base_path=data_path, num_workers=num_workers, pred_step=opt.pred_step)
    else:
        raise ValueError(f"Unknown dataset name: {opt.name}")
    
    print(f"prepare_data took {time.time() - start_time:.2f} seconds")
    return train_loader, val_loader, test_loader, d_inp, max_len, total_len, n_classes

def prepare_model(opt, d_inp, seq_len, total_len, n_classes, graph, device="cuda"):
    start_time = time.time()
    # check if the input dimension and max length matches the model
    checks = [
        (d_inp, opt.d_inp if opt.d_inp != 'auto' else d_inp, "Input dimension mismatch"),
        (seq_len, opt.max_len if opt.max_len != 'auto' else seq_len, "Max length mismatch"),
        (total_len, opt.total_len if opt.total_len != 'auto' else total_len, "Total length mismatch"),
        (n_classes, opt.n_classes if opt.n_classes != 'auto' else n_classes, "Number of classes mismatch")
    ]

    for actual, expected, message in checks:
        if actual != expected:
            raise ValueError(f"{message}: {actual} != {expected}")
        
    model = MultiModel(d_inp, seq_len, n_classes, graph, opt)
    model = model.to(device)
    print(f"prepare_model took {time.time() - start_time:.2f} seconds")
    return model

def prepare_graph(opt, device="cuda"):
    start_time = time.time()
    from dataprocess.graph import load_graph
    graph = load_graph(opt, device)
    print(f"prepare_graph took {time.time() - start_time:.2f} seconds")
    return graph

def train_model(opt, n_classes, model, train_loader, val_loader, writer, device="cuda"):
    start_time = time.time()
    model, loss, best_acc = train(opt, model, train_loader, val_loader, writer, device)

    print(f"train_model took {time.time() - start_time:.2f} seconds")
    print(f"Training finished. Loss: {loss}, Best Accuracy: {best_acc}")
    return model, loss, best_acc

def test_model(opt, model, test_loader, writer, device="cuda"):
    start_time = time.time()
    save_path = os.path.join(opt.dir_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    spath = save_path
    metric = test(
        model,
        test_loader,
        writer=writer,
        device=device,
    )
    print(f"test_model took {time.time() - start_time:.2f} seconds")
    return metric
