import sys
import os
from os.path import join as opj
from os.path import dirname as opd
from typing import Dict, Union
import time
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import random
from datetime import datetime

timestamp = time.strftime("_%Y_%m%d_%H%M%S")

class MyLogger():
    def __init__(self, log_dir: str, stdlog: bool=True, tensorboard: bool=True):
        self.project_dir = log_dir
        self.stdlog = stdlog
        self.tensorboard = tensorboard
        timestamp = datetime.now().strftime("_%Y_%m%d_%H%M%S_%f")
        self.project_dir += timestamp
        temp_name = self.project_dir
        for i in range(10):
            if not os.path.exists(temp_name):
                break
            temp_name = self.project_dir + '-' + str(i)
        self.project_dir = temp_name
        self.logdir = self.project_dir
        self.logger_dict = {}
        if tensorboard:
            self.tensorboard_init()
        else:
            os.makedirs(self.project_dir, exist_ok=True)
        if stdlog:
            self.stdlog_init()
        self.dir_init()

    def stdlog_init(self):
        stderr_handler = open(os.path.join(self.logdir, 'stderr.log'), 'w')
        sys.stderr = stderr_handler
        
    def tensorboard_init(self):
        os.makedirs(self.logdir, exist_ok=True)  # 确保目录存在
        self.tblogger = SummaryWriter(self.logdir, flush_secs=30)
        self.logger_dict['tblogger'] = self.tblogger
    
    def dir_init(self):
        self.script_dir = os.path.join(self.project_dir, 'script')
        self.model_dir = os.path.join(self.project_dir, 'model')
        os.makedirs(self.script_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def log_metrics(self, metrics_dict: Dict[str, float], iters):
        for logger_name in self.logger_dict.keys():
            if logger_name == 'tblogger':
                for k in metrics_dict.keys():
                    self.logger_dict[logger_name].add_scalar(k, metrics_dict[k], iters)

    def log_opt(self, opt):
        with open(os.path.join(self.logdir, 'config.yaml'), 'w') as f:
            f.write(str(opt))

    def log_info(self, message: str):
        print(message)
        with open(os.path.join(self.logdir, 'info.log'), 'a') as f:
            f.write(message + '\n')

    def close(self):
        for logger_name in self.logger_dict.keys():
            if logger_name == 'tblogger':
                self.logger_dict[logger_name].close()
       

def reproduc(seed: int, benchmark: bool, deterministic: bool):
    """Make experiments reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic