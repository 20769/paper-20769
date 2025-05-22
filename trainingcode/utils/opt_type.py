from dataclasses import dataclass
from typing import Any


@dataclass
class ReproducOpt:    
    seed: int
    benchmark: bool
    deterministic: bool

@dataclass
class CUTSopt:
    dir_name: str
    task_name: str
    
    @dataclass
    class CUTSargs:
        n_nodes: int
        input_step: int
        batch_size: int
        data_dim: int
        total_epoch: int
        update_every: int
        show_graph_every: int
        
        @dataclass
        class data_pred:
            model: str
            multi_scale: bool
            multi_scale_periods: list
            pred_step: int
            mlp_hid: int
            mlp_layers: int
            lr_data_start: float
            lr_data_end: float
            weight_decay: int

        @dataclass
        class graph_discov:
            lambda_s: 0.1
            lr_graph_start: float
            lr_graph_end: float
            start_tau: 0.3
            end_tau: 0.01
            dynamic_sampling_milestones: list
            dynamic_sampling_periods: list

    causal_thres: str
    reproduc: ReproducOpt
    log: Any
@dataclass
class Mylogger:
    log_dir: str
    log_file: str
    log_level: str
    log_format: str
    log_console: bool
    log_filemode: str

@dataclass
class MultiCADopt:
    dir_name: str
    task_name: str
    
    @dataclass
    class MultiCADargs:
        n_nodes: int
        input_step: int
        window_step: int
        stride: int
        batch_size: int
        sample_per_epoch: int
        data_dim: int
        total_epoch: int
        
        patience: int
        warmup: Any
        
        show_graph_every: int
        val_every: int
        
        n_groups: int
        group_policy: Any
        causal_thres: str
        
        @dataclass
        class data_pred:
            model: str
            merge_policy: str
            lr_data_start: float
            lr_data_end: float
            weight_decay: int
            prob: bool

        @dataclass
        class graph_discov:
            lr_graph_start: float
            lr_graph_end: float
            lambda_s_start: float
            lambda_s_end: float
            tau_start: float
            tau_end: float
            disable_bwd: bool
            separate_bwd: bool
            disable_ind: bool
            disable_graph: bool
            use_true_graph: bool
    
    reproduc: ReproducOpt
    log: Any