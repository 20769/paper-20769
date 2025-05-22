import numpy as np
import torch
import os

def load_graph(opt, device="cuda"):
    graph_path = './dataset/all_six_datasets/graph'
    if opt.name == 'forcast':
        data_name = opt.data_name
        data_path = f'{graph_path}/{data_name}.npy'
        if os.path.exists(data_path):
            graph = np.load(data_path)
        else:
            graph = np.ones(shape=(21, 21))
        print(f"Graph loaded from {data_path}")
        print(f'Ratio of 1 in binary graph: {np.sum(graph)/graph.size}')
        return torch.tensor(graph).to(device)
    else:
        return None