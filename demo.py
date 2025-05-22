import os
import torch
from omegaconf import OmegaConf
import sys
from os.path import join as opj
from os.path import dirname as opd
sys.path.append(opd(opd(__file__)))
from trainingcode.training.train_main import prepare_data, prepare_model, train_model, test_model
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from os.path import join as opj

# Load configuration
base_dir = os.getenv('EXPLAINTS_BASE_DIR', '.')  # Use environment variable or default to current directory
config_path = os.path.join(base_dir, './exp_settings/train_full.yaml')
opt = OmegaConf.load(config_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Prepare data
train_loader, val_loader, test_loader, d_inp, max_len, total_len, n_classes = prepare_data(opt.data, device)

# Prepare model
graph = torch.ones((d_inp, d_inp)).to(device)  # Dummy graph
model = prepare_model(opt.model, d_inp, max_len, total_len, n_classes, graph, device)

# Train the model
timestamp = datetime.now().strftime("_%Y_%m%d_%H%M%S_%f")
proj_path = opj(opt.dir_name, 'demo', timestamp)
writer = SummaryWriter(proj_path)
model, loss, best_acc = train_model(opt.train, n_classes, model, train_loader, val_loader, writer, device)
print(f"Training completed. Loss: {loss}, Best Accuracy: {best_acc}")

# load trained model
# model.load_state_dict(torch.load('./best_model.pth'))
model.load_state_dict(torch.load(opj(proj_path, 'best_model.pth')))
model.to(device)

# Test the model
metric = test_model(opt.test, model, test_loader, writer, device)
print(f"Testing completed. Metric: {metric}")

# Close the writer
writer.close()
