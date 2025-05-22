import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, mean_squared_error, mean_absolute_error
import tqdm
torch.autograd.set_detect_anomaly(True)
import sys
from os.path import dirname as opd
sys.path.append(opd(opd(__file__)))
from utils.loss_pareto import InterpretableLoss
from utils.plot_train import plot_roc_curve
import torch.cuda.amp as amp
from training.visualize import ModelVisualizer
from utils.EarlyStop import EarlyStopping

class ModelWrapper(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def train(opt, model, train_loader, val_loader, writer, device="cuda", mask_train_loader=None, mask_val_loader=None, trainset=None):
    task_type = model.task_type
    visualize_flag = opt.visualize
    model = model.to(device)
    # model = ModelWrapper(model) if torch.cuda.device_count() > 1 else model
    if opt.train_refer_model:
        out_path = '/'.join(writer.get_logdir().split('/')[:-2])
        model.train_refer_model(train_loader, val_loader, writer, save_path=out_path)
        print('Refer model training completed, saving at {}'.format(out_path))

    if task_type == 'classify':
        labels = []
        for _, _, l in train_loader:
            labels.extend(l.cpu().numpy())
        labels = torch.tensor(labels)
        class_sample_count = torch.tensor(
            [(labels == t).sum() for t in torch.unique(labels, sorted=True)])
        weight = 1. / class_sample_count.float()
        # criterion = InterpretableLoss(class_weights=weight, task_type='classify', alpha=opt.alpha, beta=opt.beta, gamma=opt.gamma).to(device)
        criterion = InterpretableLoss(class_weights=weight, task_type='classify', alpha_flag=opt.alpha).to(device)
    else:
        criterion = InterpretableLoss(task_type='regression', alpha_flag=opt.alpha).to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    num_epochs = opt.epochs
    best_metric = float('inf') if task_type != 'classify' else 0.0
    loss = 0.0
    visualizer = ModelVisualizer(model, writer)

    early_stopping = EarlyStopping(patience=7, verbose=True, delta=0.001)


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (inputs, times, labels) in progress_bar:
            inputs, times, labels = inputs.to(device), times.to(device), labels.to(device)
            
            inputs.requires_grad = True
            times.requires_grad = True

            optimizer.zero_grad()
            with amp.autocast():
                outputs = model(inputs, times)
                loss, classification_loss, distance_loss, clustering_loss, regularization_loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
 
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            criterion.update(model, optimizer, inputs, times, labels, epoch, num_epochs, writer)
            if i % 5 == 0 and i > 0:
                progress_bar.set_description(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
                writer.add_scalar('train/classification_loss', classification_loss.item(), i + epoch * len(train_loader))
                if isinstance(distance_loss, torch.Tensor):
                    writer.add_scalar('train/distance_loss', distance_loss.item(), i + epoch * len(train_loader))
                    writer.add_scalar('train/clustering_loss', clustering_loss.item(), i + epoch * len(train_loader))
            if i % 50 == 0:
                if task_type == 'classify':
                    val_acc, val_auc, fpr, tpr = evaluate_classify(model, val_loader, device, visualizer, in_train=True, writer=writer, global_step=i + epoch * len(train_loader), visualize = visualize_flag)
                    writer.add_scalar('train/Accuracy', val_acc, i + epoch * len(train_loader))
                    writer.add_scalar('train/AUC', val_auc, i + epoch * len(train_loader))
                    writer.add_figure('train/ROC', plot_roc_curve(fpr, tpr), i + epoch * len(train_loader))
                    current_metric = val_auc
                else:
                    mse, mae = evaluate_prediction(model, val_loader, device, in_train=True)
                    val_acc, val_auc = mse, mae
                    writer.add_scalar('train/MSE', mse, i + epoch * len(train_loader))
                    writer.add_scalar('train/MAE', mae, i + epoch * len(train_loader))
                    current_metric = mse
                if (task_type == 'classify' and current_metric > best_metric) or (task_type != 'classify' and current_metric < best_metric):
                    best_metric = current_metric
                    writer.add_scalar('Best Metric', best_metric, epoch)
                    save_model(model, f'{writer.get_logdir()}/best_model.pth')
            if i % 100 == 0 and i > 0:
                for inputs, times, labels in val_loader:
                    inputs, times, labels = inputs.to(device), times.to(device), labels.to(device)
                    criterion.update(model, optimizer, inputs, times, labels, epoch, num_epochs, writer)
                    break
        epoch_loss = running_loss / len(train_loader.dataset)
        if task_type == 'classify':
            val_acc, val_auc, fpr, tpr = evaluate_classify(model, val_loader, device, visualizer, in_train=False, writer=writer, global_step=i + epoch * len(train_loader), visualize = visualize_flag)
            writer.add_scalar('val/Accuracy', val_acc, epoch)
            writer.add_scalar('val/AUC', val_auc, epoch)
            writer.add_figure('val/ROC', plot_roc_curve(fpr, tpr), epoch)
            writer.add_scalar('val/Loss', epoch_loss, epoch)
            current_metric = val_acc
        else:
            mse, mae = evaluate_prediction(model, val_loader, device)
            val_acc, val_auc = mse, mae  # For consistency in logging
            writer.add_scalar('val/MSE', mse, epoch)
            writer.add_scalar('val/MAE', mae, epoch)
            writer.add_scalar('val/Loss', epoch_loss, epoch)
            current_metric = mse
    
        early_stopping(epoch_loss, model, writer.get_logdir())
        if early_stopping.early_stop:
            print("Early stopping")
            break

    save_model(model, f'{writer.get_logdir()}/final_model.pth')
    # load best model
    model.load_state_dict(torch.load(f'{writer.get_logdir()}/best_model.pth'))

    return model, loss, best_metric

def save_model(model, path):
    torch.save(model.state_dict(), path)

def evaluate_prediction(model, val_loader, device, in_train=False):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, times, labels in val_loader:
            inputs, times, labels = inputs.to(device), times.to(device), labels.to(device)
            outputs = model(inputs, times)['high_attention_output']
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if in_train and len(all_preds) > 1000:
                break

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if np.isnan(all_preds).any() or np.isnan(all_labels).any():
        print("NaN values found in predictions or labels")
        all_preds = np.nan_to_num(all_preds)
        all_labels = np.nan_to_num(all_labels)

    all_preds = all_preds.reshape(-1, all_preds.shape[-1])
    all_labels = all_labels.reshape(-1, all_labels.shape[-1])

    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    model.train()

    return mse, mae

def evaluate_classify(model, val_loader, device, visualizer, in_train=False, visualize=True, writer=None, global_step=0):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    model_outputs = [] #Store model outputs for visualization

    with torch.no_grad():
        for inputs, times, labels in val_loader: # Assuming val_loader yields model outputs
            inputs, times, labels = inputs.to(device), times.to(device), labels.to(device)
            outputs_visual = model(inputs, times)
            outputs = outputs_visual['high_attention_output']
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            model_outputs.append(outputs_visual) #Store output
            if in_train and len(all_preds) > 1000:
                break

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Check for NaN values and handle them
    if np.isnan(all_probs).any() or np.isnan(all_labels).any():
        print("NaN values found in probabilities or labels")
        all_probs = np.nan_to_num(all_probs)
        all_labels = np.nan_to_num(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    if all_probs.shape[1] == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        all_probs = F.softmax(torch.tensor(all_probs), dim=1).numpy()
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

    fpr = {}
    tpr = {}
    for i in range(all_probs.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(all_labels, all_probs[:, i], pos_label=i)
    model.train()

    if visualize and len(model_outputs) > 0:
        first_output = model_outputs[0]
        first_input = val_loader.dataset[0][0].unsqueeze(0)
        visualizer.visualize_all(first_output, first_input, global_step)


    return acc, auc, fpr, tpr
