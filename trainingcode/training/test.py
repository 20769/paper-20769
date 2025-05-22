import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os

import sys
from os.path import dirname as opd
sys.path.append(opd(opd(__file__)))
from utils.plot_train import plot_roc_curve
from training.visualize import ModelVisualizer

def test(model, test_loader, writer, device="cuda"):
    task_type = model.task_type
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    model_outputs = []  # Store model outputs for visualization
    save_path = writer.log_dir
    visualizer = ModelVisualizer(model, writer)
    with torch.no_grad():
        for inputs, times, labels in test_loader:
            inputs, times, labels = inputs.to(device), times.to(device), labels.to(device)
            outputs_visual = model(inputs, times)
            outputs = outputs_visual['high_attention_output']
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            model_outputs.append(outputs_visual)  # Store output
            if len(all_preds) > 1000:
                break
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    if task_type == 'classify':
        # Check for NaN values and handle them
        if np.isnan(all_probs).any() or np.isnan(all_labels).any():
            print("NaN values found in probabilities or labels")
            all_probs = np.nan_to_num(all_probs)
            all_labels = np.nan_to_num(all_labels)

        acc = accuracy_score(all_labels, all_preds)
        if all_probs.shape[1] == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        metric = {'accuracy': acc, 'auc': auc}
        fpr = {}
        tpr = {}
        for i in range(all_probs.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(all_labels, all_probs[:, i], pos_label=i)
    elif task_type == 'forcast':
        mse = mean_squared_error(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        metric = {'mse': mse, 'mae': mae}
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    # Save metrics and plots
    np.save(os.path.join(save_path, 'predictions.npy'), all_preds)
    np.save(os.path.join(save_path, 'labels.npy'), all_labels)
    np.save(os.path.join(save_path, 'probabilities.npy'), all_probs)

    if task_type == 'classify':
        fig_roc = plot_roc_curve(fpr, tpr)
        fig_roc.savefig(os.path.join(save_path, 'roc_curve.png'))
        print(f"Test Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    
    #Visualize only the first batch in model outputs
    first_output = model_outputs[0]
    first_input = test_loader.dataset[0][0].unsqueeze(0).to(device)
    visualizer.visualize_all(first_output, first_input, 0)
    return metric

