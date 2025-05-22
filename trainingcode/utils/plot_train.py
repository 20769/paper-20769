import matplotlib.pyplot as plt
import numpy as np
import os

def plot_prediction_results(all_labels, all_preds):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(all_labels, label='True values')
    ax.plot(all_preds, label='Predicted values', linestyle='--')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Value')
    ax.set_title('Prediction Results')
    ax.legend(loc='best')
    return fig

def plot_roc_curve(fpr, tpr):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for i in fpr:
        ax.plot(fpr[i], tpr[i], label=f'Class {i} ROC curve')
    ax.plot([0, 1], [0, 1], 'k--', label='Random guess')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc='best')
    return fig