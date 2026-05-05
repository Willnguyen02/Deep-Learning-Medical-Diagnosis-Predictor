# data utility script

import os
import kagglehub
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    precision_recall_curve, average_precision_score
)

path = kagglehub.dataset_download("nih-chest-xrays/data")

column_order=[
    'image_index',
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Effusion',
    'Emphysema',
    'Fibrosis',
    'Hernia',
    'Infiltration',
    'Mass',
    'No Finding',
    'Nodule',
    'Pleural_Thickening',
    'Pneumonia',
    'Pneumothorax',
    'follow_up_number',
    'patient_id',
    'patient_age',
    'patient_gender',
    'view_position'
]

labels=[
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Effusion',
    'Emphysema',
    'Fibrosis',
    'Hernia',
    'Infiltration',
    'Mass',
    'No Finding',
    'Nodule',
    'Pleural_Thickening',
    'Pneumonia',
    'Pneumothorax'
]

# view image by index
def view_img(file_path, index):
    return Image.open(file_path.iloc[index,0])

def calculate_featuremap_size(input_size, padding, stride, kernel_size):
    return ((input_size + (2 * padding) - kernel_size) // stride) + 1

def maxpool2d_size(input_size, padding, stride, kernel_size):
    return ((input_size - kernel_size + 2 * padding ) // stride) + 1


# focal loss - down-weights easy examples so model focuses on hard ones
class Focal_loss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none',
            pos_weight=self.pos_weight
        )

        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma

        return (focal_weight * bce).mean()


# find the best threshold per class by maximizing F1 on a given set
def optimize_thresholds(all_targets, all_probs, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.05)

    num_classes = all_targets.shape[1]
    best_thresholds = np.full(num_classes, 0.5)

    for c in range(num_classes):
        best_f1 = 0.0

        for t in thresholds:
            preds = (all_probs[:, c] >= t).astype(int)
            f1 = f1_score(all_targets[:, c], preds, zero_division=0)

            if f1 > best_f1:

                best_f1 = f1
                best_thresholds[c] = t

    return best_thresholds


# run model on a dataloader and return predictions + metrics
def model_evaluation(model, dataloader, label_names, device, thresholds=None):
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())


    all_targets = np.concatenate(all_targets, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    if thresholds is None:
        thresholds = np.full(len(label_names), 0.5)

    all_preds = (all_probs >= thresholds).astype(int)

    # auc per class
    per_class_auc = {}
    for i, name in enumerate(label_names):
        try:
            per_class_auc[name] = roc_auc_score(all_targets[:, i], all_probs[:, i])
        except ValueError:
            per_class_auc[name] = float('nan')


    valid_aucs = [v for v in per_class_auc.values() if not np.isnan(v)]
    mean_auc = np.mean(valid_aucs) if valid_aucs else float('nan')

    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_targets, all_preds, average='micro', zero_division=0)


    try:
        mAP = average_precision_score(all_targets, all_probs, average='macro')
    except ValueError:
        mAP = float('nan')

    report = classification_report(all_targets, all_preds, target_names=label_names, zero_division=0)

    return {
        'targets': all_targets,
        'probs': all_probs,
        'preds': all_preds,
        'per_class_auc': per_class_auc,
        'mean_auc': mean_auc,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'mAP': mAP,
        'report': report,
        'thresholds': thresholds
    }


# side by side bar charts comparing per-class AUC and summary metrics across models
def comparison_plot(results_dict, label_names):
    plt.figure(figsize=(20, 8))

    x = np.arange(len(label_names))
    width = 0.8 / len(results_dict)

    # per-class AUC bars
    plt.subplot(1, 2, 1)
    for i, (model_name, results) in enumerate(results_dict.items()):

        aucs = [results['per_class_auc'].get(name, 0) for name in label_names]
        plt.bar(x + i * width, aucs, width, label=model_name)

    plt.xlabel('Disease Class')
    plt.ylabel('AUC-ROC')
    plt.title('Per-Class AUC-ROC Comparison')
    plt.xticks(x + width * (len(results_dict) - 1) / 2, label_names, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)

    # summary metrics bars
    metrics = ['mean_auc', 'macro_f1', 'micro_f1', 'mAP']
    metric_labels = ['Mean AUC', 'Macro F1', 'Micro F1', 'mAP']
    x2 = np.arange(len(metrics))

    plt.subplot(1, 2, 2)
    for i, (model_name, results) in enumerate(results_dict.items()):

        values = [results[m] for m in metrics]
        plt.bar(x2 + i * width, values, width, label=model_name)

    plt.ylabel('Score')
    plt.title('Summary Metrics Comparison')
    plt.xticks(x2 + width * (len(results_dict) - 1) / 2, metric_labels)
    plt.legend()
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_train_val_curves(train_losses, val_losses, title="Training Curves"):
    plt.figure(figsize=(10, 6))

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# plot precision-recall curves for the rarest classes
def plot_precision_recall_curves(targets, probs, label_names, top_n=5):

    class_counts = targets.sum(axis=0)
    rarest_indices = np.argsort(class_counts)[:top_n]

    plt.figure(figsize=(4 * top_n, 4))
    for i, idx in enumerate(rarest_indices):

        precision, recall, _ = precision_recall_curve(targets[:, idx], probs[:, idx])
        ap = average_precision_score(targets[:, idx], probs[:, idx])

        plt.subplot(1, top_n, i + 1)
        plt.plot(recall, precision)

        plt.title(f'{label_names[idx]}\nAP={ap:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)

    plt.suptitle(f'Precision-Recall Curves ({top_n} Rarest Classes)', y=1.02)
    plt.tight_layout()
    plt.show()
