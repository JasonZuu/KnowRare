from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, balanced_accuracy_score, precision_score, recall_score, f1_score
from copy import deepcopy
from collections import Counter
from typing import List, Union
from sklearn.preprocessing import label_binarize

from models.lstm_based import LSTMBasedModel
from utils.metrics import get_optimal_f1_cutoff
from dataset.dataset_load_fn import get_dataloader_fn


def test_fn(config,
            model: Union[list, torch.nn.Module],
            optimal_cutoff: float,
            test_datasets: List[torch.utils.data.Dataset],
            average: str = "micro"  # 'macro' or ‘micro’
            )-> dict:
    if isinstance(model, torch.nn.Module):
        model.eval()
        models = [model]*len(test_datasets)
    elif isinstance(model, list):
        models = model
        for m in models:
            m.eval()

    if average == 'macro':
        test_metrics = {}
        for model, test_dataset in zip(models, test_datasets):
            test_loader = get_dataloader_fn(algo_config=config, dataset=test_dataset, mode="test")
            _test_metric = _test_loop(model=model, test_loader=test_loader, device=config.device,
                                    optimal_cutoff=optimal_cutoff)
            for metric, value in _test_metric.items():
                test_metrics.get(metric, []).append(value)
        
        test_metric = {}
        for metric, values in test_metrics.items():
            clean_values = [v if v is not None else 0 for v in values] # replace None with 0
            mean = np.mean(clean_values)
            test_metric[metric] = mean
    elif average == "micro":
        y_trues = None
        y_scores = None
        # collect all y_trues and y_scores
        for model, test_dataset in zip(models, test_datasets):
            test_loader = get_dataloader_fn(algo_config=config, dataset=test_dataset, mode="test")
            y_dict = _test_loop(model=model, test_loader=test_loader, device=config.device,
                                    optimal_cutoff=optimal_cutoff, return_metrics=False)
            y_trues = y_dict["y_tures"] if y_trues is None else np.concatenate([y_trues, y_dict["y_tures"]], axis=0)
            y_scores = y_dict["y_scores"] if y_scores is None else np.concatenate([y_scores, y_dict["y_scores"]], axis=0)
        test_metric = calculate_metrics(y_trues=y_trues, y_scores=y_scores, optimal_cutoff=optimal_cutoff)
    else:
        raise ValueError(f"Invalid average method {average}")

    return test_metric


@torch.no_grad()
def _test_loop(model, test_loader, device: str, optimal_cutoff:float=None, 
               show_progress: bool = True, return_metrics: bool = True):
    """
    Function to test the net for binary or multi-class classification.
    Task type is determined by label.shape[-1].
    """
    y_scores, y_trues = [], []
    if show_progress:
        pbar = tqdm(total=len(test_loader), desc=f'Testing', unit='batch')

    for data in test_loader:
        demo, ts, label = data['demography'], data['time_series'], data['label']
        demo, ts, label = demo.to(device), ts.to(device), label.to(device)

        logits = model(demo, ts, use_output_activate=False)  # Logits directly

        # Check task type based on label.shape[-1]
        task_type = "binary" if label.shape[-1] == 1 else "multiclass"

        if task_type == "binary":
            y_score = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)  # Sigmoid for binary classification
            y_true = label.cpu().numpy().reshape(-1)
        else:
            y_score = torch.softmax(logits, dim=-1).detach().cpu().numpy()  # Softmax for multi-class classification
            y_true = label.argmax(dim=-1).cpu().numpy()  # Convert one-hot labels to class indices

        y_scores.append(y_score)
        y_trues.append(y_true)

        if show_progress:
            pbar.update()

    total_y_score = np.concatenate(y_scores, axis=0)
    total_y_true = np.concatenate(y_trues, axis=0)

    if return_metrics:
        test_metrics = calculate_metrics(y_trues=total_y_true, y_scores=total_y_score, optimal_cutoff=optimal_cutoff)
        return test_metrics
    else:
        return {"y_tures": total_y_true, "y_scores": total_y_score}


def calculate_metrics(y_trues, y_scores, optimal_cutoff=None):
    # Check task type based on label.shape[-1]
    task_type = "binary" if len(y_scores.shape) == 1 else "multiclass"

    # calculate metrics based on task type
    if task_type == "binary":
        if y_scores.sum() < 1e-6 or y_trues.sum() < 1e-6:  # No discrimination or positive samples
            auprc, auroc, precision, recall, specificity, f1, balanced_acc = 0.0, 0.5, 0, 0, 1, 0, 0.5
            optimal_cutoff = 0.0
        else:
            if optimal_cutoff is None:
                optimal_cutoff = get_optimal_f1_cutoff(y_trues, y_scores)
            
            total_y_pred = (y_scores > optimal_cutoff).astype(int)
            
            precisions, recalls, thresholds = precision_recall_curve(y_trues, y_scores)
            auprc = auc(recalls, precisions)
            try:
                auroc = roc_auc_score(y_trues, y_scores)
            except:
                auroc = 0.5
            balanced_acc = balanced_accuracy_score(y_trues, total_y_pred)
            tn, fp, fn, tp = confusion_matrix(y_trues, total_y_pred).ravel()
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            specificity = tn / (tn + fp) if tn + fp > 0 else 1
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        test_metrics = {
            "auprc": auprc,
            "auroc": auroc,
            "balanced_accuracy": balanced_acc,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "optimal_cutoff": optimal_cutoff,
            "f1": f1,
        }

    else:  # Multi-class classification
        # Compute AUPRC and AUROC
        num_classes = y_scores.shape[1]
        y_trues_one_hot = label_binarize(y_trues, classes=range(num_classes))

        # Initialize lists to store AUPRC and AUROC for each class
        auprc_per_class = []
        auroc_per_class = []

        for i in range(num_classes):
            if y_trues_one_hot[:, i].sum() < 1e-6: # skip class if no positive samples
                continue
            elif y_scores[:, i].sum() < 1e-6: # No discrimination
                _auprc, _auroc = 0.0, 0.5
            else:
                try:
                    
                    # Compute Precision-Recall Curve for the current class
                    precision_vals, recall_vals, _ = precision_recall_curve(
                        y_trues_one_hot[:, i], y_scores[:, i]
                    )
                    # Compute AUPRC for the current class
                    _auprc = auc(recall_vals, precision_vals)
                except ValueError:
                    # Handle any unexpected issues during calculation
                    _auprc = 0.0
                
                 # try except block to handle auroc
                try:
                    _auroc = roc_auc_score(y_trues_one_hot[:, i], y_scores[:, i])
                except ValueError:
                    _auroc = 0.5
                    
            auprc_per_class.append(_auprc)
            auroc_per_class.append(_auroc)

        # Compute macro-averaged AUPRC and AUROC
        auprc = np.mean(auprc_per_class)
        auroc = np.mean(auroc_per_class)

        total_y_pred = np.argmax(y_scores, axis=1)
        precision = precision_score(y_trues, total_y_pred, average="macro", zero_division=0)
        recall = recall_score(y_trues, total_y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_trues, total_y_pred, average="macro", zero_division=0)
        balanced_acc = balanced_accuracy_score(y_trues, total_y_pred)

        test_metrics = {
            "auroc": auroc,
            "auprc": auprc,
            "balanced_accuracy": balanced_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return test_metrics