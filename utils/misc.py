import pandas as pd
import wandb
import numpy as np
import torch.nn as nn

from .best_hparams import mortality_day90_best_hparams, readmission_day30_best_hparams, icu_mortality_best_hparams, remaining_los_best_hparams
from .best_hparams import mimic_graph_embedding_best_hparams, eicu_graph_embedding_best_hparams, mimic_pretrain_best_hparams, eicu_pretrain_best_hparams


def get_log_dir(dataset: str,
                algo: str,
                task: str,
                n_source: int,
                seed: int):
    """
    Get the log directory for the experiment
    """ 
    log_dir = f"log/{dataset}/{task}/{n_source}/{algo}/{seed}"
    return log_dir



def load_best_hparams(algo_config, task, dataset=None):
        """
        Load the best hyperparameters from the database
        """
        if task == "mortality_day90":
            best_params = mortality_day90_best_hparams
        elif task == "readmission_day30":
            best_params = readmission_day30_best_hparams
        elif task == "icu_mortality":
            best_params = icu_mortality_best_hparams
        elif task == "remaining_los":
            best_params = remaining_los_best_hparams
        elif task == "graph_embedding":
            assert dataset is not None, "dataset must be provided for graph embedding task (mimic or eicu)"
            if dataset == "mimic":
                best_params = mimic_graph_embedding_best_hparams
            elif dataset == "eicu":
                best_params = eicu_graph_embedding_best_hparams
        elif task == "pretrain":
            assert dataset is not None, "dataset must be provided for pretrain task (mimic or eicu)"
            if dataset == "mimic":
                best_params = mimic_pretrain_best_hparams
            elif dataset == "eicu":
                best_params = eicu_pretrain_best_hparams

        for key, value in best_params.items():
            setattr(algo_config, key, value)

        return algo_config


def num2one_hot(index, num_classes):
    """
    Converts a single integer index to a one-hot encoded tensor.
    
    Args:
        index (int): Index to convert into one-hot.
        num_classes (int): Total number of classes.
    
    Returns:
        torch.Tensor: A tensor representing the one-hot encoded format of the index.
    """
    one_hot = np.zeros(num_classes)
    one_hot[index] = 1
    return one_hot


def set_grad_flag(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag
