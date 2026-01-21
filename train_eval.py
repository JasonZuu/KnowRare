import argparse
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from copy import deepcopy

from configs.algo_config import AdvConfig
from configs.dataset_config import MIMICDatasetConfig, EICUDatasetConfig
from run_fn import knowrare_train_fn, test_fn
from models import LSTMBasedModel
from models.discriminator import ConditionLinearDiscriminator
from dataset.dataset_load_fn import get_dataset_fn
from utils.seed import set_random_seed
from utils.misc import get_log_dir, load_best_hparams
from utils.constants import MIMIC_RARE_ICD_CODES, EICU_RARE_ICD_CODES, task_label_dict
from utils.load_pretrain import load_pretrain_weights


def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default="mimic",
                        choices=["mimic", "eicu"],
                        help='dataset to use for training')
    parser.add_argument('--algo', type=str, choices=['knowrare'], default='knowrare')
    parser.add_argument('--task', type=str,
                        choices=["readmission_day30", "mortality_day90", 'icu_mortality', 'remaining_los'],
                        default="readmission_day30")
    parser.add_argument('--n_source', type=float, default=0.1, help='ratio_source_icd9')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help='device to use for training')
    parser.add_argument("--use_best_hparams", action="store_true", help="whether to use best hparams")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    set_random_seed(args.seed)

    # initialise dataset_config
    if args.dataset == "mimic":
        dataset_config = MIMICDatasetConfig()
        RARE_ICD_CODES = MIMIC_RARE_ICD_CODES
    elif args.dataset == "eicu":
        dataset_config = EICUDatasetConfig()
        RARE_ICD_CODES = EICU_RARE_ICD_CODES
    algo_config = AdvConfig()
    
    # set configs
    algo_config.resampling = True
    dataset_config.source_domain_selection = "top_n"
    dataset_config.dataset_class = "multi_model"
    dataset_config.label_name = task_label_dict[args.task]
    algo_config.device = args.device
    algo_config.demo_dims = dataset_config.num_demo_features
    algo_config.ts_dims = dataset_config.num_ts_features
    
    # set n_source
    if args.n_source is not None:
        dataset_config.n_source = args.n_source
        algo_config.n_source = args.n_source

    # load graph
    if args.dataset == "mimic":
        graph_path = dataset_config.graph_tucker_tope06_path
    elif args.dataset == "eicu":
        graph_path = dataset_config.graph_tucker_tope005_path
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    log_dir = get_log_dir(dataset=args.dataset, algo=args.algo, task=args.task, n_source=args.n_source, seed=args.seed)
    algo_config.log_dir = log_dir

    multi_model_weights_paths = [os.path.join(log_dir, f"model-{rare_icd}.pth") for rare_icd in RARE_ICD_CODES]
    if all([os.path.exists(weights_path) for weights_path in multi_model_weights_paths]):
        print(f"All multi-model weights exist. Skipping this seed.")
        exit()
    
    Path(log_dir).mkdir(parents=True, exist_ok=True)

     # init datasetsp
    train_dataset, val_dataset, test_datasets, dataset_info = get_dataset_fn(dataset_config=dataset_config,
                                                                target_icd9_codes=RARE_ICD_CODES,
                                                                graph=graph,
                                                                dataset=args.dataset)

    # init model
    multi_cls = True if args.task == 'remaining_los' else False
    model = LSTMBasedModel(num_classes=dataset_config.num_classes_dict[dataset_config.label_name], 
                            demo_dims=algo_config.demo_dims, ts_dims=algo_config.ts_dims, multi_cls=multi_cls)
    # init discriminator
    icd9_codes = train_dataset[0].get_icd9_codes()
    num_pred_classes = dataset_config.num_classes_dict[dataset_config.label_name]
    dis_model = ConditionLinearDiscriminator(num_classes=len(icd9_codes), num_pred_classes=num_pred_classes)

    # load pretrain weights
    load_pretrain_weights(model, dataset_config, args, algo=args.algo)
    dis_model.to(args.device)
    model.to(algo_config.device)

    # load best hparameters
    args.use_best_hparams = True  # for debug
    if args.use_best_hparams:
        algo_config = load_best_hparams(algo_config, task=args.task, dataset=args.dataset)

    # train
    val_result = {}
    test_result = {}
    models = []
    if type(val_dataset) != list:
        val_dataset = [val_dataset]*len(train_dataset)
    for _train_dataset, _val_dataset, _dataset_info in zip(train_dataset, val_dataset, dataset_info):
        _model = deepcopy(model)
        _dis_model = deepcopy(dis_model)
        _val_result = knowrare_train_fn(config=algo_config,
                                        model=_model,
                                        dis_model=_dis_model,
                                        train_dataset=_train_dataset,
                                        val_dataset=_val_dataset,
                                        write_log=True,
                                        target_icd9=_dataset_info['target_domain'][0])
        for metric, value in _val_result.items():
            val_result[metric] = val_result.get(metric, []) + [value]

        # collect models
        models.append(_model)
    
    algo_dir, _ = os.path.split(algo_config.log_dir)
    val_csv_path = f"{algo_dir}/val_result.csv"
    test_csv_path = f"{algo_dir}/test_result.csv"
    val_result = {metric: np.mean(values) for metric, values in val_result.items()}
    
    if not os.path.exists(val_csv_path):
        val_result_df = pd.DataFrame(val_result, index=[0])
        val_result_df.to_csv(val_csv_path)

    # get test result
    try:
        optimal_cutoff = val_result['optimal_cutoff']
    except:
        optimal_cutoff = None
    test_result = test_fn(config=algo_config,
                            model=models,
                            test_datasets=test_datasets,
                            optimal_cutoff=optimal_cutoff)
    test_result["seed"] = args.seed
    test_resulf_df = pd.DataFrame(test_result, index=[0])
    test_resulf_df.to_csv(test_csv_path)
