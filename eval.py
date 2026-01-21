import argparse
import os
import torch
import pandas as pd
import pickle
from copy import deepcopy

from configs.algo_config import AdvConfig
from configs.dataset_config import MIMICDatasetConfig, EICUDatasetConfig
from utils.seed import set_random_seed
from dataset.dataset_load_fn import get_dataset_fn
from utils.misc import get_log_dir
from models import LSTMBasedModel
from run_fn import test_fn
from utils.constants import MIMIC_RARE_ICD_CODES, EICU_RARE_ICD_CODES, task_label_dict


def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--algo', type=str, choices=['knowrare'], default='knowrare')
    parser.add_argument('--dataset', type=str, default="mimic",
                        choices=["mimic", "eicu"],
                        help='dataset to use for training')
    parser.add_argument('--task', type=str,
                        choices=["readmission_day30", "mortality_day90", "icu_mortality", 'remaining_los'],
                        default="mortality_day90")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help='device to use for training')
    parser.add_argument('--average', type=str, default='micro', help='result merge method')
    parser.add_argument('--n_source', type=float, default=0.1, help='num_source_icd9')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    set_random_seed(args.seed)

    # initialize config and graph
    if args.dataset == "mimic":
        dataset_config = MIMICDatasetConfig()
        RARE_ICD_CODES = MIMIC_RARE_ICD_CODES
    elif args.dataset == "eicu":
        dataset_config = EICUDatasetConfig()
        RARE_ICD_CODES = EICU_RARE_ICD_CODES
    algo_config = AdvConfig()

    # set config
    algo_config.resampling = True
    dataset_config.source_domain_selection = "top_n"
    dataset_config.dataset_class = "multi_model"
    dataset_config.label_name = task_label_dict[args.task]
    algo_config.device = args.device
    algo_config.demo_dims = dataset_config.num_demo_features
    algo_config.ts_dims = dataset_config.num_ts_features
    optimal_cutoff = None

    # load graph
    if args.dataset == "mimic":
        graph_path = dataset_config.graph_tucker_tope06_path
    elif args.dataset == "eicu":
        graph_path = dataset_config.graph_tucker_tope005_path

    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    # init dataset
    _, val_dataset, test_datasets, dataset_info = get_dataset_fn(dataset_config=dataset_config,
                                                                target_icd9_codes=RARE_ICD_CODES,
                                                                graph=graph,
                                                                dataset=args.dataset)

    # init model
    multi_cls = True if args.task == 'remaining_los' else False
    model = LSTMBasedModel(num_classes=dataset_config.num_classes_dict[dataset_config.label_name], 
                            demo_dims=algo_config.demo_dims, ts_dims=algo_config.ts_dims, multi_cls=multi_cls)
    model.to(algo_config.device)

    # load model
    log_dir = get_log_dir(dataset=args.dataset, algo=args.algo, task=args.task, seed=args.seed, n_source=args.n_source)
    models = []
    for _dataset_info in dataset_info:
        target_icd9 = _dataset_info["target_domain"][0]
        weights_fpath = os.path.join(log_dir, f"model-{target_icd9}.pth")
        model.load_state_dict(torch.load(weights_fpath, weights_only=True)["model"])
        models.append(deepcopy(model))
    
    # evaluate
    if optimal_cutoff is None:
        out_dir = os.path.split(log_dir)[0]
        val_result_df = pd.read_csv(f"{out_dir}/val_results_stats.csv", index_col=0)
        try:
            optimal_cutoff = val_result_df.loc["optimal_cutoff", "mean"]
        except:
            optimal_cutoff = None
    test_result = test_fn(config=algo_config,
                            model=models,
                            optimal_cutoff=optimal_cutoff,
                            test_datasets=test_datasets,
                            average=args.average)
    
    test_result["seed"] = args.seed

    test_result_path = os.path.join(log_dir, "test_result.csv")
    test_result_df = pd.DataFrame(test_result, index=[0])
    test_result_df.to_csv(test_result_path)
