import argparse
import os
from pathlib import Path
import torch

from configs.dataset_config import EICUDatasetConfig, MIMICDatasetConfig
from configs.algo_config import ReconstructionConfig
from utils.misc import load_best_hparams
from utils.seed import set_random_seed
from dataset.dataset_load_fn import get_dataset_fn
from run_fn import reconstruct_pretrain_fn
from utils.constants import MIMIC_RARE_ICD_CODES, EICU_RARE_ICD_CODES, task_label_dict
from models import LSTMBasedModel, AutoregressiveDecoder


def get_args():
    parser = argparse.ArgumentParser(description="Pretrain the classification model (First-stage Training)")
    parser.add_argument("--algo", type=str, default="reconstruction", choices=["reconstruction"],)
    parser.add_argument('--dataset', type=str, default="eicu", choices=["mimic", "eicu"], help='dataset to use for training')
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    args.n_source = None
    set_random_seed(args.seed)

    # Dataset configuration
    if args.dataset == "mimic":
        dataset_config = MIMICDatasetConfig()
        RARE_ICD_CODES = MIMIC_RARE_ICD_CODES
    elif args.dataset == "eicu":
        dataset_config = EICUDatasetConfig()
        RARE_ICD_CODES = EICU_RARE_ICD_CODES

    log_dir = os.path.join(dataset_config.root_dir, "pretrain_weights", args.algo)
    pretrain_weight_path = os.path.join(log_dir, f"lstm.pth")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    if os.path.exists(pretrain_weight_path):
        print(f"Pretrained model weights already exist in {pretrain_weight_path}. Exiting...")
        exit()

    dataset_config.source_domain_selection = "all"
    dataset_config.dataset_class = "baseline"
    train_dataset, val_dataset, test_dataset, _ = get_dataset_fn(dataset_config=dataset_config, 
                                                                 target_icd9_codes=RARE_ICD_CODES,
                                                                 dataset=args.dataset)

    # Load configuration and best hyperparameters
    algo_config = ReconstructionConfig()
    algo_config = load_best_hparams(algo_config, task="pretrain", dataset=args.dataset)
    algo_config.log_dir = log_dir
    algo_config.ts_dims = dataset_config.num_ts_features
    algo_config.demo_dims = dataset_config.num_demo_features

    # init models
    multi_cls = True if args.algo == 'cls_remaining_los' else False
    model = LSTMBasedModel(num_classes=dataset_config.num_classes_dict[dataset_config.label_name], 
                            demo_dims=algo_config.demo_dims, ts_dims=algo_config.ts_dims, multi_cls=multi_cls)
    model.to(algo_config.device)

    decoder = AutoregressiveDecoder(hidden_dims=2*model.hidden_dims, ts_dims=algo_config.ts_dims)
    decoder.to(args.device)

    # Pretrain model using reconstruction auto-regressively
    val_results = reconstruct_pretrain_fn(
        config=algo_config,
        model=model,  # Will be initialized in the train function
        decoder=decoder,  # Will be initialized in the train function
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        write_log=True
    )
