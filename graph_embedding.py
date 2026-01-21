import argparse
import pickle
from pathlib import Path
import torch

from run_fn.graph_fn import tuckER_train_fn
from dataset.graph_dataset import TuckERDataset
from configs.algo_config import TuckERConfig
from configs.dataset_config import MIMICGraphDatasetConfig, EICUGraphDatasetConfig
from utils.misc import  load_best_hparams
from utils.seed import set_random_seed


def get_args():
    parser = argparse.ArgumentParser(description="Train Graph Embedding Model")
    parser.add_argument("--model", type=str, default="tuckER", help="Model to train", choices=["tuckER"])
    parser.add_argument("--dataset", type=str, default="mimic", help="Dataset to use for training",
                        choices=["mimic", "eicu"])
    parser.add_argument("--early_stop_epochs", type=int, default=50, help="Early stop if no improvement after n epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                         help="Device to use for training")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    set_random_seed(args.seed)

    # load dataset setting
    if args.dataset == "mimic":
        dataset_config = MIMICGraphDatasetConfig()
    elif args.dataset == "eicu":
        dataset_config = EICUGraphDatasetConfig()
    
    # load graph
    with open(dataset_config.graph_path, "rb") as f:
        graph = pickle.load(f)

    train_dataset = TuckERDataset(graph=graph, config=dataset_config, top_ratio=dataset_config.top_edge_ratio, set_type="train")
    test_dataset = TuckERDataset(graph=graph, config=dataset_config, top_ratio=dataset_config.top_edge_ratio, set_type="test")

    # load algo_config
    if args.model == "tuckER":
        algo_config = TuckERConfig()
    else:
        raise NotImplementedError(f"Model {args.model} is not implemented")
    algo_config.log_dir = dataset_config.log_dir
    algo_config.early_stop_epochs = args.early_stop_epochs
    algo_config.entity_num = train_dataset.N
    algo_config.device = args.device
    Path(algo_config.log_dir).mkdir(parents=True, exist_ok=True)
    
    algo_config = load_best_hparams(algo_config, task="graph_embedding", dataset=args.dataset)

    # train, test, and save the graph embedding
    if args.model == "tuckER":
        model, metric_dict = tuckER_train_fn(train_dataset, test_dataset, algo_config, save_log=True)
    else:
        raise NotImplementedError(f"Model {args.model} is not implemented")
    