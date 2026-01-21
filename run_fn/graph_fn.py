import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
from tqdm import tqdm
import os
import pickle
import wandb
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score
from pathlib import Path

from configs.algo_config import TuckERConfig
from models.tucker_model import TuckERModel
from dataset.graph_dataset import TuckERDataset
from models.tracker import PerformanceTracker
from configs.dataset_config import MIMICGraphDatasetConfig, EICUGraphDatasetConfig
from utils.metrics import get_optimal_f1_cutoff


def tuckER_train_fn(train_dataset, test_dataset, config: TuckERConfig, 
                    save_log=True):
    """
    Training process for the TuckER model.

    Parameters:
    - train_dataset: TuckERDataset object used for training data
    - test_dataset: TuckERDataset object used for evaluation
    - config: Configuration object containing learning rate, batch size, negative sampling ratio, etc.
    - save_log: Whether to save training logs

    Returns:
    - model: Trained TuckER model
    """
    model = TuckERModel(d1=config.d, d2=config.r, config=config).to(config.device)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    tracker = PerformanceTracker(early_stop_epochs=config.early_stop_epochs, 
                                 metric='auroc', direction='maximize')
    
    # Start training
    for epoch in range(config.num_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        model.train()
        for h_idx, r_idx, label in pbar:
            h_idx, r_idx, label = h_idx.to(config.device), r_idx.to(config.device), label.to(config.device)

            optimizer.zero_grad()
            y_score = model(h_idx, r_idx)
            loss = F.binary_cross_entropy(y_score, label)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})
            pbar.update()

        # Evaluation on test data
        metric_dict = tucker_test_fn(model, test_dataset, config)

        # Update tracker
        early_stop_flag = tracker.update(metric_dict, model.state_dict())
        if early_stop_flag:
            break
        if (epoch+1) > config.epoch_num_lr_decay_start:
            lr_scheduler.step()
    
    best_model_state_dict = tracker.export_best_model_state_dict()
    test_result = tracker.export_best_metric_dict()
    model.load_state_dict(best_model_state_dict)
    node_embedding = model.get_node_embedding()

    if save_log:
        edge_top_ratio = train_dataset.get_edge_top_ratio()
        log_dir = os.path.join(config.log_dir, f"edge_top_ratio_{edge_top_ratio}")
        Path(log_dir).mkdir(exist_ok=True, parents=True)

        weights_path = os.path.join(log_dir, f"tucker.pth")
        test_result_path = os.path.join(log_dir, f"test_result.csv")
        node_embedding_path = os.path.join(log_dir, f"node_embedding.npy")
        tucker_graph_path = os.path.join(log_dir, f"tucker_graph.pkl")
        idx_to_node_path = os.path.join(log_dir, f"idx_to_node.pkl")

        torch.save(model.state_dict(), weights_path)
        test_result_df = pd.DataFrame(test_result, index=[0])
        test_result_df.to_csv(test_result_path, index=False)
        np.save(node_embedding_path, node_embedding)
        
        # Generate the similarity graph for target nodes
        idx_to_node = train_dataset.get_idx_to_node()
        tucker_graph = get_similarity_graph_fn(node_embedding, idx_to_node)
        with open(tucker_graph_path, "wb") as f:
            pickle.dump(tucker_graph, f)
        with open(idx_to_node_path, "wb") as f:
            pickle.dump(idx_to_node, f)
    
    return model, test_result



@torch.no_grad()
def tucker_test_fn(model, test_dataset, config:TuckERConfig,
                   cutoff=None):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size*4, shuffle=False)

    with torch.no_grad():
        total_y_score = []
        total_y_true = []
        for h_idx, r_idx, t_idx in test_loader:
            h_idx, r_idx = h_idx.to(config.device), r_idx.to(config.device)
            y_true = t_idx.cpu().numpy()
            y_score = model(h_idx, r_idx).cpu().numpy()
            total_y_score.extend(list(y_score.reshape(-1)))
            total_y_true.extend(list(y_true.reshape(-1)))
        
        total_y_score = np.array(total_y_score)
        total_y_true = np.array(total_y_true)
        # calculate metrics
        auroc = roc_auc_score(total_y_true, total_y_score)
        precision, recall, _ = precision_recall_curve(total_y_true, total_y_score)
        auprc = auc(recall, precision)
        if cutoff is None:
            cutoff = get_optimal_f1_cutoff(total_y_true, total_y_score)
        total_y_pred = [1 if score > cutoff else 0 for score in total_y_score]
        f1 = f1_score(total_y_true, total_y_pred)
        acc = accuracy_score(total_y_true, total_y_pred)
        
        metrics_dict = {"auroc": auroc,
                        "auprc": auprc,
                        "f1": f1,
                        "accuracy": acc,
                        'cutoff': cutoff}
    return metrics_dict


def graph_objective_fn(args):
    """
    Input:
    - args: a dictionary containing the following

    Returns:
    - metric_dict: a dictionary containing the metric
    """
    # load dataset config
    if args.dataset == "mimic":
        dataset_config = MIMICGraphDatasetConfig()
    elif args.dataset == "eicu":
        dataset_config = EICUGraphDatasetConfig()

    with open(dataset_config.graph_path, "rb") as f:
        graph = pickle.load(f)
    train_dataset = TuckERDataset(graph=graph, config=dataset_config, top_ratio=args.edge_top_ratio, set_type="train")
    test_dataset = TuckERDataset(graph=graph, config=dataset_config, top_ratio=args.edge_top_ratio, set_type="test")

    # load config
    if args.model == "tuckER":
        algo_config = TuckERConfig()
        algo_config.log_dir = dataset_config.log_dir
        algo_config.early_stop_epochs = args.early_stop_epochs
        algo_config.entity_num = train_dataset.N
    else:
        raise NotImplementedError(f"Model {args.model} is not implemented")
    
    # get hyper-parameters
    study_name = args.model
    with wandb.init(config=args, project=args.project, entity=args.entity, name=study_name) as run:
        algo_config.lr = run.config.lr
        algo_config.batch_size = run.config.batch_size
        algo_config.d = run.config.d
        algo_config.r = run.config.r
            
        # train
        if args.model == "tuckER":
            model, metric_dict = tuckER_train_fn(train_dataset, test_dataset, algo_config, save_log=False)
        else:
            raise NotImplementedError(f"Model {args.model} is not implemented")
        
        # log metrics
        run.log(metric_dict)


def get_similarity_graph_fn(node_embedding, idx_to_node):
    """
    calculate the cosine similarity between the target node and all other nodes
    and create a graph where the target node is connected to all other nodes

    Input:
    - A: a numpy array of size N x d, where N is the number of nodes
    - idx_to_node: a dictionary mapping indices to nodes
    - target_node: the target node

    Returns:
    - target_similarity_graph: a NetworkX graph where the target node is connected to all other nodes
    """
    edges = []
    for head_node_idx in list(idx_to_node.keys()):
        for tail_node_idx in list(idx_to_node.keys()):
            if head_node_idx == tail_node_idx:
                continue
            head_node = idx_to_node[head_node_idx]
            tail_node = idx_to_node[tail_node_idx]
            # use euclidean distance as similarity
            cos_similarity = F.cosine_similarity(torch.tensor(node_embedding[head_node_idx]).unsqueeze(0),
                                                 torch.tensor(node_embedding[tail_node_idx]).unsqueeze(0))
            edges.append((head_node, tail_node, cos_similarity.item()))

    # create graph
    similarity_graph = nx.Graph()
    similarity_graph.add_weighted_edges_from(edges)

    return similarity_graph