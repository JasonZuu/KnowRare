import pickle
import networkx as nx
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from typing import Union, Dict, List, Optional
import os
from tqdm import tqdm

from configs.dataset_config import MIMICDatasetConfig, EICUDatasetConfig
from dataset.datasets import MIMICDataset, EICUDataset
from dataset.sampler import IPSWeightedSampler


def _get_all_icd9(demo_csv_path):
    demo_df = pd.read_csv(demo_csv_path)
    return demo_df['icd9_code'].unique().tolist()


def _get_source_domain(all_icd9_codes: list,
                        target_icd9_codes: list,
                        source_domain_selection:str,
                        n_source: int,
                        graph=None)-> Dict:
    info = {}
    # select source domain
    if source_domain_selection == "all":
        source_domain_dict = {'all': all_icd9_codes}
    elif source_domain_selection == "top_n" and graph is not None:
        source_domain_dict = {}
        source_weights = {}
        for target_icd9 in target_icd9_codes:
            if not graph.has_node(target_icd9):
                raise ValueError(f"Node {target_icd9} not found in graph")

            # get neighbors and sort neighbors by weight
            neighbors = graph[target_icd9]
            sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1]['weight'], reverse=True)
            source_domain = [target_icd9] + [neighbor for neighbor, data in sorted_neighbors[:n_source]]
            source_domain_dict[target_icd9] = source_domain

            source_weights[target_icd9] = {target_icd9: 1}
            source_weights[target_icd9].update({neighbor: data['weight']\
                                                for neighbor, data in sorted_neighbors[:n_source]}) # take the rest n-1 neighbors
        info['source_weights'] = source_weights
        info['val_domain'] = [neightbor for neightbor, _ in sorted_neighbors[:int(0.1*len(sorted_neighbors))]] # top 10% neighbors for validation
    else:
        raise ValueError(f"Invalid source_domain_selection {source_domain_selection}")

    return source_domain_dict, info


def get_dataset_fn(dataset_config: Union[MIMICDatasetConfig, EICUDatasetConfig],
                    target_icd9_codes: list,
                    graph: nx.Graph = None,
                    dataset: str='mimic',
                    target_used_ratio: float=1.0):
    """
    Get the dataset for the given configuration

    Args:
        config: MIMICDatasetConfig
        target_icd9_codes: list of target icd9 codes
        graph: nx.Graph
        dataset: str
        target_used_ratio: the sample used ratio of target domain

    Returns:
        train_dataset:
        val_dataset:
        test_dataset:
        info
    """
    if dataset == 'mimic':
        dataset_cls = MIMICDataset
    elif dataset == 'eicu':
        dataset_cls = EICUDataset
    if dataset_config.source_domain_selection == "top_n" and graph is None:
        raise ValueError("graph must be provided when source_domain_selection is top_n")

    # determine the n_source
    train_demo_csv_path = os.path.join(dataset_config.train_dir, dataset_config.demo_csv_name)
    all_icd9_codes = _get_all_icd9(train_demo_csv_path)

    n_source = int(dataset_config.n_source * len(all_icd9_codes))+1
    source_domain_dict, domain_info = _get_source_domain(all_icd9_codes,
                                                        target_icd9_codes,
                                                        dataset_config.source_domain_selection,
                                                        n_source=n_source,
                                                        graph=graph)
    # load train and validation dataset
    if dataset_config.dataset_class == "baseline":
        source_domain = list(source_domain_dict.values())[0]
        train_dataset = dataset_cls(dataset_config, source_domain, "train")
        val_dataset = dataset_cls(dataset_config, source_domain, "val")
        info = {"source_domain": source_domain, "target_domain": target_icd9_codes,
                'domain_info': domain_info}
    elif dataset_config.dataset_class == "multi_model":
        train_dataset = []
        val_dataset = []
        info = []
        target_icd9_codes = []
        for target_icd9, source_domain in source_domain_dict.items():
            target_domain = [target_icd9]
            _train_dataset = dataset_cls(dataset_config, source_domain, "train", target_used_ratio=target_used_ratio)
            # init val dataset, if val_domain is provided, use val_domain
            if 'val_domain' in domain_info:
                val_domain = domain_info['val_domain']
                _val_dataset = dataset_cls(dataset_config, val_domain, "val")
            else:
                _val_dataset = dataset_cls(dataset_config, source_domain, "val")
            
            _info = {"source_domain": source_domain, "target_domain": target_domain,
                     'domain_info': domain_info}
            train_dataset.append(_train_dataset)
            val_dataset.append(_val_dataset)
            info.append(_info)
            target_icd9_codes.append(target_icd9) # keep track of target_icd9_codes
    
    
    # init test dataset
    test_dataset_list = []
    for target_icd9_code in target_icd9_codes:
        test_dataset = dataset_cls(dataset_config, [target_icd9_code], "test")
        test_dataset_list.append(test_dataset)
        
    
    if dataset_config.source_domain_selection == "top_n":
        source_weights = domain_info["source_weights"]
        for _info in info:
            target_icd9 = _info["target_domain"][0]
            _info["source_weights"] = source_weights[target_icd9]

    return train_dataset, val_dataset, test_dataset_list, info


def get_dataloader_fn(algo_config, dataset, mode: str,
                      batch_size: int = None, drop_last: bool = True):
    if batch_size is None:
        batch_size = algo_config.batch_size # default batch size
    if mode == "train":
        loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=drop_last)
    elif mode == "resampling":
        sampler = IPSWeightedSampler(dataset, batch_size, shuffle=True)
        loader = DataLoader(dataset, batch_sampler=sampler)
    elif mode == "test" or mode == "val":
        loader = DataLoader(dataset, 4*batch_size, shuffle=False, drop_last=False)
    else:
        raise ValueError(f"Invalid mode {mode}")
    return loader
