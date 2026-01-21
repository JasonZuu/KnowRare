from dataclasses import dataclass
import torch


@dataclass
class MIMICDatasetConfig:
    """
    Configuration for the dataset loading
    """
    graph_c_path = "data/MIMICIII_last48h_ts2h/train/comorbidity_graph.pkl"
    graph_u_path = "data/MIMICIII_last48h_ts2h/train/usability_graph.pkl"
    graph_d_path = "data/MIMICIII_last48h_ts2h/train/drug_graph.pkl"
    graph_sum_path = "data/MIMICIII_last48h_ts2h/train/sum_graph.pkl"
    graph_tucker_tope1_path = "data/MIMICIII_last48h_ts2h/graph/edge_top_ratio_1.0/tucker_graph.pkl"
    graph_tucker_tope09_path = "data/MIMICIII_last48h_ts2h/graph/edge_top_ratio_0.9/tucker_graph.pkl"
    graph_tucker_tope08_path = "data/MIMICIII_last48h_ts2h/graph/edge_top_ratio_0.8/tucker_graph.pkl"
    graph_tucker_tope07_path = "data/MIMICIII_last48h_ts2h/graph/edge_top_ratio_0.7/tucker_graph.pkl"
    graph_tucker_tope06_path = "data/MIMICIII_last48h_ts2h/graph/edge_top_ratio_0.6/tucker_graph.pkl"
    graph_tucker_tope05_path = "data/MIMICIII_last48h_ts2h/graph/edge_top_ratio_0.5/tucker_graph.pkl"
    graph_tucker_tope04_path = "data/MIMICIII_last48h_ts2h/graph/edge_top_ratio_0.4/tucker_graph.pkl"
    graph_tucker_tope03_path = "data/MIMICIII_last48h_ts2h/graph/edge_top_ratio_0.3/tucker_graph.pkl"
    graph_tucker_tope02_path = "data/MIMICIII_last48h_ts2h/graph/edge_top_ratio_0.2/tucker_graph.pkl"
    graph_tucker_tope01_path = "data/MIMICIII_last48h_ts2h/graph/edge_top_ratio_0.1/tucker_graph.pkl"
    graph_tucker_tope005_path = "data/MIMICIII_last48h_ts2h/graph/edge_top_ratio_0.05/tucker_graph.pkl"
    graph_tucker_tope001_path = "data/MIMICIII_last48h_ts2h/graph/edge_top_ratio_0.01/tucker_graph.pkl"
    
    train_dir = "data/MIMICIII_last48h_ts2h/train"
    val_dir = "data/MIMICIII_last48h_ts2h/val"
    test_dir = "data/MIMICIII_last48h_ts2h/test"
    root_dir = "data/MIMICIII_last48h_ts2h"

    demo_csv_name = 'demographics.csv'
    ts_csv_name = 'time-series.csv'
    label_csv_name = "label.csv"

    label_name = "days_90_expire_flag"  # "days_30_readmission_flag", "days_90_expire_flag"
    dataset_class = "baseline"  # "baseline", "multi_model", "meta"
    num_classes_dict = {"days_30_readmission_flag": 1, "days_90_expire_flag": 1}
    num_demo_features = 3
    num_ts_features = 43
    num_timesteps = 24
    tucker_tope = 0.6

    # source domain selection
    source_domain_selection = "all"  # "all", "random", "top_n", 'icd_same_category', 'icd_same_diff_category'
    n_source = 0.1  # 10% of all source domains


@dataclass
class EICUDatasetConfig:
    """
    Configuration for the dataset loading
    """
    graph_c_path = "data/eICU_first24h_ts1h/train/comorbidity_graph.pkl"
    graph_u_path = "data/eICU_first24h_ts1h/train/usability_graph.pkl"
    graph_d_path = "data/eICU_first24h_ts1h/train/drug_graph.pkl"
    graph_sum_path = "data/eICU_first24h_ts1h/train/sum_graph.pkl"
    graph_tucker_tope1_path = "data/eICU_first24h_ts1h/graph/edge_top_ratio_1.0/tucker_graph.pkl"
    graph_tucker_tope09_path = "data/eICU_first24h_ts1h/graph/edge_top_ratio_0.9/tucker_graph.pkl"
    graph_tucker_tope08_path = "data/eICU_first24h_ts1h/graph/edge_top_ratio_0.8/tucker_graph.pkl"
    graph_tucker_tope07_path = "data/eICU_first24h_ts1h/graph/edge_top_ratio_0.7/tucker_graph.pkl"
    graph_tucker_tope06_path = "data/eICU_first24h_ts1h/graph/edge_top_ratio_0.6/tucker_graph.pkl"
    graph_tucker_tope05_path = "data/eICU_first24h_ts1h/graph/edge_top_ratio_0.5/tucker_graph.pkl"
    graph_tucker_tope04_path = "data/eICU_first24h_ts1h/graph/edge_top_ratio_0.4/tucker_graph.pkl"
    graph_tucker_tope03_path = "data/eICU_first24h_ts1h/graph/edge_top_ratio_0.3/tucker_graph.pkl"
    graph_tucker_tope02_path = "data/eICU_first24h_ts1h/graph/edge_top_ratio_0.2/tucker_graph.pkl"
    graph_tucker_tope01_path = "data/eICU_first24h_ts1h/graph/edge_top_ratio_0.1/tucker_graph.pkl"
    graph_tucker_tope005_path = "data/eICU_first24h_ts1h/graph/edge_top_ratio_0.05/tucker_graph.pkl"
    graph_tucker_tope001_path = "data/eICU_first24h_ts1h/graph/edge_top_ratio_0.01/tucker_graph.pkl"
    
    train_dir = "data/eICU_first24h_ts1h/train"
    val_dir = "data/eICU_first24h_ts1h/val"
    test_dir = "data/eICU_first24h_ts1h/test"
    root_dir = "data/eICU_first24h_ts1h"

    demo_csv_name = 'demographics.csv'
    ts_csv_name = 'time-series.csv'
    label_icumortality_csv_name = "label_icumortality.csv"
    label_los_csv_name = "label_los.csv"
    label_medication_csv_name = "label_medication.csv"

    label_name = "icu_mortality" # "icu_mortality", "remaining_los", "drugname_category"
    dataset_class = "baseline"  # "baseline", "multi_model", "meta"
    num_classes_dict = {"icu_mortality": 1, "remaining_los": 10, "drugname_category": 400}
    num_demo_features = 3
    num_ts_features = 34
    num_timesteps = 24
    tucker_tope = 0.05

    # source domain selection
    source_domain_selection = "all"  # "all", "random", "top_n", 'icd_same_category', 'icd_same_diff_category'
    n_source = 0.1  # 10% of all source domains


@dataclass
class MIMICGraphDatasetConfig:
    """
    Configuration for the graph dataset
    """
    log_dir = "data/MIMICIII_last48h_ts2h/graph"
    graph_path = "data/MIMICIII_last48h_ts2h/train/multi_graph.pkl"
    relations = ['comorbidity', 'usability', 'drug']
    
    test_size = 0.1
    random_seed = 42
    top_edge_ratio = 0.6


@dataclass
class EICUGraphDatasetConfig:
    """
    Configuration for the graph dataset
    """
    log_dir = "data/eICU_first24h_ts1h/graph"
    graph_path = "data/eICU_first24h_ts1h/train/multi_graph.pkl"
    relations = ['comorbidity', 'usability', 'drug']
    
    test_size = 0.1
    random_seed = 42
    top_edge_ratio = 0.05
