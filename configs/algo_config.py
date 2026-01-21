from dataclasses import dataclass
import torch


@dataclass
class AdvConfig:
    project = "knowrare"
    study_name = None
    seed = 0
    val_method = "transfer"
    
    epochs_num = 100
    early_stop_epochs = 10
    lr_warmup_epochs = 10
    lr_decay_steps = 1
    lr_decay_gamma = 0.95
    demo_dims = 3
    ts_dims = 43
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = None  # provided before training
    select_metric = "auprc"
    k_shot = 10
    num_classes = 2
    batch_size = 32
    source_domain = []
    resampling = True  # whether to leverage IPS-based resampling
    n_source = 20  # 10% of all source domains

    # optimized hyper-parameters
    lr = 1e-4 
    dis_lr = 1e-3
    dis_coeff = 1.0
    dis_step = 1
    batch_size = 32
    


@dataclass
class TuckERConfig:
    project = "knowrare"
    study_name = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = 'log/graph'
    target_icd9 = None
    early_stop_epochs = 50 
    relations = ['comorbidity', 'usability', 'drug']
    relation_num = 3
    entity_num = 214
    input_dropout=0.3
    hidden_dropout1=0.4
    hidden_dropout2=0.5
    epoch_num_lr_decay_start = 25
    num_epochs=1000

    # optimized hyper-parameters
    batch_size=64
    lr=0.001
    d=32
    r=16


@dataclass
class ReconstructionConfig:
    project = "knowrare"
    study_name = None
    seed = 0
    val_method = "transfer"
    model = "lstm"
    
    epochs_num = 100
    early_stop_epochs = 10
    lr_warmup_epochs = 10
    lr_decay_steps = 1
    lr_decay_gamma = 0.95
    demo_dims = 3
    ts_dims = 43
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = None  # provided before training
    select_metric = "mse"

    # optimized hyper-parameterslog/graph/tucker_graph.pkl
    lr = 1e-4
    batch_size = 32

