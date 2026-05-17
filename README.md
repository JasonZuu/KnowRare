# Bridging Data Gaps of Rare Conditions in ICU: A Multi-Disease Adaptation Approach for Disease-specific Clinical Prediction

## Introduction

KnowRare is a deep learning framework designed to improve clinical outcome predictions for rare ICU conditions with data scarcity and intra-condition heterogeneity issues. It utilises condition-agnostic self-supervised pre-training to capture general temporal patterns and a condition knowledge graph to selectively adapt insights from clinically similar conditions through joint adversarial domain adaptation.

## Getting Started

### Prerequisites

Before running the project, ensure you have the following prerequisites installed:

- Python 3.10+
- PyTorch
- Other dependencies listed in `requirements.txt`

### Installation

This is a demo installation guide for anonymity. The actual installation instructions will be provided in the official repository.

```bash
git clone <https://github.com/knowrare.git>
cd knowrare
pip install -r requirements.txt
```

### Requesting Access to the Data
The EHR datasets used in this study require additional access permissions. Please request access before proceeding. Additionally, you should install the datasets in a PostgreSQL database to run the data processing code.

### Data Processing
Once you have obtained access and installed the datasets in PostgreSQL, you can run the data processing code located in the data_processing directory. There are two subdirectories, MIMIC-III and eICU, each containing scripts for processing the respective datasets. The data processing pipeline consists of seven steps:

1. Data Extraction – Extract necessary raw data. Please install the MIMIC-III and eICU datasets in PostgreSQL before running the data processing code. If you have not installed the datasets, please follow the instructions in the [MIMIC-III](https://mimic.mit.edu/docs/gettingstarted/) and [eICU](https://eicu-crd.mit.edu/gettingstarted/overview/) websites.
2. Convert to Time Series – Transform raw data into time series format. Support 30min, 1h, and 2h time intervals. The default one is 2h for MIMIC-III and 1h for eICU.
3. Data Splitting – Partition the dataset into training, validation, and test sets. To guarantee reproducibility, we provide the exact patient IDs used for each split.
4. Data Imputation – Handle missing data using imputation techniques. Support backward-forward-filling, mean imputation, and linear interpolation. The default one is backward-forward-filling.
5. Data Normalization – Normalize the data using statistics from the training set.
6. Graph Construction – Construct a disease knowledge graph.
7. Data Validation – Verify the correctness of the processed data and check the statistics of rare diseases.

### Running a Demo
After completing the above steps, you can run the pretraining code, which is the first step in KnowRare training. Execute the following command to start:
```bash
python pretrain.py --dataset mimic
```

## Usage

### Step 1: Pretraining
You should start with the disease-agnostic pretraining using
```bash
python pretrain.py --dataset mimic
python pretrain.py --dataset eicu
```

### Step 2: Knowledge Graph Embedding
You can now extract the disease embedding with the knowledge graph embeding method using
```bash
python graph_embedding.py --dataset mimic
python graph_embedding.py --dataset eicu
```

### Step 3: Train with KnowRare and evaluate it on the test set
Now you have completed the pretraining and domain selection part. let's start the training using
```bash
python train_eval.py --algo knowrare --dataset mimic --task mortality_day90 --use_best_hparams
python train_eval.py --algo knowrare --dataset mimic --task readmission_day30 --use_best_hparams
python train_eval.py --algo knowrare --dataset eicu --task icu_mortality --use_best_hparams
python train_eval.py --algo knowrare --dataset eicu --task remaining_los --use_best_hparams
```
Afterward, you can see KnowRare's performance on the test set for the ten rare diseases.

We also provide the trained model weights for each task. You can find them at [Kaggle](https://www.kaggle.com/datasets/mingchengzhu/knowrare-model-weights). Please download them and put them in the "log/{dataset}/{task}/0.1/knowrare/1" directory so that the rest of the code can load them.

### Step 4. Evaluation Only
You can evaluate saved model weights using the following command:
```bash
python train_eval.py --algo knowrare --dataset mimic --task mortality_day90 --eval_only
python train_eval.py --algo knowrare --dataset mimic --task readmission_day30 --eval_only
python train_eval.py --algo knowrare --dataset eicu --task icu_mortality --eval_only
python train_eval.py --algo knowrare --dataset eicu --task remaining_los --eval_only
```


## Project Structure

The stucture of this project is as follows:

```bash
knowrare/
│
├── README.md         # Project documentation
├── requirements.txt  # List of dependencies
├── pretrain.py       # the pretraining code
├── graph_embedding.py  # the graph embedding code
├── train_eval.py     # the KnowRare training/evaluation entrypoint
├── configs/                # configuration directory
│   ├── algo_config.py      # algorithm configs
│   └── dataset_config.py   # dataset configs
├── dataset/                # datasets
│   ├── dataset_load_fn.py  # dataset loading functions
│   ├──datasets.py          # dataset for MIMIC-III and eICU
│   ├── graph_dataset.py    # dataset for graph embedding
│   └── sampler.py          # data sampler for resampling
├── run_fn/                 # training and testing functions
│   ├── __init__.py
│   ├── graph.py            # graph embedding function
│   ├── knowrare.py         # knowware adaptation process
│   ├── pretrain_recons.py  # pretraining 
│   └── test_fn.py          # testing
├── models/                 # code for backbone models 
│   └── ...
└── utils/                  # useful tools
    └── ...
```

## Citation
If you find this project helpful, please cite our paper:
```bibtex
@article{zhu2026bridging,
  title={Bridging data gaps of rare conditions in ICU: a multi-disease adaptation approach for clinical prediction},
  author={Zhu, Mingcheng and Liu, Yu and Luo, Zhiyao and Zhu, Tingting},
  journal={npj Digital Medicine},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```

## License
Shield: [![CC BY-NC-ND 4.0][cc-by-nc-nd-shield]][cc-by-nc-nd]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License][cc-by-nc-nd].

[![CC BY-NC-ND 4.0][cc-by-nc-nd-image]][cc-by-nc-nd]

[cc-by-nc-nd]: http://creativecommons.org/licenses/by-nc-nd/4.0/
[cc-by-nc-nd-image]: https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png
[cc-by-nc-nd-shield]: https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg
