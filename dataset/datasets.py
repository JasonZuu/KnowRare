import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import os

from configs.dataset_config import MIMICDatasetConfig, EICUDatasetConfig


class MIMICDataset(Dataset):
    """
    Randomly sampling from all records from included sets
    """
    def __init__(self, config: MIMICDatasetConfig,
                 source_domain: list,
                 dataset_type: str,
                 target_used_ratio: float=1.0):
        super().__init__()
        if dataset_type == "train":
            self.data_dir = config.train_dir
        elif dataset_type == "val":
            self.data_dir = config.val_dir
        elif dataset_type == "test":
            self.data_dir = config.test_dir
        else:
            raise ValueError(f"Invalid dataset type {dataset_type}")

        self.demo_df = pd.read_csv(os.path.join(self.data_dir, config.demo_csv_name))
        self.ts_df = pd.read_csv(os.path.join(self.data_dir, config.ts_csv_name))
        self.label_df = pd.read_csv(os.path.join(self.data_dir, config.label_csv_name))
        self.source_domain = source_domain
        self.dataset_type = dataset_type
        self.config = config
        self.num_ts_features = config.num_ts_features
        self.target_used_ratio = target_used_ratio

        target_domain = [self.source_domain[0]]
        source_domain = self.source_domain[1:]
        target_demo_df = self.demo_df[self.demo_df['icd9_code'].isin(target_domain)]
        target_hadm_id_list = list(target_demo_df['hadm_id'].values)
        target_hadm_id_list = target_hadm_id_list[:int(len(target_hadm_id_list) * self.target_used_ratio)]
        if len(source_domain) > 0:
            source_demo_df = self.demo_df[self.demo_df['icd9_code'].isin(source_domain)]
            source_hadm_id_list = list(source_demo_df['hadm_id'].values)
            hadm_id_list = set(target_hadm_id_list + source_hadm_id_list).intersection(set(self.label_df['hadm_id'].values)).intersection(set(self.ts_df['hadm_id'].values))
        else:
            hadm_id_list = set(target_hadm_id_list).intersection(set(self.label_df['hadm_id'].values)).intersection(set(self.ts_df['hadm_id'].values))
        self.hadm_id_list = list(hadm_id_list)
        self.icd9_codes = self.demo_df['icd9_code'].unique()

        self.demo_df = self.demo_df[self.demo_df['hadm_id'].isin(self.hadm_id_list)]
        self.ts_df = self.ts_df[self.ts_df['hadm_id'].isin(self.hadm_id_list)]
        self.label_df = self.label_df[self.label_df['hadm_id'].isin(self.hadm_id_list)]
        self.hadm_to_icd9 = dict(zip(self.demo_df['hadm_id'], self.demo_df['icd9_code']))

    def __len__(self):
        return len(self.hadm_id_list)

    def __getitem__(self, idx):
        hadm_id = self.hadm_id_list[idx]

        demo = self.demo_df[self.demo_df['hadm_id'] == hadm_id].loc[:,
               ['ethnicity_category', 'gender_category', 'age']].values
        label = self.label_df[self.label_df['hadm_id'] == hadm_id].loc[:, [self.config.label_name]].values
        ts = self.ts_df[self.ts_df['hadm_id'] == hadm_id].iloc[:, -self.num_ts_features:].values
        icd9_code = self.demo_df[self.demo_df['hadm_id'] == hadm_id]['icd9_code'].values[0]
        icd9_code_idx = np.where(self.icd9_codes == icd9_code)[0][0]

        return {"demography": torch.tensor(demo).squeeze(0).float(),
                "time_series": torch.tensor(ts).float(),
                "label": torch.tensor(label, requires_grad=False).squeeze(0).float(),
                'icd9_code': icd9_code,
                'icd9_code_idx': icd9_code_idx,
                "hadm_id": hadm_id,
                'idx': idx}
    
    def get_labels(self):
        return self.label_df[self.config.label_name].values
    
    def get_icd9_codes(self):
        return self.icd9_codes
    
    def get_icd9s(self):
        return [self.hadm_to_icd9[hadm_id] for hadm_id in self.hadm_id_list]


class EICUDataset(Dataset):
    """
    Randomly sampling from all records from included sets
    """
    def __init__(self, config: EICUDatasetConfig,
                 source_domain: list,
                 dataset_type: str,
                 target_used_ratio: float=1.0):
        super().__init__()
        if dataset_type == "train":
            self.data_dir = config.train_dir
        elif dataset_type == "val":
            self.data_dir = config.val_dir
        elif dataset_type == "test":
            self.data_dir = config.test_dir
        else:
            raise ValueError(f"Invalid dataset type {dataset_type}")
        
        self.ts_features = ['heartrate_min', 'heartrate_max', 'heartrate_mean', 'systemicsystolic_min', 'systemicsystolic_max', 'systemicsystolic_mean', 
                            'systemicdiastolic_min', 'systemicdiastolic_max', 'systemicdiastolic_mean', 'systemicmean_min', 'systemicmean_max', 'systemicmean_mean',
                            'respiration_min', 'respiration_max', 'respiration_mean', 'temperature_min', 'temperature_max',
                            'temperature_mean', 'glucose_min', 'glucose_max', 'glucose_mean', 
                            'HCO3', 'Hct', 'Hgb', 'PT', 'PTT', 'albumin', 'anion gap',
                            'chloride', 'creatinine', 'lactate', 'platelets x 1000', 'sodium', 'total bilirubin']

        self.demo_df = pd.read_csv(os.path.join(self.data_dir, config.demo_csv_name))
        self.ts_df = pd.read_csv(os.path.join(self.data_dir, config.ts_csv_name))
        self.label_name = config.label_name
        if self.label_name == 'icu_mortality':
            self.label_df = pd.read_csv(os.path.join(self.data_dir, config.label_icumortality_csv_name))
        elif self.label_name == 'remaining_los':
            self.label_df = pd.read_csv(os.path.join(self.data_dir, config.label_los_csv_name))
        else:
            raise ValueError(f"Invalid label name {self.label_name}")
        self.source_domain = source_domain
        self.dataset_type = dataset_type
        self.config = config
        self.num_ts_features = config.num_ts_features
        self.all_classes = list(self.label_df[self.label_name].unique())
        self.target_used_ratio = target_used_ratio

        target_domain = [self.source_domain[0]]
        source_domain = self.source_domain[1:]

        target_demo_df = self.demo_df[self.demo_df['icd9_code'].isin(target_domain)]
        target_stay_id_list = list(target_demo_df['stay_id'].values)
        target_stay_id_list = target_stay_id_list[:int(len(target_demo_df) * self.target_used_ratio)]
        if len(source_domain) > 0:
            source_demo_df = self.demo_df[self.demo_df['icd9_code'].isin(source_domain)]
            source_stay_id_list = list(source_demo_df['stay_id'].values)
            stay_id_list = set(target_stay_id_list + source_stay_id_list).intersection(set(self.label_df['stay_id'].values)).intersection(set(self.ts_df['stay_id'].values))
        else:
            stay_id_list = set(target_stay_id_list).intersection(set(self.label_df['stay_id'].values)).intersection(set(self.ts_df['stay_id'].values))
        self.stay_id_list = list(stay_id_list)
        self.icd9_codes = self.demo_df['icd9_code'].unique()

        self.demo_df = self.demo_df[self.demo_df['stay_id'].isin(self.stay_id_list)]
        self.ts_df = self.ts_df[self.ts_df['stay_id'].isin(self.stay_id_list)]
        self.label_df = self.label_df[self.label_df['stay_id'].isin(self.stay_id_list)]
        self.stay_to_icd9 = dict(zip(self.demo_df['stay_id'], self.demo_df['icd9_code']))

    def __len__(self):
        return len(self.stay_id_list)

    def __getitem__(self, idx):
        stay_id = self.stay_id_list[idx]

        demo = self.demo_df[self.demo_df['stay_id'] == stay_id].loc[:,
               ['ethnicity_category', 'gender_category', 'age']].values
        if self.label_name == 'icu_mortality':
            label = self.label_df[self.label_df['stay_id'] == stay_id].loc[:, [self.label_name]].values
        elif self.label_name == 'drugname_category' or self.label_name == 'remaining_los':
            label = self.label_df[self.label_df['stay_id'] == stay_id].loc[:, [self.label_name]].values
            label = self.get_one_hot(label)

        ts = self.ts_df[self.ts_df['stay_id'] == stay_id].loc[:, self.ts_features].values
        icd9_code = self.demo_df[self.demo_df['stay_id'] == stay_id]['icd9_code'].values[0]
        icd9_code_idx = np.where(self.icd9_codes == icd9_code)[0][0]

        return {"demography": torch.tensor(demo).squeeze(0).float(),
                "time_series": torch.tensor(ts).float(),
                "label": torch.tensor(label, requires_grad=False).squeeze(0).float(),
                'icd9_code': icd9_code,
                "stay_id": stay_id,
                'icd9_code_idx': icd9_code_idx,
                'idx': idx}
    
    def get_labels(self):
        return self.label_df[self.config.label_name].values
    
    def get_one_hot(self, labels):
        # initialize one-hot vector as all zeros
        one_hot = np.zeros(len(self.all_classes), dtype=int)
        
        for label in labels:
            if label in self.all_classes:
                one_hot[self.all_classes.index(label)] = 1
        
        return one_hot
    
    def get_icd9_codes(self):
        return self.icd9_codes
    
    def get_icd9s(self):
        return [self.stay_to_icd9[stay_id] for stay_id in self.stay_id_list]
    


# class MIMICDomainDataset(MIMICDataset):
#     """
#     Randomly sampling from all records from included sets
#     """
#     def __init__(self, config: MIMICDatasetConfig,
#                  domain: list):
#         super().__init__()
#         self.demo_df = pd.read_csv(config.demo_csv_path)
#         self.ts_df = pd.read_csv(config.ts_csv_path)
#         self.label_df = pd.read_csv(config.label_csv_path)
#         self.icd_df = self.demo_df[['hadm_id', 'icd9_code']].copy()
#         self.config = config

#         # get subject ids
#         self.demo_df = self.demo_df[self.demo_df['icd9_code'].isin(domain)]
#         self.subject_ids = self.demo_df['subject_id'].unique()
#         subject_id_to_hadm_ids = {subject_id: list(self.demo_df[self.demo_df['subject_id'] == subject_id]['hadm_id'].values) for subject_id in self.subject_ids}
        
#         self.hadm_id_list = []
#         for subject_id in self.subject_ids:
#             self.hadm_id_list.extend(subject_id_to_hadm_ids[subject_id])

#         self.demo_df = self.demo_df[self.demo_df['hadm_id'].isin(self.hadm_id_list)]
#         self.ts_df = self.ts_df[self.ts_df['hadm_id'].isin(self.hadm_id_list)]
#         self.label_df = self.label_df[self.label_df['hadm_id'].isin(self.hadm_id_list)]

#     def __len__(self):
#         return len(self.hadm_id_list)

#     def __getitem__(self, idx):
#         hadm_id = self.hadm_id_list[idx]

#         demo = self.demo_df[self.demo_df['hadm_id'] == hadm_id].loc[:,
#                ['ethnicity_category', 'gender_category', 'age', "staytime"]].values
#         demo_np = np.array(demo, dtype=float)
#         ts = self.ts_df[self.ts_df['hadm_id'] == hadm_id].iloc[:, 3:].values
#         ts_np = np.array(ts, dtype=float)
#         label = self.label_df[self.label_df['hadm_id'] == hadm_id].loc[:, [self.config.label_name]].values

#         return {"demography": torch.tensor(demo_np).squeeze(0).float(),
#                 "time_series": torch.tensor(ts_np).float(),
#                 "label": torch.tensor(label).float()}
    
#     def get_labels(self):
#         return self.label_df[self.config.label_name].values


def test_eicu_dataset():
    config = EICUDatasetConfig()
    config.label_name = 'drugname_category'

    # Mock source domain (ICD-9 codes)
    source_domain = ["ICD_410", "ICD_038"]

    # Instantiate the dataset
    train_dataset = EICUDataset(config, source_domain, dataset_type="train")
    print(f"Number of samples in train dataset: {len(train_dataset)}")

    # Test __getitem__ method
    sample_idx = 0
    if len(train_dataset) > 0:
        sample = train_dataset[sample_idx]
        print("Sample:")
        print(f"  Demography: {sample['demography'].shape}")
        print(f"  Time series: {sample['time_series'].shape}")
        print(f"  Label: {sample['label']}")
        print(f"  ICD-9 Code: {sample['icd9_code']}")
        print(f"  ICD-9 Code Index: {sample['icd9_code_idx']}")
        print(f"  Stay ID: {sample['stay_id']}")
    else:
        print("No samples available in the train dataset.")

    # Test get_labels and get_icd9s methods
    labels = train_dataset.get_labels()
    icd9s = train_dataset.get_icd9_codes()

    print(f"Labels: {labels[:5]}")  # Print first 5 labels
    print(f"ICD-9 Codes: {icd9s[:5]}")  # Print first 5 ICD-9 codes

if __name__ == "__main__":
    test_eicu_dataset()
