import torch
import numpy as np
from torch.utils.data import Sampler, BatchSampler



class IPSWeightedSampler(BatchSampler):
    '''
    Sampler that samples indices with replacement according to the IPS weights
    '''
    def __init__(self, dataset, batch_size, shuffle: bool = True):
        """
        Inputs:
            dataset - PyTorch Dataset, assumed to have binary labels accessible through dataset[idx][1]
                      and a method `get_labels()` returning a list of all labels for computing weights.
            batch_size - Number of samples in each batch
            shuffle - If True, shuffle indices each iteration
        """
        super().__init__(dataset, batch_size, shuffle)
        
        self.dataset = dataset

        # Get all labels from the dataset
        self.icd9s = dataset.get_icd9s()  # Assuming the dataset provides this method
        unique_icd9s = np.unique(self.icd9s)
        icd9_to_id = {icd9: idx for idx, icd9 in enumerate(unique_icd9s)}
        self.icd9_ids = [icd9_to_id[icd9] for icd9 in self.icd9s]

        # Compute class counts
        self.disease_counts = np.bincount(self.icd9_ids)
        self.disease_freqs = self.disease_counts / len(self.icd9_ids)
        self.class_weights = 1.0 / (self.disease_freqs + 1e-8) # IPS weights

        # Normalize weights
        self.class_weights /= np.sum(self.class_weights)

        # Create sampling probabilities for each sample
        self.sample_weights = [self.class_weights[label] for label in self.icd9_ids]
        self.sample_weights = torch.tensor(self.sample_weights, dtype=torch.float32)

    def __iter__(self):
        # Randomly sample indices with replacement according to the sample weights
        for _ in range(len(self)):
            batch_indices = torch.multinomial(
                self.sample_weights, 
                self.batch_size, 
                replacement=True
            ).tolist()
            yield batch_indices

    def __len__(self):
        # Define the number of iterations as the dataset size divided by the batch size
        return len(self.dataset) // self.batch_size