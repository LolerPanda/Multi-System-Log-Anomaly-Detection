import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LogDataset(Dataset):
    """
    Dataset class for log sequences
    """
    def __init__(self, data):
        """
        Initialize dataset
        
        Args:
            data: Dictionary containing 'features', 'labels', and 'sequences'
        """
        self.features = data['features']
        self.labels = data['labels']
        self.sequences = data['sequences']
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Args:
            idx: Index of the sample
            
        Returns:
            feature: The feature vector
            label: The label (0 for normal, 1 for anomaly)
            sequence: The original log sequence (for reference)
        """
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        sequence = self.sequences[idx]
        
        return feature, label, sequence
    
def create_data_loaders(train_data, test_data, batch_size=512, num_workers=4):
    """
    Create PyTorch data loaders for training and testing
    
    Args:
        train_data: Training data dictionary
        test_data: Testing data dictionary
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, test_loader: DataLoader objects for training and testing
    """
    train_dataset = LogDataset(train_data)
    test_dataset = LogDataset(test_data)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader 