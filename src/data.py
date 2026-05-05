"""Data loading and preprocessing utilities for continual learning."""

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


class ContinualLearningDataset(Dataset):
    """Dataset for continual learning with task-specific data."""
    
    def __init__(self, data: torch.Tensor, targets: torch.Tensor, 
                 task_id: int, transform: Optional[Any] = None) -> None:
        """Initialize dataset.
        
        Args:
            data: Input data tensor
            targets: Target labels tensor
            task_id: Task identifier
            transform: Optional data transformation
        """
        self.data = data
        self.targets = targets
        self.task_id = task_id
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Get item with task ID."""
        data = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            data = self.transform(data)
            
        return data, target, self.task_id


def create_digits_tasks(num_tasks: int = 2, test_size: float = 0.2, 
                       random_state: int = 42) -> Dict[str, List[DataLoader]]:
    """Create continual learning tasks from digits dataset.
    
    Args:
        num_tasks: Number of tasks to create
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing train and test loaders for each task
    """
    # Load digits dataset
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0  # Normalize to [0, 1]
    y = digits.target
    
    # Split classes into tasks
    classes_per_task = 10 // num_tasks
    task_loaders = {'train': [], 'test': []}
    
    for task_id in range(num_tasks):
        start_class = task_id * classes_per_task
        end_class = (task_id + 1) * classes_per_task
        
        # Get data for this task
        task_mask = (y >= start_class) & (y < end_class)
        X_task = X[task_mask]
        y_task = y[task_mask] - start_class  # Remap to 0-based indexing
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_task, y_task, test_size=test_size, random_state=random_state
        )
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        # Create datasets
        train_dataset = ContinualLearningDataset(X_train_tensor, y_train_tensor, task_id)
        test_dataset = ContinualLearningDataset(X_test_tensor, y_test_tensor, task_id)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        task_loaders['train'].append(train_loader)
        task_loaders['test'].append(test_loader)
        
    return task_loaders


def create_synthetic_tasks(num_tasks: int = 2, samples_per_task: int = 1000,
                          features: int = 64, classes_per_task: int = 5,
                          test_size: float = 0.2, random_state: int = 42) -> Dict[str, List[DataLoader]]:
    """Create synthetic continual learning tasks.
    
    Args:
        num_tasks: Number of tasks to create
        samples_per_task: Number of samples per task
        features: Number of input features
        classes_per_task: Number of classes per task
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing train and test loaders for each task
    """
    task_loaders = {'train': [], 'test': []}
    
    for task_id in range(num_tasks):
        # Generate synthetic data for this task
        X, y = make_classification(
            n_samples=samples_per_task,
            n_features=features,
            n_classes=classes_per_task,
            n_redundant=0,
            n_informative=features,
            random_state=random_state + task_id
        )
        
        # Normalize features
        X = X.astype(np.float32)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state + task_id
        )
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        # Create datasets
        train_dataset = ContinualLearningDataset(X_train_tensor, y_train_tensor, task_id)
        test_dataset = ContinualLearningDataset(X_test_tensor, y_test_tensor, task_id)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        task_loaders['train'].append(train_loader)
        task_loaders['test'].append(test_loader)
        
    return task_loaders


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
