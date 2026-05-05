"""Core models for continual learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron for continual learning tasks."""
    
    def __init__(self, input_dim: int = 64, hidden_dims: List[int] = [128, 64], 
                 output_dim: int = 10, dropout: float = 0.1) -> None:
        """Initialize the MLP.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (number of classes)
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class ContinualLearningModel(nn.Module):
    """Base class for continual learning models with task-specific heads."""
    
    def __init__(self, backbone: nn.Module, num_tasks: int = 2, 
                 task_output_dims: Optional[List[int]] = None) -> None:
        """Initialize continual learning model.
        
        Args:
            backbone: Shared backbone network
            num_tasks: Number of tasks
            task_output_dims: Output dimensions for each task
        """
        super().__init__()
        self.backbone = backbone
        self.num_tasks = num_tasks
        
        if task_output_dims is None:
            task_output_dims = [10] * num_tasks
            
        self.task_heads = nn.ModuleList([
            nn.Linear(backbone.network[-2].out_features, dim) 
            for dim in task_output_dims
        ])
        
    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """Forward pass with optional task-specific head."""
        features = self.backbone.network[:-1](x)  # All layers except final linear
        
        if task_id is not None:
            return self.task_heads[task_id](features)
        else:
            # Return features for all tasks
            return torch.stack([head(features) for head in self.task_heads], dim=1)
