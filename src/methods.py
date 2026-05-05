"""Continual learning algorithms and regularization methods."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from abc import ABC, abstractmethod


class ContinualLearningMethod(ABC):
    """Abstract base class for continual learning methods."""
    
    @abstractmethod
    def compute_regularization_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute regularization loss for continual learning."""
        pass
    
    @abstractmethod
    def update_after_task(self, model: nn.Module, dataloader: torch.utils.data.DataLoader) -> None:
        """Update method-specific parameters after training on a task."""
        pass


class ElasticWeightConsolidation(ContinualLearningMethod):
    """Elastic Weight Consolidation (EWC) for continual learning."""
    
    def __init__(self, importance_factor: float = 1000.0) -> None:
        """Initialize EWC.
        
        Args:
            importance_factor: Weight for the EWC regularization term
        """
        self.importance_factor = importance_factor
        self.saved_params: Dict[str, torch.Tensor] = {}
        self.fisher_information: Dict[str, torch.Tensor] = {}
        
    def compute_regularization_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute EWC regularization loss."""
        if not self.saved_params:
            return torch.tensor(0.0, device=next(model.parameters()).device)
            
        ewc_loss = 0.0
        for name, param in model.named_parameters():
            if name in self.saved_params and name in self.fisher_information:
                fisher = self.fisher_information[name]
                old_param = self.saved_params[name]
                ewc_loss += (fisher * (param - old_param) ** 2).sum()
                
        return self.importance_factor * ewc_loss
    
    def update_after_task(self, model: nn.Module, dataloader: torch.utils.data.DataLoader) -> None:
        """Update Fisher information matrix after training on a task."""
        # Save current parameters
        self.saved_params = {name: param.clone().detach() 
                           for name, param in model.named_parameters()}
        
        # Compute Fisher information matrix
        fisher_information = {name: torch.zeros_like(param) 
                            for name, param in model.named_parameters()}
        
        model.eval()
        for data, target in dataloader:
            model.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher_information[name] += param.grad ** 2 / len(dataloader)
                    
        self.fisher_information = fisher_information


class L2Regularization(ContinualLearningMethod):
    """L2 regularization to prevent catastrophic forgetting."""
    
    def __init__(self, lambda_reg: float = 0.01) -> None:
        """Initialize L2 regularization.
        
        Args:
            lambda_reg: Regularization strength
        """
        self.lambda_reg = lambda_reg
        self.initial_params: Dict[str, torch.Tensor] = {}
        
    def compute_regularization_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute L2 regularization loss."""
        if not self.initial_params:
            return torch.tensor(0.0, device=next(model.parameters()).device)
            
        l2_loss = 0.0
        for name, param in model.named_parameters():
            if name in self.initial_params:
                l2_loss += ((param - self.initial_params[name]) ** 2).sum()
                
        return self.lambda_reg * l2_loss
    
    def update_after_task(self, model: nn.Module, dataloader: torch.utils.data.DataLoader) -> None:
        """Store initial parameters after first task."""
        if not self.initial_params:
            self.initial_params = {name: param.clone().detach() 
                                 for name, param in model.named_parameters()}


class MemoryAwareSynapses(ContinualLearningMethod):
    """Memory Aware Synapses (MAS) for continual learning."""
    
    def __init__(self, lambda_reg: float = 1.0) -> None:
        """Initialize MAS.
        
        Args:
            lambda_reg: Regularization strength
        """
        self.lambda_reg = lambda_reg
        self.omega: Dict[str, torch.Tensor] = {}
        
    def compute_regularization_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute MAS regularization loss."""
        if not self.omega:
            return torch.tensor(0.0, device=next(model.parameters()).device)
            
        mas_loss = 0.0
        for name, param in model.named_parameters():
            if name in self.omega:
                mas_loss += (self.omega[name] * param ** 2).sum()
                
        return self.lambda_reg * mas_loss
    
    def update_after_task(self, model: nn.Module, dataloader: torch.utils.data.DataLoader) -> None:
        """Update importance weights using MAS."""
        omega = {name: torch.zeros_like(param) 
                for name, param in model.named_parameters()}
        
        model.eval()
        for data, target in dataloader:
            model.zero_grad()
            output = model(data)
            # Use L2 norm of output as importance signal
            importance = torch.norm(output, p=2, dim=1).mean()
            importance.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    omega[name] += torch.abs(param.grad) / len(dataloader)
                    
        # Accumulate importance weights
        if self.omega:
            for name in omega:
                self.omega[name] += omega[name]
        else:
            self.omega = omega


class ReplayBuffer:
    """Simple replay buffer for experience replay in continual learning."""
    
    def __init__(self, capacity: int = 1000) -> None:
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum number of samples to store
        """
        self.capacity = capacity
        self.buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        
    def add(self, data: torch.Tensor, target: torch.Tensor) -> None:
        """Add samples to buffer."""
        for i in range(data.size(0)):
            if len(self.buffer) >= self.capacity:
                self.buffer.pop(0)
            self.buffer.append((data[i], target[i]))
            
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample batch from buffer."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch_data = torch.stack([self.buffer[i][0] for i in indices])
        batch_target = torch.stack([self.buffer[i][1] for i in indices])
        
        return batch_data, batch_target
    
    def __len__(self) -> int:
        return len(self.buffer)
