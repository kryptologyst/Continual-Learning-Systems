"""Training utilities for continual learning."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import random
from tqdm import tqdm
import logging

from .models import SimpleMLP, ContinualLearningModel
from .methods import ContinualLearningMethod, ReplayBuffer
from .metrics import ContinualLearningMetrics
from .data import get_device


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ContinualLearningTrainer:
    """Trainer for continual learning experiments."""
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None,
                 learning_rate: float = 0.001, weight_decay: float = 1e-4) -> None:
        """Initialize trainer.
        
        Args:
            model: Model to train
            device: Device to use for training
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        
        self.metrics = ContinualLearningMetrics(num_tasks=0)  # Will be updated
        self.replay_buffer = ReplayBuffer(capacity=1000)
        
    def train_task(self, task_id: int, train_loader: DataLoader, 
                   epochs: int = 10, cl_method: Optional[ContinualLearningMethod] = None,
                   use_replay: bool = False) -> Dict[str, float]:
        """Train model on a single task.
        
        Args:
            task_id: Task identifier
            train_loader: Training data loader
            epochs: Number of training epochs
            cl_method: Continual learning method to use
            use_replay: Whether to use experience replay
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        train_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_data, batch_targets, batch_task_ids in tqdm(train_loader, 
                                                                desc=f"Task {task_id}, Epoch {epoch+1}"):
                batch_data = batch_data.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_data)
                task_loss = self.criterion(outputs, batch_targets)
                
                # Add continual learning regularization
                total_loss = task_loss
                if cl_method:
                    reg_loss = cl_method.compute_regularization_loss(self.model)
                    total_loss += reg_loss
                
                # Add replay loss if using experience replay
                if use_replay and len(self.replay_buffer) > 0:
                    replay_data, replay_targets = self.replay_buffer.sample(32)
                    replay_data = replay_data.to(self.device)
                    replay_targets = replay_targets.to(self.device)
                    
                    replay_outputs = self.model(replay_data)
                    replay_loss = self.criterion(replay_outputs, replay_targets)
                    total_loss += 0.5 * replay_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # Store samples in replay buffer
                if use_replay:
                    self.replay_buffer.add(batch_data, batch_targets)
                
                epoch_loss += total_loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
            
            logging.info(f"Task {task_id}, Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
        # Update continual learning method after task completion
        if cl_method:
            cl_method.update_after_task(self.model, train_loader)
            
        return {
            'final_loss': train_losses[-1],
            'avg_loss': np.mean(train_losses),
            'min_loss': np.min(train_losses)
        }
    
    def evaluate_all_tasks(self, test_loaders: List[DataLoader]) -> List[float]:
        """Evaluate model on all tasks.
        
        Args:
            test_loaders: List of test data loaders for each task
            
        Returns:
            List of accuracies for each task
        """
        self.model.eval()
        task_accuracies = []
        
        with torch.no_grad():
            for task_id, test_loader in enumerate(test_loaders):
                correct = 0
                total = 0
                
                for batch_data, batch_targets, _ in test_loader:
                    batch_data = batch_data.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    outputs = self.model(batch_data)
                    _, predicted = torch.max(outputs, 1)
                    
                    total += batch_targets.size(0)
                    correct += (predicted == batch_targets).sum().item()
                
                accuracy = correct / total
                task_accuracies.append(accuracy)
                
                logging.info(f"Task {task_id} Accuracy: {accuracy:.4f}")
        
        return task_accuracies
    
    def run_continual_learning_experiment(self, task_loaders: Dict[str, List[DataLoader]],
                                        cl_method: Optional[ContinualLearningMethod] = None,
                                        use_replay: bool = False, epochs_per_task: int = 10) -> Dict[str, Any]:
        """Run complete continual learning experiment.
        
        Args:
            task_loaders: Dictionary with 'train' and 'test' keys containing lists of loaders
            cl_method: Continual learning method to use
            use_replay: Whether to use experience replay
            epochs_per_task: Number of epochs per task
            
        Returns:
            Dictionary containing experiment results
        """
        train_loaders = task_loaders['train']
        test_loaders = task_loaders['test']
        num_tasks = len(train_loaders)
        
        # Update metrics tracker
        self.metrics = ContinualLearningMetrics(num_tasks)
        
        results = {
            'task_results': [],
            'final_accuracies': [],
            'training_losses': []
        }
        
        logging.info(f"Starting continual learning experiment with {num_tasks} tasks")
        
        for task_id in range(num_tasks):
            logging.info(f"Training on Task {task_id}")
            
            # Train on current task
            train_metrics = self.train_task(
                task_id, train_loaders[task_id], 
                epochs=epochs_per_task, cl_method=cl_method, use_replay=use_replay
            )
            
            # Evaluate on all tasks
            task_accuracies = self.evaluate_all_tasks(test_loaders)
            
            # Update metrics
            self.metrics.update(task_id, task_accuracies)
            
            # Store results
            results['task_results'].append({
                'task_id': task_id,
                'train_metrics': train_metrics,
                'test_accuracies': task_accuracies
            })
            results['final_accuracies'] = task_accuracies
            results['training_losses'].append(train_metrics['final_loss'])
            
            logging.info(f"Task {task_id} completed. Accuracies: {task_accuracies}")
        
        # Calculate final metrics
        results['summary_metrics'] = {
            'average_accuracy': self.metrics.get_average_accuracy(),
            'backward_transfer': self.metrics.get_backward_transfer(),
            'forward_transfer': self.metrics.get_forward_transfer()
        }
        
        return results
