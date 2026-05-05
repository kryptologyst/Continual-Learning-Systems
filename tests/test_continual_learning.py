"""Test suite for continual learning systems."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

# Add src to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import SimpleMLP, ContinualLearningModel
from methods import ElasticWeightConsolidation, L2Regularization, MemoryAwareSynapses, ReplayBuffer
from data import create_digits_tasks, create_synthetic_tasks, get_device
from train import ContinualLearningTrainer, set_seed
from config import ExperimentConfig
from metrics import ContinualLearningMetrics


class TestSimpleMLP:
    """Test SimpleMLP model."""
    
    def test_model_creation(self):
        """Test model creation with default parameters."""
        model = SimpleMLP()
        assert isinstance(model, SimpleMLP)
        assert model.input_dim == 64
        assert model.output_dim == 10
    
    def test_model_forward(self):
        """Test model forward pass."""
        model = SimpleMLP(input_dim=32, output_dim=5)
        x = torch.randn(10, 32)
        output = model(x)
        assert output.shape == (10, 5)
    
    def test_model_custom_architecture(self):
        """Test model with custom architecture."""
        model = SimpleMLP(
            input_dim=128,
            hidden_dims=[256, 128, 64],
            output_dim=20,
            dropout=0.2
        )
        x = torch.randn(5, 128)
        output = model(x)
        assert output.shape == (5, 20)


class TestContinualLearningMethods:
    """Test continual learning methods."""
    
    def test_ewc_creation(self):
        """Test EWC method creation."""
        ewc = ElasticWeightConsolidation(importance_factor=500.0)
        assert ewc.importance_factor == 500.0
        assert len(ewc.saved_params) == 0
        assert len(ewc.fisher_information) == 0
    
    def test_l2_creation(self):
        """Test L2 regularization creation."""
        l2 = L2Regularization(lambda_reg=0.1)
        assert l2.lambda_reg == 0.1
        assert len(l2.initial_params) == 0
    
    def test_mas_creation(self):
        """Test MAS method creation."""
        mas = MemoryAwareSynapses(lambda_reg=2.0)
        assert mas.lambda_reg == 2.0
        assert len(mas.omega) == 0
    
    def test_replay_buffer(self):
        """Test replay buffer functionality."""
        buffer = ReplayBuffer(capacity=100)
        assert buffer.capacity == 100
        assert len(buffer) == 0
        
        # Add samples
        data = torch.randn(10, 5)
        targets = torch.randint(0, 3, (10,))
        buffer.add(data, targets)
        assert len(buffer) == 10
        
        # Sample from buffer
        batch_data, batch_targets = buffer.sample(5)
        assert batch_data.shape[0] == 5
        assert batch_targets.shape[0] == 5


class TestDataLoaders:
    """Test data loading utilities."""
    
    def test_create_digits_tasks(self):
        """Test digits dataset task creation."""
        task_loaders = create_digits_tasks(num_tasks=2, test_size=0.2)
        
        assert 'train' in task_loaders
        assert 'test' in task_loaders
        assert len(task_loaders['train']) == 2
        assert len(task_loaders['test']) == 2
        
        # Test first task
        train_loader = task_loaders['train'][0]
        test_loader = task_loaders['test'][0]
        
        # Check batch structure
        for batch_data, batch_targets, batch_task_ids in train_loader:
            assert batch_data.shape[1] == 64  # Digits features
            assert batch_targets.shape[0] == batch_data.shape[0]
            assert batch_task_ids.shape[0] == batch_data.shape[0]
            break
    
    def test_create_synthetic_tasks(self):
        """Test synthetic dataset task creation."""
        task_loaders = create_synthetic_tasks(
            num_tasks=3,
            samples_per_task=100,
            features=32,
            classes_per_task=4
        )
        
        assert len(task_loaders['train']) == 3
        assert len(task_loaders['test']) == 3
        
        # Test first task
        train_loader = task_loaders['train'][0]
        for batch_data, batch_targets, batch_task_ids in train_loader:
            assert batch_data.shape[1] == 32
            assert batch_targets.shape[0] == batch_data.shape[0]
            break


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_metrics_creation(self):
        """Test metrics tracker creation."""
        metrics = ContinualLearningMetrics(num_tasks=3)
        assert metrics.num_tasks == 3
        assert len(metrics.task_accuracies) == 3
        assert len(metrics.forgetting_scores) == 3
    
    def test_metrics_update(self):
        """Test metrics update functionality."""
        metrics = ContinualLearningMetrics(num_tasks=2)
        
        # Update first task
        metrics.update(0, [0.8, 0.0])  # Task 0 accuracy, Task 1 not learned yet
        assert len(metrics.task_accuracies[0]) == 1
        assert metrics.task_accuracies[0][0] == [0.8, 0.0]
        
        # Update second task
        metrics.update(1, [0.7, 0.9])  # Task 0 forgotten, Task 1 learned
        assert len(metrics.task_accuracies[1]) == 1
        assert metrics.task_accuracies[1][0] == [0.7, 0.9]
        
        # Check forgetting score
        assert len(metrics.forgetting_scores[0]) == 1
        assert metrics.forgetting_scores[0][0] == 0.8 - 0.7  # 0.1 forgetting
    
    def test_average_accuracy(self):
        """Test average accuracy calculation."""
        metrics = ContinualLearningMetrics(num_tasks=2)
        metrics.update(0, [0.8, 0.0])
        metrics.update(1, [0.7, 0.9])
        
        avg_acc = metrics.get_average_accuracy()
        expected = (0.7 + 0.9) / 2  # Final accuracies
        assert abs(avg_acc - expected) < 1e-6


class TestConfiguration:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = ExperimentConfig()
        assert config.model.input_dim == 64
        assert config.training.learning_rate == 0.001
        assert config.data.dataset == "digits"
        assert config.cl_method.method == "ewc"
    
    def test_config_serialization(self):
        """Test configuration YAML serialization."""
        config = ExperimentConfig()
        config.experiment_name = "test_experiment"
        
        # Save to YAML
        config.to_yaml("test_config.yaml")
        
        # Load from YAML
        loaded_config = ExperimentConfig.from_yaml("test_config.yaml")
        
        assert loaded_config.experiment_name == "test_experiment"
        assert loaded_config.model.input_dim == config.model.input_dim
        assert loaded_config.training.learning_rate == config.training.learning_rate
        
        # Clean up
        import os
        os.remove("test_config.yaml")


class TestTrainer:
    """Test continual learning trainer."""
    
    def test_trainer_creation(self):
        """Test trainer creation."""
        model = SimpleMLP()
        trainer = ContinualLearningTrainer(model)
        
        assert trainer.model == model
        assert isinstance(trainer.optimizer, torch.optim.Adam)
        assert isinstance(trainer.criterion, torch.nn.CrossEntropyLoss)
    
    def test_set_seed(self):
        """Test seed setting for reproducibility."""
        set_seed(42)
        # This is hard to test directly, but we can ensure it doesn't raise errors
        assert True


@pytest.fixture
def sample_task_loaders():
    """Fixture providing sample task loaders for testing."""
    return create_digits_tasks(num_tasks=2, test_size=0.2)


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_experiment(self, sample_task_loaders):
        """Test end-to-end experiment."""
        model = SimpleMLP()
        trainer = ContinualLearningTrainer(model)
        ewc = ElasticWeightConsolidation(importance_factor=100.0)
        
        # Run a short experiment
        results = trainer.run_continual_learning_experiment(
            task_loaders=sample_task_loaders,
            cl_method=ewc,
            epochs_per_task=2
        )
        
        assert 'task_results' in results
        assert 'final_accuracies' in results
        assert 'summary_metrics' in results
        assert len(results['task_results']) == 2
        assert len(results['final_accuracies']) == 2
        
        # Check that metrics were updated
        assert trainer.metrics.num_tasks == 2
        assert len(trainer.metrics.task_accuracies[0]) > 0
        assert len(trainer.metrics.task_accuracies[1]) > 0


if __name__ == "__main__":
    pytest.main([__file__])
