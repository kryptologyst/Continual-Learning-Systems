"""Configuration management for continual learning experiments."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml
import os


@dataclass
class ModelConfig:
    """Model configuration."""
    input_dim: int = 64
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    output_dim: int = 10
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    epochs_per_task: int = 10
    batch_size: int = 32
    seed: int = 42


@dataclass
class DataConfig:
    """Data configuration."""
    dataset: str = "digits"  # "digits" or "synthetic"
    num_tasks: int = 2
    test_size: float = 0.2
    samples_per_task: int = 1000
    features: int = 64
    classes_per_task: int = 5


@dataclass
class CLMethodConfig:
    """Continual learning method configuration."""
    method: str = "ewc"  # "ewc", "l2", "mas", "replay", "none"
    importance_factor: float = 1000.0
    lambda_reg: float = 0.01
    replay_capacity: int = 1000


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    cl_method: CLMethodConfig = field(default_factory=CLMethodConfig)
    
    # Experiment metadata
    experiment_name: str = "continual_learning_experiment"
    save_results: bool = True
    save_plots: bool = True
    output_dir: str = "results"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert nested dictionaries to config objects
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        cl_method_config = CLMethodConfig(**config_dict.get('cl_method', {}))
        
        # Create main config
        config = cls(
            model=model_config,
            training=training_config,
            data=data_config,
            cl_method=cl_method_config
        )
        
        # Update with remaining fields
        for key, value in config_dict.items():
            if key not in ['model', 'training', 'data', 'cl_method']:
                setattr(config, key, value)
        
        return config
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'model': {
                'input_dim': self.model.input_dim,
                'hidden_dims': self.model.hidden_dims,
                'output_dim': self.model.output_dim,
                'dropout': self.model.dropout
            },
            'training': {
                'learning_rate': self.training.learning_rate,
                'weight_decay': self.training.weight_decay,
                'epochs_per_task': self.training.epochs_per_task,
                'batch_size': self.training.batch_size,
                'seed': self.training.seed
            },
            'data': {
                'dataset': self.data.dataset,
                'num_tasks': self.data.num_tasks,
                'test_size': self.data.test_size,
                'samples_per_task': self.data.samples_per_task,
                'features': self.data.features,
                'classes_per_task': self.data.classes_per_task
            },
            'cl_method': {
                'method': self.cl_method.method,
                'importance_factor': self.cl_method.importance_factor,
                'lambda_reg': self.cl_method.lambda_reg,
                'replay_capacity': self.cl_method.replay_capacity
            },
            'experiment_name': self.experiment_name,
            'save_results': self.save_results,
            'save_plots': self.save_plots,
            'output_dir': self.output_dir
        }
        
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
