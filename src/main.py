"""Main experiment script for continual learning."""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import SimpleMLP
from methods import ElasticWeightConsolidation, L2Regularization, MemoryAwareSynapses
from data import create_digits_tasks, create_synthetic_tasks, get_device
from train import ContinualLearningTrainer, set_seed
from config import ExperimentConfig
from metrics import ContinualLearningMetrics


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('experiment.log')
        ]
    )


def create_cl_method(config: ExperimentConfig):
    """Create continual learning method based on configuration."""
    method_name = config.cl_method.method.lower()
    
    if method_name == "ewc":
        return ElasticWeightConsolidation(
            importance_factor=config.cl_method.importance_factor
        )
    elif method_name == "l2":
        return L2Regularization(
            lambda_reg=config.cl_method.lambda_reg
        )
    elif method_name == "mas":
        return MemoryAwareSynapses(
            lambda_reg=config.cl_method.lambda_reg
        )
    elif method_name == "replay":
        return None  # Replay is handled differently
    else:
        return None


def create_data_loaders(config: ExperimentConfig) -> Dict[str, Any]:
    """Create data loaders based on configuration."""
    if config.data.dataset == "digits":
        return create_digits_tasks(
            num_tasks=config.data.num_tasks,
            test_size=config.data.test_size,
            random_state=config.training.seed
        )
    elif config.data.dataset == "synthetic":
        return create_synthetic_tasks(
            num_tasks=config.data.num_tasks,
            samples_per_task=config.data.samples_per_task,
            features=config.data.features,
            classes_per_task=config.data.classes_per_task,
            test_size=config.data.test_size,
            random_state=config.training.seed
        )
    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset}")


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run continual learning experiment."""
    # Setup
    set_seed(config.training.seed)
    device = get_device()
    logging.info(f"Using device: {device}")
    
    # Create data loaders
    task_loaders = create_data_loaders(config)
    logging.info(f"Created {len(task_loaders['train'])} tasks")
    
    # Create model
    model = SimpleMLP(
        input_dim=config.model.input_dim,
        hidden_dims=config.model.hidden_dims,
        output_dim=config.model.output_dim,
        dropout=config.model.dropout
    )
    
    # Create trainer
    trainer = ContinualLearningTrainer(
        model=model,
        device=device,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Create continual learning method
    cl_method = create_cl_method(config)
    use_replay = config.cl_method.method.lower() == "replay"
    
    logging.info(f"Using CL method: {config.cl_method.method}")
    
    # Run experiment
    results = trainer.run_continual_learning_experiment(
        task_loaders=task_loaders,
        cl_method=cl_method,
        use_replay=use_replay,
        epochs_per_task=config.training.epochs_per_task
    )
    
    # Plot results
    if config.save_plots:
        os.makedirs(config.output_dir, exist_ok=True)
        plot_path = os.path.join(config.output_dir, f"{config.experiment_name}_results.png")
        trainer.metrics.plot_results(save_path=plot_path)
        logging.info(f"Results plot saved to: {plot_path}")
    
    # Print summary
    trainer.metrics.print_summary()
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Continual Learning Experiment")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                      help="Path to configuration file")
    parser.add_argument("--experiment_name", type=str, default=None,
                      help="Name for this experiment")
    parser.add_argument("--output_dir", type=str, default="results",
                      help="Output directory for results")
    parser.add_argument("--log_level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    if os.path.exists(args.config):
        config = ExperimentConfig.from_yaml(args.config)
        logging.info(f"Loaded configuration from: {args.config}")
    else:
        config = ExperimentConfig()
        logging.info("Using default configuration")
    
    # Override with command line arguments
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config.output_dir, f"{config.experiment_name}_config.yaml")
    config.to_yaml(config_path)
    
    # Run experiment
    logging.info(f"Starting experiment: {config.experiment_name}")
    results = run_experiment(config)
    
    # Save results
    if config.save_results:
        results_path = os.path.join(config.output_dir, f"{config.experiment_name}_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logging.info(f"Results saved to: {results_path}")
    
    logging.info("Experiment completed successfully!")


if __name__ == "__main__":
    main()
