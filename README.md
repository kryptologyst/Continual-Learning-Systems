# Continual Learning Systems

A comprehensive framework for continual learning and lifelong learning systems, implementing state-of-the-art methods to prevent catastrophic forgetting.

**Author:** [kryptologyst](https://github.com/kryptologyst)  
**GitHub:** https://github.com/kryptologyst

## Safety Disclaimer

⚠️ **This is a research/educational framework. The models and results are NOT intended for production use or decision-making in critical applications.** Continual learning systems may exhibit unexpected behaviors and should be thoroughly validated before deployment.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Methods Implemented](#methods-implemented)
- [Usage](#usage)
- [Demo](#demo)
- [Configuration](#configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

Continual learning (also known as lifelong learning) is the ability of a machine learning model to learn from new data while retaining previously learned knowledge. This framework provides:

- **Multiple continual learning methods** (EWC, L2, MAS, Experience Replay)
- **Comprehensive evaluation metrics** (accuracy, backward/forward transfer, forgetting analysis)
- **Interactive demo** with Streamlit
- **Reproducible experiments** with proper configuration management
- **Modern PyTorch implementation** with type hints and documentation

## Features

### Core Capabilities
- **Multiple CL Methods**: EWC, L2 Regularization, Memory Aware Synapses, Experience Replay
- **Comprehensive Metrics**: Average accuracy, backward transfer, forward transfer, forgetting analysis
- **Interactive Demo**: Streamlit-based visualization and experimentation
- **Configurable**: YAML-based configuration for easy experimentation
- **Reproducible**: Deterministic seeding and proper logging
- **Visualization**: Learning curves, forgetting matrices, and performance analysis

### Technical Features
- **Modern Python**: Python 3.10+ with type hints
- **PyTorch 2.x**: Latest PyTorch with device fallback (CUDA → MPS → CPU)
- **Clean Architecture**: Modular design with proper separation of concerns
- **Testing**: Comprehensive test suite with pytest
- **Documentation**: Google-style docstrings and comprehensive README

## Installation

### Prerequisites
- Python 3.10 or higher
- PyTorch 2.0 or higher

### Install from Source

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Continual-Learning-Systems.git
cd Continual-Learning-Systems

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Install with Optional Dependencies

```bash
# For development
pip install -e ".[dev]"

# For demo functionality
pip install -e ".[demo]"

# For experiment tracking
pip install -e ".[tracking]"
```

## Quick Start

### Basic Usage

```python
from src.models import SimpleMLP
from src.methods import ElasticWeightConsolidation
from src.data import create_digits_tasks
from src.train import ContinualLearningTrainer

# Create data loaders
task_loaders = create_digits_tasks(num_tasks=2)

# Create model and trainer
model = SimpleMLP()
trainer = ContinualLearningTrainer(model)

# Create continual learning method
ewc = ElasticWeightConsolidation(importance_factor=1000.0)

# Run experiment
results = trainer.run_continual_learning_experiment(
    task_loaders=task_loaders,
    cl_method=ewc,
    epochs_per_task=10
)

# Print results
trainer.metrics.print_summary()
```

### Command Line Interface

```bash
# Run with default configuration
python src/main.py

# Run with custom configuration
python src/main.py --config configs/mas_experiment.yaml

# Run with custom parameters
python src/main.py --experiment_name "my_experiment" --output_dir "my_results"
```

## Methods Implemented

### 1. Elastic Weight Consolidation (EWC)
- **Paper**: [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796)
- **Key Idea**: Penalizes changes to important weights from previous tasks
- **Parameters**: `importance_factor` (default: 1000.0)

### 2. L2 Regularization
- **Key Idea**: Keeps parameters close to their initial values
- **Parameters**: `lambda_reg` (default: 0.01)

### 3. Memory Aware Synapses (MAS)
- **Paper**: [Memory Aware Synapses: Learning what (not) to forget](https://arxiv.org/abs/1711.09601)
- **Key Idea**: Estimates parameter importance using output sensitivity
- **Parameters**: `lambda_reg` (default: 0.01)

### 4. Experience Replay
- **Key Idea**: Stores and replays samples from previous tasks
- **Parameters**: `replay_capacity` (default: 1000)

### 5. Baseline (None)
- **Key Idea**: No continual learning technique (shows catastrophic forgetting)

## Usage

### Configuration

Create a YAML configuration file:

```yaml
model:
  input_dim: 64
  hidden_dims: [128, 64]
  output_dim: 10
  dropout: 0.1

training:
  learning_rate: 0.001
  weight_decay: 1e-4
  epochs_per_task: 10
  batch_size: 32
  seed: 42

data:
  dataset: "digits"  # or "synthetic"
  num_tasks: 2
  test_size: 0.2

cl_method:
  method: "ewc"
  importance_factor: 1000.0
```

### Running Experiments

```bash
# Run with specific configuration
python src/main.py --config configs/default.yaml

# Run multiple experiments
python src/main.py --config configs/ewc_experiment.yaml
python src/main.py --config configs/mas_experiment.yaml
python src/main.py --config configs/replay_experiment.yaml
```

## Demo

Launch the interactive Streamlit demo:

```bash
streamlit run demo/app.py
```

The demo provides:
- **Interactive configuration** of experiments
- **Real-time visualization** of results
- **Method comparison** and analysis
- **Downloadable results** and plots

## Configuration

### Model Configuration
- `input_dim`: Input feature dimension
- `hidden_dims`: List of hidden layer dimensions
- `output_dim`: Output dimension (number of classes)
- `dropout`: Dropout rate

### Training Configuration
- `learning_rate`: Learning rate for optimizer
- `weight_decay`: Weight decay for regularization
- `epochs_per_task`: Number of training epochs per task
- `batch_size`: Batch size for training
- `seed`: Random seed for reproducibility

### Data Configuration
- `dataset`: Dataset type ("digits" or "synthetic")
- `num_tasks`: Number of sequential tasks
- `test_size`: Fraction of data for testing
- `samples_per_task`: Number of samples per task (synthetic only)
- `features`: Number of input features (synthetic only)
- `classes_per_task`: Number of classes per task (synthetic only)

### CL Method Configuration
- `method`: Continual learning method ("ewc", "l2", "mas", "replay", "none")
- `importance_factor`: EWC importance factor
- `lambda_reg`: Regularization strength for L2/MAS
- `replay_capacity`: Replay buffer capacity

## Evaluation Metrics

### Core Metrics
- **Average Accuracy**: Final accuracy averaged across all tasks
- **Backward Transfer**: Knowledge retention (negative forgetting)
- **Forward Transfer**: Knowledge transfer to new tasks

### Detailed Analysis
- **Learning Curves**: Task accuracies over time
- **Forgetting Matrix**: Forgetting between task pairs
- **Final Accuracies**: Per-task final performance
- **Training Losses**: Loss evolution during training

### Interpretation
- **High Average Accuracy**: Good overall performance
- **Positive Backward Transfer**: Effective knowledge retention
- **Positive Forward Transfer**: Effective knowledge transfer
- **Low Forgetting**: Minimal catastrophic forgetting

## Project Structure

```
continual-learning-systems/
├── src/                    # Source code
│   ├── __init__.py
│   ├── models.py          # Neural network models
│   ├── methods.py         # Continual learning methods
│   ├── data.py            # Data loading utilities
│   ├── train.py           # Training utilities
│   ├── metrics.py         # Evaluation metrics
│   ├── config.py          # Configuration management
│   └── main.py            # Main experiment script
├── configs/               # Configuration files
│   ├── default.yaml
│   ├── mas_experiment.yaml
│   └── replay_experiment.yaml
├── demo/                  # Streamlit demo
│   └── app.py
├── tests/                 # Test suite
├── results/               # Experiment results
├── requirements.txt       # Dependencies
├── pyproject.toml        # Project configuration
└── README.md             # This file
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_models.py
```

## Development

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install
```

## Expected Results

### Digits Dataset (2 Tasks)
- **Baseline**: ~60-70% average accuracy, significant forgetting
- **EWC**: ~75-85% average accuracy, reduced forgetting
- **MAS**: ~70-80% average accuracy, moderate forgetting
- **Replay**: ~80-90% average accuracy, minimal forgetting

### Synthetic Dataset (4 Tasks)
- **Baseline**: ~40-50% average accuracy, severe forgetting
- **EWC**: ~60-70% average accuracy, moderate forgetting
- **MAS**: ~55-65% average accuracy, moderate forgetting
- **Replay**: ~70-80% average accuracy, minimal forgetting

*Note: Results may vary based on hyperparameters and random seeds.*

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Elastic Weight Consolidation**: Kirkpatrick et al. (2017)
- **Memory Aware Synapses**: Aljundi et al. (2018)
- **PyTorch Community**: For the excellent deep learning framework
- **Streamlit**: For the interactive demo framework

## References

1. Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. PNAS.
2. Aljundi, R., et al. (2018). Memory aware synapses: Learning what (not) to forget. ECCV.
3. Parisi, G. I., et al. (2019). Continual learning in neural networks. Neural Networks.

---

**⚠️ Disclaimer**: This framework is for research and educational purposes only. Do not use in production systems without thorough validation and safety considerations.
# Continual-Learning-Systems
