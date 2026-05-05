"""Evaluation metrics for continual learning."""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class ContinualLearningMetrics:
    """Metrics for evaluating continual learning performance."""
    
    def __init__(self, num_tasks: int) -> None:
        """Initialize metrics tracker.
        
        Args:
            num_tasks: Number of tasks in the continual learning scenario
        """
        self.num_tasks = num_tasks
        self.reset()
        
    def reset(self) -> None:
        """Reset all metrics."""
        self.task_accuracies: List[List[float]] = [[] for _ in range(self.num_tasks)]
        self.forgetting_scores: List[List[float]] = [[] for _ in range(self.num_tasks)]
        
    def update(self, task_id: int, accuracies: List[float]) -> None:
        """Update metrics after evaluation.
        
        Args:
            task_id: Current task ID
            accuracies: List of accuracies for all tasks
        """
        self.task_accuracies[task_id] = accuracies.copy()
        
        # Calculate forgetting scores
        if task_id > 0:
            for prev_task in range(task_id):
                if len(self.task_accuracies[prev_task]) > 0:
                    prev_acc = self.task_accuracies[prev_task][prev_task]
                    curr_acc = accuracies[prev_task]
                    forgetting = prev_acc - curr_acc
                    self.forgetting_scores[prev_task].append(forgetting)
                    
    def get_average_accuracy(self) -> float:
        """Calculate average accuracy across all tasks."""
        final_accuracies = []
        for task_id in range(self.num_tasks):
            if len(self.task_accuracies[task_id]) > 0:
                final_accuracies.append(self.task_accuracies[task_id][-1])
        return np.mean(final_accuracies) if final_accuracies else 0.0
    
    def get_backward_transfer(self) -> float:
        """Calculate backward transfer (negative forgetting)."""
        if not any(self.forgetting_scores):
            return 0.0
            
        all_forgetting = []
        for task_forgetting in self.forgetting_scores:
            all_forgetting.extend(task_forgetting)
            
        return -np.mean(all_forgetting) if all_forgetting else 0.0
    
    def get_forward_transfer(self) -> float:
        """Calculate forward transfer."""
        if self.num_tasks < 2:
            return 0.0
            
        forward_transfer = 0.0
        count = 0
        
        for task_id in range(1, self.num_tasks):
            if len(self.task_accuracies[task_id]) > 0:
                # Compare with random baseline (1/num_classes)
                baseline_acc = 1.0 / 10  # Assuming 10 classes per task
                task_acc = self.task_accuracies[task_id][task_id]
                forward_transfer += task_acc - baseline_acc
                count += 1
                
        return forward_transfer / count if count > 0 else 0.0
    
    def get_learning_curve(self) -> Dict[str, List[float]]:
        """Get learning curves for visualization."""
        curves = {}
        
        for task_id in range(self.num_tasks):
            if self.task_accuracies[task_id]:
                curves[f'Task {task_id}'] = self.task_accuracies[task_id]
                
        return curves
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """Plot continual learning results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Learning curves
        ax1 = axes[0, 0]
        for task_id in range(self.num_tasks):
            if self.task_accuracies[task_id]:
                ax1.plot(self.task_accuracies[task_id], label=f'Task {task_id}', marker='o')
        ax1.set_xlabel('Task')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Learning Curves')
        ax1.legend()
        ax1.grid(True)
        
        # Final accuracies
        ax2 = axes[0, 1]
        final_accs = [self.task_accuracies[i][-1] if self.task_accuracies[i] else 0 
                     for i in range(self.num_tasks)]
        ax2.bar(range(self.num_tasks), final_accs)
        ax2.set_xlabel('Task')
        ax2.set_ylabel('Final Accuracy')
        ax2.set_title('Final Task Accuracies')
        ax2.set_ylim(0, 1)
        
        # Forgetting matrix
        ax3 = axes[1, 0]
        forgetting_matrix = np.zeros((self.num_tasks, self.num_tasks))
        for i in range(self.num_tasks):
            for j in range(i):
                if j < len(self.forgetting_scores) and self.forgetting_scores[j]:
                    forgetting_matrix[i, j] = self.forgetting_scores[j][-1]
        
        sns.heatmap(forgetting_matrix, annot=True, fmt='.3f', ax=ax3, 
                   cmap='RdBu_r', center=0)
        ax3.set_xlabel('Previous Task')
        ax3.set_ylabel('Current Task')
        ax3.set_title('Forgetting Matrix')
        
        # Summary metrics
        ax4 = axes[1, 1]
        metrics = {
            'Avg Accuracy': self.get_average_accuracy(),
            'Backward Transfer': self.get_backward_transfer(),
            'Forward Transfer': self.get_forward_transfer()
        }
        
        bars = ax4.bar(metrics.keys(), metrics.values())
        ax4.set_ylabel('Score')
        ax4.set_title('Summary Metrics')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics.values()):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def print_summary(self) -> None:
        """Print summary of metrics."""
        print("Continual Learning Results Summary:")
        print(f"Average Accuracy: {self.get_average_accuracy():.4f}")
        print(f"Backward Transfer: {self.get_backward_transfer():.4f}")
        print(f"Forward Transfer: {self.get_forward_transfer():.4f}")
        print("\nTask-wise Final Accuracies:")
        for task_id in range(self.num_tasks):
            if self.task_accuracies[task_id]:
                print(f"Task {task_id}: {self.task_accuracies[task_id][-1]:.4f}")
