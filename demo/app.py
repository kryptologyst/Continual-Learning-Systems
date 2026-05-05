"""Streamlit demo for continual learning systems."""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import SimpleMLP
from methods import ElasticWeightConsolidation, L2Regularization, MemoryAwareSynapses, ReplayBuffer
from data import create_digits_tasks, create_synthetic_tasks, get_device
from train import ContinualLearningTrainer, set_seed
from config import ExperimentConfig
from metrics import ContinualLearningMetrics


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Continual Learning Systems Demo",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Continual Learning Systems Demo")
    st.markdown("**Author:** [kryptologyst](https://github.com/kryptologyst)")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Dataset selection
    dataset = st.sidebar.selectbox(
        "Dataset",
        ["digits", "synthetic"],
        help="Choose the dataset for continual learning"
    )
    
    # Number of tasks
    num_tasks = st.sidebar.slider(
        "Number of Tasks",
        min_value=2,
        max_value=5,
        value=3,
        help="Number of sequential tasks to learn"
    )
    
    # Continual learning method
    cl_method = st.sidebar.selectbox(
        "Continual Learning Method",
        ["none", "ewc", "l2", "mas", "replay"],
        help="Method to prevent catastrophic forgetting"
    )
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    epochs_per_task = st.sidebar.slider(
        "Epochs per Task",
        min_value=5,
        max_value=50,
        value=10,
        help="Number of training epochs per task"
    )
    
    learning_rate = st.sidebar.slider(
        "Learning Rate",
        min_value=0.0001,
        max_value=0.01,
        value=0.001,
        step=0.0001,
        format="%.4f",
        help="Learning rate for optimization"
    )
    
    # CL method parameters
    if cl_method == "ewc":
        importance_factor = st.sidebar.slider(
            "EWC Importance Factor",
            min_value=100.0,
            max_value=10000.0,
            value=1000.0,
            step=100.0,
            help="Weight for EWC regularization"
        )
    elif cl_method in ["l2", "mas"]:
        lambda_reg = st.sidebar.slider(
            f"{cl_method.upper()} Lambda",
            min_value=0.001,
            max_value=1.0,
            value=0.01,
            step=0.001,
            format="%.3f",
            help=f"Regularization strength for {cl_method.upper()}"
        )
    elif cl_method == "replay":
        replay_capacity = st.sidebar.slider(
            "Replay Buffer Capacity",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="Maximum number of samples in replay buffer"
        )
    
    # Run experiment button
    if st.sidebar.button("🚀 Run Experiment", type="primary"):
        run_experiment(dataset, num_tasks, cl_method, epochs_per_task, 
                      learning_rate, locals())
    
    # Main content area
    st.markdown("""
    ## About Continual Learning
    
    Continual learning (also known as lifelong learning) is the ability of a machine learning model 
    to learn from new data while retaining previously learned knowledge. This is crucial for AI systems 
    that need to adapt to new tasks without forgetting old ones.
    
    ### Methods Implemented:
    - **None**: Baseline without any continual learning technique
    - **EWC**: Elastic Weight Consolidation - penalizes changes to important weights
    - **L2**: L2 Regularization - keeps parameters close to initial values
    - **MAS**: Memory Aware Synapses - estimates parameter importance
    - **Replay**: Experience Replay - stores and replays old samples
    """)
    
    # Safety disclaimer
    st.warning("""
    ⚠️ **Safety Disclaimer**: This is a research/educational demo. The models and results are not 
    intended for production use or decision-making in critical applications. Continual learning 
    systems may exhibit unexpected behaviors and should be thoroughly validated before deployment.
    """)


def run_experiment(dataset: str, num_tasks: int, cl_method: str, 
                  epochs_per_task: int, learning_rate: float, params: dict):
    """Run continual learning experiment."""
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Setup
        status_text.text("Setting up experiment...")
        set_seed(42)
        device = get_device()
        
        # Create data loaders
        status_text.text("Creating data loaders...")
        progress_bar.progress(10)
        
        if dataset == "digits":
            task_loaders = create_digits_tasks(
                num_tasks=num_tasks,
                test_size=0.2,
                random_state=42
            )
        else:
            task_loaders = create_synthetic_tasks(
                num_tasks=num_tasks,
                samples_per_task=1000,
                features=64,
                classes_per_task=10 // num_tasks,
                test_size=0.2,
                random_state=42
            )
        
        # Create model
        status_text.text("Creating model...")
        progress_bar.progress(20)
        
        model = SimpleMLP(
            input_dim=64,
            hidden_dims=[128, 64],
            output_dim=10,
            dropout=0.1
        )
        
        # Create trainer
        trainer = ContinualLearningTrainer(
            model=model,
            device=device,
            learning_rate=learning_rate,
            weight_decay=1e-4
        )
        
        # Create CL method
        status_text.text("Setting up continual learning method...")
        progress_bar.progress(30)
        
        cl_method_obj = None
        use_replay = False
        
        if cl_method == "ewc":
            cl_method_obj = ElasticWeightConsolidation(
                importance_factor=params.get("importance_factor", 1000.0)
            )
        elif cl_method == "l2":
            cl_method_obj = L2Regularization(
                lambda_reg=params.get("lambda_reg", 0.01)
            )
        elif cl_method == "mas":
            cl_method_obj = MemoryAwareSynapses(
                lambda_reg=params.get("lambda_reg", 0.01)
            )
        elif cl_method == "replay":
            use_replay = True
            trainer.replay_buffer = ReplayBuffer(
                capacity=params.get("replay_capacity", 1000)
            )
        
        # Run experiment
        status_text.text("Running continual learning experiment...")
        progress_bar.progress(40)
        
        results = trainer.run_continual_learning_experiment(
            task_loaders=task_loaders,
            cl_method=cl_method_obj,
            use_replay=use_replay,
            epochs_per_task=epochs_per_task
        )
        
        progress_bar.progress(100)
        status_text.text("Experiment completed!")
        
        # Display results
        display_results(results, trainer.metrics, cl_method)
        
    except Exception as e:
        st.error(f"Error running experiment: {str(e)}")
        st.exception(e)


def display_results(results: dict, metrics: ContinualLearningMetrics, cl_method: str):
    """Display experiment results."""
    
    st.header("📊 Experiment Results")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Average Accuracy",
            f"{metrics.get_average_accuracy():.3f}",
            help="Final accuracy averaged across all tasks"
        )
    
    with col2:
        st.metric(
            "Backward Transfer",
            f"{metrics.get_backward_transfer():.3f}",
            help="Positive values indicate knowledge retention"
        )
    
    with col3:
        st.metric(
            "Forward Transfer",
            f"{metrics.get_forward_transfer():.3f}",
            help="Positive values indicate knowledge transfer to new tasks"
        )
    
    # Learning curves
    st.subheader("📈 Learning Curves")
    
    # Create learning curves plot
    fig = go.Figure()
    
    for task_id in range(len(metrics.task_accuracies)):
        if metrics.task_accuracies[task_id]:
            fig.add_trace(go.Scatter(
                x=list(range(len(metrics.task_accuracies[task_id]))),
                y=metrics.task_accuracies[task_id],
                mode='lines+markers',
                name=f'Task {task_id}',
                line=dict(width=3)
            ))
    
    fig.update_layout(
        title="Task Accuracies Over Time",
        xaxis_title="Task",
        yaxis_title="Accuracy",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Final accuracies bar chart
    st.subheader("🎯 Final Task Accuracies")
    
    final_accs = [metrics.task_accuracies[i][-1] if metrics.task_accuracies[i] else 0 
                 for i in range(len(metrics.task_accuracies))]
    
    fig_bar = px.bar(
        x=[f"Task {i}" for i in range(len(final_accs))],
        y=final_accs,
        title="Final Accuracy per Task",
        labels={'x': 'Task', 'y': 'Accuracy'},
        color=final_accs,
        color_continuous_scale='Viridis'
    )
    
    fig_bar.update_layout(height=400)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Forgetting matrix
    st.subheader("🔄 Forgetting Analysis")
    
    forgetting_matrix = np.zeros((len(metrics.task_accuracies), len(metrics.task_accuracies)))
    for i in range(len(metrics.task_accuracies)):
        for j in range(i):
            if j < len(metrics.forgetting_scores) and metrics.forgetting_scores[j]:
                forgetting_matrix[i, j] = metrics.forgetting_scores[j][-1]
    
    fig_heatmap = px.imshow(
        forgetting_matrix,
        labels=dict(x="Previous Task", y="Current Task", color="Forgetting"),
        x=[f"Task {i}" for i in range(len(metrics.task_accuracies))],
        y=[f"Task {i}" for i in range(len(metrics.task_accuracies))],
        color_continuous_scale='RdBu_r',
        aspect="auto"
    )
    
    fig_heatmap.update_layout(
        title="Forgetting Matrix (Red = More Forgetting)",
        height=400
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Method comparison
    st.subheader("🔬 Method Analysis")
    
    analysis_text = f"""
    **Method Used:** {cl_method.upper()}
    
    **Key Insights:**
    - Average accuracy across all tasks: {metrics.get_average_accuracy():.3f}
    - Backward transfer (knowledge retention): {metrics.get_backward_transfer():.3f}
    - Forward transfer (knowledge transfer): {metrics.get_forward_transfer():.3f}
    
    **Interpretation:**
    """
    
    if cl_method == "none":
        analysis_text += """
    - This baseline shows catastrophic forgetting without any mitigation
    - Low backward transfer indicates significant forgetting of previous tasks
    """
    elif cl_method == "ewc":
        analysis_text += """
    - EWC should show better backward transfer compared to baseline
    - The importance factor controls how much previous knowledge is preserved
    """
    elif cl_method == "replay":
        analysis_text += """
    - Experience replay maintains a buffer of old samples
    - Should show good performance on all tasks due to explicit replay
    """
    
    st.markdown(analysis_text)
    
    # Download results
    st.subheader("💾 Download Results")
    
    results_df = pd.DataFrame({
        'Task': range(len(final_accs)),
        'Final_Accuracy': final_accs
    })
    
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results CSV",
        data=csv,
        file_name=f"continual_learning_results_{cl_method}.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
