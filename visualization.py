# visualization.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from typing import List, Dict, Tuple, Optional
import os


class TaskEmbeddingVisualizer:
    """Visualizer for task embeddings and drift detection results."""
    
    def __init__(self, save_dir: str = "visualizations"):
        """
        Initialize task embedding visualizer.
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_tsne_embeddings(self, embeddings: List[torch.Tensor], 
                           task_ids: List[int], save_name: str = "tsne_embeddings.png"):
        """
        Create t-SNE visualization of task embeddings.
        
        Args:
            embeddings: List of task embedding vectors
            task_ids: Corresponding task IDs
            save_name: Output filename
        """
        if len(embeddings) < 2:
            print("Need at least 2 embeddings for t-SNE visualization")
        return

        # Convert to numpy array
        embedding_array = torch.stack(embeddings).cpu().numpy()
        
        # Apply t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        embeddings_2d = tsne.fit_transform(embedding_array)
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        # Color by task ID
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=task_ids, cmap='viridis', s=100, alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Task ID')
        
        # Add task ID labels for some points
        for i, (x, y) in enumerate(embeddings_2d):
            if i % max(1, len(embeddings) // 10) == 0:  # Label every 10th point
                plt.annotate(f'T{task_ids[i]}', (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
        
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('Task Embeddings Visualization (t-SNE)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"t-SNE visualization saved to {save_path}")
        
    def plot_drift_trend(self, drift_scores: List[float], task_ids: List[int],
                        drift_threshold: float = 0.5, save_name: str = "drift_trend.png"):
        """
        Plot drift detection trend over tasks.
        
        Args:
            drift_scores: List of drift detection scores
            task_ids: Corresponding task IDs
            drift_threshold: Threshold for drift detection
            save_name: Output filename
        """
        plt.figure(figsize=(12, 6))
        
        # Plot drift scores
        plt.plot(task_ids, drift_scores, 'b-o', linewidth=2, markersize=6, 
                label='Drift Score', alpha=0.8)
        
        # Add threshold line
        plt.axhline(y=drift_threshold, color='r', linestyle='--', 
                   label=f'Threshold ({drift_threshold})', alpha=0.7)
        
        # Highlight drift detection points
        drift_detected = [i for i, score in enumerate(drift_scores) if score > drift_threshold]
        if drift_detected:
            plt.scatter([task_ids[i] for i in drift_detected], 
                       [drift_scores[i] for i in drift_detected], 
                       color='red', s=100, zorder=5, label='Drift Detected')
        
        plt.xlabel('Task ID')
        plt.ylabel('Drift Score')
        plt.title('Drift Detection Trend')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Drift trend plot saved to {save_path}")
        
    def plot_controller_parameters(self, controller_params: List[Dict], 
                                 task_ids: List[int], save_name: str = "controller_params.png"):
        """
        Plot adaptive controller parameter dynamics.
        
        Args:
            controller_params: List of controller parameter dictionaries
            task_ids: Corresponding task IDs
            save_name: Output filename
        """
        if not controller_params:
            print("No controller parameters to plot")
        return

        # Extract parameters
        lr_scales = [params.get('lr_scale', 0) for params in controller_params]
        ewc_lambdas = [params.get('ewc_lambda', 0) for params in controller_params]
        replay_weights = [params.get('replay_weight', 0) for params in controller_params]
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Learning rate scaling
        axes[0].plot(task_ids, lr_scales, 'g-o', linewidth=2, markersize=4)
        axes[0].set_ylabel('Learning Rate Scale')
        axes[0].set_title('Adaptive Learning Rate Scaling')
        axes[0].grid(True, alpha=0.3)
        
        # EWC regularization strength
        axes[1].plot(task_ids, ewc_lambdas, 'r-s', linewidth=2, markersize=4)
        axes[1].set_ylabel('EWC λ')
        axes[1].set_title('EWC Regularization Strength')
        axes[1].grid(True, alpha=0.3)
        
        # Replay buffer weight
        axes[2].plot(task_ids, replay_weights, 'b-^', linewidth=2, markersize=4)
        axes[2].set_xlabel('Task ID')
        axes[2].set_ylabel('Replay Weight')
        axes[2].set_title('Replay Buffer Weight')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Controller parameters plot saved to {save_path}")
        
    def plot_strategy_switches(self, strategies: List[str], task_ids: List[int],
                             save_name: str = "strategy_switches.png"):
        """
        Plot strategy switching over tasks.
        
        Args:
            strategies: List of strategy names
            task_ids: Corresponding task IDs
            save_name: Output filename
        """
        # Create strategy mapping
        unique_strategies = list(set(strategies))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        strategy_colors = colors[:len(unique_strategies)]
        strategy_to_color = dict(zip(unique_strategies, strategy_colors))
        
        plt.figure(figsize=(12, 6))
        
        # Plot strategy segments
        for i in range(len(task_ids) - 1):
            strategy = strategies[i]
            color = strategy_to_color[strategy]
            plt.axvspan(task_ids[i], task_ids[i + 1], alpha=0.3, color=color, 
                       label=strategy if i == 0 or strategies[i-1] != strategy else "")
        
        # Add strategy labels
        plt.xlabel('Task ID')
        plt.ylabel('Strategy')
        plt.title('Strategy Switching Over Tasks')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Strategy switches plot saved to {save_path}")
        
    def plot_performance_comparison(self, results: Dict[str, List[float]], 
                                  task_ids: List[int], save_name: str = "performance_comparison.png"):
        """
        Plot performance comparison between different methods.
        
        Args:
            results: Dictionary mapping method names to accuracy lists
            task_ids: Task IDs for x-axis
            save_name: Output filename
        """
        plt.figure(figsize=(12, 6))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, ((method, accuracies), color) in enumerate(zip(results.items(), colors)):
            plt.plot(task_ids, accuracies, 'o-', linewidth=2, markersize=4, 
                    label=method, color=color, alpha=0.8)
        
        plt.xlabel('Task ID')
        plt.ylabel('Accuracy')
        plt.title('Performance Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance comparison plot saved to {save_path}")
        
    def plot_radar_chart(self, metrics: Dict[str, float], save_name: str = "radar_chart.png"):
        """
        Create radar chart for multiple evaluation metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            save_name: Output filename
        """
        # Define metric categories
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        # Normalize values to [0, 1] range
        max_val = max(values) if values else 1
        normalized_values = [v / max_val for v in values]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        normalized_values += normalized_values[:1]  # Close the plot
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        ax.plot(angles, normalized_values, 'o-', linewidth=2, color='blue')
        ax.fill(angles, normalized_values, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Evaluation Metrics Radar Chart')
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Radar chart saved to {save_path}")
        
    def create_comprehensive_visualization(self, embeddings: List[torch.Tensor],
                                         task_ids: List[int],
                                         drift_scores: List[float],
                                         controller_params: List[Dict],
                                         strategies: List[str],
                                         save_prefix: str = "comprehensive"):
        """
        Create comprehensive visualization combining all plots.
        
        Args:
            embeddings: Task embeddings
            task_ids: Task IDs
            drift_scores: Drift detection scores
            controller_params: Controller parameters
            strategies: Strategy names
            save_prefix: Prefix for saved files
        """
        print("Creating comprehensive visualizations...")
        
        # Create all visualizations
        self.plot_tsne_embeddings(embeddings, task_ids, f"{save_prefix}_tsne.png")
        self.plot_drift_trend(drift_scores, task_ids, save_name=f"{save_prefix}_drift.png")
        self.plot_controller_parameters(controller_params, task_ids, f"{save_prefix}_controller.png")
        self.plot_strategy_switches(strategies, task_ids, f"{save_prefix}_strategies.png")
        
        print("Comprehensive visualizations completed!")
