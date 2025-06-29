# experiment_utils.py - Experiment utilities for saving results and computing metrics

import json
import os
import numpy as np
from datetime import datetime
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd

class ExperimentLogger:
    """Logger for tracking experiment metrics and results."""
    
    def __init__(self, log_dir: str = "logs", experiment_name: Optional[str] = None):
        """
        Initialize experiment logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_path = os.path.join(log_dir, f"{self.experiment_name}.json")
        
        # Initialize metrics storage
        self.metrics = {
            'task_accuracies': [],
            'drift_scores': [],
            'controller_params': [],
            'losses': [],
            'timestamps': []
        }
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
    def log_metrics(self, task_id: int, accuracy: float, drift_score: float, 
                   controller_params: Dict, loss: float):
        """Log metrics for a single task."""
        self.metrics['task_accuracies'].append(accuracy)
        self.metrics['drift_scores'].append(drift_score)
        self.metrics['controller_params'].append(controller_params)
        self.metrics['losses'].append(loss)
        self.metrics['timestamps'].append(task_id)
        
    def save_logs(self):
        """Save metrics to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                serializable_metrics[key] = [v.tolist() if isinstance(v, np.ndarray) else v for v in value]
            else:
                serializable_metrics[key] = value
                
        with open(self.log_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
            
    def load_logs(self) -> Dict:
        """Load metrics from JSON file."""
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r') as f:
                return json.load(f)
        return {}
    
    def get_summary_stats(self) -> Dict:
        """Compute summary statistics from logged metrics."""
        if not self.metrics['task_accuracies']:
            return {}
            
        accuracies = np.array(self.metrics['task_accuracies'])
        drift_scores = np.array(self.metrics['drift_scores'])
        
        return {
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'final_accuracy': float(accuracies[-1]),
            'mean_drift_score': float(np.mean(drift_scores)),
            'max_drift_score': float(np.max(drift_scores)),
            'num_tasks': len(accuracies)
        }

class MetricsCalculator:
    """Calculate various evaluation metrics for continual learning."""
    
    @staticmethod
    def calculate_nta(accuracies: List[float]) -> float:
        """Calculate Normalized Task Accuracy (NTA)."""
        if not accuracies:
            return 0.0
        return float(np.mean(accuracies))
    
    @staticmethod
    def calculate_otr(accuracies: List[float], window_size: int = 10) -> float:
        """Calculate Overall Task Retention (OTR)."""
        if len(accuracies) < window_size:
            return 0.0
        
        # Average of recent accuracies
        recent_acc = np.mean(accuracies[-window_size:])
        return float(recent_acc)
    
    @staticmethod
    def calculate_fr(accuracies: List[float]) -> float:
        """Calculate Forgetting Rate (FR)."""
        if len(accuracies) < 2:
            return 0.0
        
        # Difference between peak and final accuracy
        peak_acc = np.max(accuracies)
        final_acc = accuracies[-1]
        return float(peak_acc - final_acc)
    
    @staticmethod
    def calculate_dda(drift_scores: List[float]) -> float:
        """Calculate Drift Detection Accuracy (DDA)."""
        if not drift_scores:
            return 0.0
        
        # Average drift detection score
        return float(np.mean(drift_scores))
    
    @staticmethod
    def calculate_cas(accuracies: List[float], drift_scores: List[float]) -> float:
        """Calculate Continual Adaptation Score (CAS)."""
        if not accuracies or not drift_scores:
            return 0.0
        
        # Correlation between accuracy and drift adaptation
        min_len = min(len(accuracies), len(drift_scores))
        acc_subset = accuracies[-min_len:]
        drift_subset = drift_scores[-min_len:]
        
        correlation = np.corrcoef(acc_subset, drift_subset)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    @staticmethod
    def calculate_at(accuracies: List[float]) -> float:
        """Calculate Adaptation Time (AT)."""
        if len(accuracies) < 2:
            return 0.0
        
        # Time to reach 90% of final performance
        target_acc = 0.9 * accuracies[-1]
        
        for i, acc in enumerate(accuracies):
            if acc >= target_acc:
                return float(i + 1)  # Task index (1-based)
        
        return float(len(accuracies))  # Never reached target

class VisualizationManager:
    """Manager for creating and saving experiment visualizations."""
    
    def __init__(self, save_dir: str = "visualizations"):
        """
        Initialize visualization manager.
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style for consistent plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_accuracy_trend(self, accuracies: List[float], save_name: str = "accuracy_trend.png"):
        """Plot accuracy trend over tasks."""
        plt.figure(figsize=(10, 6))
        plt.plot(accuracies, marker='o', linewidth=2, markersize=4)
        plt.xlabel('Task Index')
        plt.ylabel('Accuracy')
        plt.title('Task Accuracy Trend')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_drift_detection(self, drift_scores: List[float], save_name: str = "drift_detection.png"):
        """Plot drift detection scores over time."""
        plt.figure(figsize=(10, 6))
        plt.plot(drift_scores, marker='s', linewidth=2, markersize=4, color='red')
        plt.xlabel('Task Index')
        plt.ylabel('Drift Score')
        plt.title('Drift Detection Trend')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_controller_dynamics(self, controller_params: List[Dict], save_name: str = "controller_dynamics.png"):
        """Plot controller parameter dynamics."""
        if not controller_params:
            return
            
        # Extract parameters
        lr_scales = [params.get('lr_scale', 0) for params in controller_params]
        ewc_lambdas = [params.get('ewc_lambda', 0) for params in controller_params]
        replay_weights = [params.get('replay_weight', 0) for params in controller_params]
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Learning rate scaling
        axes[0].plot(lr_scales, marker='o', linewidth=2, markersize=4)
        axes[0].set_ylabel('LR Scale')
        axes[0].set_title('Learning Rate Scaling')
        axes[0].grid(True, alpha=0.3)
        
        # EWC lambda
        axes[1].plot(ewc_lambdas, marker='s', linewidth=2, markersize=4, color='orange')
        axes[1].set_ylabel('EWC λ')
        axes[1].set_title('EWC Regularization Strength')
        axes[1].grid(True, alpha=0.3)
        
        # Replay weight
        axes[2].plot(replay_weights, marker='^', linewidth=2, markersize=4, color='green')
        axes[2].set_xlabel('Task Index')
        axes[2].set_ylabel('Replay Weight')
        axes[2].set_title('Replay Buffer Weight')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_radar_chart(self, metrics: Dict[str, float], save_name: str = "radar_chart.png"):
        """Create radar chart for multiple metrics."""
        # Define metric categories
        categories = ['NTA', 'OTR', 'FR', 'DDA', 'CAS', 'AT']
        values = [metrics.get(cat.lower(), 0) for cat in categories]
        
        # Normalize values to [0, 1] range
        normalized_values = [min(max(v, 0), 1) for v in values]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        normalized_values += normalized_values[:1]  # Close the plot
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, normalized_values, 'o-', linewidth=2)
        ax.fill(angles, normalized_values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Metrics Radar Chart')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_ablation_study(self, results: Dict[str, List[float]], save_name: str = "ablation_study.png"):
        """Plot ablation study results."""
        methods = list(results.keys())
        accuracies = [np.mean(results[method]) for method in methods]
        stds = [np.std(results[method]) for method in methods]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, accuracies, yerr=stds, capsize=5, alpha=0.7)
        plt.xlabel('Method')
        plt.ylabel('Average Accuracy')
        plt.title('Ablation Study Results')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()

def create_experiment_report(logger: ExperimentLogger, viz_manager: VisualizationManager,
                           save_name: str = "experiment_report.txt") -> str:
    """Create comprehensive experiment report."""
    summary = logger.get_summary_stats()
    
    report = f"""
Experiment Report
================

Experiment: {logger.experiment_name}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Summary Statistics:
------------------
Mean Accuracy: {summary.get('mean_accuracy', 0):.4f}
Std Accuracy: {summary.get('std_accuracy', 0):.4f}
Final Accuracy: {summary.get('final_accuracy', 0):.4f}
Mean Drift Score: {summary.get('mean_drift_score', 0):.4f}
Max Drift Score: {summary.get('max_drift_score', 0):.4f}
Number of Tasks: {summary.get('num_tasks', 0)}

Detailed Metrics:
----------------
"""
    
    # Calculate detailed metrics
    accuracies = logger.metrics['task_accuracies']
    drift_scores = logger.metrics['drift_scores']
    
    if accuracies and drift_scores:
        nta = MetricsCalculator.calculate_nta(accuracies)
        otr = MetricsCalculator.calculate_otr(accuracies)
        fr = MetricsCalculator.calculate_fr(accuracies)
        dda = MetricsCalculator.calculate_dda(drift_scores)
        cas = MetricsCalculator.calculate_cas(accuracies, drift_scores)
        at = MetricsCalculator.calculate_at(accuracies)
        
        report += f"""
NTA (Normalized Task Accuracy): {nta:.4f}
OTR (Overall Task Retention): {otr:.4f}
FR (Forgetting Rate): {fr:.4f}
DDA (Drift Detection Accuracy): {dda:.4f}
CAS (Continual Adaptation Score): {cas:.4f}
AT (Adaptation Time): {at:.1f} tasks
"""
    
    # Save report
    report_path = os.path.join(logger.log_dir, save_name)
    with open(report_path, 'w') as f:
        f.write(report)
    
    return report 