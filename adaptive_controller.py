# adaptive_controller.py (updated with GatedHyperNetwork)

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
from hypernet import GatedHyperNetwork
import numpy as np
from typing import Dict, Tuple, Optional, Union

class AdaptiveController:
    """
    Adaptive controller that dynamically adjusts learning parameters based on drift detection.
    
    Uses a gated hypernetwork to generate task-specific learning rates and regularization
    parameters based on drift scores and task embeddings.
    """
    
    def __init__(self,
                 embedding_dim: int = 128,
                 hidden_dim: int = 64,
                 alpha: float = 0.5,
                 beta: float = 1.0,
                 gamma: float = 1.0):
        """
        Initialize the adaptive controller.
        
        Args:
            embedding_dim: Dimension of task embeddings
            hidden_dim: Hidden dimension for hypernetwork
            alpha: Base learning rate scaling factor
            beta: EWC regularization scaling factor  
            gamma: Replay buffer scaling factor
        """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Initialize hypernetwork for parameter generation
        self.hypernetwork = GatedHyperNetwork(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim
        )
        
        # Store historical drift scores for trend analysis
        self.drift_history = []
        self.max_history = 20
        
    def compute_adaptive_parameters(self, 
                                  task_embedding: torch.Tensor,
                                  drift_score: float,
                                  current_task: int) -> Dict[str, Union[float, str]]:
        """
        Compute adaptive learning parameters based on drift detection.
        
        Args:
            task_embedding: Current task embedding vector
            drift_score: Detected drift magnitude
            current_task: Current task index
            
        Returns:
            Dictionary containing adaptive parameters
        """
        # Update drift history
        self.drift_history.append(drift_score)
        if len(self.drift_history) > self.max_history:
            self.drift_history.pop(0)
        
        # Normalize drift score to [0, 1] range
        normalized_drift = self._normalize_drift_score(drift_score)
        
        # Compute task complexity (variance of embedding) - aligned with paper formula
        task_complexity = task_embedding.var().item() / task_embedding.numel()  # Normalize by dimension
        
        # Generate adaptive parameters using hypernetwork
        with torch.no_grad():
            eta, lambda_val = self.hypernetwork.get_hyperparams(
                task_embedding.unsqueeze(0),  # Add batch dimension
                torch.tensor([drift_score], dtype=torch.float32),
                torch.tensor([task_complexity], dtype=torch.float32)
            )
        
        # Extract and scale parameters
        lr_scale = eta.item() * self.alpha
        ewc_lambda = lambda_val.item() * self.beta
        replay_weight = self.gamma * (1.0 - normalized_drift)  # Inverse relationship
        
        # Apply drift-based adjustments
        if normalized_drift > 0.7:  # High drift detected
            lr_scale *= 1.5  # Increase learning rate
            ewc_lambda *= 2.0  # Strengthen regularization
            replay_weight *= 0.5  # Reduce replay influence
        elif normalized_drift < 0.3:  # Low drift
            lr_scale *= 0.8  # Decrease learning rate
            ewc_lambda *= 0.5  # Reduce regularization
            replay_weight *= 1.2  # Increase replay influence
        
        return {
            'lr_scale': lr_scale,
            'ewc_lambda': ewc_lambda,
            'replay_weight': replay_weight,
            'drift_trend': self._compute_drift_trend(),
            'strategy': self._select_strategy(normalized_drift)
        }
    
    def _normalize_drift_score(self, drift_score: float) -> float:
        """Normalize drift score to [0, 1] range using historical statistics."""
        if len(self.drift_history) < 3:
            return min(drift_score / 10.0, 1.0)  # Simple normalization for early tasks
        
        # Use percentile-based normalization
        sorted_scores = sorted(self.drift_history)
        percentile = np.searchsorted(sorted_scores, drift_score) / len(sorted_scores)
        return float(min(percentile, 1.0))
    
    def _compute_drift_trend(self) -> float:
        """Compute drift trend over recent history."""
        if len(self.drift_history) < 5:
            return 0.0
        
        # Linear regression on recent drift scores
        recent_scores = self.drift_history[-10:]
        x = np.arange(len(recent_scores))
        y = np.array(recent_scores)
        
        # Simple linear trend
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    
    def _select_strategy(self, normalized_drift: float) -> str:
        """Select adaptation strategy based on drift level."""
        if normalized_drift > 0.8:
            return 'aggressive'  # High drift - aggressive adaptation
        elif normalized_drift > 0.5:
            return 'moderate'    # Medium drift - balanced adaptation
        else:
            return 'conservative'  # Low drift - conservative adaptation
    
    def get_controller_state(self) -> Dict:
        """Get current controller state for logging."""
        return {
            'drift_history': self.drift_history.copy(),
            'drift_trend': self._compute_drift_trend(),
            'hypernetwork_params': sum(p.numel() for p in self.hypernetwork.parameters())
        }
