"""
Replay Memory for Continual Few-Shot Learning
Implements memory buffer for task embeddings and Fisher information as described in the paper.
"""

import torch
import numpy as np
from collections import deque
from typing import List, Tuple, Optional


class ReplayMemory:
    """
    Replay memory buffer for storing task embeddings and Fisher information.
    
    As described in Section 3.4 of the paper, this memory supports:
    - Drift detection via historical task statistics
    - EWC regularization using stored Fisher matrices
    - Controller guidance for task-specific adaptation
    """
    
    def __init__(self, max_size: int = 20):
        """
        Initialize replay memory.
        
        Args:
            max_size: Maximum number of tasks to store in memory
        """
        self.max_size = max_size
        self.task_embeddings = deque(maxlen=max_size)
        self.fisher_matrices = deque(maxlen=max_size)
        self.parameter_snapshots = deque(maxlen=max_size)
        self.task_ids = deque(maxlen=max_size)
        
    def update(self, task_embedding: torch.Tensor, task_id: int, 
               fisher_matrix: Optional[torch.Tensor] = None,
               parameter_snapshot: Optional[torch.Tensor] = None):
        """
        Update memory with new task information.
        
        Args:
            task_embedding: Task embedding vector
            task_id: Task identifier
            fisher_matrix: Fisher information matrix (optional)
            parameter_snapshot: Parameter snapshot (optional)
        """
        self.task_embeddings.append(task_embedding.detach().cpu())
        self.task_ids.append(task_id)
        
        if fisher_matrix is not None:
            self.fisher_matrices.append(fisher_matrix.detach().cpu())
        else:
            self.fisher_matrices.append(None)
            
        if parameter_snapshot is not None:
            self.parameter_snapshots.append(parameter_snapshot.detach().cpu())
        else:
            self.parameter_snapshots.append(None)
    
    def get_historical_embeddings(self) -> List[torch.Tensor]:
        """Get all stored task embeddings."""
        return list(self.task_embeddings)
    
    def get_recent_embeddings(self, window_size: int = 5) -> List[torch.Tensor]:
        """Get recent task embeddings for drift detection."""
        return list(self.task_embeddings)[-window_size:]
    
    def get_past_embeddings(self, window_size: int = 5) -> List[torch.Tensor]:
        """Get past task embeddings for drift detection."""
        if len(self.task_embeddings) < 2 * window_size:
            return list(self.task_embeddings)[:-window_size]
        return list(self.task_embeddings)[-2*window_size:-window_size]
    
    def get_fisher_matrices(self) -> List[torch.Tensor]:
        """Get all stored Fisher matrices for EWC regularization."""
        return [f for f in self.fisher_matrices if f is not None]
    
    def get_parameter_snapshots(self) -> List[torch.Tensor]:
        """Get all stored parameter snapshots."""
        return [p for p in self.parameter_snapshots if p is not None]
    
    def size(self) -> int:
        """Get current memory size."""
        return len(self.task_embeddings)
    
    def is_full(self) -> bool:
        """Check if memory is full."""
        return len(self.task_embeddings) >= self.max_size
    
    def clear(self):
        """Clear all stored data."""
        self.task_embeddings.clear()
        self.fisher_matrices.clear()
        self.parameter_snapshots.clear()
        self.task_ids.clear()
    
    def sample_tasks(self, num_samples: int) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Sample random tasks from memory for replay.
        
        Args:
            num_samples: Number of tasks to sample
            
        Returns:
            Tuple of (embeddings, task_ids)
        """
        if len(self.task_embeddings) == 0:
            return [], []
        
        indices = np.random.choice(len(self.task_embeddings), 
                                 size=min(num_samples, len(self.task_embeddings)), 
                                 replace=False)
        
        sampled_embeddings = [self.task_embeddings[i] for i in indices]
        sampled_ids = [self.task_ids[i] for i in indices]
        
        return sampled_embeddings, sampled_ids 