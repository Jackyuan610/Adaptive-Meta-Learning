# hypernet.py - Gated Hypernetwork for Task-Specific Hyperparameter Modulation

import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedHyperNetwork(nn.Module):
    """
    Gated Hypernetwork that modulates learning rates and regularization strength
    based on task embeddings, drift scores, and task complexity.
    
    According to the paper:
    η_i = η_0 * (1 + α * C_i + β * D_i)
    λ_i = λ_0 * (1 + γ * D_i)
    
    where:
    - C_i: task complexity (variance of task embedding)
    - D_i: drift score
    - α, β, γ: learnable coefficients
    """
    
    def __init__(self, embedding_dim=512, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Learnable coefficients for hyperparameter modulation
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Task complexity coefficient
        self.beta = nn.Parameter(torch.tensor(1.0))   # Drift coefficient for learning rate
        self.gamma = nn.Parameter(torch.tensor(1.0))  # Drift coefficient for regularization
        
        # Base hyperparameters
        self.eta_base = nn.Parameter(torch.tensor(1e-3))  # Base learning rate
        self.lambda_base = nn.Parameter(torch.tensor(1.0))  # Base regularization strength
        
        # Gating mechanism for stability
        self.gate_network = nn.Sequential(
            nn.Linear(embedding_dim + 2, hidden_dim),  # +2 for drift_score and complexity
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # 3 gates: for α, β, γ
            nn.Sigmoid()
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(3)

    def forward(self, task_embedding, drift_score, task_complexity):
        """
        Compute task-specific hyperparameters based on task embedding, drift score, and complexity.
        
        Args:
            task_embedding: [batch_size, embedding_dim] Task embedding vector
            drift_score: [batch_size] Drift detection score
            task_complexity: [batch_size] Task complexity measure (variance)
            
        Returns:
            eta: [batch_size] Task-specific learning rate
            lambda_val: [batch_size] Task-specific regularization strength
        """
        batch_size = task_embedding.shape[0]
        
        # Prepare input for gating network
        drift_score = drift_score.view(batch_size, 1)
        task_complexity = task_complexity.view(batch_size, 1)
        
        gate_input = torch.cat([task_embedding, drift_score, task_complexity], dim=1)
        
        # Compute gating weights
        gates = self.gate_network(gate_input)  # [batch_size, 3]
        gates = self.layer_norm(gates)  # Stabilize outputs
        
        # Apply gating to coefficients
        alpha_gated = self.alpha * gates[:, 0:1]  # [batch_size, 1]
        beta_gated = self.beta * gates[:, 1:2]    # [batch_size, 1]
        gamma_gated = self.gamma * gates[:, 2:3]  # [batch_size, 1]
        
        # Compute hyperparameters according to paper formulas
        eta = self.eta_base * (1 + alpha_gated * task_complexity + beta_gated * drift_score)
        lambda_val = self.lambda_base * (1 + gamma_gated * drift_score)
        
        # Ensure positive values
        eta = F.softplus(eta).squeeze(-1)  # [batch_size]
        lambda_val = F.softplus(lambda_val).squeeze(-1)  # [batch_size]
        
        return eta, lambda_val
    
    def get_hyperparams(self, task_embedding, drift_score, task_complexity):
        """Convenience method to get hyperparameters."""
        eta, lambda_val = self.forward(task_embedding, drift_score, task_complexity)
        return eta, lambda_val
    
    def get_coefficients(self):
        """Get current coefficient values for analysis."""
        return {
            'alpha': self.alpha.item(),
            'beta': self.beta.item(), 
            'gamma': self.gamma.item(),
            'eta_base': self.eta_base.item(),
            'lambda_base': self.lambda_base.item()
        } 