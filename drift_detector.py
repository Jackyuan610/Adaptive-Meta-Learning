# drift_detector.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import scipy.stats as stats
import warnings


class HybridDriftDetector:
    """
    Hybrid drift detector combining KL divergence, mean shift, and MLP classifier.
    
    Implements the drift detection methodology from the paper with three complementary
    detection mechanisms for robust drift identification.
    """
    
    def __init__(self, 
                 embedding_dim: int = 128,
                 window_size: int = 5,
                 kl_threshold: float = 0.1,  # 降低阈值
                 mean_shift_threshold: float = 0.1,  # 降低阈值
                 mlp_hidden_dim: int = 64):
        """
        Initialize the hybrid drift detector.
        
        Args:
            embedding_dim: Dimension of task embeddings
            window_size: Size of sliding window for drift detection
            kl_threshold: Threshold for KL divergence detection
            mean_shift_threshold: Threshold for mean shift detection
            mlp_hidden_dim: Hidden dimension for MLP classifier
        """
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.kl_threshold = kl_threshold
        self.mean_shift_threshold = mean_shift_threshold
        
        # Store historical embeddings for drift analysis
        self.embedding_history = deque(maxlen=window_size * 2)
        
        # MLP classifier for drift detection (as per paper)
        self.mlp_classifier = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # MLP training state
        self.mlp_optimizer = torch.optim.Adam(self.mlp_classifier.parameters(), lr=0.001)
        self.mlp_training_data = []
        self.mlp_labels = []
        self.mlp_trained = False
        
        # Drift detection results
        self.drift_scores = []
        self.drift_confidence = []
        
    def add_embedding(self, embedding: torch.Tensor) -> None:
        """Add new task embedding to history."""
        self.embedding_history.append(embedding.detach().cpu())
    
    def detect_drift(self, current_embedding: torch.Tensor) -> Tuple[float, float]:
        """
        Perform hybrid drift detection using multiple methods.
        
        Args:
            current_embedding: Current task embedding
            
        Returns:
            Tuple of (drift_score, mlp_score)
        """
        self.add_embedding(current_embedding)
        if len(self.embedding_history) < 2:
            return 0.0, 0.0
        
        # For early stages, use available history
        if len(self.embedding_history) < self.window_size * 2:
            # Use all available history for early drift detection
            available_history = list(self.embedding_history)
            mid_point = len(available_history) // 2
            
            if mid_point < 1:
                # Still too early, use MLP only
                mlp_score = self._compute_mlp_score(current_embedding)
                return mlp_score * 0.5, mlp_score  # Reduced weight for early stage
            
            past_embeddings = torch.stack(available_history[:mid_point])
            recent_embeddings = torch.stack(available_history[mid_point:])
        else:
            # Full history available
            past_embeddings = torch.stack(list(self.embedding_history)[:self.window_size])
            recent_embeddings = torch.stack(list(self.embedding_history)[-self.window_size:])
        
        # Compute drift scores using different methods (as per paper)
        kl_score = self._compute_kl_divergence(past_embeddings, recent_embeddings)
        mean_shift_score = self._compute_mean_shift(past_embeddings, recent_embeddings)
        mlp_score = self._compute_mlp_score(current_embedding)
        
        # Debug print for all scores
        print(f"    Drift Scores: KL={kl_score:.3f}, MeanShift={mean_shift_score:.3f}, MLP={mlp_score:.3f}")
        
        # Apply thresholds for binary signals (as per paper)
        kl_signal = 1.0 if kl_score > self.kl_threshold else 0.0
        mean_shift_signal = 1.0 if mean_shift_score > self.mean_shift_threshold else 0.0
        mlp_signal = 1.0 if mlp_score > 0.3 else 0.0  # 降低MLP阈值
        
        # Majority voting for final drift decision (as per paper)
        signals = [kl_signal, mean_shift_signal, mlp_signal]
        drift_detected = sum(signals) >= 2  # At least 2 out of 3 signals
        
        # Combine scores with weighted average for continuous drift score
        drift_score = 0.4 * kl_score + 0.3 * mean_shift_score + 0.3 * mlp_score
        
        # Ensure drift_score is valid
        if np.isnan(drift_score) or np.isinf(drift_score):
            drift_score = 0.0
        
        # Boost drift score if majority voting indicates drift
        if drift_detected:
            drift_score *= 1.5
        
        # Store results
        self.drift_scores.append(drift_score)
        
        return float(drift_score), float(mlp_score)
    
    def _compute_kl_divergence(self, past_embeddings: torch.Tensor, 
                              recent_embeddings: torch.Tensor) -> float:
        """Compute KL divergence between past and recent embedding distributions."""
        try:
            # Compute empirical distributions
            past_mean = past_embeddings.mean(dim=0)
            past_var = past_embeddings.var(dim=0) + 1e-8  # 增加epsilon
            recent_mean = recent_embeddings.mean(dim=0)
            recent_var = recent_embeddings.var(dim=0) + 1e-8  # 增加epsilon
            
            # Check for valid variances
            if torch.any(past_var <= 1e-8) or torch.any(recent_var <= 1e-8):
                # Use L2 distance between means as alternative
                mean_distance = torch.norm(recent_mean - past_mean, p='fro').item()
                return min(mean_distance, 2.0)  # 降低上限
            
            # Compute KL divergence
            kl_div = 0.5 * (
                torch.log(recent_var / past_var) + 
                (past_var + (past_mean - recent_mean) ** 2) / recent_var - 1
            ).sum().item()
            
            # Check for invalid values
            if np.isnan(kl_div) or np.isinf(kl_div):
                mean_distance = torch.norm(recent_mean - past_mean, p='fro').item()
                return min(mean_distance, 2.0)
            
            # Cap KL divergence to reasonable range
            return min(max(kl_div, 0.0), 2.0)
        except Exception as e:
            print(f"Error in KL computation: {e}")
            return 0.0
    
    def _compute_mean_shift(self, past_embeddings: torch.Tensor, 
                           recent_embeddings: torch.Tensor) -> float:
        """Compute mean shift between past and recent embeddings."""
        try:
            past_mean = past_embeddings.mean(dim=0)
            recent_mean = recent_embeddings.mean(dim=0)
            
            # L2 norm of mean difference
            mean_shift = torch.norm(recent_mean - past_mean, p='fro').item()
            
            # Check for invalid values
            if np.isnan(mean_shift) or np.isinf(mean_shift):
                return 0.0
            
            return max(mean_shift, 0.0)  # Ensure non-negative
        except Exception:
            return 0.0
    
    def _compute_mlp_score(self, current_embedding: torch.Tensor) -> float:
        """Compute drift score using trained MLP classifier."""
        try:
            with torch.no_grad():
                # Normalize embedding for MLP input
                normalized_embedding = F.normalize(current_embedding, p=2, dim=0)
                
                if self.mlp_trained and len(self.mlp_training_data) > 10:
                    # Use trained MLP classifier
                    mlp_input = normalized_embedding.unsqueeze(0)
                    mlp_score = self.mlp_classifier(mlp_input).item()
                else:
                    # Fallback to heuristic method during early training
                    embedding_std = torch.std(normalized_embedding).item()
                    distance_from_unit = abs(torch.norm(normalized_embedding, p='fro').item() - 1.0)
                    mlp_score = min((embedding_std + distance_from_unit) / 2.0, 1.0)
                
                # Debug print
                print(f"    MLP Debug: trained={self.mlp_trained}, score={mlp_score:.3f}")
                
                # Check for invalid values
                if np.isnan(mlp_score) or np.isinf(mlp_score):
                    return 0.0
                
                return max(min(mlp_score, 1.0), 0.0)  # Ensure in [0, 1] range
        except Exception as e:
            print(f"MLP score computation error: {e}")
            return 0.0
    
    def train_mlp_classifier(self, current_embedding: torch.Tensor, is_drift: bool):
        """Train MLP classifier with current embedding and drift label."""
        try:
            # Normalize embedding
            normalized_embedding = F.normalize(current_embedding, p=2, dim=0)
            
            # Add to training data
            self.mlp_training_data.append(normalized_embedding.detach().cpu())
            self.mlp_labels.append(1.0 if is_drift else 0.0)
            
            # Train MLP when we have enough data
            if len(self.mlp_training_data) >= 5:
                self._update_mlp_classifier()
                
        except Exception as e:
            print(f"MLP training error: {e}")
    
    def _update_mlp_classifier(self):
        """Update MLP classifier with accumulated training data."""
        try:
            if len(self.mlp_training_data) < 5:
                return
            
            # Prepare training data
            embeddings = torch.stack(self.mlp_training_data[-10:])  # Use last 10 samples
            labels = torch.tensor(self.mlp_labels[-10:], dtype=torch.float32)
            
            # Training loop
            self.mlp_classifier.train()
            for epoch in range(10):  # 10 epochs
                self.mlp_optimizer.zero_grad()
                
                # Forward pass
                predictions = self.mlp_classifier(embeddings).squeeze()
                loss = F.binary_cross_entropy(predictions, labels)
                
                # Backward pass
                loss.backward()
                self.mlp_optimizer.step()
            
            self.mlp_trained = True
            print(f"    MLP trained with {len(embeddings)} samples, loss: {loss.item():.4f}")
            
        except Exception as e:
            print(f"MLP update error: {e}")
    
    def get_drift_trend(self, window: int = 10) -> float:
        """Compute drift trend over recent window."""
        if len(self.drift_scores) < window:
            return 0.0
        
        recent_scores = self.drift_scores[-window:]
        x = np.arange(len(recent_scores))
        y = np.array(recent_scores)
        
        # Linear trend
        try:
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)
        except:
            return 0.0
    
    def is_drift_detected(self, drift_score: float, confidence: float) -> bool:
        """Determine if drift is detected based on score and confidence."""
        return (drift_score > self.kl_threshold and confidence > 0.6)
    
    def get_detector_state(self) -> Dict:
        """Get current detector state for logging."""
        return {
            'history_size': len(self.embedding_history),
            'drift_scores': self.drift_scores[-10:] if self.drift_scores else [],
            'drift_confidence': self.drift_confidence[-10:] if self.drift_confidence else [],
            'drift_trend': self.get_drift_trend()
        }
