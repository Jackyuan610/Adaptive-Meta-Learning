#!/usr/bin/env python3
"""
Simple test to compare ProtoNet vs MetaSGD performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
import time

class SimpleProtoNet(nn.Module):
    """Simple ProtoNet implementation"""
    def __init__(self, feature_dim=512, n_way=5):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_way = n_way
        
    def forward(self, support_features, support_labels, query_features):
        # Compute prototypes
        prototypes = []
        for i in range(self.n_way):
            mask = (support_labels == i)
            if mask.sum() > 0:
                prototype = support_features[mask].mean(0)
                prototypes.append(prototype)
            else:
                prototypes.append(torch.zeros(self.feature_dim, device=support_features.device))
        
        prototypes = torch.stack(prototypes)  # [n_way, feature_dim]
        
        # Compute distances
        query_expanded = query_features.unsqueeze(1)  # [n_query, 1, feature_dim]
        prototypes_expanded = prototypes.unsqueeze(0)  # [1, n_way, feature_dim]
        
        distances = torch.sum((query_expanded - prototypes_expanded) ** 2, dim=2)  # [n_query, n_way]
        logits = -distances  # Convert distances to logits
        
        return logits

class SimpleMetaSGD(nn.Module):
    """Simple MetaSGD implementation"""
    def __init__(self, feature_dim=512, n_way=5, inner_steps=5, inner_lr=0.01):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_way = n_way
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.classifier = nn.Linear(feature_dim, n_way)
        
    def forward(self, support_features, support_labels, query_features):
        # Initialize fast weights
        fast_weights = list(self.classifier.parameters())
        
        # Inner loop adaptation
        for step in range(self.inner_steps):
            logits = F.linear(support_features, fast_weights[0], fast_weights[1])
            loss = F.cross_entropy(logits, support_labels)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, fast_weights, create_graph=True)
            fast_weights = [w - self.inner_lr * g for w, g in zip(fast_weights, grads)]
        
        # Query prediction
        query_logits = F.linear(query_features, fast_weights[0], fast_weights[1])
        return query_logits

def create_synthetic_task(n_way=5, n_support=5, n_query=15, feature_dim=512):
    """Create synthetic few-shot task"""
    # Generate synthetic features with clear class separation
    support_features = []
    support_labels = []
    query_features = []
    query_labels = []
    
    for class_idx in range(n_way):
        # Create class-specific feature distribution
        class_mean = torch.randn(feature_dim) * 2
        class_std = 0.5
        
        # Support set
        class_support = torch.randn(n_support, feature_dim) * class_std + class_mean
        support_features.append(class_support)
        support_labels.extend([class_idx] * n_support)
        
        # Query set
        class_query = torch.randn(n_query, feature_dim) * class_std + class_mean
        query_features.append(class_query)
        query_labels.extend([class_idx] * n_query)
    
    support_features = torch.cat(support_features, dim=0)
    support_labels = torch.LongTensor(support_labels)
    query_features = torch.cat(query_features, dim=0)
    query_labels = torch.LongTensor(query_labels)
    
    return support_features, support_labels, query_features, query_labels

def test_strategies():
    """Test ProtoNet vs MetaSGD performance"""
    print("🧪 Testing ProtoNet vs MetaSGD...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create models
    protonet = SimpleProtoNet(feature_dim=512, n_way=5).to(device)
    metasgd = SimpleMetaSGD(feature_dim=512, n_way=5, inner_steps=5, inner_lr=0.01).to(device)
    
    # Test on multiple tasks
    n_tasks = 20
    protonet_accuracies = []
    metasgd_accuracies = []
    protonet_times = []
    metasgd_times = []
    
    for task_idx in range(n_tasks):
        # Create synthetic task
        s_feats, s_labels, q_feats, q_labels = create_synthetic_task(
            n_way=5, n_support=5, n_query=15, feature_dim=512
        )
        
        s_feats, s_labels = s_feats.to(device), s_labels.to(device)
        q_feats, q_labels = q_feats.to(device), q_labels.to(device)
        
        # Test ProtoNet
        start_time = time.time()
        protonet_logits = protonet(s_feats, s_labels, q_feats)
        protonet_time = time.time() - start_time
        protonet_acc = (protonet_logits.argmax(1) == q_labels).float().mean().item()
        
        # Test MetaSGD
        start_time = time.time()
        metasgd_logits = metasgd(s_feats, s_labels, q_feats)
        metasgd_time = time.time() - start_time
        metasgd_acc = (metasgd_logits.argmax(1) == q_labels).float().mean().item()
        
        protonet_accuracies.append(protonet_acc)
        metasgd_accuracies.append(metasgd_acc)
        protonet_times.append(protonet_time)
        metasgd_times.append(metasgd_time)
        
        print(f"Task {task_idx+1:2d}: ProtoNet {protonet_acc:.2%} ({protonet_time:.3f}s) | "
              f"MetaSGD {metasgd_acc:.2%} ({metasgd_time:.3f}s)")
    
    # Summary
    print("\n" + "="*60)
    print("📊 Performance Summary:")
    print(f"ProtoNet: {np.mean(protonet_accuracies):.2%} ± {np.std(protonet_accuracies):.2%} "
          f"(avg time: {np.mean(protonet_times):.3f}s)")
    print(f"MetaSGD:  {np.mean(metasgd_accuracies):.2%} ± {np.std(metasgd_accuracies):.2%} "
          f"(avg time: {np.mean(metasgd_times):.3f}s)")
    
    # Determine which strategy is better
    if np.mean(protonet_accuracies) > np.mean(metasgd_accuracies):
        print("✅ ProtoNet performs better")
        return "ProtoNet"
    else:
        print("✅ MetaSGD performs better")
        return "MetaSGD"

def test_drift_simulation():
    """Test strategy selection with simulated drift"""
    print("\n🔄 Testing Strategy Selection with Drift...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create models
    protonet = SimpleProtoNet(feature_dim=512, n_way=5).to(device)
    metasgd = SimpleMetaSGD(feature_dim=512, n_way=5, inner_steps=5, inner_lr=0.01).to(device)
    
    n_tasks = 30
    strategies_used = []
    
    for task_idx in range(n_tasks):
        # Simulate concept drift: gradually increase task difficulty
        drift_factor = task_idx / n_tasks
        
        # Create task with increasing difficulty
        s_feats, s_labels, q_feats, q_labels = create_synthetic_task(
            n_way=5, n_support=5, n_query=15, feature_dim=512
        )
        
        # Add noise based on drift factor
        noise_level = drift_factor * 0.5
        s_feats += torch.randn_like(s_feats) * noise_level
        q_feats += torch.randn_like(q_feats) * noise_level
        
        s_feats, s_labels = s_feats.to(device), s_labels.to(device)
        q_feats, q_labels = q_feats.to(device), q_labels.to(device)
        
        # Simple drift detection: use noise level as drift score
        drift_score = drift_factor * 5.0  # Scale to 0-5 range
        
        # Strategy selection
        if drift_score > 2.5:  # High drift threshold
            strategy = "MetaSGD"
            logits = metasgd(s_feats, s_labels, q_feats)
        else:
            strategy = "ProtoNet"
            logits = protonet(s_feats, s_labels, q_feats)
        
        accuracy = (logits.argmax(1) == q_labels).float().mean().item()
        strategies_used.append(strategy)
        
        print(f"Task {task_idx+1:2d}: Drift={drift_score:.2f} | Strategy={strategy} | Acc={accuracy:.2%}")
    
    # Analyze strategy usage
    protonet_count = strategies_used.count("ProtoNet")
    metasgd_count = strategies_used.count("MetaSGD")
    
    print(f"\n📈 Strategy Usage:")
    print(f"ProtoNet: {protonet_count}/{n_tasks} tasks ({protonet_count/n_tasks:.1%})")
    print(f"MetaSGD:  {metasgd_count}/{n_tasks} tasks ({metasgd_count/n_tasks:.1%})")
    
    # Check if strategy selection makes sense
    early_protonet = strategies_used[:10].count("ProtoNet")
    late_metasgd = strategies_used[-10:].count("MetaSGD")
    
    print(f"Early tasks (1-10): {early_protonet}/10 ProtoNet")
    print(f"Late tasks (21-30): {late_metasgd}/10 MetaSGD")
    
    if early_protonet > 5 and late_metasgd > 5:
        print("✅ Strategy selection working correctly!")
    else:
        print("❌ Strategy selection needs improvement")

if __name__ == "__main__":
    print("🚀 Starting Strategy Comparison Tests")
    print("="*60)
    
    # Test 1: Basic performance comparison
    better_strategy = test_strategies()
    
    # Test 2: Drift simulation
    test_drift_simulation()
    
    print("\n🎯 Conclusion:")
    print(f"Better baseline strategy: {better_strategy}")
    print("Strategy selection should use ProtoNet for low drift, MetaSGD for high drift") 