#!/usr/bin/env python3
"""
Basic functionality test for continual few-shot learning components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from tqdm import tqdm
import os

def create_simple_dataset():
    """Create a simple synthetic dataset for testing"""
    # Generate synthetic data: 10 classes, 100 samples per class
    n_classes = 10
    n_samples_per_class = 100
    feature_dim = 64
    
    # Create synthetic features
    features = []
    labels = []
    
    for class_idx in range(n_classes):
        # Create class-specific feature distribution
        class_mean = np.random.randn(feature_dim) * 2
        class_features = np.random.randn(n_samples_per_class, feature_dim) * 0.5 + class_mean
        features.append(class_features)
        labels.extend([class_idx] * n_samples_per_class)
    
    features = np.vstack(features)
    labels = np.array(labels)
    
    return torch.FloatTensor(features), torch.LongTensor(labels)

class SimpleProtoNet(nn.Module):
    """Simple ProtoNet implementation"""
    def __init__(self, feature_dim=64, n_way=5):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_way = n_way
        
    def forward(self, support_features, support_labels, query_features):
        """
        Args:
            support_features: [n_support, feature_dim]
            support_labels: [n_support]
            query_features: [n_query, feature_dim]
        """
        # Compute prototypes
        prototypes = []
        for i in range(self.n_way):
            mask = (support_labels == i)
            if mask.sum() > 0:
                prototype = support_features[mask].mean(0)
                prototypes.append(prototype)
            else:
                # If no support samples for this class, use zero prototype
                prototypes.append(torch.zeros(self.feature_dim, device=support_features.device))
        
        prototypes = torch.stack(prototypes)  # [n_way, feature_dim]
        
        # Compute distances
        query_expanded = query_features.unsqueeze(1)  # [n_query, 1, feature_dim]
        prototypes_expanded = prototypes.unsqueeze(0)  # [1, n_way, feature_dim]
        
        distances = torch.sum((query_expanded - prototypes_expanded) ** 2, dim=2)  # [n_query, n_way]
        
        # Convert distances to logits (negative distances)
        logits = -distances
        
        return logits

def sample_few_shot_task(features, labels, n_way=5, n_support=5, n_query=15):
    """Sample a few-shot task"""
    # Randomly select n_way classes
    unique_classes = torch.unique(labels)
    selected_classes = torch.randperm(len(unique_classes))[:n_way]
    selected_classes = unique_classes[selected_classes]
    
    # Sample support and query sets
    support_features, support_labels = [], []
    query_features, query_labels = [], []
    
    for i, class_idx in enumerate(selected_classes):
        class_mask = (labels == class_idx)
        class_features = features[class_mask]
        
        # Randomly sample support and query
        perm = torch.randperm(len(class_features))
        support_indices = perm[:n_support]
        query_indices = perm[n_support:n_support + n_query]
        
        support_features.append(class_features[support_indices])
        support_labels.extend([i] * n_support)  # Use 0, 1, 2, ... for n_way classes
        
        query_features.append(class_features[query_indices])
        query_labels.extend([i] * n_query)
    
    support_features = torch.cat(support_features, dim=0)
    support_labels = torch.LongTensor(support_labels)
    query_features = torch.cat(query_features, dim=0)
    query_labels = torch.LongTensor(query_labels)
    
    return support_features, support_labels, query_features, query_labels

def test_protonet():
    """Test ProtoNet functionality"""
    print("🧪 Testing ProtoNet...")
    
    # Create dataset
    features, labels = create_simple_dataset()
    print(f"Dataset: {len(features)} samples, {len(torch.unique(labels))} classes")
    
    # Create model
    model = SimpleProtoNet(feature_dim=64, n_way=5)
    
    # Test on multiple tasks
    accuracies = []
    for task_idx in range(10):
        # Sample task
        s_feats, s_labels, q_feats, q_labels = sample_few_shot_task(
            features, labels, n_way=5, n_support=5, n_query=15
        )
        
        # Forward pass
        logits = model(s_feats, s_labels, q_feats)
        
        # Compute accuracy
        accuracy = (logits.argmax(1) == q_labels).float().mean().item()
        accuracies.append(accuracy)
        
        print(f"Task {task_idx+1}: {accuracy:.2%}")
    
    avg_acc = np.mean(accuracies)
    print(f"Average accuracy: {avg_acc:.2%}")
    return avg_acc

def test_continual_learning():
    """Test simple continual learning scenario"""
    print("\n🔄 Testing Continual Learning...")
    
    # Create dataset with concept drift
    n_tasks = 20
    task_accuracies = []
    
    for task_idx in range(n_tasks):
        # Simulate concept drift: gradually change feature distributions
        drift_factor = task_idx / n_tasks
        
        # Create task-specific dataset
        features, labels = create_simple_dataset()
        
        # Apply concept drift
        features = features + drift_factor * torch.randn_like(features) * 0.5
        
        # Sample task
        s_feats, s_labels, q_feats, q_labels = sample_few_shot_task(
            features, labels, n_way=5, n_support=5, n_query=15
        )
        
        # Use ProtoNet
        model = SimpleProtoNet(feature_dim=64, n_way=5)
        logits = model(s_feats, s_labels, q_feats)
        accuracy = (logits.argmax(1) == q_labels).float().mean().item()
        
        task_accuracies.append(accuracy)
        print(f"Task {task_idx+1} (drift={drift_factor:.2f}): {accuracy:.2%}")
    
    # Analyze performance degradation
    early_acc = np.mean(task_accuracies[:5])
    late_acc = np.mean(task_accuracies[-5:])
    degradation = early_acc - late_acc
    
    print(f"Early tasks (1-5): {early_acc:.2%}")
    print(f"Late tasks (16-20): {late_acc:.2%}")
    print(f"Performance degradation: {degradation:.2%}")
    
    return task_accuracies

def main():
    """Main test function"""
    print("🚀 Starting Basic Functionality Tests")
    print("=" * 50)
    
    # Test 1: ProtoNet
    protonet_acc = test_protonet()
    
    # Test 2: Continual Learning
    cl_accuracies = test_continual_learning()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"ProtoNet average accuracy: {protonet_acc:.2%}")
    print(f"Continual learning degradation: {cl_accuracies[0] - cl_accuracies[-1]:.2%}")
    
    if protonet_acc > 0.7:
        print("✅ ProtoNet working correctly")
    else:
        print("❌ ProtoNet performance too low")
    
    if cl_accuracies[0] - cl_accuracies[-1] > 0.1:
        print("✅ Concept drift detected")
    else:
        print("❌ No significant concept drift")

if __name__ == "__main__":
    main() 