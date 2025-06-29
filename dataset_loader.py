#!/usr/bin/env python3
"""
统一数据集加载器
支持 Mini-ImageNet, Omniglot, Meta-Dataset 等数据集
"""

import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any, Union
import random

class FewShotTask:
    """Represents a few-shot learning task with support and query sets."""
    
    def __init__(self, support_data: List[Tuple], query_data: List[Tuple], 
                 n_way: int, n_support: int, n_query: int):
        """
        Initialize a few-shot task.
        
        Args:
            support_data: List of (image, label) pairs for support set
            query_data: List of (image, label) pairs for query set
            n_way: Number of classes in the task
            n_support: Number of support samples per class
            n_query: Number of query samples per class
        """
        self.support_data = support_data
        self.query_data = query_data
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        
        # Extract unique class labels and create mapping
        self.classes = sorted(list(set([label for _, label in support_data])))
        self.label_mapping = {old_label: new_label for new_label, old_label in enumerate(self.classes)}
        
    def get_support_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert support data to tensors with remapped labels."""
        import torch
        from PIL import Image
        import torchvision.transforms as T
        images = []
        remapped_labels = []
        for img, label in self.support_data:
            if isinstance(img, torch.Tensor):
                images.append(img)
            elif isinstance(img, str):
                image = Image.open(img).convert('RGB')
                image = T.ToTensor()(image)
                images.append(image)
            else:
                raise TypeError(f"Unsupported image type in support_data: {type(img)}")
            # Remap label to 0-4 range
            remapped_labels.append(self.label_mapping[label])
        images = torch.stack(images)
        labels = torch.tensor(remapped_labels, dtype=torch.long)
        return images, labels
    
    def get_query_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert query data to tensors with remapped labels."""
        import torch
        from PIL import Image
        import torchvision.transforms as T
        images = []
        remapped_labels = []
        for img, label in self.query_data:
            if isinstance(img, torch.Tensor):
                images.append(img)
            elif isinstance(img, str):
                image = Image.open(img).convert('RGB')
                image = T.ToTensor()(image)
                images.append(image)
            else:
                raise TypeError(f"Unsupported image type in query_data: {type(img)}")
            # Remap label to 0-4 range
            remapped_labels.append(self.label_mapping[label])
        images = torch.stack(images)
        labels = torch.tensor(remapped_labels, dtype=torch.long)
        return images, labels

class BaseDataset(Dataset):
    """Base class for few-shot learning datasets."""
    
    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None):
        """
        Initialize base dataset.
        
        Args:
            root_dir: Root directory containing dataset
            transform: Optional image transformations
        """
        self.root_dir = root_dir
        self.transform = transform or self._get_default_transform()
        self.classes = []
        self.class_to_idx = {}
        self.samples = []
        
    def _get_default_transform(self) -> transforms.Compose:
        """Get default image transformations."""
        return transforms.Compose([
            transforms.Resize((84, 84)),  # Standard size for few-shot learning
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get sample by index."""
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        else:
            # Fallback transform to ensure tensor output
            image = transforms.ToTensor()(image)
        
        # Ensure image is a tensor
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)
            
        return image, label

class MiniImageNetDataset(BaseDataset):
    """Mini-ImageNet dataset for few-shot learning."""
    def __init__(self, root_dir: str, split: str = 'train'):
        super().__init__(root_dir)
        self.split = split
        self._load_dataset()
    
    def _load_dataset(self):
        """Load Mini-ImageNet dataset."""
        split_dir = os.path.join(self.root_dir, self.split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        # Load class directories
        self.classes = [d for d in os.listdir(split_dir) 
                       if os.path.isdir(os.path.join(split_dir, d))]
        self.classes.sort()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load samples
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(split_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

class OmniglotDataset(BaseDataset):
    """Omniglot dataset for few-shot learning."""
    def __init__(self, root_dir: str, split: str = 'background'):
        super().__init__(root_dir)
        self.split = split
        self._load_dataset()
    
    def _load_dataset(self):
        """Load Omniglot dataset."""
        split_dir = os.path.join(self.root_dir, f"images_{self.split}")
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        # Load alphabet directories
        self.samples = []
        class_idx = 0
        
        for alphabet in os.listdir(split_dir):
            alphabet_dir = os.path.join(split_dir, alphabet)
            if not os.path.isdir(alphabet_dir):
                continue
                
            for character in os.listdir(alphabet_dir):
                character_dir = os.path.join(alphabet_dir, character)
                if not os.path.isdir(character_dir):
                    continue
                    
                for img_name in os.listdir(character_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(character_dir, img_name)
                        self.samples.append((img_path, class_idx))
                
                class_idx += 1

class UnifiedDataLoader:
    """Unified data loader for multiple datasets."""
    
    def __init__(self, dataset_config: Dict[str, str]):
        """
        Initialize unified data loader.
        
        Args:
            dataset_config: Dictionary mapping dataset names to root directories
        """
        self.dataset_config = dataset_config
        self.datasets = {}
        self._load_datasets()
    
    def _load_datasets(self):
        """Load all configured datasets."""
        for dataset_name, root_dir in self.dataset_config.items():
            if dataset_name == 'mini_imagenet':
                self.datasets[dataset_name] = {
                    'train': MiniImageNetDataset(root_dir, 'train'),
                    'val': MiniImageNetDataset(root_dir, 'val'),
                    'test': MiniImageNetDataset(root_dir, 'test')
                }
            elif dataset_name == 'omniglot':
                self.datasets[dataset_name] = {
                    'background': OmniglotDataset(root_dir, 'background'),
                    'evaluation': OmniglotDataset(root_dir, 'evaluation')
                }
            elif dataset_name == 'meta_dataset':
                # Simplified meta-dataset loading
                self.datasets[dataset_name] = self._load_meta_dataset(root_dir)
    
    def _load_meta_dataset(self, root_dir: str) -> Dict[str, Any]:
        """Load simplified meta-dataset."""
        # This is a simplified version - in practice you'd use the full meta-dataset
        return {
            'train': MiniImageNetDataset(root_dir, 'train') if os.path.exists(os.path.join(root_dir, 'train')) else None,
            'val': MiniImageNetDataset(root_dir, 'val') if os.path.exists(os.path.join(root_dir, 'val')) else None,
            'test': MiniImageNetDataset(root_dir, 'test') if os.path.exists(os.path.join(root_dir, 'test')) else None
        }
    
    def sample_task(self, dataset_name: str, n_way: int = 5, n_support: int = 5, n_query: int = 15) -> FewShotTask:
        """
        Sample a few-shot task from the specified dataset.
        
        Args:
            dataset_name: Name of the dataset to sample from
            n_way: Number of classes in the task
            n_support: Number of support samples per class
            n_query: Number of query samples per class
            
        Returns:
            FewShotTask object
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        # Choose split (prefer train, fallback to others)
        dataset = self.datasets[dataset_name]
        split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
        split_dataset = dataset[split_name]
        
        if split_dataset is None:
            raise ValueError(f"No valid split found for dataset {dataset_name}")
        
        # Sample classes
        available_classes = list(set([label for _, label in split_dataset.samples]))
        if len(available_classes) < n_way:
            raise ValueError(f"Not enough classes in dataset. Need {n_way}, have {len(available_classes)}")
        
        selected_classes = random.sample(available_classes, n_way)
        
        # Sample support and query data
        support_data = []
        query_data = []
        
        for class_idx in selected_classes:
            # Get all samples for this class
            class_samples = [(img_path, class_idx) for img_path, label in split_dataset.samples if label == class_idx]
            
            if len(class_samples) < n_support + n_query:
                raise ValueError(f"Not enough samples for class {class_idx}. Need {n_support + n_query}, have {len(class_samples)}")
            
            # Randomly sample support and query sets
            random.shuffle(class_samples)
            support_data.extend(class_samples[:n_support])
            query_data.extend(class_samples[n_support:n_support + n_query])
        
        return FewShotTask(support_data, query_data, n_way, n_support, n_query)

    def get_transforms(self, split='train'):
        """Get data transforms for different splits with enhanced augmentation."""
        if split == 'train':
            return transforms.Compose([
                transforms.RandomResizedCrop(84, scale=(0.8, 1.0)),  # 增强的随机裁剪
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0),  # 颜色增强
                transforms.RandomRotation(degrees=15),  # 随机旋转
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机仿射变换
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(92),  # 稍微增大尺寸
                transforms.CenterCrop(84),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

def create_dataloader(dataset_config: Dict[str, str]) -> UnifiedDataLoader:
    """
    Create a unified data loader for the specified datasets.
    
    Args:
        dataset_config: Dictionary mapping dataset names to root directories
        
    Returns:
        UnifiedDataLoader instance
    """
    return UnifiedDataLoader(dataset_config) 