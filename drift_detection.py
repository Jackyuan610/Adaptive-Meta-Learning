# drift_detection.py

import torch
import numpy as np

class TaskMemory:
    def __init__(self, max_size=500):
        self.embeds = []       # 每个任务的任务嵌入（如 ti）
        self.label_sets = []   # 每个任务的 label name set（用于覆盖分析）
        self.samples = []      # [(support_imgs, support_labels)]
        self.max_size = max_size

    def add(self, embeds, labels, support_imgs=None, support_labels=None):
        """添加新任务信息：嵌入 + 样本"""
        if len(self.embeds) >= self.max_size:
            self.embeds.pop(0)
            self.label_sets.pop(0)
            if self.samples:
                self.samples.pop(0)

        self.embeds.append(embeds)
        self.label_sets.append(set(labels))
        if support_imgs is not None and support_labels is not None:
            self.samples.append((support_imgs, support_labels))

    def sample_replay(self, k=5, current_embedding=None, strategy="diverse"):
        """
        支持两种策略：
        - random: 随机采样任务样本
        - diverse: 与当前任务 embedding 最远的任务样本
        """
        if len(self.samples) == 0:
            return None

        if strategy == "random" or current_embedding is None:
            idxs = np.random.choice(len(self.samples), k, replace=True)
        elif strategy == "diverse":
            distances = [torch.norm(e.clone().detach() - current_embedding).item() for e in self.embeds]
            idxs = np.argsort(distances)[-k:]  # 距离最远的任务
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        replay_imgs, replay_labels = [], []
        for i in idxs:
            imgs, labels = self.samples[i]
            replay_imgs.append(imgs)
            replay_labels.append(labels)
        return torch.cat(replay_imgs), torch.cat(replay_labels)
