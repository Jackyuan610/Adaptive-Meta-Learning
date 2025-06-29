# task_loader.py

import os
import random
import pandas as pd
import torch
from collections import defaultdict
from PIL import Image

class FewShotTaskLoader:
    def __init__(self, csv_path, image_folder, transform, n_way=5, k_shot=5, q_query=15, difficulty_mode=False):
        self.data = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.transform = transform
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.difficulty_mode = difficulty_mode  # ✅ 加了这个标志
        self.label_to_paths = defaultdict(list)

        for _, row in self.data.iterrows():
            self.label_to_paths[row['label']].append(row['filename'])
        self.labels = list(self.label_to_paths.keys())

    def __iter__(self):
        while True:
            if self.difficulty_mode:
                task_labels = self.sample_easy_labels()  # 后面可以加简单难度筛选
            else:
                task_labels = random.sample(self.labels, self.n_way)

            support_imgs, support_labels = [], []
            query_imgs, query_labels = [], []
            for i, label in enumerate(task_labels):
                imgs = random.sample(self.label_to_paths[label], self.k_shot + self.q_query)
                for img in imgs[:self.k_shot]:
                    support_imgs.append(self.transform(Image.open(os.path.join(self.image_folder, img)).convert("RGB")))
                    support_labels.append(i)
                for img in imgs[self.k_shot:]:
                    query_imgs.append(self.transform(Image.open(os.path.join(self.image_folder, img)).convert("RGB")))
                    query_labels.append(i)

            yield torch.stack(support_imgs), torch.tensor(support_labels), torch.stack(query_imgs), torch.tensor(query_labels), task_labels

    def sample_easy_labels(self):
        # ✅ 暂时还是随机，可以后面进一步细化按类别区分度采样
        return random.sample(self.labels, self.n_way)
