# task_stream_simulator.py

import random

class DriftType:
    NONE = "none"
    DOMAIN = "domain_shift"
    LABEL = "label_shift"
    SEMANTIC = "semantic_shift"

class TaskStreamSimulator:
    def __init__(self, num_tasks=100, drift_interval=5, seed=42):
        random.seed(seed)
        self.num_tasks = num_tasks
        self.drift_interval = drift_interval
        self.tasks = self._generate_task_stream()

    def _generate_task_stream(self):
        stream = []
        for i in range(self.num_tasks):
            # 根据论文：每10个任务一次漂移，从任务9开始
            # 漂移点：9, 19, 29, 39, 49, 59, 69, 79, 89, 99
            if (i + 1) % self.drift_interval == 0 and i > 0:  # 从任务9开始，每10个任务一次
                drift_type = random.choice([DriftType.DOMAIN, DriftType.LABEL, DriftType.SEMANTIC])
            else:
                drift_type = DriftType.NONE
            stream.append({
                "task_id": i,
                "is_drift": drift_type != DriftType.NONE,
                "drift_type": drift_type
            })
        return stream

    def get_task_info(self, task_index):
        if task_index < len(self.tasks):
            return self.tasks[task_index]
        else:
            return {
                "task_id": task_index,
                "is_drift": False,
                "drift_type": DriftType.NONE
            }

    def get_ground_truth(self):
        return [(t["task_id"], t["is_drift"], t["drift_type"]) for t in self.tasks]
