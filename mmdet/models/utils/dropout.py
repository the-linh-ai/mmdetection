"""
Dropout utilities designed for active learning purposes.
"""


import torch.nn as nn


class SyncedDropout(nn.Dropout):
    _all_instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SyncedDropout._all_instances.append(self)  # record current instance

    @classmethod
    def train_all(cls):
        """Trigger `.train()` for all initialized instances"""
        for instance in cls._all_instances:
            instance.train()

    @classmethod
    def eval_all(cls):
        """Trigger `.eval()` for all initialized instances"""
        for instance in cls._all_instances:
            instance.eval()
