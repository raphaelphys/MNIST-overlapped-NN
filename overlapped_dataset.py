# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 19:24:35 2025

@author: robaid
"""

#!/usr/bin/env python3
# overlapped_dataset.py

import random
import torch
from torch.utils.data import Dataset

class OverlappedMultiLabelMNIST(Dataset):
    """
    Creates overlapped images by averaging 2 random MNIST digits.
    label_vec is a 10-dim multi-hot vector (1 for each digit present).
    """

    def __init__(self, mnist_dataset):
        self.mnist = mnist_dataset

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        # Pick two random indices
        idx1 = random.randint(0, len(self.mnist) - 1)
        idx2 = random.randint(0, len(self.mnist) - 1)

        # Grab two different samples
        img1, digit1 = self.mnist[idx1]  # shape [1,28,28]
        img2, digit2 = self.mnist[idx2]

        # Overlapped image => [1,28,28]
        overlapped_img = (img1 + img2) / 2.0

        # 10-dim multi-hot label
        label_vec = torch.zeros(10, dtype=torch.float)
        label_vec[digit1] = 1.0
        label_vec[digit2] = 1.0

        return overlapped_img, label_vec
