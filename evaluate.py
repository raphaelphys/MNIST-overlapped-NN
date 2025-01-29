# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 19:22:00 2025

@author: robaid
"""

#!/usr/bin/env python3


import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import dataset definition & model if needed
from overlapped_dataset import OverlappedMultiLabelMNIST

# Import or copy your model definition
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the trained model (e.g. 'mnist_standard.pth')")
    parser.add_argument('--overlapped', action='store_true',
                        help="Was the model trained on overlapped multi-label MNIST?")
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model & load weights
    model = SimpleMLP().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Prepare data
    transform = transforms.ToTensor()
    mnist_test = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    # If overlapped, wrap with OverlappedMultiLabelMNIST
    if args.overlapped:
        test_dataset = OverlappedMultiLabelMNIST(mnist_test)
        multi_label = True
    else:
        test_dataset = mnist_test
        multi_label = False

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Example evaluation:
    correct = 0
    total = 0


    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)

            # Overlapped => multi-label
            if multi_label:
                preds = (logits > 0).float()
                matches = (preds == labels).float().sum()
                correct += matches.item()
                total += labels.numel()
            else:
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Visualize a few samples
    import random
    num_samples = 5
    indices = random.sample(range(len(test_dataset)), num_samples)

    for idx in indices:
        img, label = test_dataset[idx]
        with torch.no_grad():
            logits = model(img.unsqueeze(0).to(device))
            if multi_label:
                preds = (logits > 0).float().cpu().squeeze(0)
            else:
                preds = logits.argmax(dim=1).cpu()

        # Show image
        plt.imshow(img.squeeze(0), cmap='gray')
        if multi_label:
            true_digits = [i for i, val in enumerate(label) if val == 1]
            pred_digits = [i for i, val in enumerate(preds) if val == 1]
            plt.title(f"True: {true_digits}, Pred: {pred_digits}")
        else:
            plt.title(f"True: {label}, Pred: {preds.item()}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()
