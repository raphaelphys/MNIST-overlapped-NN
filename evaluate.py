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
    
    # Number of test samples you want to visualize
    num_test_samples = 12
    
    # Grid dimensions: for example, 3 rows x 4 columns
    ncols = 4
    nrows = num_test_samples // ncols
    if nrows * ncols < num_test_samples:
        nrows += 1
    
    indices = random.sample(range(len(test_dataset)), num_test_samples)
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axs = axs.ravel()
    
    for i, idx in enumerate(indices):
        img, label = test_dataset[idx]
    
        with torch.no_grad():
            logits = model(img.unsqueeze(0).to(device))
    
        # Check if multi-label or single-label
        if isinstance(label, int):
            # SINGLE-LABEL scenario
            # label is just an integer
            true_label = label
            # find predicted digit
            pred_digit = logits.argmax(dim=1).cpu().item()
            title_str = f"True: {true_label}\nPred: {pred_digit}"
        else:
            # MULTI-LABEL scenario
            # label is a tensor of shape [10] with 0/1
            preds = (logits > 0).float().cpu().squeeze(0)
            true_digits = [j for j, val in enumerate(label) if val == 1]
            predicted_digits = [j for j, val in enumerate(preds) if val == 1]
            title_str = f"True: {true_digits}\nPred: {predicted_digits}"
    
        axs[i].imshow(img.squeeze(0), cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(title_str)
    
    # Hide any unused subplots
    for ax in axs[num_test_samples:]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()
