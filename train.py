# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 19:23:03 2025

@author: robaid
"""

#!/usr/bin/env python3
# train.py

import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from overlapped_dataset import OverlappedMultiLabelMNIST

##############################################################################
# 1) Define the model
##############################################################################
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

##############################################################################
# 2) Training function
##############################################################################
def train_one_epoch(model, loader, optimizer, criterion, device, multi_label=False):
    model.train()
    running_loss = 0
    running_mse = 0
    running_correct = 0
    total_elements = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        #Accumulate BCE loss
        running_loss += loss.item()
        
        # MSE metric
        mse_val = mse_metric(logits, labels).item()
        running_mse += mse_val

        if multi_label:
            preds = (logits > 0).float()
            matches = (preds == labels).float().sum().item()
            total_elements += labels.numel()
            running_correct += matches
        else:
            preds = logits.argmax(dim=1)
            matches = (preds == labels).sum().item()
            total_elements += labels.size(0)
            running_correct += matches

    epoch_loss = running_loss / len(loader)
    epoch_mse = running_mse/ len(loader)
    epoch_acc = 100.0 * running_correct / total_elements
    return epoch_loss, epoch_mse, epoch_acc

##############################################################################
# 3) MSE error
##############################################################################
def mse_metric(logits, labels):
    """
    logits: [batch_size, 10], raw scores from the model
    labels: [batch_size, 10], in {0,1}
    """
    probs = torch.sigmoid(logits)               # probabilities in [0,1]
    mse = F.mse_loss(probs, labels, reduction='mean')  # average MSE over all samples & classes
    return mse

##############################################################################
# 4) Evaluation function
##############################################################################
def evaluate(model, loader, criterion, device, multi_label=False):
    model.eval()
    running_loss = 0
    running_correct = 0
    running_mse = 0
    total_elements = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item()

            if multi_label:
                preds = (logits > 0).float()
                matches = (preds == labels).float().sum().item()
                total_elements += labels.numel()
                running_correct += matches
            else:
                preds = logits.argmax(dim=1)
                matches = (preds == labels).sum().item()
                total_elements += labels.size(0)
                running_correct += matches
                
            # MSE metric
            mse_val = mse_metric(logits, labels).item()
            running_mse += mse_val

    avg_loss = running_loss / len(loader)
    epoch_mse = running_mse / len(loader)
    avg_acc = 100.0 * running_correct / total_elements
    return avg_loss, epoch_mse, avg_acc

##############################################################################
# 5) Main script
##############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--overlapped', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_model', type=str, default=None,
                        help="Path to save model (e.g. 'model.pth')")
    args = parser.parse_args()
    
    #chooses between CUDA and CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data transforms
    transform = transforms.ToTensor()

    # Download MNIST
    mnist_train = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    mnist_test  = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    # Decide multi_label vs single_label
    if args.overlapped:
        print('Going in the overlapped image training...')
        train_dataset = OverlappedMultiLabelMNIST(mnist_train)
        test_dataset  = OverlappedMultiLabelMNIST(mnist_test)
        criterion = nn.BCEWithLogitsLoss()
        multi_label = True
    else:
        print('Going in the single image training...')
        train_dataset = mnist_train
        test_dataset  = mnist_test
        criterion = nn.CrossEntropyLoss()
        multi_label = False

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    # Model & optimizer
    model = SimpleMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Lists to store metrics
    train_losses, train_accs, train_mses = [], [], []
    val_losses, val_accs, val_mses = [], [], []

    # Training loop
    for epoch in range(args.epochs):
        train_loss, train_mse, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, multi_label)
        val_loss, val_mse, val_acc = evaluate(model, test_loader, criterion, device, multi_label)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_mses.append(train_mse)
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_mses.append(val_mse)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"| Train Loss: {train_loss:.4f},  Train MSE: {train_mse:.4f}, Acc: {train_acc:.2f}% "
              f"| Val Loss: {val_loss:.4f}, Val MSE: {val_mse:.4f}, Acc: {val_acc:.2f}%")

    # Optionally save the model
    if args.save_model is not None:
        torch.save(model.state_dict(), args.save_model)
        print(f"Model saved to {args.save_model}")

    # --------------------------------------------------------------------
    # PLOT TRAINING & VALIDATION METRICS
    # --------------------------------------------------------------------

    epochs_range = range(1, args.epochs+1)
    
    plt.figure(figsize=(12,5))
    
    # Subplot 1: BCE Loss
    plt.subplot(1,3,1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.title('BCE Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Subplot 2: MSE
    plt.subplot(1,3,2)
    plt.plot(epochs_range, train_mses, label='Train MSE')
    plt.plot(epochs_range, val_mses, label='Val MSE')
    plt.title('MSE Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    
    # Subplot 3: Accuracy
    plt.subplot(1,3,3)
    plt.plot(epochs_range, train_accs, label='Train Acc')
    plt.plot(epochs_range, val_accs, label='Val Acc')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
