# MNIST-overlapped-NN
 This repository contains a 'hello world' version of the MNIST image classification with single and multiple digit overlapped on top of each other.


Train and evaluate a simple MLP on:

1. **Standard MNIST** (single-label classification).
2. **Overlapped MNIST** (multi-label classification).  

In **Overlapped MNIST**, each sample is formed by picking **two random MNIST digits** and averaging their pixel intensities. This creates a **multi-label** classification problem: each overlapped image can contain 0–2 digit labels simultaneously.

---

## Table of Contents
1. [Introduction](#introduction)  
2. [Project Structure](#project-structure)  
3. [Requirements](#requirements)  
4. [Usage](#usage)  
   - [Train](#train)  
   - [Evaluate](#evaluate)  
5. [Results and Figures](#results-and-figures)

---

## Introduction

This repository contains two core scripts:

- `train.py`: Train a **simple MLP** (three hidden layers) to classify either standard MNIST digits (with **cross-entropy loss**) or overlapped MNIST (with **BCEWithLogitsLoss** and multi-hot labels).
- `evaluate.py`: Load a saved model and run evaluation or visualize random predictions from the test set.

The main difference between **standard** and **overlapped** MNIST is the dataset:
- **Standard MNIST**: single-label classification (digits 0–9).
- **Overlapped MNIST**: each image is the average of two random digits, yielding a **multi-label** problem where each image can contain **two** of the digits (0–9).

---

## Project Structure

```bash
.
├── overlapped_dataset.py     # OverlappedMultiLabelMNIST dataset definition
├── train.py                  # Main training script
├── evaluate.py               # Script for model evaluation and visualization
├── README.md                 # This file
└── data/                     # MNIST data downloaded automatically

--
##  Usage
Train

By default, train.py trains on standard single-label MNIST with CrossEntropyLoss:

bash
Copy
Edit
python train.py --epochs 5 --save_model mnist_standard.pth
Key arguments:

--overlapped: Use overlapped (multi-label) MNIST instead of single-label MNIST.
--epochs: Number of epochs to train (default: 10).
--batch_size: Batch size (default: 64).
--lr: Learning rate (default: 0.001).
--save_model: Path to save the final model weights (e.g., model.pth).
Example: train on overlapped MNIST for 10 epochs and save:

bash
Copy
Edit
python train.py --overlapped --epochs 10 --save_model overlapped_mnist.pth
Evaluate
Use evaluate.py to load a saved model and compute accuracy on the test set. It will also display a few random samples with predicted vs. ground truth labels.

Standard MNIST example:

bash
Copy
Edit
python evaluate.py --model_path mnist_standard.pth
Overlapped MNIST example:


python evaluate.py --model_path overlapped_mnist.pth --overlapped
Key arguments:

--model_path: The .pth file where your model weights were saved.
--overlapped: Specifies that the model was trained on overlapped multi-label MNIST.
Results and Figures
Training Curves
During training, the script logs training and validation losses and accuracies at each epoch. Below is an example accuracy curve (left) and loss curve (right) for a model trained on overlapped MNIST: