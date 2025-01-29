# MNIST-overlapped-NN
 This repository contains a 'hello world' version of the MNIST image classification with single and multiple digit overlapped on top of each other.


Train and evaluate a simple MLP on:

1. **Standard MNIST** (single-label classification).
2. **Overlapped MNIST** (multi-label classification).  

In **Overlapped MNIST**, each sample is formed by picking **two random MNIST digits** and averaging their pixel intensities. This creates a **multi-label** classification problem: each overlapped image can contain 0–2 digit labels simultaneously.

---

## Table of Contents
1. [Introduction](#introduction) 
2. [Reason for BCE](#BCEWithLogitsLoss) 
3. [Project Structure](#project-structure)  
4. [Requirements](#requirements)  
5. [Usage](#usage)  
   - [Train](#train)  
   - [Evaluate](#evaluate)  
6. [Results and Figures](#results-and-figures)

---

## Introduction

This repository contains two core scripts:

- `train.py`: Train a **simple MLP** (three hidden layers) to classify either standard MNIST digits (with **cross-entropy loss**) or overlapped MNIST (with **BCEWithLogitsLoss** and multi-hot labels).
- `evaluate.py`: Load a saved model and run evaluation or visualize random predictions from the test set.



The main difference between **standard** and **overlapped** MNIST is the dataset:
- **Standard MNIST**: single-label classification (digits 0–9).
- **Overlapped MNIST**: each image is the average of two random digits, yielding a **multi-label** problem where each image can contain **two** of the digits (0–9).


## What is Binary Cross-Entropy (BCE) with Logits Loss and why?

When dealing with **multi-label** problems [ref_here](https://openaccess.thecvf.com/content/CVPR2023/papers/Kobayashi_Two-Way_Multi-Label_Loss_CVPR_2023_paper.pdf) (e.g., an image might contain multiple classes simultaneously), we treat each class as an independent binary decision. Instead of the softmax function, we use the **sigmoid** function on each logit:

$$
\sigma(z_i) = \frac{1}{1 + \exp(-z_i)} \quad \text{for each class } i.
$$

Given a **multi-hot** ground-truth vector, the **Binary Cross-Entropy (BCE)** loss for each class \(i\) is:

$$
\text{BCE}(z_i, y_i) = -y_i \log \bigl(\sigma(z_i)\bigr) - (1 - y_i)\log \bigl(1 - \sigma(z_i)\bigr).
$$

We typically **average** over all \(K\) classes:

```math
\text{BCE\_with\_Logits}\bigl(\mathbf{z}, \mathbf{y}\bigr)=
\frac{1}{K} \sum_{i=1}^{K}
\Bigl[
-\,y_i \log\bigl(\sigma(z_i)\bigr)
- \bigl(1 - y_i\bigr)\log\bigl(1 - \sigma(z_i)\bigr)
\Bigr].
```

In most modern frameworks (e.g., PyTorch), we can use a numerically stable variant called **BCEWithLogitsLoss**, which takes **raw logits** and applies the sigmoid internally to avoid numerical underflow or overflow. This makes **BCE** well-suited for **multi-label** classification, where each class is independently “on” or “off” rather than exactly one class per sample.


---

## Project Structure

```bash
.
├── overlapped_dataset.py     # OverlappedMultiLabelMNIST dataset definition
├── train.py                  # Main training script
├── evaluate.py               # Script for model evaluation and visualization
├── README.md                 # This file
└── data/                     # MNIST data downloaded automatically
```
--
##  Usage
**Train**

By default, train.py trains on standard single-label MNIST with CrossEntropyLoss:

```bash

python train.py --epochs 5 --save_model mnist_standard.pth
```
Key arguments:

--overlapped: Use overlapped (multi-label) MNIST instead of single-label MNIST.

--epochs: Number of epochs to train (default: 10).

--batch_size: Batch size (default: 64).

--lr: Learning rate (default: 0.001).

--save_model: Path to save the final model weights (e.g., model.pth).

Example: train on overlapped MNIST for 10 epochs and save:

```bash

python train.py --overlapped --epochs 10 --save_model overlapped_mnist.pth
```
## Evaluate
Use evaluate.py to load a saved model and compute accuracy on the test set. It will also display a few random samples with predicted vs. ground truth labels.


Overlapped MNIST example:

```bash
python evaluate.py --model_path overlapped_mnist.pth --overlapped

```

Standard MNIST example:

```bash

python evaluate.py --model_path mnist_standard.pth
```


Key arguments:

--model_path: The .pth file where your model weights were saved.

--overlapped: Specifies that the model was trained on overlapped multi-label MNIST.

## Results and Figures
**Training Curves**


During training, the script logs training and validation losses and accuracies at each epoch. Below is an example loss curve (left), mse curve (center) and accuracy curve (right) for a model trained on overlapped MNIST with epochs = 30:

![alt text](figures/loss_curve.PNG 'Loss curve')


When you run the evaluate.py script, it will randomly display images from the test set and print the true vs. predicted labels.

The resulting sample image for overlapped MNIST digits with true digits and predicted digits are the following:

![alt text](figures/overlapped_image.PNG 'sample overlapped mnist with pred and true')


## Reference:
1) [Kobayashi, T. (2023). Two-way multi-label loss. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 7476-7485).](https://openaccess.thecvf.com/content/CVPR2023/papers/Kobayashi_Two-Way_Multi-Label_Loss_CVPR_2023_paper.pdf)
