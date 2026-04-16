# Deep Learning Assignment 1 Report

## Overview

This project builds a fully connected neural network **from scratch** (without using deep learning frameworks) to classify handwritten digits from the MNIST dataset.
The network is implemented in a fully vectorized manner, following the conventions introduced by Andrew Ng.

The notebook also presents two comparison experiments:

1. Training **with and without Batch Normalization** (Section 5).
2. Training **with and without L2 regularization** (Section 6).

---

## Project Structure

### Section 1 - Forward Propagation

Implemented the main forward-pass components:

* Initialization of parameters (`W`, `b`)
* Linear computation (`Z = WA + b`)
* Activation functions (`ReLU`, `Softmax`)
* Full forward pass across all layers
* Cost calculation using **categorical cross-entropy**
* Optional batch normalization module

### Section 2 - Backward Propagation

Implemented the backward-pass components:

* Gradients for the linear step (`dA_prev`, `dW`, `db`)
* Backward functions for `ReLU` and `Softmax`
* Full backward propagation across the network
* Parameter updates using gradient descent

### Section 3 - Training Loop

Implemented:

* `l_layer_model(...)` for iterative mini-batch training
* Validation-based monitoring resembling early stopping
* `predict(...)` for evaluating accuracy on train, validation, and test sets

### Section 4 - MNIST Baseline Experiment (BatchNorm OFF)

* Loaded MNIST using `fetch_openml`
* Normalized inputs to the range `[-1, 1]`
* Converted labels to one-hot encoding
* Trained the baseline model and reported accuracy and loss trends

### Section 5 - BatchNorm Experiment

Repeated the baseline experiment with batch normalization enabled, and compared results against the baseline.

### Section 6 - L2 Regularization Experiment

Extended the model to include L2 regularization in both the loss and parameter updates, and compared results against the baseline.

---

## Model Architecture and Parameters

**Layer dimensions:** `[784, 20, 7, 5, 10]`

* Input layer: 784 (flattened 28×28 image)
* Hidden layers: 20, 7, 5 (ReLU activation)
* Output layer: 10 classes (Softmax)

**Training configuration:**

* Learning rate: `0.009`
* Batch size: `64`
* Maximum iterations: `30,000`
* Early stopping window: `100` steps
* BatchNorm: static - without learnable parameters
* L2 regularization: `lambda = 0.1`

**Training duration:**

* Iterations: `30,000`
* Epochs: `43`

---

## Baseline Training (Section 4)

* No BatchNorm
* No L2 regularization

### Training duration

* Iterations: `30,000`
* Epochs: `43`

### Accuracy Results

| Model                   |  Train | Validation |   Test |
| ----------------------- | -----: | ---------: | -----: |
| Baseline (No BatchNorm) | 93.7% |     92.2% | 92.4% |

### Discussion

The baseline model achieved strong performance on the MNIST dataset with 92.4% test accuracy. The model was trained without batch normalization or L2 regularization, serving as baseline. The training curve shows steady convergence with validation loss stabilizing after approximately 30,000 iterations (43 epochs), indicating effective learning dynamics and appropriate early stopping criteria.

### Figure

![Section 4 - Train and validation loss](<figures/section 4 train and validation loss.png>)

---

## Batch Normalization Comparison (Section 5)

### Training duration

* Iterations: `30,000`
* Epochs: `43`

### Accuracy Results

| Model                   |  Train | Validation |   Test |
| ----------------------- | -----: | ---------: | -----: |
| Baseline (No BatchNorm) | 93.7% |     92.2% | 92.4% |
| BatchNorm Enabled       | 91.3% |     90.6% | 90.6% |

### Discussion

Enabling Batch Normalization resulted in lower accuracy across all datasets compared to the baseline. This may be because the normalization was static and not learnable, forcing activations into a fixed distribution (zero mean and unit variance), limiting the model’s flexibility.

### Figure

![Section 5 - Loss with and without BatchNorm](<figures/section 5 loss with and witout bachnorm.png>)

---

## Regularization Comparison (Section 6)

### Training duration

* Iterations: `30,000`
* Epochs: `43`

### Accuracy Results

| Model                            |  Train | Validation |   Test |
| -------------------------------- | -----: | ---------: | -----: |
| Baseline (No L2)                 | 93.7% |     92.2% | 92.4% |
| L2 Regularization (`lambda=0.1`) | 94.2% |     93.0% | 93.2% |

### Discussion

Applying L2 regularization improved training, validation, and test accuracy, indicating enhanced generalization. It also reduced weight magnitudes by penalizing large values, encouraging a more constrained and stable model. This effect helps mitigate overfitting and improves overall robustness.

### Figure

![Section 6 - Loss with and without L2 regularization](<figures/section 6 loss with and witout L2 regularization.png>)