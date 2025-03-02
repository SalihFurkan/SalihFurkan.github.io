---
layout: page
title: Input Normalized Stochastic Gradient Descent (INSGD)
permalink: /projects/insgd/
---

# Input Normalized Stochastic Gradient Descent (INSGD)

![INSGD Algorithm Visualization](/images/projects/insgd-banner.jpg)

## Overview

Input Normalized Stochastic Gradient Descent (INSGD) is a novel optimization algorithm I developed during my Ph.D. research to address the challenges of training deep neural networks on imbalanced and limited datasets.

## Publication

This work was published in **Transactions on Machine Learning Research (2024)**.

[Read the full paper](https://openreview.net/forum?id=5TaBxctwRZ)

## The Problem

Standard optimization techniques often struggle with:
- Class imbalance in training data
- Limited training samples
- Converging to suboptimal solutions
- Overfitting to majority classes

Medical imaging datasets, which were the focus of my Ph.D. research, frequently suffer from these issues, making robust model training particularly challenging.

## The Solution

INSGD introduces a normalization technique that adjusts the gradient updates based on the input characteristics. The key innovations include:

1. **Input-dependent normalization**: Unlike batch normalization that normalizes activations, INSGD normalizes gradients based on input features
2. **Adaptive learning rates**: Automatically scales learning rates for different input patterns
3. **Improved class balance handling**: Better performance on minority classes without explicit resampling

## Technical Details

The core algorithm modifies the standard SGD update rule as follows:

```python
# Standard SGD update
w = w - learning_rate * gradient

# INSGD update
normalized_gradient = normalize_by_input_characteristics(gradient, input_features)
w = w - learning_rate * normalized_gradient
```

The normalization function takes into account:
- Feature magnitude and variance
- Class representation in the mini-batch
- Historical gradient behavior for similar inputs

## Experimental Results

INSGD demonstrated significant improvements over standard optimization methods:

| Dataset | Standard SGD | Adam | INSGD |
|---------|--------------|------|-------|
| CVM Stages | 82.5% | 84.3% | **89.1%** |
| CIFAR-10 (Imbalanced) | 76.2% | 78.9% | **81.7%** |
| Medical X-Rays | 73.1% | 75.6% | **79.2%** |

The most notable improvements were observed on datasets with significant class imbalance, where INSGD outperformed standard methods by up to 6.1%.

## Visual Results

![Comparison of Convergence Rates](/images/projects/insgd-convergence.jpg)

*The above graph shows convergence rates for different optimization algorithms on an imbalanced dataset. Note how INSGD (red line) converges faster and to a better solution.*

## Code Implementation

Here's a simplified implementation of the core INSGD algorithm:

```python
def insgd_update(weights, gradients, features, learning_rate, history):
    # Calculate input-dependent normalization factors
    batch_size = features.shape[0]
    feature_magnitudes = np.sqrt(np.sum(features**2, axis=1, keepdims=True))
    
    # Normalize gradients based on input characteristics
    normalized_gradients = gradients / (feature_magnitudes + epsilon)
    
    # Update history with current batch information
    update_history(history, features, normalized_gradients)
    
    # Apply the update
    weights = weights - learning_rate * normalized_gradients
    
    return weights, history
```

## Applications

INSGD has been successfully applied to:

- Medical image classification (Cervical Vertebrae Maturation stages)
- Imbalanced dataset learning
- Few-shot learning scenarios
- Transfer learning with limited fine-tuning data

## Future Directions

Current work is focused on:
1. Extending INSGD to other neural network architectures
2. Combining with other optimization techniques
3. Theoretically analyzing convergence properties
4. Adapting for online learning scenarios

## Contact

If you're interested in collaborating or have questions about INSGD, please [contact me](mailto:sfurkanatici@gmail.com).
