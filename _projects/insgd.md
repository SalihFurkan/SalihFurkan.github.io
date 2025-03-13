---
layout: project
title: "Input Normalized Stochastic Gradient Descent Training for Deep Neural Networks"
abstract: "A novel optimization algorithm improving CNN performance using input features."
image: "/images/projects/INSGD_diag2.png"
permalink: /projects/insgd/
categories: [Deep Learning, Optimization]
tags: [CNN, SGD, Machine Learning, Neural Networks]
---

# Input Normalized Stochastic Gradient Descent Training for Deep Neural Networks

**Authors:** Salih Atici, Hongyi Pan, Ahmet Enis Cetin (University of Illinois Chicago)
**Contact:** \{hpan21, satici2, aecyy\}@uic.edu

## Abstract
In this paper, we propose Input Normalized Stochastic Gradient Descent (INSGD), a novel optimization algorithm inspired by the Normalized Least Mean Squares (NLMS) algorithm from adaptive filtering. INSGD applies $\ell_1$ and $\ell_2$-based normalizations to the learning rate in stochastic gradient descent, using the input vector to the neuron rather than the error term. Tested on ResNet-20, Vision Transformer, MobileNetV3, WResNet-18, ResNet-50, and a custom CNN, INSGD improves accuracy on benchmark datasets: ResNet-20 on CIFAR-10 from 92.57% to 92.67%, MobileNetV3 on CIFAR-10 from 90.83% to 91.13%, WResNet-18 on CIFAR-100 from 78.24% to 78.47%, and ResNet-50 on ImageNet-1K from 75.60% to 75.92%.

## Introduction
Deep Neural Networks (DNNs) excel in supervised learning tasks ([LeCun et al., 1995](#lecun1995convolutional)). Stochastic Gradient Descent (SGD) is a cornerstone optimizer ([Bottou, 2010](#bottou2010large)), but its performance hinges on learning rate tuning. Alternatives like Adam ([Kingma & Ba, 2014](#kingma2014adam)) offer adaptability but risk divergence. Inspired by NLMS ([Mathews & Xie, 1993](#mathews1993stochastic)), we propose INSGD to enhance robustness and convergence.

### Stochastic Gradient Descent
SGD updates weights iteratively:
$$
\mathbf{w}(k+1) = \mathbf{w}(k) - \lambda \nabla_{\mathbf{w}(k)} L(\mathbf{y}, f(\mathbf{x},\mathbf{w}))
$$
It excels with convex problems but struggles with deep networks ([Li et al., 2019](#li2019convergence)).

### Normalized Least Mean Squares (NLMS)
NLMS normalizes updates for better convergence:
$$
\mathbf{w}(j) = \mathbf{w}(j-1) + \lambda \frac{\mathbf{e}(j)}{||\mathbf{u}(j)||^2_2} \mathbf{u}(j)
$$
This motivates INSGD’s input-based normalization ([Theodoridis & Mandic, 2010](#theodoridis2010adaptive)).

## Related Works
SGD variants like AdaGrad ([Duchi et al., 2011](#duchi2011adaptive)), RMSProp ([Hinton et al., 2012](#hinton2012neural)), Adam ([Kingma & Ba, 2014](#kingma2014adam)), LARS ([You et al., 2017](#you2017large)), and LSALR ([Singh et al., 2015](#singh2015layer)) adapt learning rates. INSGD uniquely normalizes using input vectors.

## Methodology
### Input Normalized Stochastic Gradient Descent Algorithm
INSGD normalizes gradients with input vectors:
$$
\mathbf{w}_{k+1} = \mathbf{w}_k - \mu \frac{\nabla_{\mathbf{w}_k}L(\mathbf{e}_k)}{\epsilon + ||\mathbf{x}_k||^2_2}
$$
For non-linear neurons:
$$
\mathbf{w}_{i,1} = \mathbf{w}_{i,0} + \lambda \frac{(\phi(d_i) - \mathbf{w}_{i,0} \cdot \mathbf{x}_k)}{\epsilon + ||\mathbf{x}_k||^2} \mathbf{x}_k
$$
Input momentum $P_k$ stabilizes normalization:
$$
P_k = \beta P_{k-1} + (1-\beta)||\mathbf{x}_{k}||_2^2
$$
Final update:
$$
\mathbf{w}_{k+1} = \mathbf{w}_k - \frac{\mu}{f_{\epsilon}(\log(P_k))} \nabla_{\mathbf{w}_k}L(\mathbf{e}_k)
$$
where $f_{\epsilon}(u)$ clips values below $\epsilon = 0.01$.

![INSGD Algorithm Diagram](/images/projects/alg_diag.png)  
*Figure 1: INSGD algorithm across layers.*

#### Algorithm 1: INSGD with Momentum
For t = 1 to ...
    g_t ← ∇θ f_t(θ{t-1})           # Compute gradient
    If β ≠ 0:                        # If input momentum is enabled
        If t > 1:
            P_t ← β P_{t-1} + (1-β) ||x_{t,θ}||2^2  # Update input power
        Else:
            P_t ← ||x{t,θ}||2^2
        EndIf
    EndIf
    g_t ← g_t / f(log(P_t))          # Normalize gradient
    If λ ≠ 0:                        # Apply weight decay
        g_t ← g_t + λ θ{t-1}
    EndIf
    If γ ≠ 0:                        # Apply gradient momentum
        If t > 1:
            b_t ← γ b_{t-1} + (1-τ) g_t
        Else:
            b_t ← g_t
        EndIf
        g_t ← b_t
    EndIf
    θ_t ← θ_{t-1} - μ g_t            # Update weights
EndFor


### Models Architecture
We evaluated INSGD on:
- **ResNet-20**: CIFAR-10 ([He et al., 2016](#he2016deep)).
- **MobileNetV3**: CIFAR-10, ImageNet-1K ([Howard et al., 2019](#mobilenetv3)).
- **Vision Transformer (ViT)**: CIFAR-10, patch size 4, depth 6 ([Dosovitskiy et al., 2020](#dosovitskiy2020image)).
- **WResNet-18**: CIFAR-100 ([Zagoruyko & Komodakis, 2016](#wideresnet)).
- **ResNet-50**: ImageNet-1K.
- **Custom CNN**: 4 conv layers for CIFAR-10.

## Experimental Results
Experiments used NVIDIA GTX 1660 Ti (CIFAR-10) and RTX A6000 (CIFAR-100, ImageNet-1K).

### CIFAR-10 Classification
- **Setup**: 200 epochs, batch size 128, SGD baseline (weight decay 0.0005, momentum 0.9), data augmentation (padding, flips, normalization).
- **ResNet-20**: INSGD-$\ell_1$ hit 92.67% vs. SGD’s 92.57%.

| Optimizer      | Initial LR | Test Accuracy |
|----------------|------------|---------------|
| SGD            | 0.1        | 92.57±0.11%   |
| Adam           | 0.001      | 91.34±0.01%   |
| Adagrad        | 0.1        | 89.41±0.01%   |
| INSGD-$\ell_1$ | 0.1        | 92.67±0.13%   |
| INSGD-$\ell_2$ | 0.1        | 92.60±0.11%   |

![Convergence Plot](/images/projects/convergence.png)  
*Figure 2: Test loss convergence over 200 epochs.*

- **Batch Size Impact**: INSGD maintained performance across 128, 256, and 512.

| Optimizer      | Batch | LR   | Test Accuracy |
|----------------|-------|------|---------------|
| SGD            | 128   | 0.1  | 92.57±0.11%   |
| INSGD-$\ell_1$ | 128   | 0.1  | 92.67±0.13%   |
| INSGD-$\ell_2$ | 128   | 0.1  | 92.60±0.11%   |
| SGD            | 512   | 0.2  | 92.25±0.26%   |
| INSGD-$\ell_1$ | 512   | 0.2  | 92.38±0.21%   |

- **MobileNetV3**: Improved from 93.06% to 93.10%.

| Optimizer      | Initial LR | Test Accuracy |
|----------------|------------|---------------|
| SGD            | 0.05       | 93.06±0.11%   |
| INSGD-$\ell_2$ | 0.05       | 93.10±0.01%   |

- **Custom CNN**: Robust at varying LRs.

| Optimizer      | Initial LR | Test Accuracy |
|----------------|------------|---------------|
| SGD            | 0.25       | 65.92±1.32%   |
| INSGD-$\ell_1$ | 0.25       | 73.24±1.17%   |

- **No Batch Norm**: INSGD stabilized training.

| Model       | Optimizer      | Initial LR | Test Accuracy |
|-------------|----------------|------------|---------------|
| Custom CNN  | SGD            | 0.1        | 11.07±2.74%   |
| Custom CNN  | INSGD-$\ell_2$ | 0.1        | 74.78±7.84%   |
| ResNet-20   | INSGD-$\ell_2$ | 0.1        | 90.70±0.15%   |

- **Vision Transformer**: Competitive with Adam.

| Optimizer      | LayerNorm | LR    | Test Accuracy |
|----------------|-----------|-------|---------------|
| Adam           | Yes       | 0.0005| 85.38±0.44%   |
| INSGD-$\ell_1$ | Yes       | 0.05  | 84.52±0.43%   |
| INSGD-$\ell_1$ | No        | 0.05  | 83.50±0.18%   |

### CIFAR-100 Experiment
- **Setup**: WResNet-18, 200 epochs, batch size 256.
- **Results**: INSGD-$\ell_2$ reached 78.47% vs. SGD’s 78.24%.

| Optimizer      | Batch | Top-1 Acc.  | Top-5 Acc.  |
|----------------|-------|-------------|-------------|
| SGD            | 128   | 78.24±0.43%| 94.31±0.09%|
| INSGD-$\ell_2$ | 128   | 78.47±0.16%| 94.39±0.20%|

### ImageNet-1K Results
- **Setup**: ResNet-50, 90 epochs, batch size 256, LR 0.1 with 1/10 reduction every 30 epochs.
- **Results**: INSGD-$\ell_1$ achieved 75.92% vs. SGD’s 75.60%.

| Optimizer      | LR   | Top-1 Acc.  | Top-5 Acc.  |
|----------------|------|-------------|-------------|
| SGD            | 0.1  | 75.60±0.14%| 92.67±0.12%|
| INSGD-$\ell_1$ | 0.1  | 75.92±0.43%| 92.75±0.10%|
| INSGD-$\ell_2$ | 0.1  | 75.90±0.14%| 92.81±0.04%|

## Conclusion
INSGD integrates NLMS-inspired input normalization into SGD, improving accuracy across CIFAR-10, CIFAR-100, and ImageNet-1K. Its flexibility in hyperparameter tuning and potential for online learning suggest broad applicability. Future work could explore real-time processing scenarios.

## References
- <a name="lecun1995convolutional"></a>LeCun et al., "Convolutional Networks," 1995.
- <a name="he2016deep"></a>He et al., "Deep Residual Learning," 2016.
- <a name="bottou2010large"></a>Bottou, "Large-Scale Machine Learning," 2010.
- <a name="kingma2014adam"></a>Kingma & Ba, "Adam," 2014.
- <a name="mathews1993stochastic"></a>Mathews & Xie, "Stochastic NLMS," 1993.
- <a name="li2019convergence"></a>Li et al., "Convergence of SGD," 2019.
- <a name="theodoridis2010adaptive"></a>Theodoridis & Mandic, "Adaptive Filtering," 2010.
- <a name="duchi2011adaptive"></a>Duchi et al., "AdaGrad," 2011.
- <a name="hinton2012neural"></a>Hinton et al., "RMSProp," 2012.
- <a name="you2017large"></a>You et al., "LARS," 2017.
- <a name="singh2015layer"></a>Singh et al., "LSALR," 2015.
- <a name="mobilenetv3"></a>Howard et al., "MobileNetV3," 2019.
- <a name="dosovitskiy2020image"></a>Dosovitskiy et al., "Vision Transformer," 2020.
- <a name="wideresnet"></a>Zagoruyko & Komodakis, "Wide Residual Networks," 2016.

*Full bibliography in the [original PDF](https://openreview.net/forum?id=5TaBxctwRZ).*
