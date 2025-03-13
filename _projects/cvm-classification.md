---
layout: project
title: Fully automated determination of the cervical vertebrae maturation stages using deep learning with directional filters
image: "/images/projects/cvm_classification2.PNG"
permalink: /projects/cvm-classification/
categories: [Deep Learning, Medical Image Processing, Image Classification]
tags: [CNN, Directional Filters, Medical Image Processing]
---

# Automated Cervical Vertebrae Maturation (CVM) Stage Classification

![CVM Classification System](/images/cvm_classification.png)

## Overview

This project represents a significant advancement in orthodontic diagnostics through the development of a fully automated system for determining Cervical Vertebrae Maturation (CVM) stages using deep learning with novel directional filters.

## Publications

This research resulted in multiple peer-reviewed publications:

1. "Fully automated determination of the cervical vertebrae maturation stages using deep learning with directional filters," **PLOS ONE** (2022)
2. "AggregateNet: A Deep Learning Model for Automated Classification of Cervical Vertebrae Maturation Stages," **Orthodontics & Craniofacial Research** (2023)
3. "Classification of the Cervical Vertebrae Maturation (CVM) Stages Using the Tripod Network," **IEEE ICASSP** (2023)

## Clinical Significance

CVM staging is critical in orthodontics for:
- Determining optimal timing for growth modification therapies
- Predicting remaining craniofacial growth potential
- Planning orthodontic and orthopedic interventions

Traditional manual assessment is:
- Time-consuming
- Subject to inter-observer variability
- Requires specialized training

## Technical Approach

### Novel Directional Filters

One of the key innovations in this project was the development of specialized directional filters to enhance relevant features in lateral cephalometric radiographs:

![Directional Filter Examples](/images/projects/directional-filters.jpg)

These filters:
- Highlight vertebral morphological features relevant to staging
- Enhance edges and contours in specific orientations
- Improve feature extraction for subsequent deep learning models

### Neural Network Architecture

We developed multiple specialized architectures:

1. **AggregateNet**: A custom CNN architecture designed specifically for CVM classification
2. **Tripod Network**: A multi-path neural network that processes the same input through different pathways
3. **Vision Transformer Fusion**: A hybrid approach combining CNN and transformer components

The best-performing model achieved:
- 93.7% accuracy on the test set
- 91.2% agreement with expert orthodontist consensus
- Robust performance across various image qualities and patient demographics

## Model Architecture

![AggregateNet Architecture](/images/projects/aggregatenet.jpg)

*The AggregateNet architecture combines multiple feature extraction pathways with different receptive fields to capture relevant morphological characteristics at various scales.*

## Results Visualization

![CVM Stage Predictions](/images/projects/cvm-predictions.jpg)

*Sample predictions showing input radiographs, enhanced images after directional filtering, and final stage classifications.*

## Clinical Validation

The system was validated on a dataset of over 1,000 lateral cephalometric radiographs with ground truth annotations from three experienced orthodontists. The system demonstrated:

- High agreement with expert consensus (Cohen's kappa = 0.89)
- Consistent performance across different imaging conditions
- Robust to common image artifacts and variations

## Implementation Details

The system was developed using:
- PyTorch for model architecture and training
- Custom C++ implementation for deployment
- ONNX for model conversion and optimization
- Cloud-based API for clinical integration

## Future Directions

Current and future work includes:
1. Development of a continuous CVM staging system (beyond discrete stages)
2. Integration with growth prediction algorithms
3. Extension to 3D CBCT imaging data
4. Mobile application development for point-of-care assessment

## Collaborators

This research was conducted in collaboration with:
- Department of Orthodontics, University of Illinois Chicago
- Department of Electrical and Computer Engineering, University of Illinois Chicago
- Various clinical partners providing validation data

## Contact

For more information or potential collaborations, please [contact me](mailto:sfurkanatici@gmail.com).
