# Grad-CV-I_Final-Project
Repo for Final Project of Grad Computer Vision I

Group 2: 
- Gurpreet Singh 
- Logan Sehr 
- Chinmaykumar Brahmbhatt

## Dataset
[Ships from Satellite](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery)
- 4000 total samples
    - 1000 Valid
    - 3000 Invalid
    - Imbalanced
- Binary object classification task
- High inter-class variance (cropped images, partial representations, etc)
- Works in conjunction with [Planet Developer Center](https://developers.planet.com/) API Key for additional scene context
- Benefit: "Automating this process can be applied to many issues including monitoring port activity levels and supply chain analysis."

## Preprocessing
### A: Image Augmentation
- Random rotations (0-360 degrees)
- Random Horizontal Flip
- Random Vertical Flip
- Gaussian Blur (kernel size = (7, 7))

### B: Image Padding
- Supplement the image label pairs with augmentations to match the desired quantity
- Desired total size of dataset size: 8000 samples
  - Ship: 4000 samples (Valid)
  - Non-Ship: 4000 samples (Invalid)

### C: Dataset Split
- Train: 70% $\rightarrow$  5600
- Validation: 10% $\rightarrow$ 800
- Test: 20% $\rightarrow$ 1600


## Task
Binary object classification on ships within the San Francisco Port

## Approach

### A: Transformer Based Neural Network
- Self-Attention Head
- Encoder Only 
- Patched input 
- Sigmoid Activation 

### B: CNN with K Nearest Neighbors 
- Simple
- Fast
- Lightweight
- Baseline Model

Model Architecture:
- Stem Portion
  - Flatten
  - FC Linear Layer from 80 $\times$ 80 $\rightarrow$ 64
  - BatchNorm1d(64)
  - ReLU Activation
- Inception Module with Custom Aggreation (1)
  - 1x1 Convolutional Layer $\rightarrow$ 3x3 Convolutional Layer 
  - 1x1 Convolutional Layer $\rightarrow$ 5x5 Convolutional Layer 
  - Max Pooling Layer 3x3 $\rightarrow$ 1x1 Convolution Layer

![Inception Module](https://d2l.ai/_images/inception.svg)

Produces a feature map to be used within the K Nearest Neighbor Model.
- Euclidean Distance 
- KNN Classifier with K=2 (Binary classification)

## Results (Abstract)