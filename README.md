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
Apply one of each to each image
- Random rotations (0-360 degrees)
- Random Horizontal Flip
- Random Vertical Flip
- Gaussian Blur (kernel size = (3, 3))
- Color Jitter

### B: Image Padding
- Supplement the image label pairs with augmentations to match the desired quantity
- Desired total size of train set size: 6K samples
  - Ship: 650 Samples $\rightarrow$ 3K samples (+2.65K Samples)  (Valid)
  - Non-Ship: 2.65K Samples $\rightarrow$ 3K Samples (+350 Samples) (Invalid)

### C: Dataset Split
- Train: 6K Samples
- Validation: 200 Samples
- Test: 500 Samples 
- Total Size: 6.7K Sampples

## Task
Binary object classification on ships within the San Francisco Port

## Approach

### A: Transformer Based Neural Network
- Self-Attention Head
- Encoder Only 
- Patched input 
- Sigmoid Activation (within Loss Function, implementation performance nuance in PyTorch)

Architecture:
- Same as ViT-Base from ["An Image with Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/pdf/2010.11929)
- Image Patch Size: 10x10
- Number of Transformer Layers: 6
- Number of Attention Heads: 2
- Hidden Dimension: 768
- MLP Dimension: 3072
- Hyperameters:
  - Batch Size: 64
  - Epochs: 100
  - Learning Rate: 0.0001
  - Attention Dropout: 0.1
  - Dropout: 0.1

Code sourced in `transformer_sandbox.ipynb` and can be run using the driver code `main.py`

Driver code usage:
  - `python main.py --model transformer`
  - Results saved in `results/transformer`
    - Metrics
    - Saved Model
    - Performance plots

### B: CNN with K Nearest Neighbors 
- Simple
- Fast
- Lightweight
- Baseline Model

- Pretrained ResNet 18:
  - kNN Classifier with k=2
  - SVM Classifier

Code sourced in `cnn.ipynb`.

## Results (Abstract)
Presented in provided PDF Paper.