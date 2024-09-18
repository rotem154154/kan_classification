[![Website](https://img.shields.io/badge/Website-Visit%20Site-blue)](https://rotem154154.github.io/)

# KAN Architectures for Dimensionality-Reduced Image Classification

![alt text](https://github.com/rotem154154/kan_classification/blob/main/optimized_model_performance.png?raw=true "Title")

The new **KAN** architecture showcases promising results on toy tasks, and our goal is to extend its application to real-world challenges. This repository features implementations and experiments with various KAN-based models, all aimed at classifying images using feature vectors extracted from the foundation model DinoV2. We employ neural architecture search (NAS) to finely tune these models, striving to optimize their structure by reducing the number of parameters without compromising performance. Here's a brief overview of our findings:

## Key Findings
1.  **KAN Alone**: Employing feature vectors from a foundation model, KAN alone struggled with direct classification tasks, suggesting the need for further model adjustments.
2.  **Hybrid Models**: Integrating a simple linear layer to reduce dimensions allowed even modestly sized KAN configurations (hidden size: 10) to surpass traditional MLPs in performance.
3.  **Reduced Parameter Count**: Minimizing data dimensions significantly cut down the number of parameters, enhancing model efficiency.
4.  **RBFKAN Performance**: Among our models, RBFKAN exhibited the most favorable balance between loss and parameter count, as well as training speed.

## Repository Structure
- **`image_feature_extractor.py`**: Extracts features from the image dataset using a pre-trained model (DINO).
- **`train_nas.py`**: Trains different models using neural architecture search and stores results.
- **`plot_models.py`**: Plots the results of the grid search in a comprehensible visualization.
- **`kans/`**: Contains the implementation of various KAN models and related utilities.

## Installation
Clone this repository:
```bash
    git clone https://github.com/rotem154154/kan_classification.git
    cd kan_classification
    pip install -r requirements.txt
```
## Usage
### Feature Extraction
Run the following command to extract features from your dataset:
```bash
python image_feature_extractor.py --save_path <path/to/dataset>
``` 

### Training

Use the `train_nas.py` script to train and evaluate models:

### Plot Results

Generate a plot comparing different models using  `plot_models.py` 

## Configuration Options

1.  **Feature Extraction**:
    
    -   `--model_name`: Pre-trained model for feature extraction (default: `vit_large_patch14_dinov2.lvd142m`).
    -   `--dataset_path`: Path to save the extracted features.
2.  **Training**:
   
    -   `model_name`: List of models to train (`mlp`, `efficient-kan`, etc.).
    -   `lr`: Learning rate.
    -   `n_units_l1`: Units in the first hidden layer.
    -   `n_units_l2`: Units in the second hidden layer.
    
## Results Visualization

The grid search results are visualized in `optimized_model_performance.png`, showing a comparison of model performance based on the number of trainable parameters and validation loss.

