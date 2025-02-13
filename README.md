# TRPCA: Transformer-based Robust Principal Component Analysis for Microbiome Data

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red)]()

A PyTorch-based framework for analyzing microbiome data using normalized transformers with multi-task learning capabilities. This implementation includes both single-task regression and multi-task learning approaches, specifically designed for microbiome feature analysis.

A PyTorch-based framework for analyzing microbiome data using normalized transformers with multi-task learning capabilities. This framework supports both single-task regression and multi-task learning (regression + classification) scenarios.

## 🌟 Features

- **Normalized Transformer Architecture**
  - Custom transformer blocks with normalization layers
  - Learnable scaling parameters for attention and MLP updates
  - Positional encoding for sequence information

- **Multi-Task Learning Support**
  - Joint regression and classification tasks
  - Uncertainty-weighted loss functions
  - Automatic task balancing during training

- **Data Processing**
  - Built-in support for microbiome data
  - Robust preprocessing with RCLR transformation
  - PCA dimensionality reduction
  - Group-aware train/test splitting

- **Analysis Tools**
  - Feature importance analysis using SHAP
  - Performance visualization
  - Comprehensive evaluation metrics

## 📋 Prerequisites

Before installing, ensure you have Python 3.8+ and PyTorch 1.8+ installed. Then install the following dependencies:

```bash
# Core dependencies
pip install torch numpy pandas scikit-learn

# Microbiome analysis
pip install biom-format
pip install gemelli

# Visualization and analysis
pip install matplotlib seaborn shap optuna tqdm
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/tydymy/TRPCA.git
cd TRPCA
git checkout dev_2
```

2. Install in development mode:
```bash
pip install -e .
```

## 🚀 Quick Start

### Single-Task Regression

```python
import pandas as pd
from TRPCA.utils import preprocess, train_model, predict_and_evaluate

# Load your data
df = pd.read_csv('your_feature_table.csv', index_col=0)
target = pd.read_csv('your_metadata.csv', index_col=0)['target_column']

# Set model parameters
model_params = {
    'input_dim': 48,
    'hidden_dim': 256,
    'num_layers': 1,
    'output_dim': 1,
    'projection_dim': 4
}

# Preprocess data
train_loader, test_loader, pca = preprocess(
    df=df,
    series=target,
    num_pcs=48,
    test_split=0.1,
    batch_size=128
)

# Train model
model, history, training_fig = train_model(
    train_loader=train_loader,
    model_params=model_params,
    num_epochs=1000,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    use_optuna=True
)

# Evaluate
metrics, predictions, eval_fig = predict_and_evaluate(
    model=model,
    test_loader=test_loader,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
```

### Multi-Task Learning

```python
from TRPCA.utils import preprocess_mtl, train_model_mtl, predict_and_evaluate_mtl

# Load data for both tasks
regression_target = metadata['continuous_variable']
classification_target = metadata['categorical_variable']

# Define parameters
model_params = {
    'input_dim': 256,
    'hidden_dim': 256,
    'num_layers': 1,
    'num_classes': n_classes,
    'output_dim': 1,
    'projection_dim': 4
}

# Preprocess
train_loader, test_loader, pca, label_encoder = preprocess_mtl(
    df=df,
    reg_series=regression_target,
    cls_series=classification_target,
    num_pcs=256,
    test_split=0.1,
    batch_size=128
)

# Train
model, history, training_fig = train_model_mtl(
    train_loader=train_loader,
    model_params=model_params,
    num_epochs=1000,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Evaluate
metrics, predictions, eval_fig = predict_and_evaluate_mtl(
    model=model,
    test_loader=test_loader,
    label_encoder=label_encoder,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
```

## 📊 Model Architecture

### Normalized Transformer
- **Input Processing**: PCA-reduced features are projected to a higher dimension
- **View Generation**: Creates multiple views of the data for robust learning
- **Transformer Blocks**: Self-attention with normalization and learnable scaling
- **Output Heads**: Task-specific layers for regression and classification

### Multi-Task Learning
- **Shared Backbone**: Common feature extraction through transformer layers
- **Uncertainty Weighting**: Automatic task balancing using learnable parameters
- **Task-Specific Heads**: Separate outputs for regression and classification

## 🔍 Advanced Features

### Feature Importance Analysis
```python
from TRPCA.utils import analyze_features

results = analyze_features(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    pca=pca,
    original_features=df.columns
)

# Access results
feature_importance = results['feature_importance_df']
sample_importance = results['sample_importance_df']
```

### Model Comparison
```python
from TRPCA.utils import compare_regressors

results, comparison_plot = compare_regressors(
    X=df,
    y=target,
    train_loader=train_loader,
    test_loader=test_loader,
    pca=pca,
    regressors=regressors
)
```

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@article{your-paper,
    title={NA},
    author={NA},
    journal={Pending},
    year={2025}
}
```

