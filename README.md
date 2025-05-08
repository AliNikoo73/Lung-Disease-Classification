# Lung Disease Classification

[![CI](https://github.com/yourusername/lung-disease-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/lung-disease-classification/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/yourusername/lung-disease-classification/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/lung-disease-classification)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A deep learning project for classifying lung diseases from chest X-ray images using transfer learning and attention mechanisms.

## Features

- Multi-class classification of lung diseases:
  - Bacterial Pneumonia
  - COVID-19
  - Tuberculosis
  - Viral Pneumonia
  - Normal
- Support for multiple pre-trained architectures:
  - DenseNet201
  - ResNet152V2
  - VGG16
- Transfer learning with EfficientNetV2
- Attention mechanism for better feature extraction
- Advanced data augmentation and preprocessing
- Model interpretation using Grad-CAM
- Comprehensive evaluation metrics and visualizations
- Experiment tracking and logging
- Automated testing and CI/CD

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/lung-disease-classification.git
cd lung-disease-classification
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:

```bash
pip install -e ".[dev]"  # For development
# or
pip install -e .  # For production
```

## Dataset

Follow the instructions in [data/README.md](data/README.md) to download and set up the dataset.

## Usage

### Training

```bash
python scripts/train.py \
    --config configs/training_config.yaml \
    --data-dir data/processed \
    --output-dir output
```

### Prediction

```bash
python scripts/predict.py \
    --model-path output/best_model.h5 \
    --image-path path/to/image.jpg
```

### Evaluation

```bash
python scripts/evaluate.py \
    --model-path output/best_model.h5 \
    --test-dir data/processed/test
```

### Training a Model (Python API)

```python
from src.data import DataLoader
from src.model import LungDiseaseModel
from src.evaluate import ModelEvaluator

# Load and preprocess data
data_loader = DataLoader("data/")
train_gen, val_gen = data_loader.load_and_preprocess()

# Create and train model
model = LungDiseaseModel(model_name='densenet')
history = model.train(train_gen, val_gen, epochs=50)

# Evaluate model
evaluator = ModelEvaluator(model.model, class_names=data_loader.classes)
evaluator.plot_training_history(history)
evaluator.compute_confusion_matrix(val_gen)
print(evaluator.generate_classification_report(val_gen))
```

### Generating Grad-CAM Visualizations

```python
evaluator.generate_gradcam("path/to/image.jpg")
```

## Development

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Run tests:
```bash
pytest tests/
```

3. Run linting:
```bash
flake8 src tests
black src tests
```

## Project Structure

```
lung-disease-classification/
├── configs/                     # Configuration files
│   ├── model_config.yaml       # Model architecture configuration
│   └── training_config.yaml    # Training hyperparameters
├── data/                       # Dataset directory
│   ├── raw/                   # Original dataset
│   ├── processed/             # Preprocessed data
│   └── README.md             # Dataset documentation
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── exploratory_analysis.ipynb
│   └── model_experiments.ipynb
├── src/                       # Source code
│   ├── data/                 # Data handling modules
│   │   ├── __init__.py
│   │   ├── dataset.py       # Dataset class
│   │   └── preprocessing.py # Data preprocessing
│   ├── models/              # Model architecture modules
│   │   ├── __init__.py
│   │   ├── base.py         # Base model class
│   │   └── attention.py    # Attention mechanism
│   ├── training/           # Training modules
│   │   ├── __init__.py
│   │   ├── trainer.py     # Training loop
│   │   └── callbacks.py   # Custom callbacks
│   ├── evaluation/        # Evaluation modules
│   │   ├── __init__.py
│   │   ├── metrics.py    # Custom metrics
│   │   └── visualization.py # Plotting utilities
│   └── utils/            # Utility functions
│       ├── __init__.py
│       └── helpers.py    # Helper functions
├── tests/                # Unit tests
│   ├── test_data.py
│   ├── test_models.py
│   └── test_training.py
├── scripts/             # Command-line scripts
│   ├── train.py        # Training script
│   ├── predict.py      # Prediction script
│   └── evaluate.py     # Evaluation script
├── docs/               # Documentation
│   ├── api/           # API documentation
│   └── examples/      # Usage examples
├── .github/           # GitHub configuration
│   └── workflows/     # CI/CD workflows
├── requirements/      # Project dependencies
│   ├── base.txt      # Base requirements
│   ├── dev.txt       # Development requirements
│   └── test.txt      # Testing requirements
├── setup.py          # Package setup
├── pyproject.toml    # Project metadata
├── .pre-commit-config.yaml  # Pre-commit hooks
├── .gitignore
├── LICENSE
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Model architecture: [EfficientNetV2](https://arxiv.org/abs/2104.00298)
- Pre-trained models: [Keras Applications](https://keras.io/api/applications/)
