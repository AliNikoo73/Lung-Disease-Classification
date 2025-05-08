# Lung Disease Classification

A deep learning project for classifying lung diseases from chest X-ray images using transfer learning and attention mechanisms.

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

## Features

- Multi-class classification of lung diseases
- Transfer learning with EfficientNetV2
- Attention mechanism for better feature extraction
- Data augmentation for improved generalization
- Comprehensive evaluation metrics
- Model interpretability with Grad-CAM
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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Model architecture: [EfficientNetV2](https://arxiv.org/abs/2104.00298)
