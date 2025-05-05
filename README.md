# Lung Disease Classification

[![CI](https://github.com/yourusername/lung-disease-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/lung-disease-classification/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/yourusername/lung-disease-classification/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/lung-disease-classification)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A deep learning project for classifying lung diseases from X-ray images using state-of-the-art convolutional neural networks.

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
- Advanced data augmentation and preprocessing
- Model interpretation using Grad-CAM
- Comprehensive evaluation metrics and visualizations

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

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Follow the instructions in [data/README.md](data/README.md) to download and set up the dataset.

## Usage

### Training a Model

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

## Project Structure

```
lung-disease-classification/
├── src/
│   ├── data.py          # Data loading and preprocessing
│   ├── model.py         # Model architecture and training
│   └── evaluate.py      # Evaluation metrics and visualization
├── tests/
│   └── test_model.py    # Unit tests
├── notebooks/
│   └── example.ipynb    # Example usage notebook
├── data/                # Dataset directory
├── models/              # Saved model weights
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
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

- Dataset source: [Kaggle](https://www.kaggle.com/datasets)
- Pre-trained models: [Keras Applications](https://keras.io/api/applications/)
