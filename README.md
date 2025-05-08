# Lung Disease Classification

A deep learning model for classifying lung diseases from chest X-ray images using transfer learning and attention mechanisms.

## Features

- Multi-class classification of lung diseases
- Transfer learning with state-of-the-art models
- Attention mechanism for improved feature extraction
- Data augmentation for better generalization
- Comprehensive evaluation metrics and visualizations
- Grad-CAM visualization for model interpretability

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
pip install -e .  # For development
# or
pip install -e ".[dev]"  # For development with testing tools
```

## Dataset

The project uses the Chest X-Ray Images (Pneumonia) dataset. You need to:
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Extract it to the `data` directory
3. The data will be automatically organized into train/val/test splits

## Usage

### Training

To train the model:

```bash
python src/train_enhanced.py \
    --data_dir data \
    --model_name EfficientNetV2B0 \
    --img_size 224 \
    --batch_size 32 \
    --epochs 50 \
    --output_dir output
```

Arguments:
- `--data_dir`: Path to dataset directory
- `--model_name`: Base model architecture (default: EfficientNetV2B0)
- `--img_size`: Image size (default: 224)
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of epochs (default: 50)
- `--output_dir`: Output directory for results (default: output)

### Making Predictions

To make predictions on new images:

```bash
python src/predict.py \
    --model_path output/EfficientNetV2B0_final.h5 \
    --image_path path/to/image.jpg \
    --class_names "NORMAL" "PNEUMONIA" \
    --output_dir predictions
```

## Project Structure

```
lung-disease-classification/
├── data/               # Dataset directory
├── src/               # Source code
│   ├── __init__.py
│   ├── data.py        # Data loading and preprocessing
│   ├── model.py       # Base model implementation
│   ├── enhanced_model.py  # Enhanced model with attention
│   ├── train_enhanced.py  # Training script
│   ├── predict.py     # Prediction script
│   └── evaluate.py    # Evaluation utilities
├── tests/             # Test files
├── output/            # Training outputs
├── requirements.txt   # Project dependencies
├── setup.py          # Package configuration
└── README.md         # Project documentation
```

## Model Architecture

The enhanced model includes:
1. Pre-trained base model (e.g., EfficientNetV2)
2. Attention mechanism using MultiHeadAttention
3. Dense layers with batch normalization and dropout
4. Softmax output layer

## Training Process

The training process consists of two phases:
1. Initial training with frozen base model
2. Fine-tuning with unfrozen base model

## Evaluation

The model evaluation includes:
- Classification metrics (accuracy, precision, recall, F1-score)
- ROC curves and AUC scores
- Confusion matrix
- Grad-CAM visualizations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Model architectures from TensorFlow/Keras
- Grad-CAM implementation from [Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam)
