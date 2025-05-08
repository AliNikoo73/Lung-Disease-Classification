# Dataset Directory

This directory contains the chest X-ray images dataset for lung disease classification.

## Directory Structure

```
data/
├── train/          # Training images
│   ├── Bacterial Pneumonia/
│   ├── Corona Virus Disease/
│   ├── NORMAL/
│   ├── Tuberculosis/
│   └── Viral Pneumonia/
├── val/            # Validation images
│   ├── Bacterial Pneumonia/
│   ├── Corona Virus Disease/
│   ├── NORMAL/
│   ├── Tuberculosis/
│   └── Viral Pneumonia/
└── test/           # Test images
    ├── Bacterial Pneumonia/
    ├── Corona Virus Disease/
    ├── NORMAL/
    ├── Tuberculosis/
    └── Viral Pneumonia/
```

## Dataset Information

- Source: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Classes: 5 classes of lung conditions
- Image Format: JPEG/PNG
- Image Size: Various (will be resized to 224x224 during training)

## Usage

1. Download the dataset from Kaggle
2. Extract the contents to this directory
3. The data will be automatically organized into train/val/test splits by the `download_dataset.py` script

## Data Organization

The dataset is organized into three splits:
- Training set (70% of data)
- Validation set (15% of data)
- Test set (15% of data)

Each split contains images from all five classes, maintaining the class distribution from the original dataset.
