# Data Directory

This directory should contain the lung disease X-ray image dataset. Due to size constraints, the actual dataset is not included in the repository.

## Dataset Structure

The dataset should be organized in the following structure:

```
data/
├── train/
│   ├── Bacterial_Pneumonia/
│   ├── COVID-19/
│   ├── Normal/
│   ├── Tuberculosis/
│   └── Viral_Pneumonia/
└── test/
    ├── Bacterial_Pneumonia/
    ├── COVID-19/
    ├── Normal/
    ├── Tuberculosis/
    └── Viral_Pneumonia/
```

## Getting the Dataset

1. Download the dataset from Kaggle: [Lung Disease Classification Dataset](https://www.kaggle.com/datasets)
2. Extract the downloaded archive
3. Place the images in their respective directories as shown in the structure above

## Data Format

- Images should be in common formats (jpg, png)
- Images will be automatically resized to 224x224 pixels during preprocessing
- Each image should be properly labeled by placing it in the corresponding disease category folder
