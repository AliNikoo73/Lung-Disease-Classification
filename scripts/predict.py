#!/usr/bin/env python3
"""Prediction script for lung disease classification."""
import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
import yaml

from src.data.dataset import LungDiseaseDataset
from src.models.base import BaseModel


def load_image(
    image_path: str, img_size: Tuple[int, int]
) -> np.ndarray:
    """Load and preprocess image.

    Args:
        image_path: Path to image file
        img_size: Target image size (height, width)

    Returns:
        Preprocessed image array
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def main(args: Optional[argparse.Namespace] = None) -> None:
    """Main prediction function.

    Args:
        args: Command-line arguments
    """
    if args is None:
        parser = argparse.ArgumentParser(description="Make predictions on new images")
        parser.add_argument(
            "--model-path",
            type=str,
            required=True,
            help="Path to trained model",
        )
        parser.add_argument(
            "--image-path",
            type=str,
            required=True,
            help="Path to input image",
        )
        parser.add_argument(
            "--config",
            type=str,
            required=True,
            help="Path to configuration file",
        )
        args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Create dataset to get class names
    dataset = LungDiseaseDataset(
        data_dir=os.path.dirname(args.image_path),
        img_size=tuple(config["model"]["img_size"]),
    )
    class_names = dataset.get_class_names()

    # Load model
    model = BaseModel(
        img_size=tuple(config["model"]["img_size"]),
        num_classes=config["model"]["num_classes"],
    )
    model.load_model(args.model_path)

    # Load and preprocess image
    image = load_image(args.image_path, tuple(config["model"]["img_size"]))

    # Make prediction
    predictions = model.model.predict(image)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    # Print results
    print("\nPrediction Results:")
    print(f"Predicted Class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.4f}")
    print("\nClass Probabilities:")
    for i, (class_name, prob) in enumerate(zip(class_names, predictions[0])):
        print(f"{class_name}: {prob:.4f}")


if __name__ == "__main__":
    main() 