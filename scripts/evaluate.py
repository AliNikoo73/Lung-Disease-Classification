#!/usr/bin/env python3
"""Evaluation script for lung disease classification."""
import argparse
import os
from typing import Dict, Optional

import yaml
import tensorflow as tf
import numpy as np

from src.data.dataset import LungDiseaseDataset
from src.evaluation.metrics import ModelEvaluator
from src.models.base import BaseModel
from src.training.trainer import ModelTrainer
from src.enhanced_model import EnhancedLungDiseaseModel


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config = load_config('configs/config.yaml')
    
    # The classes used during training
    training_classes = [
        "Bacterial Pneumonia",
        "Corona Virus Disease",
        "NORMAL"
    ]
    
    # Initialize model with original number of classes
    model = EnhancedLungDiseaseModel(
        img_size=tuple(config['model']['img_size']),
        num_classes=len(training_classes)  # Use original number of classes
    )
    
    # Create data generators
    datagen = model.create_enhanced_datagen()
    
    # Load test data
    test_generator = datagen.flow_from_directory(
        config['data']['test_dir'],
        target_size=config['model']['img_size'],
        batch_size=config['training']['batch_size'],
        class_mode='categorical',
        classes=training_classes,  # Only use the original classes
        shuffle=False
    )
    
    # Load the best model
    model_path = os.path.join(config['output']['model_dir'], 'experiment1', 'best_model')
    loaded_model = tf.keras.models.load_model(model_path)
    
    # Initialize evaluator with original classes
    evaluator = ModelEvaluator(
        class_names=training_classes,
        output_dir=config['output']['results_dir'],
        experiment_name='experiment1'
    )
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = loaded_model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Evaluate and visualize results
    print("\nEvaluation Metrics:")
    metrics = evaluator.evaluate_model(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_proba=predictions,
        save_plots=True
    )
    
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    print("\nResults have been saved to:", config['output']['results_dir'])


if __name__ == "__main__":
    main() 