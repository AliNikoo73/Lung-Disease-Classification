#!/usr/bin/env python3
"""Training script for lung disease classification."""
import argparse
import os
from typing import Dict, Optional

import yaml

from src.data.dataset import LungDiseaseDataset
from src.evaluation.metrics import ModelEvaluator
from src.models.efficientnet import EfficientNetModel
from src.training.trainer import ModelTrainer


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main(args: Optional[argparse.Namespace] = None) -> None:
    """Main training function.

    Args:
        args: Command-line arguments
    """
    if args is None:
        parser = argparse.ArgumentParser(description="Train lung disease classification model")
        parser.add_argument(
            "--config",
            type=str,
            required=True,
            help="Path to configuration file",
        )
        parser.add_argument(
            "--data-dir",
            type=str,
            required=True,
            help="Path to dataset directory",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            required=True,
            help="Path to output directory",
        )
        parser.add_argument(
            "--experiment-name",
            type=str,
            required=True,
            help="Name of the experiment",
        )
        args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create dataset
    dataset = LungDiseaseDataset(
        data_dir=args.data_dir,
        img_size=tuple(config["model"]["img_size"]),
        batch_size=config["training"]["batch_size"],
        validation_split=config["training"]["validation_split"],
    )

    # Create data generators
    train_generator, val_generator, test_generator = dataset.create_data_generators(
        augmentation_config=config["data"]["augmentation"]
    )

    # Create model
    model = EfficientNetModel(
        img_size=tuple(config["model"]["img_size"]),
        num_classes=config["model"]["num_classes"],
        dropout_rate=config["model"]["dense_layers"]["dropout"],
        l2_factor=config["model"]["regularization"]["l2_factor"],
    )
    model.build_base_model()

    # Create trainer
    trainer = ModelTrainer(
        model=model,
        dataset=dataset,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )

    # Create callbacks
    callbacks = trainer.create_callbacks(
        early_stopping_config=config["training"]["early_stopping"],
        reduce_lr_config=config["training"]["reduce_lr"],
    )

    # Train model
    history = trainer.train(
        train_generator=train_generator,
        val_generator=val_generator,
        epochs=config["training"]["epochs"],
        callbacks=callbacks,
        class_weights=dataset.get_class_weights() if config["training"]["class_weights"] else None,
    )

    # Evaluate model
    evaluator = ModelEvaluator(
        class_names=dataset.get_class_names(),
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )

    # Plot training history
    evaluator.plot_training_history(history)

    # Generate predictions
    y_pred_proba, y_true = trainer.predict(test_generator)
    y_pred = y_pred_proba.argmax(axis=1)

    # Evaluate model
    metrics = evaluator.evaluate_model(y_true, y_pred, y_pred_proba)
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(metrics["classification_report"])

    # Save model
    trainer.save_model()


if __name__ == "__main__":
    main() 