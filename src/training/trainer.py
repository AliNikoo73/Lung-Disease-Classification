"""Training module for lung disease classification."""
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.models import Model
from tqdm.keras import TqdmCallback

from src.data.dataset import LungDiseaseDataset
from src.models.base import BaseModel


class ModelTrainer:
    """Trainer class for model training and evaluation."""

    def __init__(
        self,
        model: BaseModel,
        dataset: LungDiseaseDataset,
        output_dir: str,
        experiment_name: str,
    ):
        """Initialize the trainer.

        Args:
            model: Model to train
            dataset: Dataset for training
            output_dir: Directory to save outputs
            experiment_name: Name of the experiment
        """
        self.model = model
        self.dataset = dataset
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.history: Optional[Dict] = None

        # Create output directories
        self.model_dir = os.path.join(output_dir, "models", experiment_name)
        self.results_dir = os.path.join(output_dir, "results", experiment_name)
        self.tensorboard_dir = os.path.join(output_dir, "tensorboard", experiment_name)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)

    def create_callbacks(
        self,
        early_stopping_config: Dict,
        reduce_lr_config: Dict,
    ) -> List[tf.keras.callbacks.Callback]:
        """Create training callbacks.

        Args:
            early_stopping_config: Early stopping configuration
            reduce_lr_config: Learning rate reduction configuration

        Returns:
            List of callbacks
        """
        callbacks = [
            # Model checkpointing
            ModelCheckpoint(
                filepath=os.path.join(self.model_dir, "best_model"),
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1,
                save_format="tf",
            ),
            # Early stopping
            EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_config["patience"],
                min_delta=early_stopping_config["min_delta"],
                restore_best_weights=True,
                verbose=1,
            ),
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=reduce_lr_config["factor"],
                patience=reduce_lr_config["patience"],
                min_lr=reduce_lr_config["min_lr"],
                verbose=1,
            ),
            # TensorBoard logging
            TensorBoard(
                log_dir=self.tensorboard_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq="epoch",
                profile_batch=2,
            ),
            # Progress bar
            TqdmCallback(verbose=1),
        ]
        return callbacks

    def train(
        self,
        train_generator: tf.keras.preprocessing.image.ImageDataGenerator,
        val_generator: tf.keras.preprocessing.image.ImageDataGenerator,
        epochs: int,
        callbacks: List[tf.keras.callbacks.Callback],
        class_weights: Optional[Dict[int, float]] = None,
    ) -> tf.keras.callbacks.History:
        """Train the model.

        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Number of epochs to train
            callbacks: List of callbacks to use
            class_weights: Optional class weights for imbalanced datasets

        Returns:
            Training history
        """
        print("\nStarting training...")
        print(f"Training samples: {len(train_generator)}")
        print(f"Validation samples: {len(val_generator)}")
        if class_weights:
            print("Using class weights:", class_weights)

        # Train model
        self.history = self.model.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
        )

        return self.history

    def evaluate(
        self, test_generator: tf.keras.preprocessing.image.DirectoryIterator
    ) -> Dict[str, float]:
        """Evaluate the model on test data.

        Args:
            test_generator: Test data generator

        Returns:
            Dictionary of evaluation metrics
        """
        if self.model.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        evaluation = self.model.model.evaluate(test_generator)
        metrics = dict(zip(self.model.model.metrics_names, evaluation))

        return metrics

    def predict(
        self, test_generator: tf.keras.preprocessing.image.ImageDataGenerator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions on test data.

        Args:
            test_generator: Test data generator

        Returns:
            Tuple of (predicted probabilities, true labels)
        """
        print("\nGenerating predictions...")
        y_pred_proba = self.model.model.predict(test_generator, verbose=1)
        y_true = test_generator.classes
        return y_pred_proba, y_true

    def save_model(self) -> None:
        """Save the trained model."""
        print("\nSaving model...")
        self.model.model.save(os.path.join(self.model_dir, "final_model"), save_format="tf")
        print(f"Model saved to {self.model_dir}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model.

        Args:
            filepath: Path to the saved model
        """
        self.model.model = tf.keras.models.load_model(filepath)

    def get_training_history(self) -> Dict:
        """Get training history.

        Returns:
            Dictionary containing training history
        """
        if self.history is None:
            raise ValueError("Model not trained yet. Call train() first.")

        return self.history 