"""Evaluation metrics for lung disease classification."""
import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf


class ModelEvaluator:
    """Evaluator class for model evaluation and visualization."""

    def __init__(
        self,
        class_names: List[str],
        output_dir: str,
        experiment_name: str,
    ):
        """Initialize the evaluator.

        Args:
            class_names: List of class names
            output_dir: Directory to save outputs
            experiment_name: Name of the experiment
        """
        self.class_names = class_names
        self.output_dir = output_dir
        self.experiment_name = experiment_name

        # Create output directories
        self.plot_dir = os.path.join(output_dir, "plots", experiment_name)
        os.makedirs(self.plot_dir, exist_ok=True)

    def plot_training_history(self, history: tf.keras.callbacks.History) -> None:
        """Plot training history.

        Args:
            history: Training history object
        """
        plt.figure(figsize=(12, 4))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{self.experiment_name}_training_history.png"))
        plt.close()

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = True,
        save: bool = True,
    ) -> None:
        """Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize the confusion matrix
            save: Whether to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.plot_dir, "confusion_matrix.png"))
            plt.close()
        else:
            plt.show()

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save: bool = True,
    ) -> None:
        """Plot ROC curves for each class.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save: Whether to save the plot
        """
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))

        plt.figure(figsize=(10, 8))

        # Plot ROC curve for each class
        for i in range(len(self.class_names)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
            plt.plot(
                fpr,
                tpr,
                label=f"{self.class_names[i]} (AUC = {auc:.2f})",
            )

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend(loc="lower right")
        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.plot_dir, "roc_curves.png"))
            plt.close()
        else:
            plt.show()

    def generate_classification_report(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> str:
        """Generate classification report.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Classification report as string
        """
        return classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            digits=4,
        )

    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        save_plots: bool = True,
    ) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            save_plots: Whether to save plots

        Returns:
            Dictionary of evaluation metrics
        """
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
        roc_auc = roc_auc_score(y_true_bin, y_pred_proba, multi_class="ovr")

        # Generate plots
        self.plot_confusion_matrix(y_true, y_pred, save=save_plots)
        self.plot_roc_curves(y_true, y_pred_proba, save=save_plots)

        # Generate classification report
        report = self.generate_classification_report(y_true, y_pred)

        # Save classification report
        if save_plots:
            with open(os.path.join(self.plot_dir, "classification_report.txt"), "w") as f:
                f.write(report)

        return {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "classification_report": report,
        } 