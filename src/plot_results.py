import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def plot_training_history(history, save_dir):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    """Plot confusion matrix."""
    # Filter predictions to only include classes present in the model
    valid_indices = np.where(y_true < len(class_names))[0]
    y_true_filtered = y_true[valid_indices]
    y_pred_filtered = y_pred[valid_indices]
    
    cm = confusion_matrix(y_true_filtered, y_pred_filtered)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def plot_roc_curves(y_true, y_pred_proba, class_names, save_dir):
    """Plot ROC curves for each class."""
    plt.figure(figsize=(10, 8))
    
    # Filter data to only include classes present in the model
    valid_indices = np.where(y_true < len(class_names))[0]
    y_true_filtered = y_true[valid_indices]
    y_pred_proba_filtered = y_pred_proba[valid_indices]
    
    # Convert y_true to one-hot encoding
    y_true_bin = np.zeros((len(y_true_filtered), len(class_names)))
    for i, label in enumerate(y_true_filtered):
        y_true_bin[i, label] = 1
    
    # Plot ROC curve for each class
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba_filtered[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'))
    plt.close()

def plot_class_distribution(generator, class_names, save_dir):
    """Plot class distribution in the dataset."""
    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(generator.classes, return_counts=True)
    
    # Filter to only include classes present in the model
    valid_indices = np.where(unique_classes < len(class_names))[0]
    unique_classes = unique_classes[valid_indices]
    class_counts = class_counts[valid_indices]
    
    # Filter class_names to match the actual classes in the generator
    filtered_class_names = [class_names[i] for i in unique_classes]
    
    plt.figure(figsize=(10, 6))
    plt.bar(filtered_class_names, class_counts)
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_distribution.png'))
    plt.close()

def main():
    # Load the model
    model_path = 'output/EfficientNetV2B0_best.h5'
    model = load_model(model_path)
    
    # Define class names
    class_names = ['Bacterial Pneumonia', 'Corona Virus Disease', 'NORMAL', 'Tuberculosis', 'Viral Pneumonia']
    
    # Create data generator
    datagen = ImageDataGenerator(rescale=1./255)
    
    # Load test data
    test_generator = datagen.flow_from_directory(
        'data/test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Get predictions
    y_pred_proba = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred_proba, axis=1)
    y_true = test_generator.classes
    
    # Create output directory for plots
    plot_dir = 'output/plots'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred_classes, class_names, plot_dir)
    
    # Plot ROC curves
    plot_roc_curves(y_true, y_pred_proba, class_names, plot_dir)
    
    # Plot class distribution
    plot_class_distribution(test_generator, class_names, plot_dir)
    
    # Print classification report
    print("\nClassification Report:")
    # Filter predictions to only include classes present in the model
    valid_indices = np.where(y_true < len(class_names))[0]
    y_true_filtered = y_true[valid_indices]
    y_pred_filtered = y_pred_classes[valid_indices]
    print(classification_report(y_true_filtered, y_pred_filtered, target_names=class_names))
    
    print(f"\nPlots have been saved to {plot_dir}")

if __name__ == '__main__':
    main() 