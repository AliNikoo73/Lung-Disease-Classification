"""
Model evaluation and visualization module for lung disease classification.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2

class ModelEvaluator:
    def __init__(self, model, class_names):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained Keras model or path to model file
            class_names (list): List of class names
        """
        if isinstance(model, str):
            self.model = load_model(model)
        else:
            self.model = model
        self.class_names = class_names
        
    def plot_training_history(self, history):
        """
        Plot training history metrics.
        
        Args:
            history (dict): Training history from model.fit()
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history['accuracy'], label='Training')
        ax1.plot(history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history['loss'], label='Training')
        ax2.plot(history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
    def compute_confusion_matrix(self, validation_generator):
        """
        Compute and plot confusion matrix.
        
        Args:
            validation_generator: Validation data generator
            
        Returns:
            numpy.ndarray: Confusion matrix
        """
        # Get predictions
        y_pred = []
        y_true = []
        
        for i in range(len(validation_generator)):
            x, y = validation_generator[i]
            pred = self.model.predict(x)
            y_pred.extend(np.argmax(pred, axis=1))
            y_true.extend(np.argmax(y, axis=1))
            
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        
        return cm
    
    def generate_classification_report(self, validation_generator):
        """
        Generate classification report with precision, recall, and F1-score.
        
        Args:
            validation_generator: Validation data generator
            
        Returns:
            str: Classification report
        """
        y_pred = []
        y_true = []
        
        for i in range(len(validation_generator)):
            x, y = validation_generator[i]
            pred = self.model.predict(x)
            y_pred.extend(np.argmax(pred, axis=1))
            y_true.extend(np.argmax(y, axis=1))
            
        return classification_report(y_true, y_pred, target_names=self.class_names)
    
    def generate_gradcam(self, image_path, layer_name=None):
        """
        Generate Grad-CAM visualization for a single image.
        
        Args:
            image_path (str): Path to the image file
            layer_name (str): Name of the layer to use for Grad-CAM
        """
        if layer_name is None:
            layer_name = self.model.layers[-3].name  # Usually the last conv layer
            
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(layer_name).output, self.model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]
            
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        plt.figure(figsize=(8, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Original Image')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.imshow(cv2.resize(heatmap, (224, 224)), alpha=0.6, cmap='jet')
        plt.title('Grad-CAM')
        
        plt.tight_layout()
        plt.show() 