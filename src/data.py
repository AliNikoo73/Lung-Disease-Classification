"""
Data loading and preprocessing module for lung disease classification.
"""

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataLoader:
    def __init__(self, data_dir, img_size=(224, 224)):
        """
        Initialize the data loader.
        
        Args:
            data_dir (str): Path to the data directory
            img_size (tuple): Target image size (height, width)
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.classes = ['Bacterial_Pneumonia', 'COVID-19', 'Tuberculosis', 'Viral_Pneumonia', 'Normal']
        
    def load_and_preprocess(self, batch_size=32, split_ratio=0.2):
        """
        Load and preprocess the dataset with data augmentation.
        
        Args:
            batch_size (int): Batch size for training
            split_ratio (float): Validation split ratio
            
        Returns:
            tuple: (train_generator, validation_generator)
        """
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=split_ratio
        )
        
        train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        validation_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        return train_generator, validation_generator
    
    def preprocess_single_image(self, image_path):
        """
        Preprocess a single image for prediction.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img / 255.0
        return np.expand_dims(img, axis=0) 