"""
Model training and tuning module for lung disease classification.
"""

from tensorflow.keras.applications import DenseNet201, ResNet152V2, VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class LungDiseaseModel:
    def __init__(self, model_name='densenet', num_classes=5):
        """
        Initialize the model.
        
        Args:
            model_name (str): Name of the base model ('densenet', 'resnet', or 'vgg')
            num_classes (int): Number of disease classes
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Build and compile the model architecture.
        
        Returns:
            tensorflow.keras.Model: Compiled model
        """
        # Select base model
        if self.model_name == 'densenet':
            base_model = DenseNet201(weights='imagenet', include_top=False)
        elif self.model_name == 'resnet':
            base_model = ResNet152V2(weights='imagenet', include_top=False)
        elif self.model_name == 'vgg':
            base_model = VGG16(weights='imagenet', include_top=False)
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, train_generator, validation_generator, epochs=50, callbacks=None):
        """
        Train the model.
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs (int): Number of epochs to train
            callbacks (list): List of Keras callbacks
            
        Returns:
            dict: Training history
        """
        if callbacks is None:
            callbacks = [
                ModelCheckpoint(
                    f'models/{self.model_name}_best.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max'
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
        
        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history.history
    
    def fine_tune(self, train_generator, validation_generator, num_layers=10):
        """
        Fine-tune the model by unfreezing some layers.
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            num_layers (int): Number of layers to unfreeze from the top
        """
        # Unfreeze the last num_layers layers
        for layer in self.model.layers[-num_layers:]:
            layer.trainable = True
            
        # Recompile with a lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        ) 