import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import (
    VGG16, VGG19, Xception, InceptionV3, MobileNetV2,
    DenseNet201, NASNetLarge, InceptionResNetV2, ResNet152V2,
    EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2
)
from tensorflow.keras.layers import (
    Input, Dense, Flatten, BatchNormalization, Dropout,
    Conv2D, GlobalAveragePooling2D, MultiHeadAttention, LayerNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
)
import tensorflow_addons as tfa

class EnhancedLungDiseaseModel:
    def __init__(self, img_size=(224, 224), num_classes=5):
        self.img_size = img_size
        self.num_classes = num_classes
        self.base_models = {
            "EfficientNetV2B0": EfficientNetV2B0,
            "EfficientNetV2B1": EfficientNetV2B1,
            "EfficientNetV2B2": EfficientNetV2B2,
            "ResNet152V2": ResNet152V2,
            "DenseNet201": DenseNet201,
            "VGG16": VGG16,
            "VGG19": VGG19,
            "Xception": Xception,
            "InceptionV3": InceptionV3,
            "InceptionResNetV2": InceptionResNetV2,
            "MobileNetV2": MobileNetV2,
            "NASNetLarge": NASNetLarge,
        }

    def create_enhanced_datagen(self):
        """Create enhanced data generator with more augmentation techniques"""
        return tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    def create_model_with_attention(self, base_model_name, NNeuron=256, DO_factor=0.5):
        """Create model with attention mechanism"""
        base_model_class = self.base_models[base_model_name]
        base_model = base_model_class(
            weights="imagenet",
            include_top=False,
            input_shape=(*self.img_size, 3)
        )

        # Get the output from the base model
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        
        # Reshape for attention mechanism
        x = tf.expand_dims(x, axis=1)  # Add sequence dimension
        
        # Add attention mechanism
        x = LayerNormalization(epsilon=1e-6)(x)
        attn = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = tf.keras.layers.Add()([x, attn])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Squeeze back the sequence dimension
        x = tf.squeeze(x, axis=1)
        
        # Dense layers
        x = Dense(NNeuron, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(DO_factor)(x)
        predictions = Dense(self.num_classes, activation="softmax")(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        return model, base_model

    def create_ensemble_model(self, models, weights=None):
        """Create ensemble model from multiple trained models"""
        if weights is None:
            weights = [1/len(models)] * len(models)
        
        def ensemble_predict(x):
            predictions = [model.predict(x) for model in models]
            weighted_pred = np.average(predictions, weights=weights, axis=0)
            return weighted_pred

        return ensemble_predict

    def get_callbacks(self, model_path):
        """Get enhanced callbacks for training"""
        return [
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

    def train_with_mixup(self, model, train_generator, validation_generator, epochs=10):
        """Train model with Mixup augmentation"""
        def mixup(x, y, alpha=0.2):
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1

            batch_size = tf.shape(x)[0]
            index = tf.random.shuffle(tf.range(batch_size))
            
            mixed_x = lam * x + (1 - lam) * tf.gather(x, index)
            mixed_y = lam * y + (1 - lam) * tf.gather(y, index)
            
            return mixed_x, mixed_y

        # Custom training loop with Mixup
        optimizer = Adam(learning_rate=0.001)
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        
        for epoch in range(epochs):
            for x_batch, y_batch in train_generator:
                mixed_x, mixed_y = mixup(x_batch, y_batch)
                
                with tf.GradientTape() as tape:
                    predictions = model(mixed_x, training=True)
                    loss = loss_fn(mixed_y, predictions)
                
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    def evaluate_model(self, model, test_generator):
        """Enhanced model evaluation with multiple metrics"""
        predictions = model.predict(test_generator)
        y_true = test_generator.classes
        y_pred = np.argmax(predictions, axis=1)

        # Calculate various metrics
        from sklearn.metrics import (
            confusion_matrix, classification_report,
            roc_auc_score, precision_recall_curve
        )

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification Report
        report = classification_report(y_true, y_pred)
        
        # ROC AUC Score
        roc_auc = roc_auc_score(
            tf.keras.utils.to_categorical(y_true),
            predictions,
            multi_class='ovr'
        )

        return {
            'confusion_matrix': cm,
            'classification_report': report,
            'roc_auc': roc_auc
        }

    def visualize_results(self, model, test_generator, save_path):
        """Enhanced visualization of model results"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        # Get predictions
        predictions = model.predict(test_generator)
        y_true = test_generator.classes
        y_pred = np.argmax(predictions, axis=1)

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
        plt.close()

        # Plot ROC curves
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc

        classes = range(self.num_classes)
        y_true_bin = label_binarize(y_true, classes=classes)
        
        plt.figure(figsize=(10, 8))
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], predictions[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC curve (class {i}) (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for All Classes')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_path, 'roc_curves.png'))
        plt.close() 