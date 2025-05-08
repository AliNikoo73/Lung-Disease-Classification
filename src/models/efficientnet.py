"""EfficientNet model implementation for lung disease classification."""
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import Model

from src.models.attention import SpatialAttention
from src.models.base import BaseModel


class EfficientNetModel(BaseModel):
    """EfficientNet-based model for lung disease classification."""

    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        num_classes: int = 5,
        dropout_rate: float = 0.5,
        l2_factor: float = 0.0001,
    ):
        """Initialize the EfficientNet model.

        Args:
            img_size: Input image size (height, width)
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
            l2_factor: L2 regularization factor
        """
        super().__init__(img_size, num_classes, dropout_rate, l2_factor)

    def build_base_model(self) -> Model:
        """Build the EfficientNet model with attention mechanism.

        Returns:
            Compiled Keras model
        """
        # Load pre-trained EfficientNetV2B0
        base_model = EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            input_shape=(*self.img_size, 3),
        )

        # Freeze base model layers
        base_model.trainable = False

        # Add spatial attention
        x = base_model.output
        x = SpatialAttention()(x)

        # Global average pooling
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)

        # Add dense layers
        x = Dense(
            256,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_factor),
            name="dense_1",
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)

        # Output layer
        outputs = Dense(
            self.num_classes,
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_factor),
            name="output",
        )(x)

        # Create model
        model = Model(inputs=base_model.input, outputs=outputs)

        # Compile model with a higher learning rate since we're only training the top layers
        model = self.compile_model(
            model,
            learning_rate=0.001,  # Higher learning rate for transfer learning
        )

        self.model = model
        return model 