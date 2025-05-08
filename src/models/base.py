"""Base model class for lung disease classification."""
from typing import Dict, List, Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2


class BaseModel:
    """Base model class that implements common functionality for all models."""

    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        num_classes: int = 5,
        dropout_rate: float = 0.5,
        l2_factor: float = 0.0001,
    ):
        """Initialize the base model.

        Args:
            img_size: Input image size (height, width)
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
            l2_factor: L2 regularization factor
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.l2_factor = l2_factor
        self.model: Optional[Model] = None

    def build_base_model(self) -> Model:
        """Build the base model architecture.

        Returns:
            Compiled Keras model
        """
        raise NotImplementedError("Subclasses must implement build_base_model()")

    def add_classification_head(
        self, base_model: Model, trainable: bool = True
    ) -> Model:
        """Add classification head to the base model.

        Args:
            base_model: Base model to add classification head to
            trainable: Whether the base model layers should be trainable

        Returns:
            Model with classification head
        """
        # Set base model trainability
        base_model.trainable = trainable

        # Add classification head
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(
            256,
            activation="relu",
            kernel_regularizer=l2(self.l2_factor),
            name="dense_1",
        )(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(
            self.num_classes,
            activation="softmax",
            kernel_regularizer=l2(self.l2_factor),
            name="output",
        )(x)

        return Model(inputs=base_model.input, outputs=outputs)

    def compile_model(
        self,
        model: Model,
        learning_rate: float = 0.001,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
    ) -> Model:
        """Compile the model with specified optimizer and loss.

        Args:
            model: Model to compile
            learning_rate: Learning rate for optimizer
            optimizer: Custom optimizer to use (if None, Adam is used)

        Returns:
            Compiled model
        """
        if optimizer is None:
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def get_model_summary(self) -> str:
        """Get model summary as string.

        Returns:
            Model summary string
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_base_model() first.")

        string_list = []
        self.model.summary(print_fn=lambda x: string_list.append(x))
        return "\n".join(string_list) 