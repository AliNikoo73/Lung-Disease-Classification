"""Attention mechanism for lung disease classification."""
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Layer, MultiHeadAttention


class AttentionBlock(Layer):
    """Multi-head attention block for feature refinement."""

    def __init__(
        self,
        num_heads: int = 4,
        key_dim: int = 64,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        """Initialize the attention block.

        Args:
            num_heads: Number of attention heads
            key_dim: Dimension of key vectors
            dropout_rate: Dropout rate for attention weights
            **kwargs: Additional layer arguments
        """
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout_rate,
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(
        self, inputs: tf.Tensor, training: Optional[bool] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply attention mechanism to inputs.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, features)
            training: Whether the layer is in training mode

        Returns:
            Tuple of (output tensor, attention weights)
        """
        # Layer normalization
        normalized = self.layer_norm(inputs)

        # Apply multi-head attention
        attention_output, attention_weights = self.attention(
            normalized,
            normalized,
            return_attention_scores=True,
            training=training,
        )

        # Residual connection
        output = inputs + attention_output

        return output, attention_weights

    def get_config(self) -> dict:
        """Get layer configuration.

        Returns:
            Layer configuration dictionary
        """
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class SpatialAttention(Layer):
    """Spatial attention mechanism for focusing on relevant image regions."""

    def __init__(self, reduction_ratio: int = 8, **kwargs):
        """Initialize spatial attention.

        Args:
            reduction_ratio: Channel reduction ratio
            **kwargs: Additional layer arguments
        """
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape: Tuple[int, ...]):
        """Build the layer.

        Args:
            input_shape: Input tensor shape
        """
        channels = input_shape[-1]
        self.channel_attention = tf.keras.Sequential(
            [
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(
                    channels // self.reduction_ratio,
                    activation="relu",
                ),
                tf.keras.layers.Dense(channels, activation="sigmoid"),
            ]
        )
        self.spatial_attention = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    1,
                    kernel_size=7,
                    padding="same",
                    activation="sigmoid",
                )
            ]
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply spatial attention to inputs.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels)

        Returns:
            Attention-weighted tensor
        """
        # Channel attention
        channel_weights = self.channel_attention(inputs)
        channel_weights = tf.expand_dims(channel_weights, axis=1)
        channel_weights = tf.expand_dims(channel_weights, axis=1)
        channel_out = inputs * channel_weights

        # Spatial attention
        spatial_weights = self.spatial_attention(channel_out)
        output = channel_out * spatial_weights

        return output

    def get_config(self) -> dict:
        """Get layer configuration.

        Returns:
            Layer configuration dictionary
        """
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config 