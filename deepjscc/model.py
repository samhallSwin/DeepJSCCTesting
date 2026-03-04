"""DeepJSCC autoencoder model definition."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from deepjscc.channels import apply_channel


class DeepJSCC(keras.Model):
    """Convolutional autoencoder with differentiable channel layer."""

    def __init__(
        self,
        image_size: int = 64,
        channel_uses: int = 256,
        latent_channels: int = 128,
        channel_type: str = "awgn",
        snr_db: float = 10.0,
        rician_k: float = 5.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.channel_uses = channel_uses
        self.latent_channels = latent_channels
        self.channel_type = channel_type
        self.snr_db = snr_db
        self.rician_k = rician_k

        self.encoder = keras.Sequential(
            [
                layers.Input(shape=(image_size, image_size, 3)),
                layers.Conv2D(64, 5, strides=2, padding="same", activation="relu"),
                layers.Conv2D(128, 5, strides=2, padding="same", activation="relu"),
                layers.Conv2D(latent_channels, 3, strides=2, padding="same", activation="relu"),
                layers.Flatten(),
                layers.Dense(2 * channel_uses),
            ],
            name="encoder",
        )

        reduced_size = image_size // 8
        self.decoder = keras.Sequential(
            [
                layers.Input(shape=(2 * channel_uses,)),
                layers.Dense(reduced_size * reduced_size * latent_channels, activation="relu"),
                layers.Reshape((reduced_size, reduced_size, latent_channels)),
                layers.Conv2DTranspose(128, 5, strides=2, padding="same", activation="relu"),
                layers.Conv2DTranspose(64, 5, strides=2, padding="same", activation="relu"),
                layers.Conv2DTranspose(32, 5, strides=2, padding="same", activation="relu"),
                layers.Conv2D(3, 3, padding="same", activation="sigmoid"),
            ],
            name="decoder",
        )

    def call(self, inputs, training=False):
        symbols_ri = self.encoder(inputs, training=training)
        rx_symbols = apply_channel(
            symbols_ri,
            channel_type=self.channel_type,
            snr_db=self.snr_db,
            rician_k=self.rician_k,
        )
        return self.decoder(rx_symbols, training=training)


class PSNRMetric(keras.metrics.Metric):
    """Average PSNR over mini-batches."""

    def __init__(self, name="psnr", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.image.psnr(y_true, y_pred, max_val=1.0)
        self.total.assign_add(tf.reduce_sum(values))
        self.count.assign_add(tf.cast(tf.size(values), tf.float32))

    def result(self):
        return self.total / (self.count + 1e-8)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)
