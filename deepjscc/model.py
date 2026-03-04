"""DeepJSCC autoencoder model definition."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from deepjscc.channels import apply_channel


@dataclass(frozen=True)
class ModelVariant:
    encoder_filters: tuple[int, ...]
    decoder_filters: tuple[int, ...]
    conv_kernel: int = 5
    bottleneck_kernel: int = 3
    activation: str = "relu"


MODEL_VARIANTS = {
    "tiny": ModelVariant(
        encoder_filters=(32,),
        decoder_filters=(64, 32),
    ),
    "small": ModelVariant(
        encoder_filters=(32, 64),
        decoder_filters=(64, 32, 16),
    ),
    "base": ModelVariant(
        encoder_filters=(64, 128),
        decoder_filters=(128, 64, 32),
    ),
    "large": ModelVariant(
        encoder_filters=(64, 128, 256),
        decoder_filters=(256, 128, 64, 32),
    ),
}


class DeepJSCC(keras.Model):
    """Convolutional autoencoder with differentiable channel layer."""

    def __init__(
        self,
        image_size: int = 64,
        channel_uses: int = 256,
        latent_channels: int = 128,
        model_variant: str = "base",
        channel_type: str = "awgn",
        snr_db: float = 10.0,
        rician_k: float = 5.0,
    ):
        super().__init__()
        if model_variant not in MODEL_VARIANTS:
            raise ValueError(
                f"Unsupported model variant '{model_variant}'. Use one of: {tuple(MODEL_VARIANTS.keys())}"
            )

        variant = MODEL_VARIANTS[model_variant]
        downsample_stages = len(variant.encoder_filters) + 1
        downsample_factor = 2**downsample_stages
        if image_size % downsample_factor != 0:
            raise ValueError(
                f"image_size={image_size} must be divisible by {downsample_factor} for model_variant='{model_variant}'."
            )
        if len(variant.decoder_filters) != downsample_stages:
            raise ValueError(
                f"model_variant '{model_variant}' has inconsistent depth; expected "
                f"{downsample_stages} decoder stages, got {len(variant.decoder_filters)}."
            )

        self.image_size = image_size
        self.channel_uses = channel_uses
        self.latent_channels = latent_channels
        self.model_variant = model_variant
        self.variant = variant
        self.channel_type = channel_type
        self.snr_db = snr_db
        self.rician_k = rician_k

        encoder_layers = [layers.Input(shape=(image_size, image_size, 3))]
        for filters in variant.encoder_filters:
            encoder_layers.append(
                layers.Conv2D(
                    filters,
                    variant.conv_kernel,
                    strides=2,
                    padding="same",
                    activation=variant.activation,
                )
            )
        encoder_layers.extend(
            [
                layers.Conv2D(
                    latent_channels,
                    variant.bottleneck_kernel,
                    strides=2,
                    padding="same",
                    activation=variant.activation,
                ),
                layers.Flatten(),
                layers.Dense(2 * channel_uses),
            ]
        )
        self.encoder = keras.Sequential(encoder_layers, name="encoder")

        reduced_size = image_size // downsample_factor
        decoder_layers = [
            layers.Input(shape=(2 * channel_uses,)),
            layers.Dense(reduced_size * reduced_size * latent_channels, activation=variant.activation),
            layers.Reshape((reduced_size, reduced_size, latent_channels)),
        ]
        for filters in variant.decoder_filters:
            decoder_layers.append(
                layers.Conv2DTranspose(
                    filters,
                    variant.conv_kernel,
                    strides=2,
                    padding="same",
                    activation=variant.activation,
                )
            )
        decoder_layers.append(layers.Conv2D(3, 3, padding="same", activation="sigmoid"))
        self.decoder = keras.Sequential(decoder_layers, name="decoder")

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
