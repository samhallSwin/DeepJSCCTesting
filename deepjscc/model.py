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
    "base_wide": ModelVariant(
        encoder_filters=(96, 192),
        decoder_filters=(192, 96, 48),
    ),
    "large_wide": ModelVariant(
        encoder_filters=(96, 192, 384),
        decoder_filters=(384, 192, 96, 48),
    ),
}


def _make_activation(name: str) -> layers.Layer:
    return layers.Activation(name)


class FiLM(layers.Layer):
    """Feature-wise linear modulation driven by an SNR embedding."""

    def __init__(self, channels: int, max_scale_delta: float = 0.25, max_shift: float = 0.25, name: str | None = None):
        super().__init__(name=name)
        self.channels = channels
        self.max_scale_delta = max_scale_delta
        self.max_shift = max_shift
        self.gamma = layers.Dense(
            channels,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name=f"{name}_gamma" if name else None,
        )
        self.beta = layers.Dense(
            channels,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name=f"{name}_beta" if name else None,
        )

    def call(self, inputs, training=False):
        x, snr_embedding = inputs
        gamma = self.gamma(snr_embedding)
        beta = self.beta(snr_embedding)
        gamma = tf.tanh(gamma) * self.max_scale_delta
        beta = tf.tanh(beta) * self.max_shift
        gamma = tf.reshape(gamma, (-1, 1, 1, self.channels))
        beta = tf.reshape(beta, (-1, 1, 1, self.channels))
        gamma = tf.cast(gamma, x.dtype)
        beta = tf.cast(beta, x.dtype)
        return x * (1.0 + gamma) + beta


class ConvFiLMBlock(layers.Layer):
    def __init__(self, filters: int, kernel_size: int, activation: str, name: str):
        super().__init__(name=name)
        self.conv = layers.Conv2D(filters, kernel_size, strides=2, padding="same", activation=None)
        self.film = FiLM(filters, name=f"{name}_film")
        self.activation = _make_activation(activation)

    def call(self, inputs, training=False):
        x, snr_embedding = inputs
        x = self.conv(x, training=training)
        x = self.film((x, snr_embedding), training=training)
        return self.activation(x)


class DeconvFiLMBlock(layers.Layer):
    def __init__(self, filters: int, kernel_size: int, activation: str, name: str):
        super().__init__(name=name)
        self.deconv = layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=2,
            padding="same",
            activation=None,
        )
        self.film = FiLM(filters, name=f"{name}_film")
        self.activation = _make_activation(activation)

    def call(self, inputs, training=False):
        x, snr_embedding = inputs
        x = self.deconv(x, training=training)
        x = self.film((x, snr_embedding), training=training)
        return self.activation(x)


class DeepJSCCEncoder(keras.Model):
    def __init__(
        self,
        image_size: int,
        channel_uses: int,
        latent_channels: int,
        variant: ModelVariant,
        name: str = "encoder",
    ):
        super().__init__(name=name)
        self.snr_mlp = keras.Sequential(
            [
                layers.Input(shape=(1,)),
                layers.Dense(64, activation=variant.activation),
                layers.Dense(128, activation=variant.activation),
            ],
            name="snr_encoder_mlp",
        )
        self.blocks = [
            ConvFiLMBlock(
                filters=filters,
                kernel_size=variant.conv_kernel,
                activation=variant.activation,
                name=f"enc_block_{idx}",
            )
            for idx, filters in enumerate(variant.encoder_filters)
        ]
        self.bottleneck = ConvFiLMBlock(
            filters=latent_channels,
            kernel_size=variant.bottleneck_kernel,
            activation=variant.activation,
            name="enc_bottleneck",
        )
        self.flatten = layers.Flatten()
        self.out_dense = layers.Dense(2 * channel_uses)

    def call(self, inputs, training=False):
        image, snr_db = inputs
        snr_embedding = self.snr_mlp(snr_db, training=training)
        x = image
        for block in self.blocks:
            x = block((x, snr_embedding), training=training)
        x = self.bottleneck((x, snr_embedding), training=training)
        x = self.flatten(x)
        dense_input = tf.concat([x, tf.cast(snr_embedding, x.dtype)], axis=-1)
        return self.out_dense(dense_input, training=training)


class DeepJSCCDecoder(keras.Model):
    def __init__(
        self,
        image_size: int,
        channel_uses: int,
        latent_channels: int,
        variant: ModelVariant,
        reduced_size: int,
        name: str = "decoder",
    ):
        super().__init__(name=name)
        self.snr_mlp = keras.Sequential(
            [
                layers.Input(shape=(1,)),
                layers.Dense(64, activation=variant.activation),
                layers.Dense(128, activation=variant.activation),
            ],
            name="snr_decoder_mlp",
        )
        self.pre_dense = layers.Dense(reduced_size * reduced_size * latent_channels, activation=None)
        self.pre_film = FiLM(latent_channels, name="dec_pre_film")
        self.pre_activation = _make_activation(variant.activation)
        self.reshape_layer = layers.Reshape((reduced_size, reduced_size, latent_channels))
        self.blocks = [
            DeconvFiLMBlock(
                filters=filters,
                kernel_size=variant.conv_kernel,
                activation=variant.activation,
                name=f"dec_block_{idx}",
            )
            for idx, filters in enumerate(variant.decoder_filters)
        ]
        self.out_conv = layers.Conv2D(3, 3, padding="same", activation="sigmoid")
        self.latent_channels = latent_channels
        self.reduced_size = reduced_size

    def call(self, inputs, training=False):
        rx_symbols, snr_db = inputs
        snr_embedding = self.snr_mlp(snr_db, training=training)
        dense_input = tf.concat([rx_symbols, tf.cast(snr_embedding, rx_symbols.dtype)], axis=-1)
        x = self.pre_dense(dense_input, training=training)
        x = self.reshape_layer(x)
        x = self.pre_film((x, snr_embedding), training=training)
        x = self.pre_activation(x)
        for block in self.blocks:
            x = block((x, snr_embedding), training=training)
        return self.out_conv(x, training=training)


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

        reduced_size = image_size // downsample_factor
        self.encoder = DeepJSCCEncoder(
            image_size=image_size,
            channel_uses=channel_uses,
            latent_channels=latent_channels,
            variant=variant,
            name="encoder",
        )
        self.decoder = DeepJSCCDecoder(
            image_size=image_size,
            channel_uses=channel_uses,
            latent_channels=latent_channels,
            variant=variant,
            reduced_size=reduced_size,
            name="decoder",
        )

    def _split_inputs(self, inputs):
        if not isinstance(inputs, dict):
            raise TypeError("DeepJSCC expects inputs as a dict with keys 'image' and 'snr_db'.")
        image = tf.cast(inputs["image"], tf.float32)
        snr_db = tf.cast(inputs["snr_db"], tf.float32)
        if snr_db.shape.rank == 1:
            snr_db = tf.expand_dims(snr_db, axis=-1)
        return image, snr_db

    def call(self, inputs, training=False):
        image, snr_db = self._split_inputs(inputs)
        symbols_ri = self.encoder((image, snr_db), training=training)
        rx_symbols = apply_channel(
            symbols_ri,
            channel_type=self.channel_type,
            snr_db=tf.squeeze(snr_db, axis=-1),
            rician_k=self.rician_k,
        )
        return self.decoder((rx_symbols, tf.cast(snr_db, rx_symbols.dtype)), training=training)


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


class SSIMMetric(keras.metrics.Metric):
    """Average SSIM over mini-batches."""

    def __init__(self, name="ssim", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.image.ssim(y_true, y_pred, max_val=1.0)
        self.total.assign_add(tf.reduce_sum(values))
        self.count.assign_add(tf.cast(tf.size(values), tf.float32))

    def result(self):
        return self.total / (self.count + 1e-8)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


class ReconstructionLoss(keras.losses.Loss):
    """Hybrid pixel- and structure-aware reconstruction loss."""

    def __init__(
        self,
        l1_weight: float = 0.8,
        ssim_weight: float = 0.2,
        name: str = "reconstruction_loss",
    ):
        super().__init__(name=name)
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        l1 = tf.reduce_mean(tf.abs(y_true - y_pred), axis=(1, 2, 3))
        ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
        return self.l1_weight * l1 + self.ssim_weight * (1.0 - ssim)

    def get_config(self):
        return {
            "l1_weight": self.l1_weight,
            "ssim_weight": self.ssim_weight,
            "name": self.name,
        }
