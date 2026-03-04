"""Channel simulation layers for DeepJSCC."""

from __future__ import annotations

import tensorflow as tf


CHANNEL_CHOICES = ("none", "awgn", "rayleigh", "rician")


def snr_db_to_noise_std(snr_db: tf.Tensor) -> tf.Tensor:
    """Convert SNR (dB) to noise std for complex AWGN with unit signal power."""
    snr_linear = tf.pow(10.0, snr_db / 10.0)
    noise_var = 1.0 / (2.0 * snr_linear)
    return tf.sqrt(noise_var)


def _to_complex(x_ri: tf.Tensor) -> tf.Tensor:
    """Convert [..., 2 * N] real-imag tensor to complex [..., N]."""
    real, imag = tf.split(x_ri, num_or_size_splits=2, axis=-1)
    if real.dtype in (tf.float16, tf.bfloat16):
        real = tf.cast(real, tf.float32)
        imag = tf.cast(imag, tf.float32)
    return tf.complex(real, imag)


def _to_ri(x_c: tf.Tensor) -> tf.Tensor:
    """Convert complex [..., N] tensor to [..., 2 * N] real-imag representation."""
    return tf.concat([tf.math.real(x_c), tf.math.imag(x_c)], axis=-1)


def normalize_complex_power(x_c: tf.Tensor, eps: float = 1e-8) -> tf.Tensor:
    """Normalize average per-batch symbol power to 1."""
    power = tf.reduce_mean(tf.abs(x_c) ** 2, axis=-1, keepdims=True)
    return x_c / tf.cast(tf.sqrt(power + eps), x_c.dtype)


def apply_channel(
    symbols_ri: tf.Tensor,
    channel_type: str,
    snr_db: float,
    rician_k: float = 5.0,
) -> tf.Tensor:
    """Apply selected channel to real-imag symbols and return same shape."""
    input_dtype = symbols_ri.dtype
    channel_type = channel_type.lower()
    if channel_type not in CHANNEL_CHOICES:
        raise ValueError(f"Unsupported channel type '{channel_type}'. Use {CHANNEL_CHOICES}.")

    x_c = _to_complex(symbols_ri)
    x_c = normalize_complex_power(x_c)

    if channel_type == "none":
        return tf.cast(_to_ri(x_c), input_dtype)

    batch_shape = tf.shape(x_c)
    noise_std = snr_db_to_noise_std(tf.cast(snr_db, tf.float32))

    noise = tf.complex(
        tf.random.normal(batch_shape, stddev=noise_std),
        tf.random.normal(batch_shape, stddev=noise_std),
    )

    if channel_type == "awgn":
        y_c = x_c + noise
    elif channel_type == "rayleigh":
        h = tf.complex(
            tf.random.normal(batch_shape, stddev=tf.sqrt(0.5)),
            tf.random.normal(batch_shape, stddev=tf.sqrt(0.5)),
        )
        y_c = h * x_c + noise
        y_c = y_c / (h + 1e-8)
    else:  # rician
        k = tf.cast(rician_k, tf.float32)
        los_amp = tf.sqrt(k / (k + 1.0))
        scatter_amp = tf.sqrt(1.0 / (k + 1.0))
        h_scatter = tf.complex(
            tf.random.normal(batch_shape, stddev=tf.sqrt(0.5)),
            tf.random.normal(batch_shape, stddev=tf.sqrt(0.5)),
        )
        h = tf.complex(los_amp, 0.0) + tf.cast(scatter_amp, tf.complex64) * h_scatter
        y_c = h * x_c + noise
        y_c = y_c / (h + 1e-8)

    return tf.cast(_to_ri(y_c), input_dtype)
