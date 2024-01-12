import jax
import jax.numpy as jnp


def standardize(
    xs: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    mean = jnp.mean(xs, axis=0)
    std = jnp.std(xs, axis=0)
    return (xs - mean) / std, mean, std


def inverse_standardize(xs: jax.Array, mean: jax.Array, std: jax.Array) -> jax.Array:
    return xs * std + mean


def inverse_standardize_std(xs: jax.Array, std: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.square(xs) * jnp.square(std))


def min_max_scale(
    xs: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    min = jnp.min(xs, axis=0)
    max = jnp.max(xs, axis=0)
    return (xs - min) / (max - min), min, max
