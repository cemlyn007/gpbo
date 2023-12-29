import typing
from typing import Callable
import jax.numpy as jnp
import jax
import jax.typing


class State(typing.NamedTuple):
    log_amplitude: jax.Array
    log_length_scale: jax.Array
    log_noise_scale: jax.Array


def amplitude_squared(state: State) -> jax.Array:
    return jnp.exp(state.log_amplitude * 2)


def length_scale(state: State) -> jax.Array:
    return jnp.exp(state.log_length_scale)


def noise_scale_squared(state: State) -> jax.Array:
    return jnp.exp(state.log_noise_scale * 2)


Kernel = Callable[[State, jax.Array, jax.Array], jax.Array]


def euclidean_squared_distance_matrix(xs: jax.Array, ys: jax.Array) -> jax.Array:
    if xs.ndim == 1:
        xs = jnp.expand_dims(xs, axis=-1)
    if ys.ndim == 1:
        ys = jnp.expand_dims(ys, axis=-1)
    differences = xs[:, None] - ys[None, :]
    squared_differences = jnp.square(differences)
    squared_distances = jnp.sum(squared_differences, axis=2)
    return squared_distances


def gaussian(state: State, xs: jax.Array, ys: jax.Array) -> jax.Array:
    if xs.ndim == 1:
        xs = jnp.expand_dims(xs, axis=-1)
    if ys.ndim == 1:
        ys = jnp.expand_dims(ys, axis=-1)
    squared_distances = euclidean_squared_distance_matrix(xs, ys)
    return amplitude_squared(state) * jnp.exp(
        -squared_distances / (2 * jnp.square(length_scale(state)))
    )


def matern(state: State, xs: jax.Array, ys: jax.Array) -> jax.Array:
    if xs.ndim == 1:
        xs = jnp.expand_dims(xs, axis=-1)
    if ys.ndim == 1:
        ys = jnp.expand_dims(ys, axis=-1)
    xnorms_2 = jnp.expand_dims(jnp.diag(xs.dot(xs.T)), axis=-1)
    ynorms_2 = jnp.expand_dims(jnp.diag(ys.dot(ys.T)), axis=-1)
    xnorms_2 = xnorms_2 @ jnp.ones((1, ynorms_2.shape[0]))
    ynorms_2 = jnp.ones((xnorms_2.shape[0], 1)) @ ynorms_2.T
    tmp_calc = xnorms_2 + ynorms_2 - 2 * xs @ ys.T
    tmp_calc = jnp.sqrt(3 * tmp_calc) / length_scale(state)
    return amplitude_squared(state) * (1 + tmp_calc) * jnp.exp(-tmp_calc)
