import typing
from typing import Callable
import jax.numpy as jnp
import jax
import jax.typing


class State(typing.NamedTuple):
    log_amplitude: jax.Array
    log_length_scale: jax.Array
    log_noise_scale: jax.Array

    @property
    def amplitude_squared(self) -> jax.Array:
        return jnp.exp(self.log_amplitude * 2)

    @property
    def length_scale(self) -> jax.Array:
        return jnp.exp(self.log_length_scale)

    @property
    def noise_scale_squared(self) -> jax.Array:
        return jnp.exp(self.log_noise_scale * 2)


Kernel = Callable[[State, jax.Array, jax.Array], jax.Array]


def gaussian(state: State, xs: jax.Array, ys: jax.Array) -> jax.Array:
    if xs.ndim == 1:
        xs = jnp.expand_dims(xs, axis=-1)
    if ys.ndim == 1:
        ys = jnp.expand_dims(ys, axis=-1)
    differences = xs[:, None] - ys[None, :]
    squared_differences = jnp.square(differences)
    squared_distances = jnp.sum(squared_differences, axis=2)
    return state.amplitude_squared * jnp.exp(
        -squared_distances / jnp.square(2 * state.length_scale)
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
    tmp_calc = jnp.sqrt(3 * tmp_calc) / state.length_scale
    return state.amplitude_squared * (1 + tmp_calc) * jnp.exp(-tmp_calc)
