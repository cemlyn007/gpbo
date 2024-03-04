
from gpbo._src.objective_functions import core
import jax
import jax.numpy as jnp
import jax.random
import matplotlib.axes
import jax.typing
import numpy


class MeshGridObjectiveFunction(core.ObjectiveFunction):
    def __init__(self, xs: jax.Array, ys: jax.Array) -> None:
        self._xs = xs
        for _ in range(1, self._xs.ndim):
            if self._xs.shape[1] == 1:
                self._xs = self._xs[:, 0]
        self._ys = ys

    def evaluate(self, key: jax.Array, *xs: jax.Array) -> jax.Array:
        if len(xs) != 1:
            broadcasted_arrays = jnp.broadcast_arrays(*xs)
            dense_xs =  jnp.stack(broadcasted_arrays, axis=-1)
        else:
            dense_xs = xs[0]

        has_batch_dimension = len(self._xs.shape) == len(dense_xs.shape)
        if has_batch_dimension:
            if dense_xs.shape[1:] != self._xs.shape[1:]:
                raise ValueError('The input array must have the same shape as the mesh grid.')
        else:
            if dense_xs.shape != self._xs.shape[1:]:
                raise ValueError('The input array must have the same shape as the mesh grid.')
            
        if has_batch_dimension:
            if self._xs.ndim == 1:
                indices = jnp.nonzero(jnp.equal(self._xs, jnp.expand_dims(dense_xs, 1)))[1]
            else:
                indices = jnp.nonzero(jnp.equal(self._xs, dense_xs).all(-1))[-1]
        else:
            if self._xs.ndim == 1:
                indices, = jnp.nonzero(jnp.equal(self._xs, dense_xs))
            else:
                indices, = jnp.nonzero(jnp.all(jnp.equal(self._xs, dense_xs), -1))

        if has_batch_dimension:
            if (dense_xs.shape[0] if dense_xs.shape else 1) != len(indices):
                raise ValueError('Some values are not in the dataset.')
        else:
            if len(indices) == 0:
                raise ValueError('Some values are not in the dataset.')
        ys = self._ys[indices].copy()
        if dense_xs.ndim == 0:
            ys = jnp.reshape(ys, ())
        return ys

    @property
    def dataset_bounds(self) -> tuple[core.Boundary, ...]:
        min_x = jnp.min(self._xs, axis=0)
        max_x = jnp.max(self._xs, axis=0)
        if self._xs.ndim == 1:
            return tuple(
                core.Boundary(min_x, max_x, dtype=self._dtype_to_python_type(self._xs.dtype))
                for i in range(self._xs.ndim)
            )
        else:
            return tuple(
                core.Boundary(min_x[i], max_x[i], dtype=self._dtype_to_python_type(self._xs.dtype))
                for i in range(self._xs.shape[1])
            )

    def plot(self, axis: matplotlib.axes.Axes, *xs: jax.Array) -> None:
        raise NotImplementedError
    
    def _dtype_to_python_type(self, dtype) -> type:
        return {numpy.dtypes.Float32DType(): float, numpy.dtypes.Float64DType(): float}[dtype]
