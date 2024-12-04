import csv

import jax
import jax.numpy as jnp
import jax.random
import jax.typing
import matplotlib.axes

from gpbo._src.objective_functions import core


class CsvObjectiveFunction(core.ObjectiveFunction):
    def __init__(self, file_path: str) -> None:
        super().__init__()
        self._file_path = file_path

        with open(file_path, "r") as f:
            reader = csv.reader(f)
            data = [list(map(float, row)) for row in reader]

        self._data = jnp.array(data)
        self._xs = self._data[:, :-1]
        for _ in range(1, self._xs.ndim):
            if self._xs.shape[1] == 1:
                self._xs = self._xs[:, 0]
        self._ys = self._data[:, -1]

    def evaluate(self, key: jax.Array, *xs: jax.Array) -> jax.Array:
        if len(xs) != 1:
            raise ValueError("CsvObjectiveFunction only supports dense arrays.")
        # else...
        dense_xs = xs[0]
        mask = jnp.all(jnp.isin(self._xs, dense_xs), axis=list(range(1, self._xs.ndim)))
        if dense_xs.shape == (self._xs.ndim,):
            if sum(mask) == 0:
                raise ValueError("Some values are not in the dataset.")
        else:
            if (dense_xs.shape[0] if dense_xs.shape else 1) != sum(mask):
                raise ValueError("Some values are not in the dataset.")
        ys = self._ys[mask].copy()
        return ys

    @property
    def dataset_bounds(self) -> tuple[core.Boundary, ...]:
        min_x = jnp.min(self._xs, axis=0)
        max_x = jnp.max(self._xs, axis=0)
        if self._xs.ndim == 1:
            min_x = jnp.expand_dims(min_x, 0)
            max_x = jnp.expand_dims(max_x, 0)
        return tuple(
            core.Boundary(
                min_x[i],
                max_x[i],
                dtype={jnp.float32: float, jnp.float64: float}[self._xs.dtype],
            )
            for i in range(self._xs.ndim)
        )

    def plot(self, axis: matplotlib.axes.Axes, *xs: jax.Array) -> None:
        raise NotImplementedError
