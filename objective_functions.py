from typing import Iterable
import jax
import jax.numpy as jnp
import jax.random
import abc
import typing
import matplotlib.axes
import jax.typing


class Boundary[T: int | float](typing.NamedTuple):
    min_value: T
    max_value: T
    dtype: type[T]


class ObjectiveFunction(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, key: jax.Array, *xs: jax.Array) -> jax.Array:
        pass

    @property
    @abc.abstractmethod
    def dataset_bounds(self) -> tuple[Boundary, ...]:
        pass

    @abc.abstractmethod
    def plot(self, axis: matplotlib.axes.Axes, *xs: jax.Array) -> None:
        pass


class NoisyObjectiveFunction[T: ObjectiveFunction](ObjectiveFunction):
    def __init__(
        self,
        objective_function: T,
        additional_gaussian_noise_std: float,
    ) -> None:
        self._assert_compatible(objective_function)
        self._objective_function = objective_function
        self._additional_gaussian_noise_std = additional_gaussian_noise_std

    def _assert_compatible(self, objective_function: ObjectiveFunction) -> None:
        for i, boundary in enumerate(objective_function.dataset_bounds):
            if boundary.dtype != float:
                raise ValueError(
                    f"Only float dtypes are supported. boundary.dtype at position {i}: {boundary.dtype}"
                )

    def evaluate(self, key: jax.Array, *xs: jax.Array) -> jax.Array:
        key, noise_key = jax.random.split(key)
        y = self._objective_function.evaluate(key, *xs)
        noise = jax.random.normal(noise_key, y.shape, y.dtype)
        noisy_y = self._additional_gaussian_noise_std * noise + y
        return noisy_y

    def dataset_bounds(self) -> tuple[Boundary, ...]:
        return self._objective_function.dataset_bounds

    def plot(self, axis: matplotlib.axes.Axes, *xs: jax.Array) -> None:
        self._objective_function.plot(axis, *xs)


class UnivariateObjectiveFunction(ObjectiveFunction):
    def evaluate(self, key: jax.Array, xs: jax.Array) -> jax.Array:
        result = jnp.sin(xs) + jnp.sin((10.0 / 3.0) * xs)
        return result

    @property
    def dataset_bounds(self) -> tuple[Boundary, ...]:
        return (Boundary(2.0, 8.0, float),)

    def plot(self, axis: matplotlib.axes.Axes, xs: jax.Array, ys: jax.Array) -> None:
        if xs.shape != ys.shape:
            raise ValueError(
                f"xs and ys must have the same shape. xs.shape: {xs.shape}, ys.shape: {ys.shape}"
            )
        # else...
        axis.scatter(xs, ys)


class SixHumpCamelObjectiveFunction(ObjectiveFunction):
    def evaluate(self, key: jax.Array, xs: jax.Array, ys: jax.Array) -> jax.Array:
        x2 = xs**2
        x4 = xs**4
        y2 = ys**2
        return (4.0 - 2.1 * x2 + (x4 / 3.0)) * x2 + xs * ys + (-4.0 + 4.0 * y2) * y2

    @property
    def dataset_bounds(self) -> tuple[Boundary, ...]:
        return (
            Boundary(-3.0, 3.0, float),
            Boundary(-2.0, 2.0, float),
        )

    def plot(
        self, axis: matplotlib.axes.Axes, xs: jax.Array, ys: jax.Array, zs: jax.Array
    ) -> None:
        """xs and ys must be 1-dimensional - likely made from using meshgrid with flatten.
        zs must be 2-dimensional."""
        if xs.ndim != 1:
            raise ValueError(f"xs must be 1-dimensional. xs.ndim: {xs.ndim}")
        elif ys.ndim != 1:
            raise ValueError(f"ys must be 1-dimensional. ys.ndim: {ys.ndim}")
        elif zs.ndim != 2:
            raise ValueError(f"zs must be 2-dimensional. zs.ndim: {zs.ndim}")
        # else...
        levels = jnp.arange(-1.5, 10, 0.5, dtype=xs.dtype)
        axis.contourf(xs, ys, zs, levels=levels)


def sample(key: jax.Array, n: int, dataset_bounds: tuple[Boundary, ...]) -> jax.Array:
    keys = jax.random.split(key, len(dataset_bounds))
    xs = jnp.stack(
        [
            jax.random.uniform(
                key,
                (n,),
                minval=boundary.min_value,
                maxval=boundary.max_value,
                dtype=boundary.dtype,
            )
            if boundary.dtype == float
            else jax.random.randint(
                key,
                (n,),
                minval=boundary.min_value,
                maxval=boundary.max_value + 1,
                dtype=boundary.dtype,
            )
            for key, boundary in zip(keys, dataset_bounds, strict=True)
        ],
        axis=1,
    )
    if len(dataset_bounds) == 1:
        xs = xs.squeeze(axis=1)
    return xs


def get_mesh_grid(boundary_ticks: Iterable[tuple[Boundary, int]]) -> list[jax.Array]:
    """Note that when dealing with integer dtypes, the number of points is not guaranteed to be respected."""
    list_grid_points = []
    for boundary, number_of_points in boundary_ticks:
        ticks = jnp.linspace(
            boundary.min_value,
            boundary.max_value,
            number_of_points,
            dtype=jax.dtypes.canonicalize_dtype(boundary.dtype),
        )
        # When using integer dtypes, you can end up with duplicate values.
        ticks = jnp.unique(ticks)
        list_grid_points.append(ticks)
    return jnp.meshgrid(*list_grid_points, sparse=True)
