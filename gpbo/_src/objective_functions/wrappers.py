from gpbo._src.objective_functions import core
import jax
import jax.numpy as jnp
import matplotlib.axes


class DtypeCasterObjectiveFunction[T: core.ObjectiveFunction](core.ObjectiveFunction):
    def __init__(self, objective_function: T) -> None:
        self._objective_function = objective_function

    def evaluate(self, key: jax.Array, *xs: jax.Array) -> jax.Array:
        xs = jax.tree_map(lambda x: x.astype(jnp.float32), xs)
        y = self._objective_function.evaluate(key, *xs)
        return y.astype(jnp.float64)

    @property
    def dataset_bounds(self) -> tuple[core.Boundary, ...]:
        return self._objective_function.dataset_bounds

    def plot(self, axis: matplotlib.axes.Axes, *xs: jax.Array) -> None:
        self._objective_function.plot(axis, *xs)


class NoisyObjectiveFunction[T: core.ObjectiveFunction](core.ObjectiveFunction):
    def __init__(
        self,
        objective_function: T,
        additional_gaussian_noise_std: float,
    ) -> None:
        self._assert_compatible(objective_function)
        self._objective_function = objective_function
        self._additional_gaussian_noise_std = additional_gaussian_noise_std

    def _assert_compatible(self, objective_function: core.ObjectiveFunction) -> None:
        for i, boundary in enumerate(objective_function.dataset_bounds):
            if boundary.dtype != float:
                raise ValueError(
                    f"Only float dtypes are supported. boundary.dtype at position {i}: {boundary.dtype}"
                )

    def evaluate(self, key: jax.Array, *xs: jax.Array) -> jax.Array:
        key, noise_key = jax.random.split(key)
        y = self._objective_function.evaluate(key, *xs)
        noise = jax.random.normal(noise_key, y.shape, y.dtype)
        noisy_y = y + self._additional_gaussian_noise_std * noise
        return noisy_y

    @property
    def dataset_bounds(self) -> tuple[core.Boundary, ...]:
        return self._objective_function.dataset_bounds

    def plot(self, axis: matplotlib.axes.Axes, *xs: jax.Array) -> None:
        self._objective_function.plot(axis, *xs)


class JitObjectiveFunction[T: core.ObjectiveFunction](core.ObjectiveFunction):
    def __init__(self, objective_function: T) -> None:
        self._objective_function = objective_function
        self._evaluate = jax.jit(self._objective_function.evaluate)

    def evaluate(self, key: jax.Array, *xs: jax.Array) -> jax.Array:
        return self._evaluate(key, *xs)

    @property
    def dataset_bounds(self) -> tuple[core.Boundary, ...]:
        return self._objective_function.dataset_bounds

    def plot(self, axis: matplotlib.axes.Axes, *xs: jax.Array) -> None:
        self._objective_function.plot(axis, *xs)
