import jax
import jax.numpy as jnp
import jax.random
import jax.typing
import matplotlib.axes

from gpbo._src.objective_functions import core


class UnivariateObjectiveFunction(core.ObjectiveFunction):
    def evaluate(self, key: jax.Array, xs: jax.Array) -> jax.Array:
        result = jnp.sin(xs) + jnp.sin((10.0 / 3.0) * xs)
        return result

    @property
    def dataset_bounds(self) -> tuple[core.Boundary, ...]:
        return (core.Boundary(2.0, 8.0, float),)

    def plot(self, axis: matplotlib.axes.Axes, xs: jax.Array, ys: jax.Array) -> None:
        if xs.shape != ys.shape:
            raise ValueError(
                f"xs and ys must have the same shape. xs.shape: {xs.shape}, ys.shape: {ys.shape}"
            )
        # else...
        axis.scatter(xs, ys)
