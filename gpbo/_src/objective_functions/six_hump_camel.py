import jax
import jax.numpy as jnp
import jax.random
import jax.typing
import matplotlib.axes

from gpbo._src.objective_functions import core


class SixHumpCamelObjectiveFunction(core.ObjectiveFunction):
    def evaluate(self, key: jax.Array, xs: jax.Array, ys: jax.Array) -> jax.Array:
        x2 = xs**2
        x4 = xs**4
        y2 = ys**2
        return (4.0 - 2.1 * x2 + (x4 / 3.0)) * x2 + xs * ys + (-4.0 + 4.0 * y2) * y2

    @property
    def dataset_bounds(self) -> tuple[core.Boundary, ...]:
        return (
            core.Boundary(-3.0, 3.0, float),
            core.Boundary(-2.0, 2.0, float),
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
