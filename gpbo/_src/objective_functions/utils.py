from typing import Iterable
import jax
import jax.numpy as jnp
import jax.random
from gpbo._src.objective_functions import core
import jax.typing


def sample(
    key: jax.Array, n: int, dataset_bounds: tuple[core.Boundary, ...]
) -> jax.Array:
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


def get_ticks(
    boundary: core.Boundary,
    max_number_of_points: int,
) -> jax.Array:
    ticks = jnp.linspace(
        boundary.min_value,
        boundary.max_value,
        max_number_of_points,
        dtype=jax.dtypes.canonicalize_dtype(boundary.dtype),
    )
    # When using integer dtypes, you can end up with duplicate values.
    ticks = jnp.unique(ticks)
    return ticks


def get_mesh_grid(
    boundary_ticks: Iterable[tuple[core.Boundary, int]], sparse: bool, indexing='xy'
) -> list[jax.Array]:
    """Note that when dealing with integer dtypes, the number of points is not guaranteed to be respected."""
    grid_points = (
        get_ticks(boundary, number_of_points)
        for boundary, number_of_points in boundary_ticks
    )
    return jnp.meshgrid(*grid_points, sparse=sparse, indexing=indexing)
