import jax.numpy as jnp
import jax
import typing


class Dataset(typing.NamedTuple):
    xs: jax.Array
    ys: jax.Array


def standardize(
    xs: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    mean = jnp.mean(xs, axis=0)
    std = jnp.std(xs, axis=0)
    return (xs - mean) / std, mean, std


def standardize_dataset(
    dataset: Dataset,
) -> tuple[Dataset, Dataset, Dataset]:
    xs, xs_mean, xs_std = standardize(dataset.xs)
    ys, ys_mean, ys_std = standardize(dataset.ys)
    return Dataset(xs, ys), Dataset(xs_mean, ys_mean), Dataset(xs_std, ys_std)


def min_max_scale(
    xs: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    min = jnp.min(xs, axis=0)
    max = jnp.max(xs, axis=0)
    return (xs - min) / (max - min), min, max


def min_max_scale_dataset(
    dataset: Dataset,
) -> tuple[Dataset, Dataset, Dataset]:
    xs, xs_min, xs_max = min_max_scale(dataset.xs)
    ys, ys_min, ys_max = min_max_scale(dataset.ys)
    return Dataset(xs, ys), Dataset(xs_min, ys_min), Dataset(xs_max, ys_max)


def mean_center_dataset(
    dataset: Dataset,
) -> tuple[Dataset, Dataset]:
    xs_mean = jnp.mean(dataset.xs, axis=0)
    ys_mean = jnp.mean(dataset.ys, axis=0)
    return Dataset(dataset.xs - xs_mean, dataset.ys - ys_mean), Dataset(
        xs_mean, ys_mean
    )
