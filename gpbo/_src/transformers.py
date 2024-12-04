import abc

import jax
import jax.numpy as jnp

from gpbo import datasets, scalers


class Transformer(abc.ABC):
    @abc.abstractmethod
    def transform_dataset(
        self,
        dataset: datasets.Dataset,
    ) -> tuple[datasets.Dataset, datasets.Dataset | None, datasets.Dataset | None]:
        pass

    @abc.abstractmethod
    def transform_values(
        self,
        values: jax.Array,
        center: jax.Array | None,
        scale: jax.Array | None,
    ) -> jax.Array:
        pass

    @abc.abstractmethod
    def inverse_transform_values(
        self,
        values: jax.Array,
        center: jax.Array | None,
        scale: jax.Array | None,
    ) -> jax.Array:
        pass

    @abc.abstractmethod
    def inverse_transform_y_stds(
        self,
        stds: jax.Array,
        center: jax.Array | None,
        scale: jax.Array | None,
    ) -> jax.Array:
        pass


class Identity(Transformer):
    def transform_dataset(
        self,
        dataset: datasets.Dataset,
    ) -> tuple[datasets.Dataset, datasets.Dataset | None, datasets.Dataset | None]:
        return (
            dataset,
            datasets.Dataset(
                jnp.zeros((), dtype=dataset.xs.dtype),
                jnp.zeros((), dtype=dataset.ys.dtype),
            ),
            datasets.Dataset(
                jnp.ones((), dtype=dataset.xs.dtype),
                jnp.ones((), dtype=dataset.ys.dtype),
            ),
        )

    def transform_values(
        self,
        values: jax.Array,
        center: jax.Array | None,
        scale: jax.Array | None,
    ) -> jax.Array:
        return values

    def inverse_transform_values(
        self,
        values: jax.Array,
        center: jax.Array | None,
        scale: jax.Array | None,
    ) -> jax.Array:
        return values

    def inverse_transform_y_stds(
        self,
        stds: jax.Array,
        center: jax.Array | None,
        scale: jax.Array | None,
    ) -> jax.Array:
        return stds


class Standardizer(Transformer):
    def transform_dataset(
        self,
        dataset: datasets.Dataset,
    ) -> tuple[datasets.Dataset, datasets.Dataset | None, datasets.Dataset | None]:
        return datasets.standardize_dataset(dataset)

    def transform_values(
        self,
        values: jax.Array,
        center: jax.Array | None,
        scale: jax.Array | None,
    ) -> jax.Array:
        return (values - center) / scale

    def inverse_transform_values(
        self,
        values: jax.Array,
        center: jax.Array | None,
        scale: jax.Array | None,
    ) -> jax.Array:
        return values * scale + center

    def inverse_transform_y_stds(
        self,
        stds: jax.Array,
        center: jax.Array | None,
        scale: jax.Array | None,
    ) -> jax.Array:
        return stds * scale


class StandardizerXsOnly(Transformer):
    def transform_dataset(
        self,
        dataset: datasets.Dataset,
    ) -> tuple[datasets.Dataset, datasets.Dataset | None, datasets.Dataset | None]:
        xs, xs_mean, xs_std = scalers.standardize(dataset.xs)
        return (
            datasets.Dataset(xs, dataset.ys),
            datasets.Dataset(xs_mean, jnp.zeros((), dtype=dataset.ys.dtype)),
            datasets.Dataset(xs_std, jnp.ones((), dtype=dataset.ys.dtype)),
        )

    def transform_values(
        self,
        values: jax.Array,
        center: jax.Array | None,
        scale: jax.Array | None,
    ) -> jax.Array:
        return (values - center) / scale

    def inverse_transform_values(
        self,
        values: jax.Array,
        center: jax.Array | None,
        scale: jax.Array | None,
    ) -> jax.Array:
        return values * scale + center

    def inverse_transform_y_stds(
        self,
        stds: jax.Array,
        center: jax.Array | None,
        scale: jax.Array | None,
    ) -> jax.Array:
        return stds


class MinMaxScaler(Transformer):
    def transform_dataset(
        self,
        dataset: datasets.Dataset,
    ) -> tuple[datasets.Dataset, datasets.Dataset | None, datasets.Dataset | None]:
        (
            transformed_dataset,
            dataset_mins,
            dataset_maxs,
        ) = datasets.min_max_scale_dataset(dataset)
        return (
            transformed_dataset,
            datasets.Dataset(dataset_mins.xs, dataset_mins.ys),
            datasets.Dataset(
                dataset_maxs.xs - dataset_mins.xs, dataset_maxs.ys - dataset_mins.ys
            ),
        )

    def transform_values(
        self,
        values: jax.Array,
        center: jax.Array | None,
        scale: jax.Array | None,
    ) -> jax.Array:
        return (values - center) / scale

    def inverse_transform_values(
        self,
        values: jax.Array,
        center: jax.Array | None,
        scale: jax.Array | None,
    ) -> jax.Array:
        return values * scale + center

    def inverse_transform_y_stds(
        self,
        stds: jax.Array,
        center: jax.Array | None,
        scale: jax.Array | None,
    ) -> jax.Array:
        return stds * scale


class MeanCenterer(Transformer):
    def transform_dataset(
        self,
        dataset: datasets.Dataset,
    ) -> tuple[datasets.Dataset, datasets.Dataset | None, datasets.Dataset | None]:
        return (
            *datasets.mean_center_dataset(dataset),
            datasets.Dataset(
                jnp.ones((), dtype=dataset.xs.dtype),
                jnp.ones((), dtype=dataset.ys.dtype),
            ),
        )

    def transform_values(
        self,
        values: jax.Array,
        center: jax.Array | None,
        scale: jax.Array | None,
    ) -> jax.Array:
        return values - center

    def inverse_transform_values(
        self,
        values: jax.Array,
        center: jax.Array | None,
        scale: jax.Array | None,
    ) -> jax.Array:
        return values + center

    def inverse_transform_y_stds(
        self,
        stds: jax.Array,
        center: jax.Array | None,
        scale: jax.Array | None,
    ) -> jax.Array:
        return stds


class MeanCentererXsOnly(Transformer):
    def transform_dataset(
        self,
        dataset: datasets.Dataset,
    ) -> tuple[datasets.Dataset, datasets.Dataset | None, datasets.Dataset | None]:
        (
            transformed_dataset,
            dataset_means,
        ) = datasets.mean_center_dataset(dataset)
        return (
            transformed_dataset,
            datasets.Dataset(dataset_means.xs, jnp.zeros((), dtype=dataset.ys.dtype)),
            datasets.Dataset(
                jnp.ones((), dtype=dataset.xs.dtype),
                jnp.ones((), dtype=dataset.ys.dtype),
            ),
        )

    def transform_values(
        self,
        values: jax.Array,
        center: jax.Array | None,
        scale: jax.Array | None,
    ) -> jax.Array:
        return values - center

    def inverse_transform_values(
        self,
        values: jax.Array,
        center: jax.Array | None,
        scale: jax.Array | None,
    ) -> jax.Array:
        return values + center

    def inverse_transform_y_stds(
        self,
        stds: jax.Array,
        center: jax.Array | None,
        scale: jax.Array | None,
    ) -> jax.Array:
        return stds
