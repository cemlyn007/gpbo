import abc

import jax
import jax.numpy as jnp
import jax.random
import jax.scipy.stats.norm
import jax.typing

from gpbo._src import datasets, gaussian_process, kernels


class AcquisitionFunction(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        kernel: kernels.Kernel,
        state: kernels.State,
        dataset: datasets.Dataset,
        xs: jax.Array,
    ) -> jax.Array:
        "Compute the utility of the given points."

    @abc.abstractmethod
    def compute_arg_sort(
        self,
        kernel: kernels.Kernel,
        state: kernels.State,
        dataset: datasets.Dataset,
        xs: jax.Array,
    ) -> jax.Array:
        pass


def expected_improvement(
    mean: jax.Array, std: jax.Array, min_y: jax.Array
) -> jax.Array:
    gamma = (min_y - mean) / std
    utility = std * (
        gamma * jax.scipy.stats.norm.cdf(gamma) + jax.scipy.stats.norm.pdf(gamma)
    )
    return utility


class ExpectedImprovement(AcquisitionFunction):
    def __call__(
        self,
        kernel: kernels.Kernel,
        state: kernels.State,
        dataset: datasets.Dataset,
        xs: jax.Array,
    ) -> jax.Array:
        mean, std = gaussian_process.get_mean_and_std(kernel, state, dataset, xs)
        return expected_improvement(mean, std, jnp.min(dataset.ys))

    def compute_arg_sort(
        self,
        kernel: kernels.Kernel,
        state: kernels.State,
        dataset: datasets.Dataset,
        xs: jax.Array,
    ) -> jax.Array:
        utility = self(kernel, state, dataset, xs)
        return jnp.flip(jnp.argsort(utility))


def lower_confidence_bound(
    mean: jax.Array, std: jax.Array, confidence_rate: jax.Array
) -> jax.Array:
    return -1 * (mean - jnp.sqrt(confidence_rate) * std)


class LowerConfidenceBound(AcquisitionFunction):
    def __init__(self, confidence_rate: float) -> None:
        super(LowerConfidenceBound, self).__init__()
        if confidence_rate < 0:
            raise ValueError(
                f"Confidence rate must be non-negative. confidence_rate: {
                    confidence_rate}"
            )
        self._confidence_rate = confidence_rate

    def __call__(
        self,
        kernel: kernels.Kernel,
        state: kernels.State,
        dataset: datasets.Dataset,
        xs: jax.Array,
    ) -> jax.Array:
        mean, std = gaussian_process.get_mean_and_std(kernel, state, dataset, xs)
        utility = lower_confidence_bound(mean, std, self._confidence_rate)
        return utility

    def compute_arg_sort(
        self,
        kernel: kernels.Kernel,
        state: kernels.State,
        dataset: datasets.Dataset,
        xs: jax.Array,
    ) -> jax.Array:
        utility = self(kernel, state, dataset, xs)
        return jnp.flip(jnp.argsort(utility))
