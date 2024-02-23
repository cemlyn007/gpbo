import jax
import jax.numpy as jnp
import jax.random
import abc
import jax.typing
import jax.scipy.stats.norm
from gpbo._src import kernels
from gpbo._src import gaussian_process
from gpbo._src import datasets


class AcquisitionFunction(abc.ABC):
    @abc.abstractmethod
    def __call__(self, kernel: kernels.Kernel,
                 state: kernels.State,
                 dataset: datasets.Dataset,
                 xs: jax.Array,) -> jax.Array:
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


def expected_improvement(mean: jax.Array, std: jax.Array, min_y: jax.Array) -> jax.Array:
    gamma = (min_y - mean) / std
    utility = std * (
        gamma * jax.scipy.stats.norm.cdf(gamma) +
        jax.scipy.stats.norm.pdf(gamma)
    )
    return utility


class ExpectedImprovement(AcquisitionFunction):
    def __call__(self, kernel: kernels.Kernel,
                 state: kernels.State,
                 dataset: datasets.Dataset,
                 xs: jax.Array) -> jax.Array:
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


class LowerConfidenceBound(AcquisitionFunction):
    def __init__(self, confidence_rate: float) -> None:
        super(LowerConfidenceBound, self).__init__()
        if confidence_rate < 0:
            raise ValueError(
                f"Confidence rate must be non-negative. confidence_rate: {
                    confidence_rate}"
            )
        self._confidence_rate = confidence_rate

    def __call__(self, kernel: kernels.Kernel,
                 state: kernels.State,
                 dataset: datasets.Dataset,
                 xs: jax.Array) -> jax.Array:
        mean, std = gaussian_process.get_mean_and_std(
            kernel, state, dataset, xs)
        utility = -1 * (mean - jnp.sqrt(self._confidence_rate) * std)
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
