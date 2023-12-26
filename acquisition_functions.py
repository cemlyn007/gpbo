import jax
import jax.numpy as jnp
import jax.random
import abc
import jax.typing
import jax.scipy.stats.norm
import kernels
import gaussian_process


class AcquisitionFunction(abc.ABC):
    @abc.abstractmethod
    def compute_arg_sort(
        self,
        kernel: kernels.Kernel,
        state: kernels.State,
        dataset: gaussian_process.Dataset,
        xs: jax.Array,
    ) -> jax.Array:
        pass


class ExpectedImprovement(AcquisitionFunction):
    def compute_arg_sort(
        self,
        kernel: kernels.Kernel,
        state: kernels.State,
        dataset: gaussian_process.Dataset,
        xs: jax.Array,
    ) -> jax.Array:
        mean, std = gaussian_process.get_mean_and_std(kernel, state, dataset, xs)
        gamma = (jnp.min(dataset.ys) - mean) / std
        utility = std * (
            gamma * jax.scipy.stats.norm.cdf(gamma) + jax.scipy.stats.norm.pdf(gamma)
        )
        return jnp.flip(jnp.argsort(utility))


class LowerConfidenceBound(AcquisitionFunction):
    def __init__(self, confidence_rate: float) -> None:
        super(LowerConfidenceBound, self).__init__()
        if confidence_rate < 0:
            raise ValueError(
                f"Confidence rate must be non-negative. confidence_rate: {confidence_rate}"
            )
        self._confidence_rate = confidence_rate

    def compute_arg_sort(
        self,
        kernel: kernels.Kernel,
        state: kernels.State,
        dataset: gaussian_process.Dataset,
        xs: jax.Array,
    ) -> jax.Array:
        mean, std = gaussian_process.get_mean_and_std(kernel, state, dataset, xs)
        utility = -1 * (mean - jnp.sqrt(self._confidence_rate) * std)
        return jnp.flip(jnp.argsort(utility))
