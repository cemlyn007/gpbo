import jax
import jaxlib.xla_extension
import tqdm
import jax.numpy as jnp
from gpbo._src import kernels
from gpbo._src import datasets
from typing import Callable


class Utility:
    def __init__(self, acquisition_function: Callable[[jax.Array, jax.Array, jax.Array], jax.Array], initial_batch_size: int, device: jax.Device, fallback_device: jax.Device) -> None:
        self._acquisition_function = acquisition_function
        self._device = device
        self._fallback_device = fallback_device
        self._initial_batch_size = initial_batch_size
        self._batch_size = initial_batch_size

    def get_candidate_utilities(self, mean, std, dataset):
        total_candidates = mean.shape[0]
        self._batch_size = min(self._batch_size, total_candidates)
        mean = jax.device_put(mean, self._device)
        std = jax.device_put(std, self._device)
        dataset = jax.device_put(dataset, self._device)
        while True:
            total_batches, remainder = divmod(total_candidates, self._batch_size)
            if remainder > 0:
                raise ValueError("Failed to get candidate utilities")
            batched_mean = mean.reshape(total_batches, self._batch_size)
            batched_std = std.reshape(total_batches, self._batch_size)
            try:
                return jnp.concatenate(
                    [self._acquisition_function(
                        batch_mean, batch_std, dataset
                    ) for batch_mean, batch_std in tqdm.tqdm(zip(batched_mean, batched_std), desc="Getting candidate utilities", total=total_batches)],
                    axis=None,
                ).flatten()
            except jaxlib.xla_extension.XlaRuntimeError:
                self._batch_size = self._batch_size // 2
                if self._batch_size == 0:
                    if self._device == self._fallback_device:
                        raise ValueError("Failed to get candidate utilities")
                    # else...
                    self._device = self._fallback_device
                    self._batch_size = min(total_candidates, self._initial_batch_size)


class DynamicGetMeanAndStd:
    def __init__(self, 
                 get_mean_and_std: Callable[[kernels.Kernel, kernels.State, datasets.Dataset, jax.Array], tuple[jax.Array, jax.Array]], 
                 initial_batch_size: int,
                 device: jax.Device, 
                 fallback_device: jax.Device) -> None:
        self._get_mean_and_std = get_mean_and_std
        self._device = device
        self._fallback_device = fallback_device
        self._initial_batch_size = initial_batch_size
        self._batch_size = initial_batch_size

    def get_mean_and_std(self, kernel: kernels.Kernel,
                            state: kernels.State,
                            dataset: datasets.Dataset,
                            xs: jax.Array):
        total_candidates = xs.shape[0]
        self._batch_size = min(self._batch_size, xs.shape[0])
        state = jax.device_put(state, self._device)
        dataset = jax.device_put(dataset, self._device)
        xs = jax.device_put(xs, self._device)
        while True:
            batched_xs = xs.reshape(-1, self._batch_size, xs.shape[-1])
            try:
                batched_means_and_stds = [self._get_mean_and_std(
                    kernel, state, dataset, batch_xs
                ) for batch_xs in tqdm.tqdm(batched_xs, desc="Getting mean and std")]
                return tuple(
                    jnp.concatenate(
                        [mean_and_std[i] for mean_and_std in batched_means_and_stds],
                        axis=None,
                    ).flatten()
                    for i in range(2)
                )
            except jaxlib.xla_extension.XlaRuntimeError:
                self._batch_size = self._batch_size // 2
                if self._batch_size == 0:
                    if self._device == self._fallback_device:
                        raise ValueError("Failed to get candidate utilities")
                    # else...
                    self._device = self._fallback_device
                    self._batch_size = min(total_candidates, self._initial_batch_size)
