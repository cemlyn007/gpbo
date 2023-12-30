from typing import Iterable
import jax
import jax.numpy as jnp
import jax.random
import abc
import typing
import matplotlib.axes
import jax.typing
import http.client
import os
import gzip
import optax
from flax import linen as nn
from flax.training import train_state
import math


class Boundary[T: int | float](typing.NamedTuple):
    min_value: T
    max_value: T
    dtype: type[T]


class ObjectiveFunction(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, key: jax.Array, *xs: jax.Array) -> jax.Array:
        pass

    @property
    @abc.abstractmethod
    def dataset_bounds(self) -> tuple[Boundary, ...]:
        pass

    @abc.abstractmethod
    def plot(self, axis: matplotlib.axes.Axes, *xs: jax.Array) -> None:
        pass


class NoisyObjectiveFunction[T: ObjectiveFunction](ObjectiveFunction):
    def __init__(
        self,
        objective_function: T,
        additional_gaussian_noise_std: float,
    ) -> None:
        self._assert_compatible(objective_function)
        self._objective_function = objective_function
        self._additional_gaussian_noise_std = additional_gaussian_noise_std

    def _assert_compatible(self, objective_function: ObjectiveFunction) -> None:
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
    def dataset_bounds(self) -> tuple[Boundary, ...]:
        return self._objective_function.dataset_bounds

    def plot(self, axis: matplotlib.axes.Axes, *xs: jax.Array) -> None:
        self._objective_function.plot(axis, *xs)


class JitObjectiveFunction[T: ObjectiveFunction](ObjectiveFunction):
    def __init__(self, objective_function: T) -> None:
        self._objective_function = objective_function
        self._evaluate = jax.jit(self._objective_function.evaluate)

    def evaluate(self, key: jax.Array, *xs: jax.Array) -> jax.Array:
        return self._evaluate(key, *xs)

    @property
    def dataset_bounds(self) -> tuple[Boundary, ...]:
        return self._objective_function.dataset_bounds

    def plot(self, axis: matplotlib.axes.Axes, *xs: jax.Array) -> None:
        self._objective_function.plot(axis, *xs)


class UnivariateObjectiveFunction(ObjectiveFunction):
    def evaluate(self, key: jax.Array, xs: jax.Array) -> jax.Array:
        result = jnp.sin(xs) + jnp.sin((10.0 / 3.0) * xs)
        return result

    @property
    def dataset_bounds(self) -> tuple[Boundary, ...]:
        return (Boundary(2.0, 8.0, float),)

    def plot(self, axis: matplotlib.axes.Axes, xs: jax.Array, ys: jax.Array) -> None:
        if xs.shape != ys.shape:
            raise ValueError(
                f"xs and ys must have the same shape. xs.shape: {xs.shape}, ys.shape: {ys.shape}"
            )
        # else...
        axis.scatter(xs, ys)


class SixHumpCamelObjectiveFunction(ObjectiveFunction):
    def evaluate(self, key: jax.Array, xs: jax.Array, ys: jax.Array) -> jax.Array:
        x2 = xs**2
        x4 = xs**4
        y2 = ys**2
        return (4.0 - 2.1 * x2 + (x4 / 3.0)) * x2 + xs * ys + (-4.0 + 4.0 * y2) * y2

    @property
    def dataset_bounds(self) -> tuple[Boundary, ...]:
        return (
            Boundary(-3.0, 3.0, float),
            Boundary(-2.0, 2.0, float),
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


class MnistDataset:
    # http://yann.lecun.com/exdb/mnist/
    HOST_URL = "yann.lecun.com"
    TRAIN_IMAGES_RELATIVE_URL = "/exdb/mnist/train-images-idx3-ubyte.gz"
    TRAIN_LABELS_RELATIVE_URL = "/exdb/mnist/train-labels-idx1-ubyte.gz"
    TEST_IMAGES_RELATIVE_URL = "/exdb/mnist/t10k-images-idx3-ubyte.gz"
    TEST_LABELS_RELATIVE_URL = "/exdb/mnist/t10k-labels-idx1-ubyte.gz"

    TRAIN_IMAGES_FILE_NAME = "train-images-idx3-ubyte.gz"
    TRAIN_LABELS_FILE_NAME = "train-labels-idx1-ubyte.gz"
    TEST_IMAGES_FILE_NAME = "t10k-images-idx3-ubyte.gz"
    TEST_LABELS_FILE_NAME = "t10k-labels-idx1-ubyte.gz"

    def __init__(self, cache_directory: str) -> None:
        self._cache_directory = cache_directory

    def download(self) -> None:
        if not os.path.exists(self._cache_directory):
            os.makedirs(self._cache_directory)

        for relative_url, file_name in [
            (
                MnistDataset.TRAIN_IMAGES_RELATIVE_URL,
                MnistDataset.TRAIN_IMAGES_FILE_NAME,
            ),
            (
                MnistDataset.TRAIN_LABELS_RELATIVE_URL,
                MnistDataset.TRAIN_LABELS_FILE_NAME,
            ),
            (MnistDataset.TEST_IMAGES_RELATIVE_URL, MnistDataset.TEST_IMAGES_FILE_NAME),
            (MnistDataset.TEST_LABELS_RELATIVE_URL, MnistDataset.TEST_LABELS_FILE_NAME),
        ]:
            file_path = os.path.join(self._cache_directory, file_name)
            if not os.path.exists(file_path):
                self._download_file(relative_url, file_path)

    def _download_file(self, relative_url: str, file_path: str) -> None:
        connection = http.client.HTTPConnection(MnistDataset.HOST_URL)
        try:
            connection.request("GET", relative_url)
            response = connection.getresponse()
            if response.status != http.HTTPStatus.OK:
                raise RuntimeError(
                    f"Failed to download {relative_url} with status {response.status}"
                )
            # else...
            with open(file_path, "wb") as file:
                file.write(response.read())
        finally:
            connection.close()

    def load_train_images(self, device: jax.Device) -> jax.Array:
        return self._load_images(
            os.path.join(self._cache_directory, MnistDataset.TRAIN_IMAGES_FILE_NAME),
            2051,
            device,
        )

    def load_test_images(self, device: jax.Device) -> jax.Array:
        return self._load_images(
            os.path.join(self._cache_directory, MnistDataset.TEST_IMAGES_FILE_NAME),
            2051,
            device,
        )

    def _load_images(
        self, file_path: str, expected_magic_number: int, device: jax.Device
    ) -> jax.Array:
        with gzip.open(file_path) as f:
            magic_number = int.from_bytes(f.read(4), byteorder="big", signed=True)
            if magic_number != expected_magic_number:
                raise AssertionError(
                    f"Invalid magic number {magic_number} for MNIST images"
                )
            # else...
            number_of_images = int.from_bytes(f.read(4), byteorder="big", signed=True)
            number_of_rows = int.from_bytes(f.read(4), byteorder="big", signed=True)
            number_of_columns = int.from_bytes(f.read(4), byteorder="big", signed=True)
            images_in_bytes = f.read(
                number_of_images * number_of_rows * number_of_columns
            )

        with jax.default_device(device):
            flat_images = jnp.frombuffer(images_in_bytes, dtype=jnp.uint8)

        # JAX-Metal does not support reshaping on the GPU, so we fallback onto CPU and later
        # will put it back on the GPU.
        if device.platform == "METAL":
            device_for_reshape = jax.devices("cpu")[0]
        else:
            device_for_reshape = device
        images = jax.device_put(flat_images, device_for_reshape).reshape(
            number_of_images, number_of_rows, number_of_columns
        )
        return jax.device_put(images, device)

    def load_train_labels(self, device: jax.Device) -> jax.Array:
        return self._load_labels(
            os.path.join(self._cache_directory, MnistDataset.TRAIN_LABELS_FILE_NAME),
            2049,
            device,
        )

    def load_test_labels(self, device: jax.Device) -> jax.Array:
        return self._load_labels(
            os.path.join(self._cache_directory, MnistDataset.TEST_LABELS_FILE_NAME),
            2049,
            device,
        )

    def _load_labels(
        self, file_path: str, expected_magic_number: int, device: jax.Device
    ) -> jax.Array:
        with gzip.open(file_path) as f:
            magic_number = int.from_bytes(f.read(4), byteorder="big", signed=True)
            if magic_number != expected_magic_number:
                raise AssertionError(
                    f"Invalid magic number {magic_number} for MNIST labels"
                )
            # else...
            number_of_labels = int.from_bytes(f.read(4), byteorder="big", signed=True)
            labels_in_bytes = f.read(number_of_labels)
        with jax.default_device(device):
            labels = jnp.frombuffer(labels_in_bytes, dtype=jnp.uint8)
        return labels


class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype
        x = nn.Conv(features=32, kernel_size=(3, 3), param_dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3), param_dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256, param_dtype=dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10, param_dtype=dtype)(x)
        return x


class MnistObjectiveFunction(ObjectiveFunction):
    def __init__(
        self, cache_directory: str, add_momentum_dimension: bool, device: jax.Device
    ) -> None:
        super().__init__()
        self._dataset = MnistDataset(cache_directory)
        self._dataset.download()
        self._train_images = (
            jnp.expand_dims(self._dataset.load_train_images(device), axis=-1) / 255.0
        )
        self._train_labels = self._dataset.load_train_labels(device)
        self._test_images = (
            jnp.expand_dims(self._dataset.load_test_images(device), axis=-1) / 255.0
        )
        self._test_labels = self._dataset.load_test_labels(device)
        self._add_momentum_dimension = add_momentum_dimension

    def evaluate(
        self,
        key: jax.Array,
        log_learning_rates: jax.Array,
        log_momentums: jax.Array | None = None,
    ) -> jax.Array:
        learning_rates = jnp.exp(log_learning_rates)
        keys = (
            jax.random.split(key, learning_rates.size)
            if learning_rates.size
            else jnp.expand_dims(key, axis=0)
        )

        if self._add_momentum_dimension:
            if log_momentums is None:
                raise ValueError(
                    "log_momentums cannot be None when add_momentum_dimension is True"
                )
            # else...
            momentums = jnp.exp(log_momentums)
            if learning_rates.shape != momentums.shape:
                raise ValueError(
                    f"learning_rates and momentums must have the same shape. learning_rates.shape: {learning_rates.shape}, momentums.shape: {momentums.shape}"
                )
            # else...
            return jax.vmap(self._single_evaluate)(
                keys, learning_rates.flatten(), momentums.flatten()
            ).reshape(learning_rates.shape)
        else:
            return jax.vmap(self._single_evaluate, in_axes=(0, 0, None))(
                keys, learning_rates.flatten(), None
            ).reshape(learning_rates.shape)

    def create_train_state(self, rng, config):
        """Creates initial `TrainState`."""
        cnn = CNN()
        params = cnn.init(rng, jnp.ones([1, 28, 28, 1], dtype=float))["params"]
        tx = optax.sgd(config["learning_rate"], config["momentum"], accumulator_dtype=float)
        return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)

    def _single_evaluate(
        self, key: jax.Array, learning_rate: float, momentum: float | None
    ) -> float:
        init_key, sample_key = jax.random.split(key)
        state = self.create_train_state(
            init_key, {"learning_rate": learning_rate, "momentum": momentum}
        )

        def train_step(i, val):
            key, state = val
            key, sub_key = jax.random.split(key)
            shape = (32,)
            train_images = jax.random.choice(sub_key, self._train_images, shape)
            train_labels = jax.random.choice(sub_key, self._train_labels, shape)
            gradient, loss, accuracy = self.apply_model(
                state, train_images, train_labels
            )
            new_state = state.apply_gradients(grads=gradient)
            return key, new_state

        _, trained_state = jax.lax.fori_loop(
            0,
            1000,
            train_step,
            (sample_key, state),
        )

        _, _, accuracy = self.apply_model(
            trained_state, self._test_images, self._test_labels
        )

        return -accuracy

    def apply_model(self, state, images, labels):
        """Computes gradients, loss and accuracy for a single batch."""

        def loss_fn(params):
            logits = state.apply_fn({"params": params}, images)
            one_hot = jax.nn.one_hot(labels, 10)
            loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
            return loss, logits

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels, dtype=float)
        return grads, loss, accuracy

    @property
    def dataset_bounds(self) -> tuple[Boundary, ...]:
        bounds = [
            Boundary(math.log(1e-8), math.log(1e-0), float),
        ]  # Learning Rate
        if self._add_momentum_dimension:
            bounds.append(Boundary(math.log(1e-8), math.log(1e-0), float))
        return tuple(bounds)

    def plot(self, axis: matplotlib.axes.Axes, xs: jax.Array, ys: jax.Array) -> None:
        if xs.shape != ys.shape:
            raise ValueError(
                f"xs and ys must have the same shape. xs.shape: {xs.shape}, ys.shape: {ys.shape}"
            )
        # else...
        axis.scatter(xs, ys)


def sample(key: jax.Array, n: int, dataset_bounds: tuple[Boundary, ...]) -> jax.Array:
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
    boundary: Boundary,
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
    boundary_ticks: Iterable[tuple[Boundary, int]], sparse: bool
) -> list[jax.Array]:
    """Note that when dealing with integer dtypes, the number of points is not guaranteed to be respected."""
    grid_points = (
        get_ticks(boundary, number_of_points)
        for boundary, number_of_points in boundary_ticks
    )
    return jnp.meshgrid(*grid_points, sparse=sparse)
