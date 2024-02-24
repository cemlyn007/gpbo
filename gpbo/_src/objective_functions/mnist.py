from gpbo._src.objective_functions import core
import jax.numpy as jnp
import jax.random
import matplotlib.axes
import jax.typing
import http.client
import os
import gzip
import optax
from flax import linen as nn
from flax.training import train_state
import math


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


class MnistObjectiveFunction(core.ObjectiveFunction):
    def __init__(
        self,
        cache_directory: str,
        add_momentum_dimension: bool,
        add_learning_rate_decay: bool,
        n_epochs: int,
        device: jax.Device,
    ) -> None:
        super().__init__()
        self._n_epochs = n_epochs
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
        self._add_learning_rate_decay = add_learning_rate_decay
        if self._add_learning_rate_decay and not self._add_momentum_dimension:
            raise NotImplementedError(
                "Learning rate decay is only implemented for the case where add_momentum_dimension is True"
            )

    def evaluate(
        self,
        key: jax.Array,
        log_learning_rates: jax.Array,
        log_momentums: jax.Array | None = None,
        log_last_learning_rates: jax.Array | None = None,
    ) -> jax.Array:
        learning_rates = jnp.exp(log_learning_rates)
        keys = (
            jax.random.split(key, learning_rates.size)
            if learning_rates.size
            else jnp.expand_dims(key, axis=0)
        )
        dtype = learning_rates.dtype
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
            if learning_rates.dtype != momentums.dtype:
                raise ValueError(
                    f"learning_rates and momentums must have the same dtype. learning_rates.dtype: {learning_rates.dtype}, momentums.dtype: {momentums.dtype}"
                )
            # else...
            if self._add_learning_rate_decay:
                if log_last_learning_rates is None:
                    raise ValueError(
                        "log_last_learning_rates cannot be None when add_learning_rate_decay is True"
                    )
                # else...
                last_learning_rates = jnp.exp(log_last_learning_rates)
                if learning_rates.shape != last_learning_rates.shape:
                    raise ValueError(
                        f"learning_rates and last_learning_rates must have the same shape. learning_rates.shape: {learning_rates.shape}, last_learning_rates.shape: {last_learning_rates.shape}"
                    )
                if learning_rates.dtype != last_learning_rates.dtype:
                    raise ValueError(
                        f"learning_rates and last_learning_rates must have the same dtype. learning_rates.dtype: {learning_rates.dtype}, last_learning_rates.dtype: {last_learning_rates.dtype}"
                    )
                return jax.vmap(self._single_evaluate, in_axes=(0, 0, 0, 0, None))(
                    keys,
                    learning_rates.flatten(),
                    momentums.flatten(),
                    last_learning_rates.flatten(),
                    dtype,  # type: ignore
                ).reshape(learning_rates.shape)
            else:
                return jax.vmap(self._single_evaluate, in_axes=(0, 0, 0, None, None))(
                    keys, learning_rates.flatten(), momentums.flatten(), None, dtype  # type: ignore
                ).reshape(learning_rates.shape)
        else:
            return jax.vmap(self._single_evaluate, in_axes=(0, 0, None, None, None))(
                keys, learning_rates.flatten(), None, None, dtype  # type: ignore
            ).reshape(learning_rates.shape)

    def _create_train_state(
        self,
        rng: jax.Array,
        learning_rate: float,
        momentum: float | None,
        last_learning_rate: float | None,
        dtype: jnp.dtype,
    ):
        """Creates initial `TrainState`."""
        cnn = CNN()
        params = cnn.init(rng, jnp.ones([1, 28, 28, 1], dtype=dtype))["params"]
        if last_learning_rate is None:
            tx = optax.sgd(learning_rate, momentum, accumulator_dtype=dtype)
        else:
            tx = optax.sgd(optax.linear_schedule(learning_rate, last_learning_rate, self._n_epochs), momentum, accumulator_dtype=dtype)
        tx = optax.sgd(learning_rate, momentum, accumulator_dtype=dtype)
        return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)

    def _single_evaluate(
        self,
        key: jax.Array,
        learning_rate: float,
        momentum: float | None,
        last_learning_rate: float | None,
        dtype: jnp.dtype,
    ) -> jax.Array:
        init_key, sample_key = jax.random.split(key)
        state = self._create_train_state(init_key, learning_rate, momentum, last_learning_rate, dtype)

        def train_step(
            i: int, val: tuple[jax.Array, train_state.TrainState]
        ) -> tuple[jax.Array, train_state.TrainState]:
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
            self._n_epochs,
            train_step,
            (sample_key, state),
        )

        _, _, accuracy = self.apply_model(
            trained_state, self._test_images, self._test_labels
        )

        return -accuracy

    def apply_model(
        self, state: train_state.TrainState, images: jax.Array, labels: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Computes gradients, loss and accuracy for a single batch."""

        def loss_fn(params):
            logits = state.apply_fn({"params": params}, images)
            one_hot = jax.nn.one_hot(labels, 10)
            loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
            return loss, logits

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels, dtype=logits.dtype)
        return grads, loss, accuracy

    @property
    def dataset_bounds(self) -> tuple[core.Boundary, ...]:
        bounds = [
            core.Boundary(math.log(1e-4), math.log(1e-0), float),
        ]  # Learning Rate
        if self._add_momentum_dimension:
            bounds.append(core.Boundary(math.log(1e-3), math.log(1e-0), float))
        if self._add_learning_rate_decay:
            bounds.append(core.Boundary(math.log(1e-7), math.log(1e-0), float))
        return tuple(bounds)

    def plot(self, axis: matplotlib.axes.Axes, xs: jax.Array, ys: jax.Array) -> None:
        if xs.shape != ys.shape:
            raise ValueError(
                f"xs and ys must have the same shape. xs.shape: {xs.shape}, ys.shape: {ys.shape}"
            )
        # else...
        axis.scatter(xs, ys)
