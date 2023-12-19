import kernels
import jax.numpy as jnp
import pytest
from typing import Generator
import jax.experimental


@pytest.mark.parametrize("x64", [False, True], ids=["x32", "x64"])
@pytest.mark.parametrize(
    "kernel",
    [
        kernels.gaussian,
        kernels.matern,
        jax.jit(kernels.gaussian),
        jax.jit(kernels.matern),
    ],
    ids=["gaussian", "matern", "jit_gaussian", "jit_matern"],
)
class TestKernels:
    @pytest.fixture(autouse=True)
    def toggle_x64(self, x64: bool) -> Generator[None, None, None]:
        with jax.experimental.enable_x64(x64):
            yield

    @pytest.fixture
    def state(self) -> kernels.State:
        return kernels.State(
            log_amplitude=jnp.array(0.5, dtype=float),
            log_length_scale=jnp.array(0.5, dtype=float),
            log_noise_scale=jnp.array(0.5, dtype=float),
        )

    def test_call_with_1d_inputs(
        self, kernel: kernels.Kernel, state: kernels.State
    ) -> None:
        xs = jnp.array([1, 2, 3], dtype=float)
        ys = jnp.array([1, 2], dtype=float)
        kernel_matrix = kernel(state, xs, ys)
        assert kernel_matrix.shape == (3, 2)

    def test_call_with_2d_inputs(
        self, kernel: kernels.Kernel, state: kernels.State
    ) -> None:
        xs = jnp.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=float)
        ys = jnp.array([[1, 2], [2, 3], [3, 4]], dtype=float)
        kernel_matrix = kernel(state, xs, ys)
        assert kernel_matrix.shape == (4, 3)
