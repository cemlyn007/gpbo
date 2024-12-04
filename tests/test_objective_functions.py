from typing import Generator

import jax
import jax.experimental
import jax.random
import matplotlib.pyplot as plt
import pytest

from gpbo._src import objective_functions


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize(
    "objective_function_cls",
    [
        objective_functions.UnivariateObjectiveFunction,
        objective_functions.SixHumpCamelObjectiveFunction,
    ],
)
class TestObjectiveFunction:
    @pytest.fixture(autouse=True)
    def toggle_x64(self, x64: bool) -> Generator[None, None, None]:
        with jax.experimental.enable_x64(x64):
            yield

    @pytest.fixture
    def objective_function(
        self, objective_function_cls: type[objective_functions.ObjectiveFunction]
    ) -> objective_functions.ObjectiveFunction:
        return objective_function_cls()

    def test_dataset_bounds(
        self, objective_function: objective_functions.ObjectiveFunction
    ) -> None:
        dataset_bounds = objective_function.dataset_bounds
        for boundary in dataset_bounds:
            if boundary.dtype == float:
                assert isinstance(boundary.min_value, float)
                assert isinstance(boundary.max_value, float)
            else:
                raise NotImplementedError(
                    "Only float dtypes are supported for testing at the moment."
                )

    def test_evaluate(
        self, objective_function: objective_functions.ObjectiveFunction
    ) -> None:
        n = 100
        array = objective_functions.sample(
            jax.random.PRNGKey(0), n, objective_function.dataset_bounds
        )
        if len(objective_function.dataset_bounds) == 1:
            xs = [array]
        else:
            xs = [array[:, i] for i in range(len(objective_function.dataset_bounds))]
        ys = objective_function.evaluate(jax.random.PRNGKey(0), *xs)

        assert ys.shape == (n,)
        assert ys.dtype == array.dtype

    def test_evaluate_with_meshgrid(
        self, objective_function: objective_functions.ObjectiveFunction
    ) -> None:
        n = 100
        mesh_grid = objective_functions.get_mesh_grid(
            [(boundary, n) for boundary in objective_function.dataset_bounds],
            sparse=True,
        )
        ys = objective_function.evaluate(jax.random.PRNGKey(0), *mesh_grid)
        if len(objective_function.dataset_bounds) == 1:
            assert ys.shape == (n,)
        else:
            assert ys.shape == (n, n)
        assert all(ys.dtype == ticks.dtype for ticks in mesh_grid)

    def test_plot_smoke(
        self, objective_function: objective_functions.ObjectiveFunction
    ):
        n = 100
        mesh_grid = objective_functions.get_mesh_grid(
            [(boundary, n) for boundary in objective_function.dataset_bounds],
            sparse=True,
        )
        ys = objective_function.evaluate(jax.random.PRNGKey(0), *mesh_grid)
        figure = plt.figure("test")
        try:
            objective_function.plot(
                figure.add_subplot(111),
                *(ticks.flatten() for ticks in mesh_grid),
                ys,
            )
        finally:
            plt.close(figure)
