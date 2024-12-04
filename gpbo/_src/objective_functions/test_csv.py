import jax.numpy as jnp
import jax.random
import pytest

from gpbo._src.objective_functions import core, csv


class TestCsvObjectiveFunction:
    def test_csv_1d(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        with open(csv_file, "w") as f:
            f.write("1,2\n")
            f.write("3,4\n")
        objective_function = csv.CsvObjectiveFunction(csv_file)
        assert objective_function.evaluate(
            jax.random.PRNGKey(0), jnp.array(1)
        ) == jnp.array(2)
        assert all(
            objective_function.evaluate(jax.random.PRNGKey(0), jnp.array((1, 3)))
            == jnp.array((2, 4))
        )
        with pytest.raises(ValueError):
            objective_function.evaluate(jax.random.PRNGKey(0), jnp.array((2)))
        assert objective_function.dataset_bounds == (
            core.Boundary(1, 3, dtype=jnp.float32),
        )

    def test_csv_2d(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        with open(csv_file, "w") as f:
            f.write("1,2,3\n")
            f.write("4,5,6\n")
            f.write("7,8,9\n")
        objective_function = csv.CsvObjectiveFunction(csv_file)
        assert objective_function.evaluate(
            jax.random.PRNGKey(0), jnp.array((1, 2))
        ) == jnp.array(3)
        assert all(
            objective_function.evaluate(
                jax.random.PRNGKey(0), jnp.array(((1, 2), (4, 5)))
            )
            == jnp.array((3, 6))
        )
        with pytest.raises(ValueError):
            objective_function.evaluate(jax.random.PRNGKey(0), jnp.array((1, 2, 3)))
        assert objective_function.dataset_bounds == (
            core.Boundary(1, 7, dtype=jnp.float32),
            core.Boundary(2, 8, dtype=jnp.float32),
        )
