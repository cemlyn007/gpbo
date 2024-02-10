import pytest
from gpbo._src import datasets
from gpbo._src import io
import jax.numpy as jnp


@pytest.fixture
def file_path(tmp_path):
    file_path = tmp_path / "dataset.csv"
    file_path.write_text(
        "1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0\n10.0,11.0,12.0\n13.0,14.0,15.0\n"
    )
    return file_path


def test_read_csv(file_path: str):
    dataset = io.read_csv(file_path)
    assert isinstance(dataset, datasets.Dataset)
    assert dataset.xs.shape == (5, 2)
    assert dataset.ys.shape == (5,)


def test_write_csv(tmp_path):
    dataset = datasets.Dataset(
        jnp.array([[1.0, 2.0], [3.0, 4.0]]), jnp.array([5.0, 6.0])
    )
    file_path = tmp_path / "dataset.csv"
    io.write_csv(dataset, file_path)
    assert file_path.read_text() == "1.0,2.0,5.0\n3.0,4.0,6.0\n"
