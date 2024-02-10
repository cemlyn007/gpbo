import csv
import numpy as np
import jax.numpy as jnp
from gpbo._src import datasets


def read_csv(file_path: str) -> datasets.Dataset:
    xs: list[list[float]] = []
    ys: list[float] = []
    with open(file_path, newline="") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            x = list(map(float, row[:-1]))
            y = float(row[-1])
            xs.append(x)
            ys.append(y)
    return datasets.Dataset(jnp.array(xs), jnp.array(ys))


def write_csv(dataset: datasets.Dataset, file_path: str) -> None:
    if len(dataset.xs) != len(dataset.ys):
        raise ValueError(
            "The number of x values must be equal to the number of y values."
        )
    # else...
    rows = []
    for x, y in zip(dataset.xs, dataset.ys):
        x = np.array(x).tolist()
        y = np.array(y).tolist()
        if isinstance(x, list):
            row = [*x, y]
        else:
            row = [x, y]
        rows.append(row)
    with open(file_path, "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(rows)

def write_mesh_grid_csv(mesh_grid: np.ndarray, file_path: str) -> None:
    rows = []
    for row in mesh_grid:
        row = np.array(row).tolist()
        rows.append(row)
    with open(file_path, "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(rows)

