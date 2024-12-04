import argparse

import numpy as np
import pyvista

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export 3D data to VTK")
    parser.add_argument(
        "--ticks_filepath",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--grid_ys_filepath",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--mean_filepath",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--std_filepath",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--utility_filepath",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_filepath",
        type=str,
        default=None,
    )

    arguments = parser.parse_args()

    ticks = np.load(arguments.ticks_filepath)
    grid_ys = np.load(arguments.grid_ys_filepath)

    structured_grid = pyvista.StructuredGrid(*np.meshgrid(*ticks))
    structured_grid["ys"] = grid_ys.ravel("F")
    if arguments.mean_filepath:
        mean = np.load(arguments.mean_filepath)
        structured_grid["mean"] = mean.ravel("F")
    if arguments.std_filepath:
        std = np.load(arguments.std_filepath)
        structured_grid["std"] = std.ravel("F")
    if arguments.utility_filepath:
        utility = np.load(arguments.utility_filepath)
        structured_grid["utility"] = utility.ravel("F")

    structured_grid.save(arguments.save_filepath)
