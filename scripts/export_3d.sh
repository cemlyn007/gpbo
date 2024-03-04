#!/bin/sh
for objective_function in mnist_3d
do
    for resolution in 15
    do
        venv/bin/python examples/export_3d_to_vtk.py \
            --ticks_filepath=results/gp/$objective_function/$resolution/matern/standardize/ticks.npy \
            --grid_ys_filepath=results/gp/$objective_function/$resolution/matern/standardize/grid_ys.npy \
            --mean_filepath=results/gp/$objective_function/$resolution/matern/standardize/mean.npy \
            --std_filepath=results/gp/$objective_function/$resolution/matern/standardize/std.npy \
            --save_filepath=results/gp/$objective_function/$resolution/matern/standardize/structured_grid.vtk
    done
done
