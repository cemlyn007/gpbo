#!/bin/sh
for objective_function in mnist_1d mnist_2d mnist_3d
do
    for resolution in 15 50 100
    do
        venv/bin/python examples/make_npy.py \
            --resolution=$resolution \
            --objective_function=$objective_function \
            --save_path=data/$objective_function/$resolution \
            --fallback
    done
done
