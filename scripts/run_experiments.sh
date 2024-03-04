#!/bin/sh
if [ "$1" = "cpu" ]; then
    export CUDA_VISIBLE_DEVICES=""
elif [ "$1" = "gpu" ]; then
    export CUDA_VISIBLE_DEVICES="0"
else
    echo "The argument must be either 'cpu' or 'gpu'."
    exit 1
fi

export MPLBACKEND=Agg

for objective_function in univariate six_hump_camel
do
    resolution=100
    if [ $objective_function = "univariate" ]
    then
        iterations=16
        initial_dataset_size=4
    elif [ $objective_function = "six_hump_camel" ]
    then
        iterations=32
        initial_dataset_size=8
    fi
    echo "[Started] GP for $objective_function at $resolution"
    venv/bin/python examples/gp.py \
        --plot_resolution=$resolution \
        --iterations=$iterations \
        --initial_dataset_size=$initial_dataset_size \
        --use_x64 \
        --objective_function=$objective_function \
        --transform=standardize \
        --kernel=matern \
        --save_path=results/gp/$objective_function/$resolution/matern/standardize
    echo "[Ended] GP for $objective_function at $resolution"
    echo "[Started] BO for $objective_function at $resolution"
    venv/bin/python examples/bo.py \
        --plot_resolution=$resolution \
        --iterations=$iterations \
        --initial_dataset_size=$initial_dataset_size \
        --use_x64 \
        --objective_function=$objective_function \
        --acquisition_function=expected_improvement \
        --transform=standardize \
        --kernel=matern \
        --save_path=results/bo/$objective_function/$resolution/matern/standardize
    echo "[Ended] BO for $objective_function at $resolution"
done

for mnist_objective_function in mnist_1d mnist_2d mnist_3d
do
    for resolution in 15 50
    do
        if [ $mnist_objective_function = "mnist_1d" ]
        then
            iterations=32
            initial_dataset_size=4
            optimizer_random_starts=1
            if [ $resolution = 15 ]
            then
                iterations=11
            fi
        elif [ $mnist_objective_function = "mnist_2d" ]
        then
            iterations=128
            initial_dataset_size=8
            optimizer_random_starts=3
        elif [ $mnist_objective_function = "mnist_3d" ]
        then
            iterations=512
            initial_dataset_size=16
            optimizer_random_starts=15
        fi
        echo "[Started] GP for $mnist_objective_function at $resolution"
        venv/bin/python examples/gp.py \
            --plot_resolution=$resolution \
            --iterations=$iterations \
            --initial_dataset_size=$initial_dataset_size \
            --use_x64 \
            --objective_function=npy \
            --transform=standardize \
            --kernel=matern \
            --grid_xs_npy_path=data/$mnist_objective_function/$resolution/grid_xs.npy \
            --grid_ys_npy_path=data/$mnist_objective_function/$resolution/grid_ys.npy \
            --save_path=results/gp/$mnist_objective_function/$resolution/matern/standardize
        echo "[Ended] GP for $mnist_objective_function at $resolution"
        echo "[Started] BO for $mnist_objective_function at $resolution"
        venv/bin/python examples/bo.py \
            --plot_resolution=$resolution \
            --iterations=$iterations \
            --initial_dataset_size=$initial_dataset_size \
            --use_x64 \
            --objective_function=npy \
            --acquisition_function=expected_improvement \
            --transform=standardize \
            --kernel=matern \
            --grid_xs_npy_path=data/$mnist_objective_function/$resolution/grid_xs.npy \
            --grid_ys_npy_path=data/$mnist_objective_function/$resolution/grid_ys.npy \
            --save_path=results/bo/$mnist_objective_function/$resolution/matern/standardize
        echo "[Ended] BO for $mnist_objective_function at $resolution"
    done
done
