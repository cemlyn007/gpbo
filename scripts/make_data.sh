#!/bin/sh
export MNIST_HOST_URL="ossci-datasets.s3.amazonaws.com"
export MNIST_TRAIN_IMAGES_RELATIVE_URL="/mnist/train-images-idx3-ubyte.gz"
export MNIST_TRAIN_LABELS_RELATIVE_URL="/mnist/train-labels-idx1-ubyte.gz"
export MNIST_TEST_IMAGES_RELATIVE_URL="/mnist/t10k-images-idx3-ubyte.gz"
export MNIST_TEST_LABELS_RELATIVE_URL="/mnist/t10k-labels-idx1-ubyte.gz"
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
