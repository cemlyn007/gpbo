# Installation
Requires Python 3.12 or above. Would recommend JAX 0.4.17 because 0.4.18 to 0.4.23 appears to sometimes yield CUDA internal errors randomly at runtime.
## User
### Internet
```
pip install git+https://github.com/cemlyn007/gpbo.git#egg=gpbo
```
### Locally
Clone the repository at `https://github.com/cemlyn007/gpbo` and run:
```
pip install .
```
## Development
Clone the repository at `https://github.com/cemlyn007/gpbo` and run:
```pip install -e .[dev]```
If using Visual Studio Code with Pylance, you will want to run instead:
```
pip install -e .[dev] --config-settings editable_mode=strict
```
To get the type hinter to find the package.

# Examples
You will need to install the example dependencies by running:
```pip install -e .[example]```

## Optimizing a Gaussian Process
### Univariate
```
python examples/gp.py --objective_function=univariate --use_x64 --iterations 48
```
### Six Hump Camel
```
python examples/gp.py --objective_function=six_hump_camel --use_x64 --iterations 256
```
### MNIST Log Learning Rate vs Negative Accuracy
```
python examples/gp.py --objective_function=mnist_1d --use_x64 --iterations 256
```
### MNIST Log Learning Rate and Log Momentum vs Negative Accuracy
```
python examples/gp.py --objective_function=mnist_2d --iterations 135
```

## Bayesian Optimization
### Univariate
```
python examples/bo.py --objective_function=univariate --use_x64 --iterations 16
```
### Six Hump Camel
```
python examples/bo.py --objective_function=six_hump_camel --use_x64 --iterations 48
```

## Generate Results
All commands are expected to be run from the root of the repository.
### Via Shell
Run this command to generate the MNIST dataset:
```
./scripts/make_data.sh
```
Run this command to generate the results, passing `gpu` or `cpu` as the first argument to specify the device to use:
```
./scripts/run_experiments.sh gpu
```
To export the 3D results to be visualised using VTK via something like Paraview:
```
./scripts/export_3d.sh
```

### Via Slurm
Run this command to generate the MNIST dataset:
```
sbatch --workdir=$(pwd) ./scripts/make_data.sh
```
Run this command to generate the results, passing `gpu` or `cpu` as the first argument to specify the device to use:
```
sbatch --workdir=$(pwd) ./scripts/run_experiments.sh gpu
```
