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
## Optimizing a Gaussian Process
### Univariate
```
python examples/gp.py --plot_throughout --objective_function=univariate --use_x64 --iterations 48
```
### Six Hump Camel
```
python examples/gp.py --plot_throughout --objective_function=six_hump_camel --use_x64 --iterations 256
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
