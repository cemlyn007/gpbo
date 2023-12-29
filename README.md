# Installation
Requires Python 3.12 or above.
## User
### Internet
```
pip install git+https://github.com/cemlyn007/gpbo.git#egg=gpbo
```
### Locally
Clone the repository at `https://github.com/cemlyn007/gpbo` and run
```
pip install .
```
## Development
```pip install -e .[dev]```
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
