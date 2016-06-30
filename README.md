# PRMS-Python

PRMS-Python provides a Python interface to PRMS data files and for running
PRMS simulations. This module tries to improve the management of PRMS simulation
data while also providing useful "pythonic" tools to do scenario-based PRMS
simulations.  By "scenario-based" modeling we mean, for example, parameter
sensitivity analysis, where each "scenario" is an iterative perturbation of
one or many parameters. Another example "scenario-based" modeling exercise would
be climate scenario modeling: what will happen to modeled outputs if the
input meteorological data were to change?


## Installation

Currently it's clone-then-pip:

```
git clone https://github.com/northwest-knowledge-network/prms-python
```

then

```
pip install -r requirements.txt
```


## Usage

Please read the [Online Documentation](https://prms-python.github.io/docs).


## Unit tests

I run them using nose but that's not required. From the root repo directory

```
nosetests -v
```
