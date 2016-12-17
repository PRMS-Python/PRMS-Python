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
pip install --editable .
```

## Building documentation

This project uses the [Sphinx documentation engine for Python]()
The documentation source is located in `docs/source`. Eventually we can
wrap the following steps into a script. But for now, to build the
documentation, go to the `docs/` directory and run

```
make html
```

If it fails because of missing dependencies, just install the dependencies 
it says it's missing. I don't know why it's asking for some dependencies
right now.

If it succeds and this is your first time making the html docs you will see a 
new directory, `build/html`. If this is the first time you've 
built and updated the documentation you'll need to run this first

```
git remote add docs http://github.com/PRMS-Python/docs
```

Now, we'll create a branch that consists solely 
of this built html by running the following command from the root of the 
PRMS-Python documentation directory

```
git cob new-docs \
    && git add -f docs/build/html \
    && git cm -m"built updated docs" \
    && git filter-branch -f --prune-empty --subdirectory-filter docs/build/html new-docs \
    && git push -u docs HEAD:new-docs
```

This pushed the newly built documentation to the `PRMS-Python/docs` repository.
Now we just have to move the new branch you pushed to GitHub to be the
`gh-pages` branch.

First, change directories to the PRMS-Python/docs repository. Then,

```
git fetch origin \
    && git checkout new-docs \
    && git push --delete gh-pages \
    && git push -u origin HEAD:gh-pages \
    && git push --delete origin new-docs \
    && git br -D new-docs
```


## Usage

Please read the [Online Documentation](https://prms-python.github.io/docs).


## Unit tests

I run them using nose but that's not required. From the root repo directory

```
nosetests -v
```
