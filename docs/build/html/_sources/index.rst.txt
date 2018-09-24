.. PRMS-Python documentation master file, created by
   sphinx-quickstart on Tue Jun 28 10:24:04 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PRMS-Python
===========

This module provides intuitive structures and functions for implementing and
managing common modeling workflows with `PRMS <https://wwwbrr.cr.usgs.gov/projects/SW_MoWS/PRMS.html>`_, the United States Geological
Survery's hydrologic computer model. PRMS-Python gives direct access to PRMS input 
parameters, climate forcing data, and output variables-- interfacing PRMS 
data structures with powerful scientific Python data structures particularly 
the ``numpy.ndarray`` and ``pandas.DataFrame`` objects. The module contains 
routines to systematically modify input parameters using user defined methods 
or a Monte Carlo parameter resampler; such routines then run a series of PRMS 
simulations in parralel. Metadata is also created and tracked along with each 
PRMS simulations inputs and outputs when running an automated routine. 
Post-processing of PRMS-Python managed simulations is also enhanced by tools for 
retrieving and archiving model inputs and outpts, calculate goodness-of-fit metrics, 
and produce visualizations of model input and output. 


Install 
-------

``pip install prms-python``

Developer Install
`````````````````

First, clone from GitHub

.. code-block:: sh

    git clone https://github.com/PRMS-Python/PRMS-Python.git 
    cd PRMS-Python

Then install dependencies and executable

.. code-block:: sh

    pip install --editable .


Usage
-----

..  
   If you want to dive right in, modify some parameters and run some scenarios,
   go on to the :doc:`cli` page. There you'll learn how to run `prmspy`, the 
   command-line interface to the PRMS-Python tools.

We recommend getting acquainted with PRMS-Python by reading and applying some examples and recipes found in the :doc:`tutorial` page, and consult the :doc:`api` for more details. More examples including basic and advanced workflows, and additional documentation can be found in the Jupyter notebooks found in the `notebooks directory <https://github.com/PRMS-Python/PRMS-Python/tree/master/notebooks>`_ in the PRMS-Python package. 


.. toctree::
    :includehidden:
    :maxdepth: 2

    tutorial
    cli
    api
    changelog
    

