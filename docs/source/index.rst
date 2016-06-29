.. PRMS-Python documentation master file, created by
   sphinx-quickstart on Tue Jun 28 10:24:04 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PRMS-Python
===========

This module provides intuitive functions and structures for implementing
PRMS simulations and for systematically modifying input parameter and
data files, then running a series of PRMS simulations in an automated and
intelligent way. Every Simulation's input and output data is tracked on the
filesystem with metadata. A series of scenarios (``ScenarioSeries``) are stored
in a single root directory for each new series with metadata for later 
analysis.

Below are some examples. For more details, see the :doc:`/api`. 

Installation
------------

Currently this is only available through cloning on GitHub. 

.. code-block:: sh

    git clone https://github.com/mtpain/PRMS-Python.git

Then install dependencies

.. code-block:: sh

    pip install -r requirements.txt


Usage
-----

Please see the :doc:`tutorial` page for usage recipes and examples, and consult the
:doc:`api` for more details.

.. toctree::
    :includehidden:
    :maxdepth: 2

    tutorial
    api
