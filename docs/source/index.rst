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

Questions or comments? Contact `John Volk <mailto:jmvolk@unr.edu>`_


Install (out of date version on PyPI, for new version use method below)
-------

``pip install prms-python``

Developer Install
`````````````````

First, clone from GitHub

.. code-block:: sh

    git clone https://github.com/JohnVolk/PRMS-Python && cd PRMS-Python

Then install dependencies and executable

.. code-block:: sh

    pip install --editable .


Usage
-----

If you want to dive right in, modify some parameters and run some scenarios,
go on to the :doc:`cli` page. There you'll learn how to run `prmspy`, the 
command-line interface to the PRMS-Python tools.

If instead you would rather get acquainted with the Python API for direct use,
see the :doc:`tutorial` page for usage recipes and examples, and consult the
:doc:`api` for more details. More examples and in depth documentation can be 
found in the Jupyter notebooks found in the notebooks directory in the PRMS-Python 
package.

Happy coding, y'all!


.. toctree::
    :includehidden:
    :maxdepth: 2

    cli
    tutorial
    api
