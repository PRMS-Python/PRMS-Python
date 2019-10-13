Change log
**********

Version 1.0.1
=============

Substantial improvements and updates to online documentation. Renamed two util
module functions to align with PEP style conventions specifically: 
``Kolmogorov_Smirnov`` became ``kolmogorov_smirnov`` and ``calc_emp_CDF`` 
became ``calc_emp_cdf``, also renamed ``util.load_data_file`` to 
``util.load_data`` for consistency with ``util.load_statvar``. Name changes 
are backwards compatible. Add interactive code snippets in online documentation,
add conda environment file.

Bug Fixes
---------

* Fix `prmspy` script errors after prms-python installation with Python 2

Version 1.0.0
=============

Stable version and first numbered release on PyPI. All functionality tested 
and passed on multiple platforms using several PRMS models. 

Major submodules, classes and functions 
---------------------------------------

* data
  - Data
* parameters
  - Parameters, modify_params
* simulation
  - Simulation, SimulationSeries
* scenario
  - Scenario, ScenarioSeries
* optimizer
  - Optimizer, OptimizationResult, resample_param
* util
  - load_statvar, nash_sutcliffe, and others

This version also includes the command line interface script "`prmspy`". 
Ongoing work on Docs website, includes mostly up-to-date Jupyter notebook 
examples. 

Version 0.1.0
=============

First numbered version, was not setup for installation using PyPI. Many changes
occured for initial development under this version which were not released.

