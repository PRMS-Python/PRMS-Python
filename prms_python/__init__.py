"""
PRMS-Python provides a Python interface to PRMS data files and manages 
PRMS simulations. This module aims to improve the efficiency of PRMS 
workflows by giving access to PRMS data structures while providing 
"pythonic" tools to do scenario-based PRMS simulations. By 
"scenario-based" we mean testing model hypotheses associated with model 
inputs, outputs, and model structure. For example, parameter sensitivity 
analysis, where each "scenario" is an iterative perturbation of one or 
many parameters. Another example "scenario-based" modeling exercise 
would be climate scenario modeling: what will happen to modeled outputs 
if the input meteorological data were to change?

"""

__name__ = 'prms-python'
__author__ = 'John Volk and Matthew Turner'
__version__ = '1.0.0'

from .data import Data
from .optimizer import Optimizer, OptimizationResult
from .parameters import Parameters, modify_params
from .simulation import Simulation, SimulationSeries
from .scenario import Scenario, ScenarioSeries
from .util import load_statvar, load_data_file, nash_sutcliffe
