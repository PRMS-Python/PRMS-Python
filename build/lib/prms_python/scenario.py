# -*- coding: utf-8 -*-
'''
scenario.py -- holds ``Scenario`` and ``ScenarioSeries`` classes for PRMS 
managing parameter-based model scenarios that may be used for hypotheses
testing.
'''

import inspect
import json
import multiprocessing as mp
import os
import shutil
import uuid

from datetime import datetime
from .parameters import modify_params, Parameters
from .data import Data
from .util import load_statvar
from .simulation import Simulation


class ScenarioSeries(object):
    """
    Create and manage a series of model runs where parameters are modified. 
    
    First initialize the series with an optional title and description. 
    Then to build the series the user provides a list of dictionaries with 
    parameter-function key-value pairs, and optionally a title and 
    description for each dictionary defining the individual scenario.

    The ScenarioSeries' ``build`` method creates a file structure under
    the series directory (``scenarios_dir``) where each subdirectory is 
    named with a :mod:`uuid` which can be later matched to its title using 
    the metadata in ``scenario_dir/series_metadata.json`` (see :mod:`json`). 
    In the future we may add a way for the user to access the results of 
    the scenario simulations directly through the ``ScenarioSeries`` 
    instance, but for now the results are written to disk. Therefore each 
    scenario's title metadata can be used to refer to which parmaters were 
    modified and how for post-processing and analysis. One could also use 
    the description metadata for this purpose. 

    Arguments:
        base_dir (str): path to base inputs; 'control', 'parameters',
            and 'data' must be present there
        scenarios_dir (str): directory where scenario data will be written
            to; will be overwritten or created if it does not exist

    Keyword Arguments:
        title (str, optional): title of the ScenarioSeries instance
        description (str, optional): description of the ScenarioSeries 
            instance

    Attributes:
        metadata (dict): dictionary with title, description, and UUID map
           dictionary for individiual ``Scenario`` output directories, 
           the UUID dictionary (``uuid_title_map``)is left empty until 
           calling :meth:`ScenarioSeries.build`.
        scenarios (list): empty list that will be filled with ``Scenario``s
           after defining them by calling :meth:`ScenarioSeries.build`.

    Example:
        There are three steps to both ``Scenario`` and ScenarioSeries,
        first we initialize the object

        >>> series = ScenarioSeries(
                       base_dir = 'dir_with_input_files',
                       scenarios_dir = 'dir_to_run_series',
                       title = 'title_for_group_of_scenarios',
                       description = 'description_for_scenarios'
                     )
        
        The next step is to "build" the ``ScenarioSeries`` by calling the 
        :meth:`ScenarioSeries.build` method which defines which parameters
        to modify, how to modify them, and then performs the modification
        which readies the series to be "run" (the last step). See the
        :meth:`ScenarioSeries.build` method for the next step example. 
       
        Also see :ref:`scenario_and_scenarioseries_tutorial` for full example
    """

    def __init__(self, base_dir, scenarios_dir, title=None, description=None):
        """
        """
        self.base_dir = base_dir
        self.scenarios_dir = scenarios_dir
        if os.path.exists(scenarios_dir):
            shutil.rmtree(scenarios_dir)

        os.mkdir(scenarios_dir)

        shutil.copytree(base_dir, os.path.join(scenarios_dir, 'base_inputs'))

        self.metadata = dict(title=title,
                             description=description,
                             uuid_title_map={})
        self.scenarios = []

        self.outputs = None

    @classmethod
    def from_parameters_iter(cls, base_directory, parameters_iter,
                             title=None, description=None):
        '''
        Alternative way to initialize and build a ``ScenarioSeries`` in one
        step.

        Create and build a ``ScenarioSeries`` by including the param-keyed-
        function-valued dictionary (``parameters_iter``) that is otherwise 
        passed in :meth:`ScenarioSeries.build`. 

        Arguments:
            base_directory (str): directory that contains model input files
            parameters_iter (list of dicts): list of dictionaries for each
                ``Scenario`` as described in :class:`Scenario` and 
                :meth:`ScenarioSeries.build`.
            title (str): title for group of scenarios
            description (str): description for group of scenarios

        Returns:
            None
        '''
        series = cls(base_directory, base_directory,
                     title=title, description=description)

        for parameters in parameters_iter:

            title = parameters['title'] if 'title' in parameters else None

            uu = str(uuid.uuid4())

            series.metadata['uuid_title_map'].update({uu: title})

            scenario_dir = os.path.join(series.scenarios_dir, uu)

            scenario = Scenario(series.base_dir, scenario_dir, title=title)

            scenario.build()

            series.scenarios.append(scenario)

        with open(
            os.path.join(series.scenarios_dir, 'series_metadata.json'), 'w'
        ) as f:
            f.write(json.dumps(series.metadata, indent=2))

    def __len__(self):
        return len(self.scenarios)

    def build(self, scenarios_list):
        """
        Build the scenarios from a list of scenario definitions in dicitonary
        form. 
        
        Each element of ``scenarios_list`` can have any number of parameters
        as keys with a function for each value. The other two acceptable keys
        are ``title`` and ``description`` which will be passed on to each 
        individual Scenario's metadata in ``series_metadata.json`` for future 
        lookups. The ``build`` method also creates a file structure that uses
        UUID values as individiual ``Scenario`` subdirectories as shown below.

        Arguments:
            scenarios_list (list): list of dictionaries with key-value
                pairs being parameter-function definition pairs or
                title-title string or description-description string.
        Returns:
            None

        Examples:
            Following the initialization of a ``ScenarioSeries`` instance as
            shown the example docstring there, we "build" the series by 
            defining a list of param named-keyed function-valued 
            dictionaries. This example uses arbitrary functions on two 
            PRMS parameters *snowinfil_max* and *snow_adj*,

            >>> def _function1(x): #Note, function must start with underscore
                    return x * 0.5
            >>> def _function2(x):
                    return x + 5
            >>> dic1 = {'snowinfil_max': _function1, 'title': 'scenario1'}
            >>> dic2 = {'snowinfil_max': _function2, 
                        'snow_adj': function1,
                        'title': 'scenario2',
                        'description': 'we adjusted two snow parameters'
                       }
            >>> example_scenario_list = [dic1, dic2]
            >>> # now we can build the series
            >>> series.build(example_scenario_list)

        In this example that follows from :class:`ScenarioSeries` example the
        file structure that is created by the ``build`` method is as follows::

            dir_to_run_series
            ├── 670d6352-2852-400a-997e-7b12ba34f0b0
            │   ├── control
            │   ├── data
            │   └── parameters
            ├── base_inputs
            │   ├── control
            │   ├── data
            │   └── parameters
            ├── ee9526a9-8fe6-4e88-b357-7dfd7111208a
            │   ├── control
            │   ├── data
            │   └── parameters
            └── series_metadata.json

        As shown the build method has copied the original inputs from the 
        ``base_dir`` given on initialization of ``ScenarioSeries`` to a new
        subdirectory of the ``scenarios_dir``, it also applied the 
        modifications to the parameters for both scenarios above and move the
        input files to their respective directories. At this stage the 
        ``metadata`` will not have updated the UUID map dictionary to each
        scenarios subdirectory because they have not yet been run. See the 
        :meth:`ScenarioSeries.run` method for further explanation including
        the final file structure and metadata file contents.        
        """

        title = None
        description = None
        for s in scenarios_list:

            if 'title' in s:
                title = s['title']
                del s['title']

            if 'description' in s:
                description = s['description']
                del s['description']

            uu = str(uuid.uuid4())

            self.metadata['uuid_title_map'].update({uu: title})

            scenario_path = os.path.join(self.scenarios_dir, uu)

            # create Scenario
            scenario = Scenario(
                self.base_dir, scenario_path, title=title,
                description=description
            )

            # s now only contains parameter keys and function references vals
            scenario.build(s)

            self.scenarios.append(scenario)

        with open(
            os.path.join(self.scenarios_dir, 'series_metadata.json'), 'w'
        ) as f:

            f.write(json.dumps(self.metadata, indent=2))

    def run(self, prms_exec='prms', nproc=None):
        """
        Run a "built" ``ScenarioSeries`` and make final updates
        to file structure and metadata. 

        Keyword Arguments:
            prms_exec (str): name of PRMS executable on $PATH or path to
                executable. Default = 'prms'
            nproc (int or None): number of processceors available to 
                parallelize PRMS simulations, if None (default) then use
                half of what the :mod:`multiprocessing` detects on the
                machine.

        Returns:
            None

        Examples:
            This example starts where the example ends in 
            :meth:`ScenarioSeries.build`, calling ``run`` will run the
            models for all scenarios and then update the file structure 
        as well as create individual ``Scenario`` metadata files as such::

            dir_to_run_series
            ├── 5498c21d-d064-45f4-9912-044734fd230e
            │   ├── inputs
            │   │   ├── control
            │   │   ├── data
            │   │   └── parameters
            │   ├── metadata.json
            │   └── outputs
            │       ├── prms_ic.out
            │       ├── prms.out
            │       └── statvar.dat
            ├── 9d28ec5a-b570-4abb-8000-8dac113cbed3
            │   ├── inputs
            │   │   ├── control
            │   │   ├── data
            │   │   └── parameters
            │   ├── metadata.json
            │   └── outputs
            │       ├── prms_ic.out
            │       ├── prms.out
            │       └── statvar.dat
            ├── base_inputs
            │   ├── control
            │   ├── data
            │   └── parameters
            └── series_metadata.json

        As we can see the file structure follows the combined structures 
        as defined by :class:`Simulation` and :class:`Scenario`. The content
        of the top-level metadata file ``series_metadata.json`` is as such::

            {
              "title": "title_for_group_of_scenarios",
              "description": "description_for_scenarios",
              "uuid_title_map": {
                "5498c21d-d064-45f4-9912-044734fd230e": "scenario1",
                "9d28ec5a-b570-4abb-8000-8dac113cbed3": "scenario2"
              }
            }
        
        Therefore one can use the :mod:`json` file to track between UUID's and 
        individual scenario titles. The json files are read as a Python
        dictionary which makes them particularly convenient. The contents of
        an individual scenarios ``metadata.json`` file included a string
        representation of the function(s) that were applied to the 
        paramter(s)::

            {
                "description": null,
                "end_datetime": "2018-09-03T00:00:40.793817",
                "mod_funs_dict": {
                    "snowinfil_max": "def _function1(x):
                                          return x * 0.5"
                },
                "start_datetime": "2018-09-03T00:00:30.421353",
                "title": "scenario1"
            }

        Note:
            As shown, it is important to give appropriate scenario titles when
            building a ``ScenarioSeries`` dictionary in order to later 
            understand how parameters were modified in each scenario. If not
            one would have to rely on the individual ``metadata.json`` files
            in each scenario directory which may be more cumbersome.
        """
        if not nproc:
            nproc = mp.cpu_count()//2

        pool = mp.Pool(processes=nproc)
        pool.map(_scenario_runner, self.scenarios)


# multiprocessing req the function be def'd at root scope so it's picklable
def _scenario_runner(scenario, prms_exec='prms'):
    scenario.run(prms_exec=prms_exec)


class Scenario:
    """
    Container for the process in which one modifies input parameters then
    runs a simulation while tracking metadata. 
    
    Metadata includes a title and description, if provided, plus start/end 
    datetime, and parameter names of parameters that were modified including 
    string representations of the Python modification functions that were 
    applied to each parameter. The metadata file is in :mod:`json` format
    making it conveniently read as a Python dictionary.

    Arguments:
        base_dir (str): path to directory that contains initial *control*, 
            *parameter*, and *data* files to be used for ``Scenario``. 
            The *parameters* file in ``base_dir`` will not be modifed 
            instead will be copied to ``scenario_dir`` and them modified.
        scenario_dir (str): directory path to bundle inputs and outputs
        title (str, optional): title of ``Scenario``, if given will be 
            added to ``Scenario.metadata`` attribute as well as the 
            ``metadata.json`` file in ``scenario_dir`` written after 
            calling the :func:`Scenario.build()` and :func:`Scenario.run()` 
            methods.
        description (str, optional): description of ``Scenario``, also
            is added to ``Scenario.metadata`` as ``title``.

    Attributes:
        metadata (:class:`scenario.ScenarioMetadata`): a dictionary-like 
            class in :mod:`prms_python.scenario` that tracks ``Scenario`` and 
            ``ScenarioSeries`` imformation including user-defined parameter 
            modifications and descriptions, and file structure. 

    Examples:
        This example is kept simple for clarity, here we adjust a 
        single PRMS parameter *tmin_lapse* by using a single arbitrary
        mathematical function. We use the example PRMS model included
        with PRMS-Python for this example,

        >>> input_dir = 'PRMS-Python/test/data/models/lbcd'
        >>> scenario_directory = 'scenario_testing'
        >>> title = 'Scenario example'
        >>> desc = 'adjust tmin_lapse using sine wave function'
        >>> # create Scenario instance
        >>> scenario_obj = Scenario
                (  
                  base_dir=input_dir, 
                  scenario_dir=scenario_directory, 
                  title=title, 
                  description=desc
                )
       
        Next we need to build a dictionary to modify, in this case
        *tmin_lapse*, here we use a vectorized sine function 

        >>> # build the modification function and dictionary
        >>> def a_func(arr):
                return 4 + np.sin(np.linspace(0,2*np.pi,num=len(arr)))
        >>> # make dictionary with parameter names as keys and modification
        >>> # function as values
        >>> param_mod_dic = dict(tmin_lapse=a_func) 
        >>> scenario_obj.build(param_mod_funs=param_mod_dic)

    After building a ``Scenario`` instance the input files are 
    copied to ``scenario_dir`` which was assigned 'scenario_testing'::

        scenario_testing
        ├── control
        ├── data
        └── parameters
        
      
    After calling ``build`` the input files from ``input_dir`` were 
    first copied to ``scenario_dir`` and then the functions in 
    ``param_mod_dic`` are applied the the parameters names (key)
    in ``param_mod_dic``. To run the ``Scenario`` use the the ``run``
    method
        
        >>> scenario_obj.run()

    Now the simulation is run and the ``metadata.json`` file is created,
    the final file structure will be similar to this::

        scenario_testing    
        ├── inputs
        │   ├── control
        │   ├── data
        │   └── parameters
        ├── metadata.json
        └── outputs
            ├── prms_ic.out
            ├── prms.out
            └── statvar.dat
 
    Finally, here is what is contained in ``metadata.json`` for this example 
    which is also updates in the :attr:`Scenario.metadata`

        >>> scenario_obj.metadata
            {
              'title': 'Scenario example', 
              'description': 'adjust tmin_lapse using sine wave function', 
              'start_datetime': '2018-09-01T19:20:21.723003', 
              'end_datetime': '2018-09-01T19:20:31.117004', 
              'mod_funs_dict': {
                                 'tmin_lapse': 'def parab(arr):
                                                    return 4 + np.sin(np.linspace(0,2*np.pi,num=len(arr)))' 
                               }
            }

    As shown the metadata retirieved the parameter modification function
    as a string representation of the exact Python function(s) used for 
    modifying the user-defined parameter(s). 

    Note:
        The main differentiator between :class:`Scenario` and 
        :class:`ScenarioSeries` is that ``Scenario`` is designed for modifying
        one or more parameters of a **single** *parameters* file whereas
        ``ScenarioSeries`` is designed for modifying and tracking the 
        modification of one or more parameters in **multiple** PRMS 
        *parameters* files, therefore resulting in multiple PRMS simulations. 
     
    """

    def __init__(self, base_dir, scenario_dir,
                 title=None, description=None):

        self.title = title
        self.description = description

        self.base_dir = base_dir
        self.scenario_dir = scenario_dir

        self.metadata = ScenarioMetadata(title=title, description=description)

        self.__simulation_ready = False

    def build(self, param_mod_funs=None):
        """
        Take a user-defined dictionary with param names as keys and Python
        functions as values, copy the original input files as given when
        initializing a :class:`Scenario` instance to the ``simulation_dir``
        then apply the functions in the user-defined dictionary to the 
        parameters there. The ``build`` method must be called before running 
        the ``Scenario`` (calling :func:`Scenario.run()` ). 

        Keyword Arguments:
            param_mod_funs (dict): dictionary with parameter names as keys
                and Python functions as values to apply to the names (key)

        Returns:
            None

        Example:
            see :class:`Scenario` for a full example.

        Note:
            If the ``scenario_dir`` that was assigned for the current 
            instance already exists, it will be overwritten when ``build``
            is invoked. 
        """

        if isinstance(param_mod_funs, dict):

            # create scenario_dir that will be used as Simulation input dir
            if os.path.isdir(self.scenario_dir):
                shutil.rmtree(self.scenario_dir)

            os.makedirs(self.scenario_dir)
            shutil.copy(
                os.path.join(self.base_dir, 'control'), self.scenario_dir
            )
            shutil.copy(
                os.path.join(self.base_dir, 'data'), self.scenario_dir
            )

            old_params_path = os.path.join(self.base_dir, 'parameters')
            new_params_path = os.path.join(self.scenario_dir, 'parameters')
            if not param_mod_funs:
                shutil.copy(old_params_path, self.scenario_dir)
            else:
                modify_params(old_params_path, new_params_path, param_mod_funs)

            param_mod_funs_metadata = {
                param_name: inspect.getsource(param_mod_fun)
                for param_name, param_mod_fun in param_mod_funs.items()
            }

            self.metadata['mod_funs_dict'] = param_mod_funs_metadata

            self.simulation = Simulation(self.scenario_dir, self.scenario_dir)

        else:
            self.simulation = Simulation(self.scenario_dir)

        self.__simulation_ready = True

    def run(self, prms_exec='prms'):
        """
        Run the PRMS simulation for a *built* ``Scenario`` instance.

        Keyword Arguments: 
            prms_exec (str): name of PRMS executable on $PATH or path to 
                executable
        
        Returns:
            None

        Examples:
            see :class:`Scenario` for full example

        Raises:
            RuntimeError: if the :func:`Scenario.build` method has not yet
                been called.
        """
        if not self.__simulation_ready:
            raise RuntimeError(
                'Scenario has not yet been prepared: run build_scenario first'
            )

        self.metadata['start_datetime'] = datetime.now().isoformat()
        self.simulation.run(prms_exec=prms_exec)
        self.metadata['end_datetime'] = datetime.now().isoformat()

        self.metadata.write(os.path.join(self.scenario_dir, 'metadata.json'))


class ScenarioMetadata:

    def __init__(self, title=None, description=None, start_datetime=None,
                 end_datetime=None, mod_funs_dict=None):

        self.metadata_dict = dict(title=title,
                                  description=description,
                                  start_datetime=start_datetime,
                                  end_datetime=end_datetime,
                                  mod_funs_dict=mod_funs_dict)

    def __getitem__(self, key):
        return self.metadata_dict[key]

    def __setitem__(self, key, value):
        self.metadata_dict[key] = value

    def __repr__(self):
        return self.metadata_dict.__repr__()

    def write(self, output_path):
        with open(output_path, 'w') as f:
            f.write(json.dumps(self.metadata_dict, ensure_ascii=False,\
                               indent=4, sort_keys=True))


