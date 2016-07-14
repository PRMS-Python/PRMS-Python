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
    Container for a series of related scenario model runs. The user can
    initialize the series with a title and a description. Then to build
    the series the user provides a list of dictionaries with parameter-function
    key-value pairs, and optionally a title and description for each dictionary
    defining the individual scenario.

    The ScenarioSeries' ``build`` method creates a directory structure under
    the series directory (``{series-directory}``) where each
    subdirectory is named with a UUID which can be matched to its title using
    the metadata in ``{series-directory}/series_metadata.json``.

    In the future we may add a way for the user to access the results of
    the scenario simulations diretcly through the series, but for now the
    results are written to disk and the user must load them manually. It's
    maybe a little clunky, but here we use the title metadata to be able to
    reference the statsvar file for a particular set of scale factors. See
    below how to build the statvar path after running a series of scenarios.
    See the online documentation for a full example.

    Example:
        >>> import numpy as np
        >>> sc_ser = ScenarioSeries(
                'base_inputs', 'scenario_dir', title='my title',
                description='series description'
            )
        >>> def _scale_fun(val):
                def scale(x):
                    return x*val
                return scale
        >>> sc_list = [
                {
                    'rad_trncf': _scale_fun(val),
                    'snow_adj': _scale_fun(val),
                    'title':
                        '"rad_trncf":{0:%.1f}|"snow_adj":{0:%.1f}'.format(val)
                } for val in np.arange(0.7, 0.9, 0.1)
            ]
        >>> sc_ser.build(sc_list)
        >>> sc_ser.run()
        >>> series_md = json.loads(
                open(os.path.join('scenario_dir', 'series_metadata.json'))
            )
        >>> uuid_title_map = series_md['uuid_title_map']
        >>> uu = [k for k, v in uuid_title_map.iteritems()
                    if v == '"rad_trncf":0.7|"snow_adj":0.7']
        >>> statvar_path = os.path.join(
                'scenario_dir', uu, 'outputs', 'statvar.dat'
            )
    """

    def __init__(self, base_dir, scenarios_dir, title=None, description=None):
        """
        Create a new ScenarioSeries using inputs from base_dir and writing
        outputs to scenarios_dir.

        Arguments:
            base_dir (str): path to base inputs; 'control', 'parameters',
                and 'data' must be present there
            scenarios_dir (str): directory where scenario data will be written
                to; will be overwritten or created if it does not exist
            title (str): title of the ScenarioSeries instance
            description (str): description of the ScenarioSeries instance
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
        Create a ScenarioSeries from a list of parameters and the path to a
        directory in which scenarios should be saved and a data and control
        file can be found.
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
        form. Each element of scenarios_list can have any number of parameters
        as keys with a function for each value. The other two acceptable keys
        are title and description which will be passed on to each individual
        Scenario's metadata for future lookups.

        See the example in ScenarioSeries docstring above for a usage example.

        Calling ``build`` will create the directory structure explained in
        the class docstring.

        Arguments:
            scenarios_list (list): list of dictionaries with key-value
                pairs being parameter-function definition pairs or
                title-title string or description-description string.
        Returns:
            None
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

        if not nproc:
            nproc = mp.cpu_count()//2

        pool = mp.Pool(processes=nproc)
        pool.map(_scenario_runner, self.scenarios)

        # self.outputs = [
            # ScenarioOutput(uu, os.path.join(os.curdir(), d))
            # for uu, d in self.metadata['uuid_title_map'].items()
        # ]


# multiprocessing req the function be def'd at root scope so it's picklable
def _scenario_runner(scenario, prms_exec='prms'):
    scenario.run(prms_exec=prms_exec)


class Scenario:
    """
    Container for the process in which one modifies some base files then
    runs a simulation on the modified files. This also tracks metadata,
    including a title and description, if provided, plus start/end datetime
    and strings of the modification functions python code.
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

    def write(self, output_path):
        with open(output_path, 'w') as f:
            f.write(json.dumps(self.metadata_dict))


class ScenarioOutput:

    def __init__(self, scenario_uu, scenario_directory, title=None):
        opj = os.path.join
        self.uuid = scenario_uu
        self.scenario_directory = scenario_directory
        self.title = title
        self.data = Data(opj(scenario_directory, 'data'))
        self.parameters = Parameters(opj(scenario_directory, 'parameters'))
        self.statvar = load_statvar(opj(scenario_directory, 'statvar.dat'))
        self.control = open(opj(scenario_directory, 'control')).read()
