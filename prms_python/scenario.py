import inspect
import json
import os
import shutil

from datetime import datetime

from .parameters import modify_params
from .simulation import Simulation


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

    def build_scenario(self, param_mod_funs=None):

        if not isinstance(param_mod_funs, dict):
            raise TypeError('param_mod_funs must be a dictionary')

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
            for param_name, param_mod_fun in param_mod_funs.iteritems()
        }

        self.metadata['mod_funs_dict'] = param_mod_funs_metadata

        self.simulation = Simulation(self.scenario_dir, self.scenario_dir)

        self.__simulation_ready = True

    def run(self):

        if not self.__simulation_ready:
            raise RuntimeError(
                'Scenario has not yet been prepared: run build_scenario first'
            )

        self.metadata['start_datetime'] = datetime.now().isoformat()
        self.simulation.run()
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
