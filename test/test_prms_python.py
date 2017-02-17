from copy import copy
import json
import glob
import numpy as np
import os
import re
import shutil
import unittest

from difflib import Differ
from numpy.testing import assert_array_almost_equal

from prms_python import (
    modify_params, Parameters, Scenario, ScenarioSeries, Simulation,
    SimulationSeries, Data
)


OPJ = os.path.join


class TestSimulationSeries(unittest.TestCase):
    '''
    Expect that a SimulationSeries
    '''
    def setUp(self):
        self.test_model_data_dir = os.path.join(
            'test', 'data', 'models', 'lbcd'
        )

        self.simulation_dir = os.path.join(self.test_model_data_dir, 'tmp_sim')

    def tearDown(self):
        for g in glob.glob(OPJ(self.simulation_dir, '*')):
            shutil.rmtree(g)

    def test_simulation_series(self):
        tdd = self.test_model_data_dir
        data = Data(OPJ(tdd, 'data'))
        base_parameters = Parameters(OPJ(tdd, 'parameters'))

        def _copy_mod(base_parameters, val):
            ret = copy(base_parameters)
            ret['dday_intcp'][:] = val

            return ret

        parameters_gen = (
            (val, _copy_mod(base_parameters, val))
            for val in (-50, -40, -30)
        )
        control_path = OPJ(tdd, 'control')

        series = SimulationSeries(
            Simulation.from_data(
                data, parameters, control_path,
                self.simulation_dir + str(val)
            )
            for val, parameters in parameters_gen
        )

        outputs = list(series.run().outputs_iter())

        self.assertEqual(len(series), 3)
        self.assertEqual(len(outputs), 3)

        for out in outputs:
            sdir = out['simulation_dir']
            assert os.path.isdir(out['simulation_dir'])
            assert os.path.exists(OPJ(sdir, 'outputs', 'statvar.dat'))
            assert os.path.exists(OPJ(sdir, 'inputs', 'data'))
            assert os.path.exists(OPJ(sdir, 'inputs', 'parameters'))

            shutil.rmtree(sdir)


class TestSimulation(unittest.TestCase):
    """
    Simulations should take a base directory and return a simulation directory
    """

    def setUp(self):

        self.test_data_dir = os.path.join('test', 'data')

        self.test_model_data_dir = os.path.join(
            'test', 'data', 'models', 'lbcd'
        )

        self.simulation_dir = os.path.join(self.test_data_dir, 'tmp_sim')

    def tearDown(self):

        if os.path.exists(self.simulation_dir):
            shutil.rmtree(self.simulation_dir)

    def test_simulation_no_simdir(self):
        "Simulation should run and write outputs to input directory when simulation_dir is not specified"
        s = Simulation(self.test_model_data_dir)
        s.run()

        g = [
                os.path.basename(f)
                for f in glob.glob(os.path.join(self.test_model_data_dir, '*'))
        ]

        self.assertIn('prms_ic.out', g)
        self.assertIn('prms.out', g)
        self.assertIn('statvar.dat', g)

    def test_simulation_w_simdir(self):
        "Simulation should create sim dir with inputs and outputs directory when simulation_dir is specified"
        s = Simulation(self.test_model_data_dir, self.simulation_dir)
        s.run()

        gs = [
                os.path.basename(f)
                for f in glob.glob(os.path.join(self.simulation_dir, '*'))
        ]
        self.assertIn('inputs', gs)
        self.assertIn('outputs', gs)

        assert_valid_input_dir(
            self, os.path.join(self.simulation_dir, 'inputs')
        )
        assert_valid_output_dir(
            self, os.path.join(self.simulation_dir, 'outputs')
        )

    def test_simulation_from_data(self):
        """
        Use @classmethod from_data to build a simulation from Parameters and Data instances
        """
        tdd = self.test_model_data_dir

        data = Data(OPJ(tdd, 'data'))
        parameters = Parameters(OPJ(tdd, 'parameters'))
        ctrl = OPJ(tdd, 'control')

        test_dir = OPJ(tdd, 'test-sim-dir')
        if os.path.isdir(test_dir):
            shutil.rmtree(test_dir)

        s = Simulation.from_data(data, parameters, ctrl, test_dir)

        s.run()

        g = [
                os.path.basename(f)
                for f in glob.glob(OPJ(test_dir, 'outputs', '*'))
        ]

        self.assertIn('prms_ic.out', g)
        self.assertIn('prms.out', g)
        self.assertIn('statvar.dat', g)

        # clean up
        shutil.rmtree(test_dir)


class TestScenario(unittest.TestCase):

    def setUp(self):

        self.test_data_dir = os.path.join('test', 'data')

        self.test_model_data_dir = os.path.join(
            'test', 'data', 'models', 'lbcd'
        )

        self.scenario_dir = os.path.join(self.test_data_dir, 'tmp_scenario')

    def tearDown(self):

        if os.path.exists(self.scenario_dir):
            shutil.rmtree(self.scenario_dir)

    def test_create_scenario(self):
        """
        a simulation setup should create a simulation directory with correct scenario data
        """

        s = Scenario(
            self.test_model_data_dir, self.scenario_dir,
            title='Scenario Uno', description='test scenario for prms_python'
        )

        param_mods = {
            'snow_adj': lambda x: 1.1*x,
            'rad_trncf': lambda x: 0.9*x
        }
        s.build(param_mod_funs=param_mods)

        assert_valid_input_dir(self, self.scenario_dir)  # os.path.join(self.scenario_dir, 'inputs'))

        s.run()

        assert_valid_input_dir(
            self, os.path.join(self.scenario_dir, 'inputs')
        )
        assert_valid_output_dir(
            self, os.path.join(self.scenario_dir, 'outputs')
        )

        md_json_path = os.path.join(self.scenario_dir, 'metadata.json')
        assert os.path.isfile(md_json_path)

        md_json = json.loads(open(md_json_path).read())
        assert md_json['title'] == 'Scenario Uno'
        assert md_json['description'] == 'test scenario for prms_python'
        assert 'start_datetime' in md_json
        assert 'end_datetime' in md_json
        assert 'mod_funs_dict' in md_json

        p_base = Parameters(
            os.path.join(self.test_model_data_dir, 'parameters')
        )
        p_scen = Parameters(
            os.path.join(self.scenario_dir, 'inputs', 'parameters')
        )

        assert_array_almost_equal(p_base['snow_adj']*1.1, p_scen['snow_adj'])
        assert_array_almost_equal(p_base['rad_trncf']*0.9, p_scen['rad_trncf'])


class TestScenarios(unittest.TestCase):

    def setUp(self):

        self.test_data_dir = os.path.join('test', 'data')

        self.test_model_data_dir = os.path.join(
            self.test_data_dir, 'models', 'lbcd'
        )

        self.scenarios_dir = os.path.join(self.test_data_dir, 'tmp_scenarios')

    def tearDown(self):

        if os.path.exists(self.scenarios_dir):
            shutil.rmtree(self.scenarios_dir)

    def test_scenario_series(self):
        "create_many_simulations should create many simulation directories and correct data"

        s = ScenarioSeries(
            self.test_model_data_dir, self.scenarios_dir,
            title='scenario series uno',
            description='''
Each scenario is given a title with the schema
'"<param_name1>":<scale_value1>|"<param_name2>":<scale_value2>', which can be
easily parsed into a Python dictionary later. We use the pipe instead of
comma for easier visual inspection, though this is entirely up to the
user/developer. The scenario consists of parameters scaled for each of the
specified parameters.
'''
        )

        def _scale_param(val):
            def scale_by_val(x):
                return x*val
            return scale_by_val

        scale_arange = np.arange(0.7, 0.9, 0.1)
        series_funs = [
            {
                'title': '"rad_trncf":{0:.1f}|"snow_adj":{0:.1f}'.format(val),
                'rad_trncf': _scale_param(val),
                'snow_adj': _scale_param(val)
            }
            for val in scale_arange
        ]

        s.build(series_funs)

        g_series = glob.glob(os.path.join(self.scenarios_dir, '*'))

        # this should be 5 because of series_metadata.json and inputs_dir
        assert len(g_series) == 5, g_series

        uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
        )

        uuid_dir_count = 0
        non_uuid_dir_count = 0
        series_dirs = []
        found_metadata = False
        for d in g_series:
            # XXX TODO this is in flux. at this point maybe all will be dirs
            if os.path.isdir(d):
                # at this point the files are in the base scenario dir;
                # they will be moved later when the scenario is run
                assert_valid_input_dir(self, os.path.join(d))
                if re.match(uuid_pattern, os.path.basename(d)):
                    uuid_dir_count += 1
                    assert re.match(uuid_pattern, os.path.basename(d)), os.path.basename(d)
                else:
                    non_uuid_dir_count += 1

            else:
                found_metadata = True

                series_dirs.append(d)

        assert uuid_dir_count == 3
        assert non_uuid_dir_count == 1
        assert found_metadata

        s.run()

        g_series = glob.glob(os.path.join(self.scenarios_dir, '*'))

        series_md_path = os.path.join(
            self.scenarios_dir, 'series_metadata.json'
        )
        self.assertIn(series_md_path, g_series)

        series_md = json.loads(open(series_md_path).read())
        dir_titles = series_md['uuid_title_map']
        titles = dir_titles.values()

        assert '"rad_trncf":0.7|"snow_adj":0.7' in titles
        assert '"rad_trncf":0.8|"snow_adj":0.8' in titles
        assert '"rad_trncf":0.9|"snow_adj":0.9' in titles

        p_base = Parameters(
            os.path.join(self.test_model_data_dir, 'parameters')
        )
        for g in g_series:
            if os.path.isdir(g) and 'input' not in g:
                assert_valid_output_dir(self, os.path.join(g, 'outputs'))

                md = json.loads(open(os.path.join(g, 'metadata.json')).read())

                title = md['title']
                scale_vals = eval('{' + title.replace('|', ',') + '}')

                p_scen = Parameters(os.path.join(g, 'inputs', 'parameters'))

                rad_base = p_base['rad_trncf']
                rad_scen = p_scen['rad_trncf']

                snow_base = p_base['snow_adj']
                snow_scen = p_scen['snow_adj']

                assert_array_almost_equal(
                    rad_base*scale_vals['rad_trncf'], rad_scen)
                assert_array_almost_equal(
                    snow_base*scale_vals['snow_adj'], snow_scen)


def assert_valid_input_dir(test_case, d):

    gs = [
        os.path.basename(f)
        for f in glob.glob(os.path.join(d, '*'))
    ]

    test_case.assertIn('control', gs, d)
    test_case.assertIn('parameters', gs, d)
    test_case.assertIn('data', gs, d)


def assert_valid_output_dir(test_case, d):

    go = [
            os.path.basename(f)
            for f in
            glob.glob(os.path.join(d, '*'))
    ]
    test_case.assertIn('prms_ic.out', go)
    test_case.assertIn('prms.out', go)
    test_case.assertIn('statvar.dat', go)


class TestParameters(unittest.TestCase):

    def setUp(self):

        self.test_data_dir = os.path.join('test', 'data')
        self.test_param = os.path.join(self.test_data_dir, 'parameters')
        self.temp_dir = os.path.join('test', 'data', 'tmp')
        if os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir)

        os.mkdir(self.temp_dir)

    def tearDown(self):

        if os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_faithful_copy(self):
        """
        Parameters class should load and write identical file
        """
        test_copy = os.path.join(self.temp_dir, 'param_copy_test')
        p = Parameters(self.test_param)
        p.write(test_copy)

        expected = open(self.test_param)
        generated = open(test_copy)

        # ignore first two lines ignore now date written on Parameters.write
        expected = expected.readlines()[2:]
        generated = generated.readlines()[2:]

        assert expected == generated

    def test_modify_params(self):
        "parameter and data file modification access should work as expected"

        # test parameters mod with 1d rad_trncf
        params_out = os.path.join(self.temp_dir, 'modify_params_out')

        mod_d = {'rad_trncf': lambda x: 1.1*x}
        modify_params(self.test_param, params_out, mod_d)

        generated = open(params_out).readlines()[2:]

        expected = open(
            os.path.join(
                self.test_data_dir, 'expected_rad_trncf_1.1'
            )
        ).readlines()[2:]

        assert expected == generated

        # test parameters mod with 2d snow_adj
        params_out = os.path.join(self.temp_dir, 'modify_params_out')

        mod_d = {'snow_adj': lambda x: 1.1*x}
        modify_params(self.test_param, params_out, mod_d)

        generated = open(params_out).readlines()[2:]

        expected = open(
            os.path.join(
                self.test_data_dir, 'expected_snow_adj_1.1'
            )
        ).readlines()[2:]

        for a, b in zip(expected, generated):
            try:
                a = float(a)
                b = float(b)
                self.assertAlmostEqual(a, b, places=5)
            except ValueError:
                assert a == b


def colored_string_diff(s1, s2):
    """ Writes differences between strings s1 and s2 """
    d = Differ()
    diff = d.compare(s1.splitlines(), s2.splitlines())
    diffList = [el for el in diff
                if el[0] != ' ' and el[0] != '?']

    for l in diffList:

        if l[0] == '+':
            print(bcolors.GREEN + '+' + l[1:] + bcolors.ENDC)
        elif l[0] == '-':
            print(bcolors.RED + '-' + l[1:] + bcolors.ENDC)
        else:
            assert False, 'Error, diffList entry must start with + or -'


class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
