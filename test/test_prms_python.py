import json
import glob
import os
import shutil
import unittest

from difflib import Differ
from numpy.testing import assert_array_almost_equal

from prms_python import modify_params, Parameters, Scenario, Simulation


class TestSimulations(unittest.TestCase):
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
        self.assertIn('animation.out.nhru', g)

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


class TestScenarios(unittest.TestCase):

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
        """a simulation setup should create a simulation directory with correct scenario data"""

        s = Scenario(
            self.test_model_data_dir, self.scenario_dir,
            title='Scenario Uno', description='test scenario for prms_python'
        )

        param_mods = {
            'snow_adj': lambda x: 1.1*x,
            'rad_trncf': lambda x: 0.9*x
        }
        s.build_scenario(param_mod_funs=param_mods)

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

    def test_create_many_scenarios(self):
        "create_many_simulations should create many simulation directories"
        assert False


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
    test_case.assertIn('animation.out.nhru', go)


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
            print bcolors.GREEN + '+' + l[1:] + bcolors.ENDC
        elif l[0] == '-':
            print bcolors.RED + '-' + l[1:] + bcolors.ENDC
        else:
            assert False, 'Error, diffList entry must start with + or -'


class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
