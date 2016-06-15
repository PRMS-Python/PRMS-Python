import os
import shutil
import unittest

from difflib import Differ

from prms_python import modify_params, Parameters


class TestScenarioSimulations(unittest.TestCase):
    """
    Simulations should take a base directory and return a simulation directory
    """

    def setUp(self):

        self.test_data_dir = os.path.join('test', 'data')

        td = self.test_data_dir

        self.simulation_dirs = [
            os.path.join(td, 'simdir1'), os.path.join(td, 'simdir2')
        ]

        for d in self.simulation_dirs:
            if os.path.exists(d):
                shutil.rmtree(d)

            os.mkdir(d)

    def tearDown(self):
        """docstring for tearDown"""

        for d in self.simulation_dirs:
            if os.path.exists(d):
                shutil.rmtree(d)

    def test_simulation(self):
        assert False

    def test_create_scenario(self):
        """a simulation setup should create a simulation directory"""
        assert False

    def test_create_many_scenarios(self):
        "create_many_simulations should create many simulation directories"
        assert False


class TestParameters(unittest.TestCase):

    def setUp(self):

        self.test_data_dir = os.path.join('test', 'data')
        self.test_param = os.path.join(self.test_data_dir, 'parameter')
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
