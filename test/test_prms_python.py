import os
import shutil
import unittest

from prms_python import modify_params


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

    def test_create_scenario(self):
        """a simulation setup should create a simulation directory"""
        assert False

    def test_create_many_scenarios(self):
        """create_many_simulations should create many corresponding simulation directories"""
        assert False

    def test_modify_params(self):
        "parameter and data file modification access should work as expected"

        params_out = os.path.join(self.test_data_dir, 'modify_params_out')
        mod_d = {
            'rad_trncf': lambda x: 1.1*x,
        }

        modify_params(
            os.path.join(self.starting_dirs[0]),
            params_out,
            mod_d
        )

        generated = open(params_out).read().strip()

        expected = open(
            os.path.join(
                self.test_data_dir, 'expected_mod_params'
            ).read().strip()
        )

        assert expected == generated

    def test_simulation(self):
        assert False

    def test_simulate_many(self):
        assert False
