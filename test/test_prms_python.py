import os
import shutil
import unittest


class TestScenarioSimulations(unittest.TestCase):
    """Simulations should take a base directory and return a simulation directory"""

    def setUp(self):

        self.test_data_dir = os.path.join('test', 'data', 'tmp')

        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)

        os.mkdir(self.test_data_dir)

        td = self.test_data_dir

        self.starting_dirs = [
            os.path.join(td, 'basedir1'), os.path.join(td, 'basedir2')
        ]

        self.simulation_dirs = [
            os.path.join(td, 'simdir1'), os.path.join(td, 'simdir2')
        ]

    def tearDown(self):
        """docstring for tearDown"""

        shutil.rmtree(self.test_data_dir)

    def test_create_scenario(self):
        """a simulation setup should create a simulation directory"""
        assert False

    def test_create_many_scenarios(self):
        """create_many_simulations should create many corresponding simulation directories"""
        assert False

    def test_apply_funcs(self):
        """parameter and data file modification access should work as expected"""
        assert False

    def test_simulation(self):
        assert False

    def test_simulate_many(self):
        assert False
