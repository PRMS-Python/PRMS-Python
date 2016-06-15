import glob
import os
import shutil
import subprocess


class Simulation(object):
    """
    Simulation class for tracking the inputs and outputs of a single
    PRMS simulation.
    """
    def __init__(self, input_dir, simulation_dir=None):
        """
        Create a new Simulation object from a simulation directory. Check that
        all required PRMS inputs (control, parameters, data) exist in the
        expected locations.

        Also parses the control file to make sure that the data and parameter
        file specified match the ones in the input_dir.

        If simulation_dir is provided and does not exist, it will be created.
        If it does exist it will be overwritten.

        Arguments:
            input_dir (str): location of control, parameter, and data
                files for the Simulation
            simulation_dir (str): location to bundle inputs and outputs
        """
        idir = input_dir
        self.input_dir = idir
        self.control_path = os.path.join(idir, 'control')
        self.parameters_path = os.path.join(idir, 'parameters')
        self.data_path = os.path.join(idir, 'data')

        if not os.path.exists(self.control_path):
            raise RuntimeError('Control file missing from ' + idir)

        if not os.path.exists(self.parameters_path):
            raise RuntimeError('Parameter file missing from ' + idir)

        if not os.path.exists(self.data_path):
            raise RuntimeError('Data file missing from ' + idir)

        self.simulation_dir = simulation_dir
        if simulation_dir and simulation_dir != input_dir:

            if os.path.exists(simulation_dir):
                shutil.rmtree(simulation_dir)

            os.mkdir(simulation_dir)

            shutil.copy(self.control_path, simulation_dir)
            shutil.copy(self.data_path, simulation_dir)
            shutil.copy(self.parameters_path, simulation_dir)

            self.control_path = os.path.join(simulation_dir, 'control')
            self.parameters_path = os.path.join(simulation_dir, 'parameters')
            self.data_path = os.path.join(simulation_dir, 'data')

        self.has_run = False

    def run(self):

        cwd = os.getcwdu()

        if self.simulation_dir:
            os.chdir(self.simulation_dir)

        else:
            os.chdir(self.input_dir)

        p = subprocess.Popen(
            'prms control', shell=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        prms_finished = False
        while not prms_finished:
            p.communicate()

            poll = p.poll()
            print poll
            prms_finished = poll == 0

        self.has_run = True

        if self.simulation_dir:
            os.mkdir('inputs')
            os.mkdir('outputs')
            shutil.move('data', 'inputs')
            shutil.move('parameters', 'inputs')
            shutil.move('control', 'inputs')

            # all remaining files are outputs
            for g in glob.glob('*'):
                if not os.path.isdir(g):
                    shutil.move(g, 'outputs')

        os.chdir(cwd)
