from __future__ import print_function
import glob
import multiprocessing as mp
import os
import shutil
import subprocess
import time

from .data import Data
from .parameters import Parameters
from .util import load_statvar


OPJ = os.path.join


class SimulationSeries(object):
    '''
    Series of simulations all to be run through a common interface
    '''

    def __init__(self, simulations):
        # XXX TODO would love to not have to use list here, but otherwise can't
        # access the simulations after they have run through map
        self.series = list(simulations)

    def run(self, prms_exec='prms', nproc=None):

        if not nproc:
            nproc = mp.cpu_count()//2 

        pool = mp.Pool(processes=nproc)
        pool.map(_simulation_runner, self.series)
        pool.close() 

        return self

    def outputs_iter(self):
        '''
        Return an iterator of directories with the path to the simulation_dir
        as well as a pandas.DataFrame of the statvar output, and the Data and
        Parameters representations used in the simulation.

        Example:
            >>> ser = SimulationSeries(simulations)
            >>> ser.run()
            >>> g = ser.outputs_iter()
            >>> print(g.next())

        Would return something like

            {'simulation_dir': 'path/to/sim/', 'statvar': 'path/to/statvar',
             'data': <data.Data>, 'parameters': <parameters.Parameters>}

        Returns:
            (generator(dict)):
        '''
        dirs = list(s.simulation_dir for s in self.series)
        print(dirs)

        return (
            {
                'simulation_dir': d,
                'statvar': OPJ(d, 'outputs', 'statvar.dat'),
                'data': OPJ(d, 'inputs', 'data'),
                'parameters': OPJ(d, 'inputs', 'parameters')
            }
            for d in dirs
        )        

    def __len__(self):
        return len(list(self.outputs_iter()))


def _simulation_runner(sim):
    sim.run(prms_exec='prms')


class Simulation(object):
    """
    Simulation class for tracking the inputs and outputs of a single
    PRMS simulation.
    """
    def __init__(self, input_dir=None, simulation_dir=None):
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
        self.simulation_dir = simulation_dir
        if idir is not None:
            self.control_path = os.path.join(idir, 'control')
            self.parameters_path = os.path.join(idir, 'parameters')
            self.data_path = os.path.join(idir, 'data')

            if not os.path.exists(self.control_path):
                raise RuntimeError('Control file missing from ' + idir)

            if not os.path.exists(self.parameters_path):
                raise RuntimeError('Parameter file missing from ' + idir)

            if not os.path.exists(self.data_path):
                raise RuntimeError('Data file missing from ' + idir)

            if simulation_dir is not None:
                self.simulation_dir = simulation_dir
                if simulation_dir and simulation_dir != input_dir:

                    if os.path.exists(simulation_dir):
                        shutil.rmtree(simulation_dir)

                    os.mkdir(simulation_dir)

                    shutil.copy(self.control_path, simulation_dir)
                    shutil.copy(self.data_path, simulation_dir)
                    shutil.copy(self.parameters_path, simulation_dir)

                    self.control_path = os.path.join(simulation_dir, 'control')
                    self.parameters_path = os.path.join(simulation_dir,
                                                        'parameters')
                    self.data_path = os.path.join(simulation_dir, 'data')

        else:
            self.control_path = None
            self.parameters_path = None
            self.data_path = None
            self.simulation_dir = None

        self.has_run = False

    @classmethod
    def from_data(cls, data, parameters, control_path, simulation_dir):
        '''
        Create a Simulation from a Data and Parameter object, plus a path
        to the control file, and providing a simulation_dir where the
        simulation should be run.

        Args:
            data (Data): weather station data
            parameters (Parameters): simulation parameters
            control_path (str): path to control file
            simulation_dir (str): path to directory where simulations will be
                run and output will be stored. If it exists it will be
                overwritten.

        Returns:
            (Simulation) simulation ready to be run using simulation_dir for
                inputs and outputs
        '''

        if not isinstance(data, Data):
            raise TypeError('data must be instance of Data')

        if not isinstance(parameters, Parameters):
            raise TypeError('parameters must be instance of Parameters, not ' + str(type(parameters)))

        if os.path.exists(simulation_dir):
            shutil.rmtree(simulation_dir)

        os.makedirs(simulation_dir)

        sim = cls()
        sim.simulation_dir = simulation_dir

        sd = simulation_dir

        data_path = OPJ(sd, 'data')
        data.write(data_path)
        params_path = OPJ(sd, 'parameters')
        parameters.write(params_path)
        shutil.copy(control_path, OPJ(sd, 'control'))

        return sim

    def run(self, prms_exec='prms'):

        cwd = os.getcwd()

        if self.simulation_dir:
            os.chdir(self.simulation_dir)

        else:
            os.chdir(self.input_dir)

        p = subprocess.Popen(
            prms_exec + ' control', shell=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        prms_finished = False
        checked_once = False
        while not prms_finished:

            if not checked_once:
                p.communicate()
                checked_once = True

            poll = p.poll()
            prms_finished = poll >= 0

        self.has_run = True
        # avoid too many files open error
        p.stdout.close()
        p.stderr.close()

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
