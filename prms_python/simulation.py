# -*- coding: utf-8 -*-
"""
simulation.py -- Contains ``Simulation`` and ``SimulationSeries`` classes and 
associated functions for managing PRMS simulations at a low level.
"""

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
    Series of simulations all to be run through a common interface. 
    
    Utilizes :class:`multiprocessing.Pool` class to parallelize the 
    execution of series of PRMS simulations. SimulationSeries also
    allows the user to define the PRMS executable command which is 
    set to "prms" as default. It is best to add the prms executable to 
    your $PATH environment variable. Each simulation that is run through
    ``SimulationSeries`` will follow the strict file structure as defined
    by :func:`Simulation.run()`. This class is useful particularly for
    creating new programatic workflows not provided yet by PRMS-Python.

    Arguments:
        simulations (list or tuple): list of :class:`Simulation` objects
            to be run.

    Example:
        Lets say you have already created a series of PRMS models by modifying
        the input climatic forcing data, e.g. you have 100 *data* files and
        you want to run each using the same *control* and *parameters* file.
        For simplicity lets say there is a directory that contains all 100
        *data* files e.g. data1, data2, ... or whatever they are named and 
        nothing else. This example also assumes that you want each simulation 
        to be run and stored in directories named after the *data* files as 
        shown.

        >>> data_dir = 'dir_that_contains_all_data_files'
        >>> params = Parameters('path_to_parameter_file')
        >>> control_path = 'path_to_control'
        >>> # a list comprehension to make multiple simulations with
        >>> # different data files, alternatively you could use a for loop
        >>> sims = [
                    Simulation.from_data
                      (  
                        Data(data_file),
                        params,
                        control_path,
                        simulation_dir='sim_{}'.format(data_file)
                      )
                    for data_file in os.listdir(data_dir)
                    ]

        Next we can use ``SimulationSeries`` to run all of these 
        simulations in parrallel. For example we may use 8 logical cores
        on a common desktop computer.

        >>> sim_series = SimulationSeries(sims)
        >>> sim_series.run(nprocs=8)

        The ``SimulationSeries.run()`` method will run all 100 simulations
        where chunks of 8 at a time will be run in parrallel. Inputs and 
        outputs of each simulation will be sent to each simulation's
        ``simulation_dir`` following the file structure of 
        :func:`Simulation.run()`.

    Note:
        The :class:`Simulation` and :class:`SimulationSeries` classes
        are low-level in that they alone do not create metadata for
        PRMS simulation scenarios. In other words they do not produce
        any additional files that may help the user know what differs
        between individual simulations. 
        
    '''

    def __init__(self, simulations):
        self.series = list(simulations)

    def run(self, prms_exec='prms', nproc=None):
        """
        Method to run multiple :class:`Simulation` objects in parrallel.

        Keyword Arguments:
            prms_exec (str): name of PRMS executable on $PATH or path to 
                executable
            nproc (int or None): number of logical or physical processors
                for parrallel execution of PRMS simulations.  

        Example:
            see :class:`SimulationSeries`

        Note:
            If ``nproc`` is not assigned the deault action is to use half
            of the available processecors on the machine using the Python
            :mod:`multiprocessing` module. 

        """
        if not nproc:
            nproc = mp.cpu_count() // 2 

        pool = mp.Pool(processes=nproc)
        pool.map(_simulation_runner, self.series)
        pool.close() 

        return self

    def outputs_iter(self):
        '''
        Return a :class:`generator` of directories with the path to the 
        ``simulation_dir`` as well as paths to the *statvar.dat* output 
        file, and *data* and *parameters* input files used in the simulation.

        Yields:
            :obj:`dict`: dictionary of paths to simulation directory,
                input, and output files.

        Example:
            >>> ser = SimulationSeries(simulations)
            >>> ser.run()
            >>> g = ser.outputs_iter()

            Would return something like

            >>> print(g.next())
                {
                   'simulation_dir': 'path/to/sim/', 
                   'statvar': 'path/to/statvar', 
                   'data': 'path/to/data', 
                   'parameters': 'path/to/parameters'
                 }
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
    Class that runs and manages file structure for a single PRMS simulation.
    
    The ``Simulation`` class provides low-level managment of a PRMS simulation
    by copying model input files from ``input_dir`` argument to an output dir
    ``simulation_dir``. The file stucture for an individual simulation after
    calling the ``run`` method is simple, two subdirectories "inputs" and 
    "outputs" are created under ``simulation_dir`` and the respective input
    and output files from the current PRMS simulation are transfered there after
    the ``Simulation.run()`` method is called which executes the PRMS model,
    (see examples below in :func:`Simulation.run`).

    A ``Simulation`` instance checks that all required PRMS inputs (control, 
    parameters, data) exist in the expected locations. If simulation_dir is 
    provided and does not exist, it will be created. If it does exist it will 
    be overwritten.
 
    Keyword Arguments:
        input_dir (str): path to directory that contains control, parameter, 
            and data files for the simulation
        simulation_dir (str): directory path to bundle inputs and outputs
       
    Example:
        see :func:`Simulation.run()`

    Raises:
        RuntimeError: if input directory does not contain a PRMS *data*,
            *parameters*, and *control* file.

    """
    def __init__(self, input_dir=None, simulation_dir=None):
        # check if model input paths exist
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
            # build output (simulation_dir) and move input files there
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
        Create a ``Simulation`` from a :class:`Data` and :class:`Parameter` object,
        plus a path to the *control* file, and providing a ``simulation_dir`` 
        where the simulation should be run.

        Arguments:
            data (:class:`Data`): ``Data`` object for simulation
            parameters (:class:`Parameters`): ``Parameters`` object for simulation 
            control_path (str): path to control file
            simulation_dir (str): path to directory where simulations will be
                run and where input and output will be stored. If it exists it will 
                be overwritten.

        Returns:
            :class:`Simulation` ready to be run using ``simulation_dir`` for
                inputs and outputs

        Example:

            >>> d = Data('path_to_data_file')
            >>> p = Parameters('path_to_parameters_file')
            >>> c = 'path_to_control_file'
            >>> sim_dir = 'path_to_create_simulation'
            >>> sim = Simulation.from_data(d, p, c, sim_dir)
            >>> sim.run()

        Raises:
            TypeError: if ``data`` and ``parameters`` arguments are not of type
                :class:`Data` and :class:`Parameters`
        '''

        if not isinstance(data, Data):
            raise TypeError('data must be instance of Data')

        if not isinstance(parameters, Parameters):
            raise TypeError('parameters must be instance of Parameters, not '\
                             + str(type(parameters)))

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
        """
        Run a ``Simulation`` instance using PRMS input files from ``input_dir`` 
        and copy to the ``Simulation`` file structure under ``simulation_dir`` if
        given, otherwise leave PRMS input output unstructured and in ``input_dir``

        This method runs a single PRMS simulation from a ``Simulation`` instance,
        waits until the process has completed and then transfers model input and 
        output files to respective newly created directories. See example of the 
        file structure that is created under different workflows of the ``run``
        method below.

        Keyword Arguments:
            prms_exec (str): name of PRMS executable on $PATH or path to executable
        
        Examples:
            If we create a :class:`Simulation` instance by only assigning the 
            ``input_dir`` argument and call its ``run`` method the model will be
            run in the ``input_dir`` and all model input and output files will
            remain in ``input_dir``,
    
            >>> import os
            >>> input_dir = os.path.join(
                                          'PRMS-Python',
                                          'prms_python',
                                          'models',
                                          'lbcd'
                                        )
            >>> os.listdir(input_dir)
                ['data',
                 'data_3deg_upshift',
                 'parameters',
                 'parameters_adjusted',
                 'control']
            >>> sim = Simulation(input_dir)
            >>> sim.run()
            >>> os.listdir(input_dir) # all input and outputs in input_dir
                ['data',
                 'data_3deg_upshift',
                 'parameters',
                 'parameters_adjusted',
                 'control',
                 'statvar.dat',
                 'prms_ic.out',
                 'prms.out' ]
            
            Instead if we assigned a path for ``simulation_dir`` keyword 
            argument and then called ``run``, i.e. 
            
            >>> sim = Simulation(input_dir, 'path_simulation')
            >>> sim.run()
    
        the files structure for the PRMS simulation created by ``Simulation.run()`` 
        would be::
             
             path_simulation                             
             ├── inputs
             │   ├── control
             │   ├── data
             │   └── parameters
             └── outputs
                 ├── data_3deg_upshift
                 ├── parameters_adjusted
                 ├── prms_ic.out
                 ├── prms.out
                 └── statvar.dat

        Note:
            As shown in the last example, currently the ``Simulation.run`` routine only
            recognizes the *data*, *parameters*. and *control* file as PRMS inputs,
            all other files found in ``input_dir`` before *and* after normal completion 
            of the PRMS simulation will be transferred to ``simulation_dir/outputs/``. 
        """
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
