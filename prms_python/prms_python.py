'''
PRMS-Python: Powerful, sane tools for manipulating PRMS input data to create
new scenarios or parameterizations for sensitivity analysis, scenario
modeling, or whatever other uses this might have.

The fundamental process in scenario development is to modify some "base"
starting data to create some "scenario" data. No matter what data we're using,
once it's ready, we run a PRMS "simulation" on that data.

This module presents a Simulation and Scenario class, where each tracks
relevant provenance information and input files to facilitate better
data management techniques to streamline later analyses.
'''
import os
import subprocess


class Simulation(object):
    """
    Simulation class for tracking the inputs and outputs of a single
    PRMS simulation.
    """
    def __init__(self, simulation_dir):
        """
        Create a new Simulation object from a simulation directory. Check that
        all required PRMS inputs (control, parameters, data) exist in the
        expected locations.

        Also parses the control file to make sure that the data and parameter
        file specified match the ones in the simulation_dir

        Arguments:
            simulation_dir (str): location of control, parameter, and data
                files for the Simulation
        """
        sd = simulation_dir

        self.control = os.path.join(sd, 'control')
        self.parameter = os.path.join(sd, 'parameter')
        self.data = os.path.join(sd, 'data')

        if not os.path.exists(self.control):
            raise RuntimeError('Control file missing from ' + sd)

        if not os.path.exists(self.parameter):
            raise RuntimeError('Parameter file missing from ' + sd)

        if not os.path.exists(self.data):
            raise RuntimeError('Data file missing from ' + sd)

        self.has_run = False

    def run(self):

        prms_finished = False
        while not prms_finished:

            p = subprocess.Popen(
                'prms ' + self.control, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            p.communicate()

            poll = p.poll()
            prms_finished = poll != 0

        self.has_run = True

    def visualize(self):

        if not self.has_run:
            raise RuntimeError(
                'You must first run the model before performing visualizations'
            )

        return None
