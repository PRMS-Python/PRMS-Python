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
import copy
import datetime
import itertools
import numpy as np
import os
import subprocess

from collections import OrderedDict


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


def modify_params(params_in, params_out, param_mods=None):
    '''
    Given a parameter file in and a dictionary of param_mods, write modified
    parameters to params_out.

    Example:

    Below we modify the monthly jh_coef by increasing it 10% for every month.

    >>> params_in = 'models/lbdc/params'
    >>> params_out = 'scenarios/jh_coef_1.1/params'
    >>> scale_10pct = lambda x: x * 1.1
    >>> modify_params(params_in, params_out, {'jh_coef': scale_10pct})

    So param_mods is a dictionary of with keys being parameter names and
    values a function that operates on a single value. Currently we only
    accept functions that operate on single values without reference to any
    other parameters. The function will be applied to every cell, month, or
    cascade routing rule for which the parameter is defined.

    Arguments:
        params_in (str): location on disk of the base parameter file
        params_out (str): location on disk where the modified parameters will
            be written
        param_mods (dict): param name-keyed, param modification function-valued

    Returns:
        None
    '''
    p_in = Parameters(params_in)

    for k in param_mods:
        p_in[k] = param_mods[k](p_in[k])

    p_in.write(params_out)


class Parameters(object):
    '''
    Disk-based representation of a PRMS parameter file. For the sake of
    memory efficiency, we only load parameters from ``base_file`` that get
    modifified through item assignment, for example

    >>> p = Parameters('example_params')
    >>> p['jh_coef'] = p['jh_coef']*1.1
    >>> p.write('example_modified_params')

    will read parameter information from the params file to check that
    ``jh_coef`` is present in the parameter file, read the lines corresponding
    to ``jh_coef`` data and assign the new value as requested. Internally,
    a reference is kept to only modified parameter data,
    so when ``p.write(modified_params_file)`` is called, mostly this will copy
    from ``base_file`` to ``modified_params_file``.
    '''

    def __init__(self, base_file):

        self.base_file = base_file
        self.base_file_reader = open(base_file)
        self.dimensions, self.base_params = self.__read_base(base_file)
        self.param_arrays = dict()

    def write(self, out_name):

        with open(self.base_file, 'r') as base_file:
            with open(out_name, 'w') as out_file:
                # write metadata
                out_file.write('File Auto-generated by PRMS-Python\n')
                out_file.write(datetime.datetime.now().isoformat() + '\n')

                # # write dimensions
                out_file.write('** Dimensions **\n')
                # for dimname, nvals in self.dimensions.iteritems():
                    # out_file.write('####\n')
                    # out_file.write(dimname + '\n')
                    # out_file.write(str(nvals) + '\n')

                # write parameters; pre-sorted by data start line on read
                name_is_next = False
                params_start = False
                write_params_lines = False
                for l in base_file:

                    if not params_start and l.strip() == '** Parameters **':
                        out_file.write('** Parameters **\n')
                        params_start = True

                    elif l.strip() == '####':
                        name_is_next = True

                    elif name_is_next:
                        name = l.strip()
                        if name not in self.param_arrays:
                            out_file.write('####\n')
                            out_file.write(name + '\n')
                            name_is_next = False
                            write_params_lines = True
                        else:
                            write_params_lines = False
                            name_is_next = False

                    elif write_params_lines:
                        out_file.write(l.strip() + '\n')

                # write all parameters that had been accessed and/or modified
                for param, new_arr in self.param_arrays.iteritems():

                    out_file.write('####\n')

                    param_info = [el for el in self.base_params
                                  if el['name'] == param].pop()

                    out_file.write(str(param_info['name']) + '\n')
                    out_file.write(str(param_info['ndims']) + '\n')
                    for dimname in param_info['dimnames']:
                        out_file.write(dimname + '\n')
                    out_file.write(str(param_info['length']) + '\n')
                    out_file.write(str(param_info['vartype']) + '\n')
                    out_file.writelines([str(a) + '\n'
                                         for a in new_arr.flatten()])

    def __read_base(self, base_file):
        "Read base file returning 2-tuple of dimension and params dict"

        params_startline, dimensions = self.__make_dimensions_dict(base_file)
        base_params = self.__make_parameter_dict(base_file, params_startline)

        return (dimensions, base_params)

    def __make_dimensions_dict(self, base_file):
        """
        Extract dimensions and each dimension length. Run before
        __make_parameter_dict.
        """
        ret = OrderedDict()

        dim_name = ''
        dim_len = 0
        # finished = False
        found_dim_start = False
        # while not finished:
        for idx, l in enumerate(self.base_file_reader):

            if l.strip() == '** Dimensions **':  # start of dimensions
                found_dim_start = True

            elif '#' in l:  # comments
                pass

            elif l.strip() == '** Parameters **':  # start of parameters
                dimlines = idx
                # finished = True
                break

            elif found_dim_start:

                if dim_name == '':
                    dim_name = l.strip()
                else:
                    dim_len = int(l)
                    ret.update({dim_name: dim_len})
                    dim_name = ''

        return (dimlines, ret)

    def __make_parameter_dict(self, base_file, params_startline=0):
        ret = []

        name = ''
        ndims = 0
        dimnames = []
        length = 0
        vartype = ''

        dimnames_read = 0
        data_startline = 0

        for idx, l in enumerate(self.base_file_reader):

            if '#' in l:
                # we have a comment; the next lines will be new
                # parameter metadata. No data for the first time through, so
                # we don't want to append an metadata blob with empty values
                if name:
                    ret.append(
                        dict(
                            name=name,
                            ndims=ndims,
                            dimnames=dimnames,
                            length=length,
                            vartype=vartype,
                            data_startline=data_startline
                        )
                    )

                    name = ''
                    ndims = 0
                    dimnames = []
                    length = 0
                    vartype = ''
                    dimnames_read = 0

            elif not name:
                name = l.strip()

            elif not ndims:
                ndims = int(l.strip())

            elif not (dimnames_read == ndims):
                dimnames.append(l.strip())
                dimnames_read += 1

            elif not length:
                length = int(l.strip())

            elif not vartype:
                vartype = l.strip()
                # advance one from current position and account for starting
                # to count from zero
                data_startline = params_startline + idx + 2

        # need to append one more time since iteration will have stopped after
        # last line
        ret.append(
            dict(
                name=name,
                ndims=ndims,
                dimnames=dimnames,
                length=length,
                vartype=vartype,
                data_startline=data_startline
            )
        )

        return ret

    def __getitem__(self, key):
        """
        Look up a parameter by its name.

        Raises:
            KeyError if parameter name is not valid
        """
        def load_parameter_array(param_metadata):

            startline = param_metadata['data_startline']
            endline = startline + param_metadata['length'] + 1

            arr = np.genfromtxt(
                itertools.islice(
                    open(self.base_file), startline, endline
                )
            )

            if param_metadata['ndims'] > 1:
                dimsizes = [
                    self.dimensions[d] for d in param_metadata['dimnames']
                ]
                dimsizes.reverse()
                arr = arr.reshape(dimsizes)

            return arr

        if key in self.param_arrays:
            return self.param_arrays[key]

        else:
            try:
                param_metadata = [
                    el for el in self.base_params if el['name'] == key
                ].pop()

            except IndexError:
                raise KeyError(key)

            arr = load_parameter_array(param_metadata)

            # cache the value for future access (but maybe shouldn't?)
            self.param_arrays.update({key: arr})

            return arr

    def __setitem__(self, key, value):

        if key in self.param_arrays:
            cur_arr = self.param_arrays[key]
            if not value.shape == cur_arr.shape:
                raise ValueError('New array does not match existing')

            self.param_arrays[key] = value
