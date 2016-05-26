'''
Tools for running PRMS and creating calibration and prediction scenarios.

Date: May 25 2016

Author: Matthew Turner <maturner01@gmail.com>
'''
import os
import shutil
import subprocess


__author__ = 'Matthew Turner <maturner01@gmail.com>'


CONTROL_FILENAME = 'control.txt'
DATA_FILENAME = 'data.txt'
PARAM_FILENAME = 'param.txt'


class PRMSRun:

    def __init__(self, input_dir=None, output_dir=None):
        self.input_dir = input_dir
        self.output_dir = output_dir

    @classmethod
    def scenario(cls, unmodified_dir, input_dir, output_dir=None,
                 data_mod_fun=None, param_mod_fun=None):
        _create_scenario(unmodified_dir, input_dir,
                         data_mod_fun, param_mod_fun)

        if output_dir is None:
            output_dir = input_dir + '-output'

        return cls(input_dir, output_dir)

    def run(self):
        control_file = os.path.join(self.input_dir, 'control')

        subprocess.Popen(['prms', control_file],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

        res = subprocess.communicate()

        if res.returncode > 0:
            raise RuntimeError('PRMS Error:\n' + res[1])


def _create_scenario(unmodified_dir, input_dir, data_mod_fun, param_mod_fun):
    '''
    Given an unmodified directory of PRMS data, create a scenario of modified
    data and/or parameters, saved to input_dir. `data_mod_fun` and
    `param_mod_fun` are function delegates that modify the netCDF
    representation of either the data or parameter file.
    '''
    if not os.path.isdir(input_dir):
        os.makedirs(input_dir)
    else:
        shutil.rmtree(input_dir)
        os.makedirs(input_dir)

    inputs_to_modify = PRMSInputs.from_dir(unmodified_dir)

    modified_inputs = inputs_to_modify.modify(
        data_mod_fun=data_mod_fun,
        param_mod_fun=param_mod_fun
    )

    return modified_inputs


class PRMSInputs:

    def __init__(self,
                 control_file,
                 data_file,
                 param_file):
        """docstring for __init__"""

        self.control_file = ControlFile(control_file)
        self.data_file = DataFile(data_file)
        self.param_file = ParamFile(param_file)

    @classmethod
    def from_dir(cls, d):
        control = os.path.join(d, 'control')
        data = os.path.join(d, 'data')
        param = os.path.join(d, 'param')

        if not os.path.exists(control):
            raise RuntimeError('No control file in directory ' + d)
        if not os.path.exists(data):
            raise RuntimeError('No data file in directory ' + d)
        if not os.path.exists(param):
            raise RuntimeError('No param file in directory ' + d)

        return cls(control, data, param)


class ControlFile:
    def __init__(self, path=None):
        self.data_path = None
        self.param_path = None


class ParamFile:

    def __init__(self):
        """docstring for __init__"""
        pass

    def to_netcdf(self, nc_path):
        pass


class DataFile:

    def __init__(self):
        """docstring for __init__"""
        pass

    def to_netcdf(self, nc_path):
        pass
