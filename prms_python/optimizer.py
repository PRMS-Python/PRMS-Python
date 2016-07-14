'''
optimizer.py -- Optimization routines for PRMS parameters and data.
'''
import pandas as pd
import numpy as np
import os

from .data import Data
from .parameters import Parameters
from .scenario import ScenarioSeries


class Optimizer:
    '''
    Container for a PRMS parameter optimization routine consisting of the
    four stages as described in Hay, et al, 2006
    (ftp://brrftp.cr.usgs.gov/pub/mows/software/luca_s/jawraHay.pdf).

    Example:

    >>> from prms_python import Data, Optimizer, Parameters
    >>> params = Parameters('path/to/parameters')
    >>> data = Data('path/to/data')
    >>> optr = Optimizer(params, data, title='the title', description='desc')
    >>> optr.srad('path/to/reference_data/measured_srad.csv')

    '''

    def __init__(self, parameters, data, working_dir,
                 title=None, description=None):

        if isinstance(parameters, Parameters):
            self.parameters = parameters
        else:
            raise TypeError('parameters must be instance of Parameters')

        if isinstance(data, Data):
            self.data = data
        else:
            raise TypeError('data must be instance of Data')

        self.working_dir = working_dir
        self.title = title
        self.description = description

    def srad(self, reference_srad_path, station_nhru, method='',
             dday_intcp_range=None, dday_slope_range=None,
             intcp_delta=None, slope_delta=None):
        '''
        Optimize the monthly dday_intcp and dday_slope parameters by one of
        two methods: 'uniform' or 'random' for uniform sampling

        Args:
            reference_srad_path (str): path to measured solar radiation data
        Kwargs:
            method (str): 'uniform' or 'random'; if 'random',
                intcp_delta and slope_delta are ignored, if provided
            dday_intcp_range ((float, float)): two-tuple of minimum and
                maximum value to  consider for the dday_intcp parameter
            dday_slope_range ((float, float)): two-tuple of minimum and
                maximum value to  consider for the dday_slope parameter
            intcp_delta (float): resolution of grid to test in intcp dimension
            slope_delta (float): resolution of grid to test in slope dimension

        Returns:
            (SradOptimizationResult)
        '''
        if dday_intcp_range is None:
            dday_intcp_range = (-60.0, 10.0)
            intcp_delta = 10.0
        elif intcp_delta is None:
            intcp_delta = (dday_intcp_range[1] - dday_intcp_range[0]) / 4.0

        if dday_slope_range is None:
            dday_slope_range = (0.2, 0.9)
            slope_delta = .05
        elif slope_delta is None:
            slope_delta = (dday_slope_range[1] - dday_slope_range[0]) / 4.0

        # create parameters
        ir = dday_intcp_range
        sr = dday_slope_range

        intcps = np.arange(ir[0], ir[1], intcp_delta)
        slopes = np.arange(sr[0], sr[1], slope_delta)

        param_grid = np.meshgrid(intcps, slopes)

        def _mod_params(parameters, month, intcp, slope):

            parameters['dday_intcp'][month] = intcp
            parameters['dday_slope'][month] = slope

        parameters_iter = (
            {
                'parameters':
                    _mod_params(self.parameters, month, intcp, slope),

                'title': '"month":{0},"dday_intcp":{1:.3f},'
                         '"dday_slope":{2:.3f}'.format(month, intcp, slope),
            }
            for month in range(12)
            for intcp, slope in param_grid
        )

        # create ScenarioSeries from parameters

        # XXX TODO XXX TODO
        series = ScenarioSeries.from_params_iter(
            self.working_dir,
            parameters_iter=parameters_iter,
            title=self.title,
            description=self.description
        )

        # run all scenarios
        series.run()

        def _error(x, y):
            return float(abs(x - y))/float(len(x))

        # calculate the top performing
        modeled_srads = (
            (output['title'], output['statvar']['swrad_' + str(station_nhru)])

            # XXX TODO XXX TODO
            for output in series.outputs
        )

        measured_srad = pd.read_csv(reference_srad_path, parse_dates=True)

        errors = (
            (modeled_srads[0], _error(measured_srad, modeled_srads[1]))
            for modeled_srad in modeled_srads
        )
        rankings = list(sorted(errors, key=lambda x: x[1]))

        # update internal parameters
        self.parameters = series.outputs[rankings[0]]['parameters']

        return rankings


class OptimizationResult:

    pass


class SradOptimizationResult(OptimizationResult):

    pass
