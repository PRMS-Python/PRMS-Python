'''
optimizer.py -- Optimization routines for PRMS parameters and data.
'''
import pandas as pd
import numpy as np
import os, sys

from copy import deepcopy
from numpy import log10

from .data import Data
from .parameters import Parameters
from .simulation import Simulation, SimulationSeries


OPJ = os.path.join


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

    def __init__(self, parameters, data, control_file, working_dir,
                 title=None, description=None):

        if isinstance(parameters, Parameters):
            self.parameters = parameters
        else:
            raise TypeError('parameters must be instance of Parameters')

        if isinstance(data, Data):
            self.data = data

        else:
            raise TypeError('data must be instance of Data')

        self.control_file = control_file
        self.working_dir = working_dir
        self.title = title
        self.description = description

    def srad(self, reference_srad_path, station_nhru, method='',
             dday_intcp_range=None, dday_slope_range=None,
             intcp_delta=None, slope_delta=None, nproc=None):
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

        pgrid = np.meshgrid(intcps, slopes)

        self.data.write(OPJ(self.working_dir, 'data'))

        series = SimulationSeries(
            Simulation.from_data(
                self.data, _mod_params(self.parameters, month, intcp, slope),
                self.control_file,
                os.path.join(
                    self.working_dir,
                    'month:{0}_intcp:{1}_slope:{2}'.format(month, intcp, slope)
                )
            )
            for intcp in intcps.flatten()
            for slope in slopes.flatten()
            for month in range(12)
        )

        # run all scenarios
        outputs = series.run(nproc=nproc).outputs_iter()

        def _error(x, y):

            ret = abs(log10(x) - log10(y)).dropna()

            ret = sum(ret)

            return ret

        measured_srad = pd.Series.from_csv(
            reference_srad_path, parse_dates=True
        )

        # calculate the top performing
        errors = (
            (
                output['simulation_dir'],
                _error(measured_srad,
                       output['statvar']['swrad_' + str(station_nhru)])
            )

            for output in outputs
        )

        monthly_errors = {str(mo): [] for mo in range(12)}

        for directory, error in errors:

            month, intcp, slope = (el.split(':')[1] for el in
                                   directory.split(os.sep)[-1].split('_'))

            monthly_errors[month].append((intcp, slope, error))

        rankings = {
            str(mo): list(sorted(monthly_errors[str(mo)], key=lambda x: x[-1]))
            for mo in range(12)
        }

        tops = [(mo, rankings[str(mo)][0]) for mo in range(12)]

        # update internal parameters
        for top in tops:
            mo = top[0]
            self.parameters['dday_intcp'][mo] = tops[mo][1][0]
            self.parameters['dday_slope'][mo] = tops[mo][1][1]

        return {
            'best': tops,
            'all': rankings
        }


def _mod_params(parameters, month, intcp, slope):

    ret = deepcopy(parameters)
    print (month, intcp, slope)

    ret['dday_intcp'][month] = intcp
    ret['dday_slope'][month] = slope

    return ret


class OptimizationResult:

    pass


class SradOptimizationResult(OptimizationResult):

    pass
