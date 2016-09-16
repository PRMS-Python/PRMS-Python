'''
optimizer.py -- Optimization routines for PRMS parameters and data.
'''
from __future__ import print_function
import pandas as pd
import numpy as np
import datetime as dt
import os, sys, json

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
    >>> control = 'path/to/control'
    >>> work_directory = 'path/to/create/simulations'
    >>> optr = Optimizer(params, data, control, work_directory, \
               title='the title', description='desc')
    >>> srad_hru = 2490
    >>> optr.srad('path/to/reference_data/measured_srad.csv', srad_hru)

    '''
    
    ## constant attributes for allowable range of solrad parameters
    ir = (-60.0, 10.0) # PRMS dday_intcp range
    sr = (0.2, 0.9) # dday_slope range

    def __init__(self, parameters, data, control_file, working_dir,
                 title, description=None):

        if isinstance(parameters, Parameters):
            self.parameters = parameters
        else:
            raise TypeError('parameters must be instance of Parameters')

        if isinstance(data, Data):
            self.data = data

        else:
            raise TypeError('data must be instance of Data')

        if not os.path.isdir(working_dir):
            os.mkdir(working_dir)

        ## user needs to enter a title for output info now
#	if not title: 
#            title = 'unnamed_optimization_{}'.format(\
#                    dt.datetime.today().strftime('%Y-%m-%d')) 

        self.control_file = control_file
        self.working_dir = working_dir
        self.title = title
        self.description = description

    def srad(self, reference_srad_path, station_nhru, n_sims=10, method='',\
             nproc=None):
        '''
        Optimize the monthly dday_intcp and dday_slope parameters by one of
        two methods: 'uniform' or 'random' for uniform sampling

        Args:
            reference_srad_path (str): path to measured solar radiation data
            station_nhru (int): hru index in PRMS that is geographically 
                near the measured solar radiation location. You must
                have swrad 'nhru' listed as a statvar output in your 
                control file.
        Kwargs:
            method (str): XXX not yet implemented- all uniform now 
                'uniform' or 'random'; if 'random',
                intcp_delta and slope_delta are ignored, if provided
            n_sims (int): number of simulations to conduct 
                parameter optimization/uncertaitnty analysis.
        Returns:
            (SradOptimizationResult)
        '''

	srad_start_time = dt.datetime.now()
        srad_start_time = srad_start_time.replace(second=0, microsecond=0)
  
	## shifting all monthly values by random amount from uniform distribution
        ## resampling degree day slope and intercept simoultaneously
        intcps = [_resample_param(self.parameters['dday_intcp'], Optimizer.ir[0],\
			Optimizer.ir[1]) for i in range(n_sims)]

        slopes = [_resample_param(self.parameters['dday_slope'], Optimizer.sr[0],\
			Optimizer.sr[1]) for i in range(n_sims)]

        self.data.write(OPJ(self.working_dir, 'data'))

        series = SimulationSeries(
            Simulation.from_data(
                self.data, _mod_params(self.parameters, intcps[i], slopes[i]),
                self.control_file,
                OPJ(
                    self.working_dir,
                    'intcp:{0:.2f}_slope:{1:.2f}'.format(np.mean(intcps[i]),\
                                                         np.mean(slopes[i]))
                )
            )
            for i in range(n_sims)
        )

        # run all scenarios
        outputs = series.run(nproc=nproc).outputs_iter()

	srad_end_time = dt.datetime.now()
        srad_end_time = srad_end_time.replace(second=0, microsecond=0)

        def _error(x, y):
            ret = abs(log10(x) - log10(y)).dropna()
            ret = sum(ret)
            return ret

        measured_srad = pd.Series.from_csv(
            reference_srad_path, parse_dates=True
        )

        srad_meta = {'stage' : 'swrad',
                     'hru_id' : station_nhru,
                     'optimization_title' : self.title,
                     'optimization_description' : self.description,
                     'start_time' : str(srad_start_time),
                     'end_time' : str(srad_end_time),
                     'measured_swrad' : reference_srad_path,
		     'sim_dirs' : [],
                     'original_params' : self.parameters.base_file
                    }

        for output in outputs:
            srad_meta['sim_dirs'].append(output['simulation_dir'])

        json_outfile = OPJ(self.working_dir, '{0}_swrad_opt.json'.format(self.title))  

        with open(json_outfile, 'w') as outf:  
            json.dump(srad_meta, outf, sort_keys = True, indent = 4, ensure_ascii = False)

        print('{0}\nOutput information sent to {1}\n'.format('-' * 80, json_outfile))

#         # calculate the top performing                                    
#         errors = (                                                        
#             (                                                             
#                 output['simulation_dir'],                                 
#                 _error(measured_srad,                                     
#                        output['statvar']['swrad_' + str(station_nhru)])   
#             )                                                             
#                                                                           
#             for output in outputs                                         
#         )                                                                 
#
#         print("directory, error (swrad)")
#         for directory, error in errors:
#             print(directory, error)                        #
#
#        monthly_errors = {str(mo): [] for mo in range(12)}
#
#        for directory, error in errors:
#
#            month, intcp, slope = (el.split(':')[1] for el in
#                                   directory.split(os.sep)[-1].split('_'))
#
#            monthly_errors[month].append((intcp, slope, error))
#
#        rankings = {
#            str(mo): list(sorted(monthly_errors[str(mo)], key=lambda x: x[-1]))
#            for mo in range(12)
#        }
#
#        tops = [(mo, rankings[str(mo)][0]) for mo in range(12)]
#
#        # update internal parameters
#        for top in tops:
#            mo = top[0]
#            self.parameters['dday_intcp'][mo] = tops[mo][1][0]
#            self.parameters['dday_slope'][mo] = tops[mo][1][1]
#
#        return {
#            'best': tops,
#            'all': rankings
#        }

def _resample_param(param, p_min, p_max):
    """
    Resample PRMS parameter by shifting all values by a constant that is 
    taken from a uniform distribution, where the range of the shift 
    values is equal to the difference between the min(max) of the parameter
    set and the min(max) of the allowable range from PRMS. Next add noise
    to each parameter element by adding a RV from a normal distribution 
    with mean 0, sigma = param allowable range / 10.  
    
    Arguments:
        param (numpy.ndarray): ndarray of parameter to be resampled
        p_min (float): lower bound of PRMS allowable range for param
        p_max (float): upper bound of PRMS allowable range for param
    Returns:
        tmp (numpy.ndarry): ndarray of param after uniform random mean 
            shift and element-wise noise addition (normal r.v.) 
    """
    
    low_bnd = p_min - np.min(param) # lowest param value minus allowable min
    up_bnd = p_max - np.max(param)
    s = (p_max - p_min) / 10 # one tenth of the range- small for noise
    
    shifted_param = np.random.uniform(low=low_bnd, high=up_bnd) + param
    ## add noise to each point keeping result within allowable range  
    while True:
        tmp = shifted_param + np.random.normal(0,s,size=(np.shape(param)))
        if np.max(tmp) <= p_max and np.min(tmp) >= p_min:
            return tmp
        
def _mod_params(parameters, intcp, slope):

    ret = deepcopy(parameters)
    #print (intcp, slope)

    ret['dday_intcp'] = intcp
    ret['dday_slope'] = slope

    return ret


class OptimizationResult:
    
    def __init__(self, working_dir, stage='all'):
        self.working_dir = working_dir 
        self.metadata_json_paths = self.\
                                get_optr_jsons(self.working_dir, stage)     
    @staticmethod 
    def get_optr_jsons(work_dir, stage='all'):
	"""
        Retrieve locations of optimization output jsons which contain 
        important metadata needed to understand optimization results.
        Create dictionary of each optimization stage as keys, and lists
        of corresponding json file paths for each stage as values. 

        Arguments:
            work_dir (str): path to simulation directory where 
                optimization simulations were conducted and where 
                corresponding json files should exist. 
            stage (str): the stage ('swrad', 'pet', 'flow', etc.) of 
                the optimization in which to gather the jsons, if
                stage is 'all' then each stage will be gathered.
        Returns:
            ret (dict): dictionary of stage (keys) and lists of 
                json file paths for that stage (values).  
	"""

        ret = {}
        if stage != 'all':
            ret[stage] = [OPJ(work_dir, f) for f in\
                              os.listdir(work_dir) if\
                              f.endswith('_{0}_opt.json'.format(stage)) ]
        else: 
            stages = ['swrad', 'pet', 'flow']
            for s in stages:
                ret[s] = OptimizationResult.get_optr_jsons(work_dir, s)        

        return ret

class SradOptimizationResult(OptimizationResult):

    def __init__(self, working_dir, stage='swrad'):
        self.working_dir = working_dir
        self.stage = stage

        self.metadata_json_paths =  OptimizationResult.get_optr_jsons(self.working_dir, stage)

        
        
class PetOptimizationResult(OptimizationResult):

    def __init__(self, working_dir, stage='pet'):
        self.working_dir = working_dir
        self.stage = stage

        self.metadata_json_paths =  OptimizationResult.get_optr_jsons(self.working_dir, stage)
 
