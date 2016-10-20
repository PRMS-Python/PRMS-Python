'''
optimizer.py -- Optimization routines for PRMS parameters and data.
'''
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os, sys, json, re

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
#    if not title: 
#            title = 'unnamed_optimization_{}'.format(\
#                    dt.datetime.today().strftime('%Y-%m-%d')) 

        self.control_file = control_file
        self.working_dir = working_dir
        self.title = title
        self.description = description
        self.srad_outputs = []
        self.measured_srad = None
        self.srad_hru = None

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
        # assign the optimization object a copy of measured srad for plots
        self.measured_srad = pd.Series.from_csv(
            reference_srad_path, parse_dates=True
        )

        self.srad_hru = station_nhru

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
                    'intcp:{0:.3f}_slope:{1:.3f}'.format(np.mean(intcps[i]),\
                                                         np.mean(slopes[i]))
                )
            )
            for i in range(n_sims)
        )

        # run all scenarios
        outputs = list(series.run(nproc=nproc).outputs_iter())        
        self.srad_outputs.extend(outputs) 

        srad_end_time = dt.datetime.now()
        srad_end_time = srad_end_time.replace(second=0, microsecond=0)

        def _error(x, y):
            ret = abs(log10(x) - log10(y)).dropna()
            ret = sum(ret)
            return ret

        srad_meta = {'stage' : 'swrad',
                     'swrad_hru_id' : self.srad_hru,
                     'optimization_title' : self.title,
                     'optimization_description' : self.description,
                     'start_time' : str(srad_start_time),
                     'end_time' : str(srad_end_time),
                     'measured_swrad' : reference_srad_path,
             'sim_dirs' : [],
                     'original_params' : self.parameters.base_file,
                     'n_sims' : n_sims
                    }

        for output in outputs:
            srad_meta['sim_dirs'].append(output['simulation_dir'])
        
        json_outfile = OPJ(self.working_dir, _create_metafile_name(\
                          self.working_dir, self.title, 'swrad'))
      
        with open(json_outfile, 'w') as outf:  
            json.dump(srad_meta, outf, sort_keys = True, indent = 4,\
                      ensure_ascii = False)

        print('{0}\nOutput information sent to {1}\n'.format('-' * 80, json_outfile))

    def plot_srad_optimization(self, freq='daily', method='time_series'):
        """
        Basic plotting of current srad optimization results with 
        limited options for quick viewing, measured, original, 
        and simulated swrad at the correspinding HRU is plotted
        either as time series (all three) or  scatter (measured
        versus simulated). Not recommended for plotting results 
        when n_sims is very high, use plotting options from
        an OptimizationResult object

        Kwargs:
            freq (str): frequency of time series plots, value can be 'daily'
                or 'monthly' for solar radiation !!!need to finish monthly!!!
            method (str): 'time_series' for time series sub plot of each
                simulation alongside measured radiation. Other choice is 
                'correlation' which plots each measured daily solar radiation
                value versus the corresponding simulated variable as subplots
                one for each simulation in the optimization. With coefficients
                of determiniationi i.e. square of pearson correlation coef.  
        """
        if not self.srad_outputs:
            raise ValueError('You have not run any srad optimizations')
            
        # indices that measured and simulated swrad share (the intersection)
        X = self.measured_srad
        idx = X.index.intersection(self.srad_outputs[0]['statvar']\
                          ['swrad_{}'.format(self.srad_hru)].index)
        X = X[idx]
        n = len(self.srad_outputs) # number of simulations to plot

        if freq == 'daily' and method == 'time_series':
            fig, ax = plt.subplots(n, sharex=True, sharey=True,\
                                   figsize=(12,n*3.5))
            axs = ax.ravel()
            for i,out in enumerate(self.srad_outputs):
                axs[i].plot(out['statvar']['swrad_{}'.format(self.srad_hru)]\
                       [idx], 'r.', markersize=3, label='Simulated')
                axs[i].plot(self.measured_srad[idx], 'k.', markersize=3,\
                       label='Measured')
                axs[i].set_ylabel('sim: {}'.format(out['simulation_dir'].\
                       split(os.sep)[-1].replace('_', ' ')))
                if i == 0: axs[i].legend(markerscale=5, loc='best')
            fig.subplots_adjust(hspace=0)
            fig.autofmt_xdate()
            plt.show()  

        elif method == 'correlation':
            ## number of subplots and rows (two plots per row) 
            nrow = n//2 # round down if odd n
            ncol = 2
            odd_n = False
            if n/2. - nrow == 0.5: 
                nrow+=1 # odd number need extra row
                odd_n = True
            ## figure
            fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(12,n*3))
            axs = ax.ravel()
            ## subplot dimensions
            meas_min = min(X) 
            meas_max = max(X)
            
            for i, out in enumerate(self.srad_outputs):
                Y = out['statvar']['swrad_{}'.format(self.srad_hru)][idx]
                sim_max = max(Y)
                sim_min = min(Y)
                m = max(meas_max,sim_max)
                axs[i].plot([0, m], [0, m], 'k--', lw=2) ## one to one line
                axs[i].set_xlim(meas_min,meas_max)
                axs[i].set_ylim(sim_min, sim_max)
                axs[i].scatter(X, Y, facecolors='none', edgecolor='r', s=3)
                axs[i].set_ylabel('sim: {}'.format(out['simulation_dir']\
                      .split(os.sep)[-1].replace('_', ' ')))
                axs[i].set_xlabel('Measured shortwave radiation')
                axs[i].text(0.05, 0.95,r'$R^2 = {0:.2f}$'.format(\
                            X.corr(Y)**2), fontsize=16,\
                            ha='left', va='center', transform=axs[i].transAxes)  
            if odd_n: # empty subplot if odd number of simulations 
                fig.delaxes(axs[n])      

def _create_metafile_name(out_dir, opt_title, stage):
    """
    Search through output directory where simulations are conducted
    look for all metadata simulation json files and find out if the
    current simulation is a replicate. Then use that information to 
    build the correct file name for the output json file. The series
    are typically run in parallelel that is why this step has to be 
    done after running multiple simulations from an optimization stage.

    Args:
        out_dir (str): path to directory with model results, i.e. 
            location where simulation series outputs and optimization
            json files are located, aka Optimizer.working_dir
        opt_title (str): optimization instance title for file search
    stage (str): stage of optimization, e.g. 'swrad', 'pet' 

    Returns:
    name (str): file name for the current optimization simulation series
        metadata json file. E.g 'dry_creek_swrad_opt.json', or if
        this is the second time you have run an optimization titled
        'dry_creek' the next json file will be returned as 
        'dry_creek_swrad_opt1.json' and so on with integer increments     
    """
    swrad_meta_re = re.compile(r'^{}_{}_opt(\d*)\.json'.format(opt_title, stage))
    reps = []
    for f in os.listdir(out_dir): 
        if swrad_meta_re.match(f):
            nrep = swrad_meta_re.match(f).group(1)
            if nrep == '': 
                reps.append(0)
            else:
                reps.append(nrep)

    if not reps: 
        name = '{}_{}_opt.json'.format(opt_title, stage)
    else:
        # this is the nth optimization done under the same title
        n = max(map(int, reps)) + 1
        name = '{}_{}_opt{}.json'.format(opt_title, stage, n)
    return name

def _resample_param(param, p_min, p_max, noise_factor=0.1 ):
    """
    Resample PRMS parameter by shifting all values by a constant that is 
    taken from a uniform distribution, where the range of the shift 
    values is equal to the difference between the min(max) of the parameter
    set and the min(max) of the allowable range from PRMS. Next add noise
    to each parameter element by adding a RV from a normal distribution 
    with mean 0, sigma = param allowable range / 10.  
    
    Args:
        param (numpy.ndarray): ndarray of parameter to be resampled
        p_min (float): lower bound of PRMS allowable range for param
        p_max (float): upper bound of PRMS allowable range for param
    Kwargs: 
        noise_factor (float): factor to multiply parameter range by, 
            use the result as the standard deviation for the normal rand.
            variable used to add element wise noise. i.e. higher 
            noise facter will result in higher noise added to each param
            element.
    Returns:
        tmp (numpy.ndarry): ndarray of param after uniform random mean 
            shift and element-wise noise addition (normal r.v.) 
    """
    
    low_bnd = p_min - np.min(param) # lowest param value minus allowable min
    up_bnd = p_max - np.max(param)
    s = (p_max - p_min) * noise_factor # stddev noise, default: range*(1/10)  
    
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
        self.stage = stage
        self.metadata_json_paths = self.get_optr_jsons(working_dir, stage)     

    def get_optr_jsons(self, work_dir, stage):
        """
        Retrieve locations of optimization output jsons which contain 
        important metadata needed to understand optimization results.
        Create dictionary of each optimization stage as keys, and lists
        of corresponding json file paths for each stage as values. 

        Arguments:
            work_dir (str): path to directory with model results, i.e. 
                location where simulation series outputs and optimization
                json files are located, aka Optimizer.working_dir
            stage (str): the stage ('swrad', 'pet', 'flow', etc.) of 
                the optimization in which to gather the jsons, if
                stage is 'all' then each stage will be gathered.
        Returns:
            ret (dict): dictionary of stage (keys) and lists of 
                json file paths for that stage (values).  
        """

        ret = {}
        if stage != 'all':
            optr_metafile_re = re.compile(r'^.*_{}_opt(\d*)\.json'.format(stage))
            ret[stage] = [OPJ(work_dir, f) for f in\
                              os.listdir(work_dir) if\
                              optr_metafile_re.match(f) ]
        else: 
            stages = ['swrad', 'pet', 'flow']
            for s in stages:
                optr_metafile_re = re.compile(r'^.*_{}_opt(\d*)\.json'.format(s))
                ret[s] =  [OPJ(work_dir, f) for f in\
                               os.listdir(work_dir) if\
                               optr_metafile_re.match(f) ]
        return ret

class SradOptimizationResult(OptimizationResult):

    def __init__(self, working_dir, stage='swrad' ):
        OptimizationResult.__init__(self, working_dir, stage)

        
        
class PetOptimizationResult(OptimizationResult):

    def __init__(self, working_dir, stage='pet'):
        OptimizationResult.__init__(self, working_dir, stage)
 
