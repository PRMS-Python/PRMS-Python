'''
optimizer.py -- Optimization routines for PRMS parameters and data.
'''
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os, sys, json, re, shutil
import multiprocessing as mp

from copy import copy
from copy import deepcopy
from numpy import log10

from .data import Data
from .parameters import Parameters
from .simulation import Simulation, SimulationSeries
from .util import load_statvar, nash_sutcliffe, percent_bias, rmse

OPJ = os.path.join


class Optimizer:
    '''
    Container for a PRMS parameter optimization routine consisting of 
    stages similar to what is described in Hay, et al, 2006
    (ftp://brrftp.cr.usgs.gov/pub/mows/software/luca_s/jawraHay.pdf).

    Example:

    >>> from prms_python import Data, Optimizer, Parameters
    >>> params = Parameters('path/to/parameters')
    >>> data = Data('path/to/data')
    >>> control = 'path/to/control'
    >>> work_directory = 'path/to/create/simulations'
    >>> optr = Optimizer(params, data, control, work_directory, \
               title='the title', description='desc')
    >>> measured = 'path/to/measured/csv' 
    >>> statvar_name = 'basin_cfs' # or any other valid statvar 
    >>> params_to_resample = ['dday_intcp', 'dday_slope'] # list
    >>> optr.monte_carlo(measured, params_to_resample, statvar_name)

    '''
        
    #dic for min/max of parameter allowable ranges, add more when needed
    param_ranges = {'dday_intcp': (-60.0, 10.0), 
                    'dday_slope': (0.2, 0.9),
                    'jh_coef': (0.005, 0.06), 
                    'pt_alpha': (1.0, 2.0), 
                    'potet_coef_hru_mo': (1.0, 2.0), 
                    'tmax_index': (-10.0, 110.0), 
                    'tmin_lapse': (-10.0, 10.0), 
                    'soil_moist_max': (0.001, 10.0), 
                    'rain_adj': (0.5, 2.0),
                    'ppt_rad_adj': (0.0, 0.5),
                    'radadj_intcp': (0.0, 1.0),
                    'radadj_slope': (0.0, 1.0),
                    'radj_sppt': (0.0, 1.0),
                    'radj_wppt': (0.0, 1.0),
                    'radmax': (0.1, 1.0)
                   } 

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

        input_dir = '{}'.format(os.sep).join(control_file.split(os.sep)[:-1])
        if not os.path.isfile(OPJ(input_dir, 'statvar.dat')):
            print('You have no statvar.dat file in your model directory')
            print('Running PRMS on original data in {} for later comparison'\
                  .format(input_dir))
            sim = Simulation(input_dir)
            sim.run()

        if not os.path.isdir(working_dir):
            os.mkdir(working_dir)
        
        self.input_dir = input_dir
        self.control_file = control_file
        self.working_dir = working_dir
        self.title = title
        self.description = description
        self.measured_arb = None # for arbitrary output methods
        self.statvar_name = None
        self.arb_outputs = []

    def monte_carlo(self, reference_path, param_names, statvar_name, \
                    stage, n_sims=10, method='uniform', mu_factor=1,\
                    noise_factor=0.1, nproc=None):
        '''
        Optimize the monthly dday_intcp and dday_slope parameters 
        (two key parameters in the ddsolrad module in PRMS) by one of
        multiple methods (in development): Monte Carlo default method

        Args:
            reference_path (str): path to measured data for optimization
            param_names (list): list of parameter names to resample
            statvar_name (str): name of statisical variable output name for
                optimization 
            stage (str): custom name of optimization stage e.g. 'ddsolrad' 
        Kwargs:
            n_sims (int): number of simulations to conduct 
                parameter optimization/uncertaitnty analysis.
            method (str): resampling method for parameters (normal or uniform)
            mu_factor (float): coefficient to scale mean of the parameter(s)
                to resample from when using the normal distribution to resample
                i.e. a value of 1.5 will sample from a normal rv with mean
                50% higher than the original parameter mean
            noise_factor (float): scales the variance of noise to add to
                parameter values when using normal rv (method='normal')
            nproc (int): number of processors available to run PRMS simulations
        '''
        if '_' in stage: 
            raise ValueError('stage name cannot contain an underscore')
        # assign the optimization object a copy of measured data for plots 
        self.measured_arb = pd.Series.from_csv(reference_path,\
                                               parse_dates=True)
        # statistical variable output name  
        self.statvar_name = statvar_name

        start_time = dt.datetime.now()
        start_time = start_time.replace(second=0, microsecond=0)
        # resample params for all simulations- potential place to serialize
        params = []
        for name in param_names: # create list of lists of resampled params
            tmp = []
            for idx in range(n_sims):
                tmp.append(resample_param(self.parameters, name, how=method,\
                           mu_factor=mu_factor, noise_factor=noise_factor))
            params.append(list(tmp))
        
        # SimulationSeries comprised of each resampled param set
        series = SimulationSeries(
            Simulation.from_data(
                self.data, _mod_params(self.parameters,\
                                   [params[n][i] for n in range(len(params))],\
                                   param_names),
                self.control_file,
                OPJ(
                    self.working_dir, # name of sim: first param and mean value
                    '{0}:{1:.6f}'.format(param_names[0], np.mean(params[0][i]))
                )
            )
            for i in range(n_sims)
        )

        # run 
        outputs = list(series.run(nproc=nproc).outputs_iter())        
        self.arb_outputs.extend(outputs) # for current instance- add outputs 

        end_time = dt.datetime.now()
        end_time = end_time.replace(second=0, microsecond=0)
        
        # json metadata for Monte Carlo run 
        meta = { 'params_adjusted' : param_names,
                 'statvar_name' : self.statvar_name,
                 'optimization_title' : self.title,
                 'optimization_description' : self.description,
                 'start_time' : str(start_time),
                 'end_time' : str(end_time),
                 'measured' : reference_path,
                 'method' : 'Monte Carlo',
                 'mu_factor' : mu_factor,
                 'noise_factor' : noise_factor,
                 'resample': method,
                 'sim_dirs' : [],
                 'stage': '{}'.format(stage),
                 'original_params' : self.parameters.base_file,
                 'nproc': nproc,
                 'n_sims' : n_sims
               }

        for output in outputs:
            meta['sim_dirs'].append(output['simulation_dir'])
       
        json_outfile = OPJ(self.working_dir, _create_metafile_name(\
                          self.working_dir, self.title, stage))
      
        with open(json_outfile, 'w') as outf:  
            json.dump(meta, outf, sort_keys = True, indent = 4,\
                      ensure_ascii = False)

        print('{0}\nOutput information sent to {1}\n'.\
                                               format('-' * 80, json_outfile))

    def plot_optimization(self, freq='daily', method='time_series',\
                          plot_vars='both', plot_1to1=True, return_fig=False,\
                          n_plots=4):
        """
        Basic plotting of current optimization results with limited options. 
        Plots measured, original simluated, and optimization simulated variabes
        Not recommended for plotting results when n_sims is very large, instead 
        use options from an OptimizationResult object, or employ a user-defined 
        method using the result data.

        Kwargs:
            freq (str): frequency of time series plots, value can be 'daily'
                or 'monthly' for solar radiation 
            method (str): 'time_series' for time series sub plot of each
                simulation alongside measured radiation. Another choice is 
                'correlation' which plots each measured daily solar radiation
                value versus the corresponding simulated variable as subplots
                one for each simulation in the optimization. With coefficients
                of determiniationi i.e. square of pearson correlation coef.  
            plot_vars (str): what to plot alongside simulated srad: 
                'meas': plot simulated along with measured swrad
                'orig': plot simulated along with the original simulated swrad
                'both': plot simulated, with original simulation and measured
            plot_1to1 (bool): if True plot one to one line on correlation 
                scatter plot, otherwise exclude.
            return_fig (bool): flag whether to return matplotlib figure 
        Returns: 
            f (matplotlib.figure.Figure): If kwarg return_fig=True, then return
                copy of the figure that is generated to the user. 
        """
        if not self.arb_outputs:
            raise ValueError('You have not run any optimizations')
        var_name = self.statvar_name
        X = self.measured_arb
        idx = X.index.intersection(load_statvar(self.arb_outputs[0]\
                           ['statvar'])['{}'.format(var_name)].index)
        X = X[idx]
        orig = load_statvar(OPJ(self.input_dir, 'statvar.dat'))['{}'\
                            .format(var_name)][idx]
        meas = self.measured_arb[idx]
        sims = [load_statvar(out['statvar'])['{}'.format(var_name)][idx] for \
                out in self.arb_outputs]
        simdirs = [out['simulation_dir'].split(os.sep)[-1].\
                   replace('_', ' ') for out in self.arb_outputs]
        var_name = '{}'.format(self.statvar_name)
        n = len(self.arb_outputs) # number of simulations to plot
        # user defined number of subplots from first n_plots results 
        if (n > n_plots): n = n_plots  
        # styles for each plot
        ms = 4 # markersize for all points
        orig_sty = dict(linestyle='none',markersize=ms,\
                           markerfacecolor='none', marker='s',\
                           markeredgecolor='royalblue', color='royalblue') 
        meas_sty = dict(linestyle='none',markersize=ms+1,\
                           markerfacecolor='none', marker='1',\
                           markeredgecolor='k', color='k') 
        sim_sty = dict(linestyle='none',markersize=ms,\
                           markerfacecolor='none', marker='o',\
                           markeredgecolor='r', color='r') 
        ## number of subplots and rows (two plots per row) 
        nrow = n//2 # round down if odd n
        ncol = 2
        odd_n = False
        if n/2. - nrow == 0.5: 
            nrow+=1 # odd number need extra row
            odd_n = True
        ########
        ## Start plots depnding on key word arguments
        ########
        if freq == 'daily' and method == 'time_series':
            fig, ax = plt.subplots(n, sharex=True, sharey=True,\
                                   figsize=(12,n*3.5))
            axs = ax.ravel()
            for i,sim in enumerate(sims[:n]):
                if plot_vars in ('meas', 'both'):
                    axs[i].plot(meas, label='Measured', **meas_sty)
                if plot_vars in ('orig', 'both'):
                    axs[i].plot(orig, label='Original sim.', **orig_sty)
                axs[i].plot(sim, **sim_sty)
                axs[i].set_ylabel('sim: {}'.format(simdirs[i]), fontsize=10)
                if i == 0: axs[i].legend(markerscale=5, loc='best')
            fig.subplots_adjust(hspace=0)
            fig.autofmt_xdate()
        #monthly means
        elif freq == 'monthly' and method == 'time_series':
            # compute monthly means
            meas = meas.groupby(meas.index.month).mean()
            orig = orig.groupby(orig.index.month).mean()
            # change line styles for monthly plots to lines not points
            for d in (orig_sty, meas_sty, sim_sty):
                d['linestyle'] = '-'
                d['marker'] = None

            fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(12,n*3.5))
            axs = ax.ravel()
            for i,sim in enumerate(sims[:n]):
                if plot_vars in ('meas', 'both'):
                    axs[i].plot(meas, label='Measured', **meas_sty)
                if plot_vars in ('orig', 'both'):
                    axs[i].plot(orig, label='Original sim.', **orig_sty)
                sim = sim.groupby(sim.index.month).mean()
                axs[i].plot(sim, **sim_sty)
                axs[i].set_ylabel('sim: {}\nmean'.format(simdirs[i]),\
                                  fontsize=10)
                axs[i].set_xlim(0.5,12.5)
                if i == 0: axs[i].legend(markerscale=5, loc='best')
            if odd_n: # empty subplot if odd number of simulations 
                fig.delaxes(axs[n])
            fig.text(0.5, 0.1, 'month') 
        #x-y scatter
        elif method == 'correlation':
            ## figure
            fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(12,n*3))
            axs = ax.ravel()
            ## subplot dimensions
            meas_min = min(X) 
            meas_max = max(X)
            
            for i, sim in enumerate(sims[:n]):
                Y = sim
                sim_max = max(Y)
                sim_min = min(Y)
                m = max(meas_max,sim_max)
                if plot_1to1:
                    axs[i].plot([0, m], [0, m], 'k--', lw=2) ## one to one line
                axs[i].set_xlim(meas_min,meas_max)
                axs[i].set_ylim(sim_min, sim_max)
                axs[i].plot(X, Y, **sim_sty)
                axs[i].set_ylabel('sim: {}'.format(simdirs[i]))
                axs[i].set_xlabel('Measured {}'.format(var_name))
                axs[i].text(0.05, 0.95,r'$R^2 = {0:.2f}$'.format(\
                            X.corr(Y)**2), fontsize=16,\
                            ha='left', va='center', transform=axs[i].transAxes)  
            if odd_n: # empty subplot if odd number of simulations 
                fig.delaxes(axs[n])      

        if return_fig:
            return fig

def _create_metafile_name(out_dir, opt_title, stage):
    """
    Search through output directory where simulations are conducted
    look for all metadata simulation json files and find out if the
    current simulation is a replicate. Then use that information to 
    build the correct file name for the output json file. The series
    are typically run in parallel that is why this step has to be 
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
    meta_re = re.compile(r'^{}_{}_opt(\d*)\.json'.format(opt_title, stage))
    reps = []
    for f in os.listdir(out_dir): 
        if meta_re.match(f):
            nrep = meta_re.match(f).group(1)
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

def resample_param(params, param_name, how='uniform', mu_factor=1,\
                   noise_factor=0.1):
    """
    Resample PRMS parameter by shifting all values by a constant that is 
    taken from a uniform distribution, where the range of the uniform 
    values is equal to the difference between the min(max) of the parameter
    set and the min(max) of the allowable range from PRMS. If the resampling
    method ("how" argument) is set to 'normal', randomly sample a normal 
    distribution with mean 0 and sigma = param allowable range multiplied by
    noise_factor, then add this random value to original param value. If 
    parameters have array length <= 366 then individual parameter values are 
    resampled otherwise resample all param values at once, i.e. by taking
    a single random value from the uniform distribution or by adding a single
    random value samplied from the normal distribution to all param values. 

    Args:
        params (parameters.Parameters): parameter object 
        param_name (str): name of PRMS parameter to resample
    Kwargs: 
        how (str): distribution to resample parameters from in the case 
            that each parameter element can be resampled (len <=366)
            Currently works for uniform and normal distributions. 
        noise_factor (float): factor to multiply parameter range by, 
            use the result as the standard deviation for the normal rand.
            variable used to add element wise noise. i.e. higher 
            noise facter will result in higher variance. Must be > 0.
    Returns:
        ret (numpy.ndarry): ndarray of param after uniform random mean 
            shift or element-wise noise addition (normal r.v.) 
    """
    p_min, p_max = Optimizer.param_ranges.get(param_name,(-1,-1))
    
    # create dictionary of parameter basic info (not values)
    param_dic = {param['name']: param for param in params.base_params}
    if not param_dic.get(param_name):
        raise KeyError('{} is not a valid parameter'.format(param_name))
        
    if p_min == p_max == -1:
        raise ValueError("""{} has not been added to the dictionary of
        parameters to resample, add it's allowable min and max value
        to the Optimizer.param_ranges attribute in
        Optimizer.py""".format(param_name))
        
    dim_case = None
    nhru = params.dimensions['nhru']
    ndims = param_dic.get(param_name)['ndims']
    dimnames = param_dic.get(param_name)['dimnames']
    length = param_dic.get(param_name)['length']
    param = deepcopy(params[param_name])
    
    # could expand list and check parameter name also e.g. cascade_flg
    # is a parameter that should not be changed 
    dims_to_not_change = set(['ncascade','ncascdgw','nreach',\
                             'nsegment']) 
    if (len(set.intersection(dims_to_not_change, set(dimnames))) > 0):
        raise ValueError("""{} should not be resampled as
                          it relates to the location of cascade flow
                          parameters.""".format(param_name))
        
    # use param info to get dimension info- e.g. if multidimensional
    if (ndims == 1 and length <= 366):
        dim_case = 'resample_each_value' # for smaller param dimensions
    elif (ndims == 1 and length > 366):
        dim_case = 'resample_all_values_once' # covers nssr, ngw, etc. 
    elif (ndims == 2 and dimnames[1] == 'nmonths' and \
          nhru == params.dimensions[dimnames[0]]):
        dim_case = 'nhru_nmonths'   
    elif not dim_case:
        raise ValueError('The {} parameter is not set for resampling'.\
                         format(param_name))        
#     #testing purposes    
#     print('name: ', param_name)
#     print('max_val: ', p_max)
#     print('min_val: ', p_min)
#     print('ndims: ', ndims)
#     print('dimnames: ', dimnames)
#     print('length: ', length)
#     print('resample_method: ', dim_case)

    s = (p_max - p_min) * noise_factor # std_dev (s) default: param_range/10  
    #do resampling based on param dimensions and sampling distribution
    if dim_case == 'resample_all_values_once': 
        if how == 'uniform':
            ret = np.random.uniform(low=p_min, high=p_max, size=param.shape) 
        elif how == 'normal': # scale parameter mean if mu_factor given 
            mu = np.mean(param) * mu_factor
            if mu_factor != 1:
                tmp = np.random.normal(mu, s, size=param.shape)
                ret = tmp + param
            else: # if default mu_factor, add noise from N(0,s)
                ret = np.random.normal(0, s, size=param.shape) + param

    elif dim_case == 'resample_each_value':
        ret = param
        if how == 'uniform':
            ret = np.random.uniform(low=p_min, high=p_max, size=param.shape)
        elif how == 'normal': # the original value is considered the mean
            if len(ret.shape) != 0:
                for i, el in enumerate(param):
                    mu = el * mu_factor
                    ret[i] =  np.random.normal(mu, s) 
            else: # single value parameter
                mu = param * mu_factor
                ret = np.random.normal(mu, s)

    # nhru by nmonth dimensional params
    elif dim_case == 'nhru_nmonths':
        ret = param
        if how == 'uniform':
            for month in range(12):
                ret[month] = np.random.uniform(low=p_min, high=p_max,\
                                                           size=param[0].shape)
        elif how == 'normal':
            for i, el in enumerate(param):
                mu = np.mean(el) * mu_factor
                if mu_factor != 1:
                    tmp = np.random.normal(mu, s, size=el.shape)
                    ret[i] = tmp + el
                else:
                    ret[i] = np.random.normal(0, s, size=el.shape) + el

    return ret

def _mod_params(parameters, params, param_names):
    # loop through list of params and assign their values to Parameter instance
    ret = copy(parameters)
    for idx, param in enumerate(params):
        ret[param_names[idx]] = np.array(param)
    return ret


class OptimizationResult:
    
    def __init__(self, working_dir, stage):
        self.working_dir = working_dir 
        self.stage = stage
        self.metadata_json_paths = self._get_optr_jsons(working_dir, stage) 
        self.total_sims = self._count_total_sims()
        self.statvar_name = self._get_statvar_name(stage)
        self.measured = self._get_measured(stage)
        self.input_dir = self._get_input_dir(stage)
        self.input_params = self._get_input_params(stage)

        # if there are more than one input param for given stage
        if len(self.input_params) > 1:
            print(
"""Warning: there were more than one initial parameter sets used for the 
    optimization for stage: {}. Make sure to compare the the correct input 
    params with their corresponding output sims.""".format(stage))
            print('\nThis optimization stage used the\
 following input parameter files:\n{}'.format('\n'.join(self.input_params)))
 
    def _count_total_sims(self):
        # total number of simulations of given stage in working directory
        tracked_dirs = []
        for f in self.metadata_json_paths[self.stage]:
            with open(f) as fh:
                json_data = json.load(fh)
                tracked_dirs.extend(json_data.get('sim_dirs'))
        return len(tracked_dirs)

    def _get_optr_jsons(self, work_dir, stage):
        """
        Retrieve locations of optimization output jsons which contain 
        metadata needed to understand optimization results.
        Create dictionary of each optimization with stage as key and lists
        of corresponding json file paths as values. 

        Args:
            work_dir (str): path to directory with model results, i.e. 
                location where simulation series outputs and optimization
                json files are located, aka Optimizer.working_dir
            stage (str): the stage ('ddsolrad', 'jhpet', 'flow', etc.) of 
                the optimization in which to gather the jsons
        Returns:
            ret (dict): dictionary of stage (keys) and lists of 
                json file paths for that stage (values).  
        """

        ret = {}
        optr_metafile_re = re.compile(r'^.*_{}_opt(\d*)\.json'.\
                                      format(stage))
        ret[stage] = [OPJ(work_dir, f) for f in\
                          os.listdir(work_dir) if\
                          optr_metafile_re.match(f)]
        return ret

    def _get_input_dir(self, stage):
        # retrieves the input directory from the first json file of given stage
        json_file = self.metadata_json_paths[stage][0]
        with open(json_file) as jf:
            meta_dic = json.load(jf)
        return '{}'.format(os.sep).join(meta_dic['original_params'].\
                                                            split(os.sep)[:-1]) 

    def _get_input_params(self, stage):
        json_files = self.metadata_json_paths[stage]
        param_paths = []
        for json_file in json_files:
            with open(json_file) as jf:
                meta_dic = json.load(jf)
                param_paths.append(meta_dic['original_params'])
        return list(set(param_paths))

    def _get_sim_dirs(self, stage):
        jsons = self.metadata_json_paths[stage]
        json_files = []
        sim_dirs = []
        for inf in jsons:
            with open(inf) as json_file:
                json_files.append(json.load(json_file))
        for json_file in json_files:
            sim_dirs.extend(json_file['sim_dirs'])
        # list of all simulation directory paths for stage 
        return sim_dirs

    def _get_measured(self, stage):
        # only need to open one json file to get this information
        if not self.metadata_json_paths.get(stage):
            return # no optimization json files exist for given stage
        first_json = self.metadata_json_paths[stage][0]
        with open(first_json) as json_file:
            json_data = json.load(json_file)
        measured_series = pd.Series.from_csv(json_data.get('measured'),\
                                              parse_dates=True)
        return measured_series 

    def _get_statvar_name(self, stage):
        # only need to open one json file to get this information
        try:
            first_json = self.metadata_json_paths[stage][0]
        except:
            raise ValueError("""No optimization has been run for
                              stage: {}""".format(stage))    
        with open(first_json) as json_file:
            json_data = json.load(json_file)
        var_name = json_data.get('statvar_name')

        return var_name

    def result_table(self, freq='daily', top_n=5, latex=False):
        ##TODO: add stats for freq options annual (means or sum)

        sim_dirs = self._get_sim_dirs(self.stage)
        if top_n >= len(sim_dirs): 
            top_n = len(sim_dirs) + 1 # for returning inclusive last sim
        sim_names = [path.split(os.sep)[-1] for path in sim_dirs] 
        meas_var = self._get_measured(self.stage)
        statvar_name = self._get_statvar_name(self.stage)
        orig_statvar = load_statvar(OPJ(self.input_dir,'statvar.dat'))\
                       [statvar_name]

        result_df = pd.DataFrame(columns=\
                            ['NSE','RMSE','PBIAS','COEF_DET','ABS(PBIAS)'])
        orig_results = pd.DataFrame(index=['orig_params'],\
                                 columns=['NSE','RMSE','PBIAS','COEF_DET'])
        # get datetime indices that overlap from measured and simulated
        sim_out = load_statvar(OPJ(sim_dirs[0], 'outputs', 'statvar.dat'))\
                                               [statvar_name]
        idx = meas_var.index.intersection(sim_out.index)
        meas_var = copy(meas_var[idx])
        #sim_out = sim_out[idx]    
        orig_statvar = orig_statvar[idx]
        
        if freq == 'monthly':
            meas_mo = meas_var.groupby(meas_var.index.month).mean()
            orig_mo = orig_statvar.groupby(orig_statvar.index.month).mean()            
        
        for i, sim in enumerate(sim_dirs):
            try: 
                sim_out = load_statvar(OPJ(sim, 'outputs', 'statvar.dat'))\
                                               ['{}'.format(statvar_name)]
            except: # simulation might have been removed or missing
                pass

            sim_out = sim_out[idx]    
            if freq == 'daily':
                result_df.loc[sim_names[i]] = [\
                              nash_sutcliffe(meas_var, sim_out),\
                              rmse(meas_var, sim_out),\
                              percent_bias(meas_var,sim_out),\
                              meas_var.corr(sim_out)**2,\
                              np.abs(percent_bias(meas_var, sim_out)) ]                                  
                orig_results.loc['orig_params'] = [\
                              nash_sutcliffe(orig_statvar,meas_var),\
                              rmse(orig_statvar,meas_var),\
                              percent_bias(orig_statvar,meas_var),\
                              orig_statvar.corr(meas_var)**2]
                              
            elif freq == 'monthly':
                sim_out = sim_out.groupby(sim_out.index.month).mean()
                result_df.loc[sim_names[i]] = [\
                              nash_sutcliffe(meas_mo, sim_out),\
                              rmse(meas_mo, sim_out),\
                              percent_bias(meas_mo, sim_out),\
                              meas_mo.corr(sim_out)**2,\
                              np.abs(percent_bias(meas_mo, sim_out)) ]    
                orig_results.loc['orig_params'] = [\
                              nash_sutcliffe(orig_mo,meas_mo),\
                              rmse(orig_mo,meas_mo),\
                              percent_bias(orig_mo,meas_mo),\
                              orig_mo.corr(meas_mo)**2]
                                 
        sorted_result = result_df.sort_values(by=['NSE','RMSE','ABS(PBIAS)',\
                               'COEF_DET'], ascending=[False,True,True,False])
        sorted_result.columns.name = '{} parameters'.format(self.stage)
        sorted_result = sorted_result[['NSE','RMSE','PBIAS','COEF_DET']] 
        sorted_result = pd.concat([orig_results,sorted_result])

        if latex: return sorted_result[:top_n].to_latex(escape=False)
        else: return  sorted_result[:top_n]

    def get_top_ranked_sims(self, sorted_df):
        # use result table to make dic with best param and statvar paths 
        # index of table are simulation directory names
        ret = {
              'dir_name' : [],
              'param_path' : [],
              'statvar_path' : [],
              'params_adjusted' : []
              }
        
        json_paths = self.metadata_json_paths[self.stage]         
        
        for el in sorted_df.drop('orig_params').index:
            ret['dir_name'].append(el)
            ret['param_path'].append(OPJ(self.working_dir,el,'inputs',\
                                                                 'parameters'))
            ret['statvar_path'].append(OPJ(self.working_dir,el,'outputs',\
                                                                'statvar.dat'))
            for f in json_paths:
                with open(f) as fh:
                    json_data = json.load(fh)
                    if OPJ(self.working_dir, el) in json_data.get('sim_dirs'):
                        ret['params_adjusted'].append(\
                                             json_data.get('params_adjusted'))  
                  
        return ret

    def archive(self, remove_sims=True, metric_freq='daily'):
        """
            Create archive directory to hold json files that contain 
            information of adjusted parameters, model output, and performance 
            metrics for each Optimizer simulation of the 
            OptimizationResult.stage in the OptimizationResult.working_dir.                  
                                                                                    
            Kwargs:                                                              
                remove_sims (bool) : If True recursively delete all folders 
                    and files associated with original simulations of the 
                    OptimizationResult.stage in the 
                    OptimizationResult.working_dir, if False do not delete 
                    simulations.
                metric_freq (Str) : Frequency of output metric computation 
                    for recording of model performance. Can be 'daily' 
                    (default) or 'monthly'. Note, other results can be computed 
                    later with archived results. 
            Returns:                                                                
                None                          
            """
        # create output archive directory
        archive_dir = OPJ(self.working_dir,"{}_archived".format(self.stage))
        if not os.path.isdir(archive_dir):
            os.mkdir(archive_dir)
        
        # create table and use to make mapping dic    
        table = self.result_table(freq=metric_freq,\
                                  top_n=self.total_sims, latex=False)
        map_dic = self.get_top_ranked_sims(table)
        
        metadata_json_paths = self.metadata_json_paths[self.stage]
        # get measured optimization variable path
        first_json = metadata_json_paths[0]
 
        with open(first_json) as json_file:                                                                                                                                                                 
            json_data = json.load(json_file)                                                                                                                                                                
            measured_path = json_data.get('measured')                                                                                                         
                                                     
        # record info for each simulation and archive to JSONs
        # pandas series and numpy arrays are converted to Python lists 
        # for JSON serialization
        for i, sim in enumerate(map_dic.get('dir_name')):
            json_path = OPJ(archive_dir, '{sim}.json'.format(sim=sim))
            try:
                output_series = load_statvar(OPJ(self.working_dir, sim,\
                                              'outputs', 'statvar.dat'))\
                                              [self.statvar_name]
            except: # simulation directory was already removed
                continue
            
            # look for resampling method info for the particular simulation
            for f in metadata_json_paths:
                with open(f) as tmp_file:
                    tmp = json.load(tmp_file)
                    if OPJ(self.working_dir, sim) in tmp.get('sim_dirs'):
                        resample = tmp.get('resample')
                        noise_factor = tmp.get('noise_factor')   
                        mu_factor = tmp.get('mu_factor')   

            json_data = {
                          'param_names' : [],
                          'param_values' : [],
                          'original_param_path' : self.input_params, 
                          'measured_path' : measured_path,
                          'output_name' : self.statvar_name,
                          'output_date_index' : output_series.\
                                                   index.astype(str).tolist(),
                          'output_values' : output_series.values.tolist(),
                          'metric_freq' : metric_freq,
                          'resample' : resample,
                          'mu_factor' : mu_factor,
                          'noise_factor' : noise_factor,
                          'NSE' : table.loc[sim, 'NSE'],
                          'RMSE' : table.loc[sim, 'RMSE'],
                          'PBIAS' : table.loc[sim, 'PBIAS'],
                          'COEF_DET' : table.loc[sim, 'COEF_DET']        
                        }  
           
            for param in map_dic.get('params_adjusted')[i]:  
                json_data['param_names'].append(param)
                json_data['param_values'].append(Parameters(\
                                          map_dic.get('param_path')[i])[param]\
                                          .tolist())
            # save JSON file into archive directory    
            with open(json_path, 'w') as outf:
                json.dump(json_data, outf, sort_keys = True, indent = 4,\
                          ensure_ascii = False)     

            # recursively delete all simulation directories after archiving
            if remove_sims:
                path = OPJ(self.working_dir, sim)
                for dirpath, dirnames, filenames in os.walk(path,\
                                                                topdown=False):
                    shutil.rmtree(dirpath, ignore_errors=True)
            else:
                continue

        # after archiving all simulations delete the original JSON metadata
        for meta_file in self.metadata_json_paths[self.stage]:
            try:
                os.remove(meta_file)
            except:
                continue


