# -*- coding: utf-8 -*-
"""
util.py -- Utilities for working with PRMS data or other functionality that aren't
appropriate to put elsewhere at this time.
"""
import warnings
import os, shutil, json
import numpy as np
import pandas as pd

def calc_emp_CDF(data):
    # changed function name for PEP 8 style
    warnings.warn("calc_emp_CDF is deprecated, please use "+\
                  "util.calc_emp_cdf instead", DeprecationWarning)
    return calc_emp_cdf(data)

def calc_emp_cdf(data):
    """
    Create empirical CDF of arbitrary data
    
    Arguments:
        data (array_like) : array to calculate CDF on

    Returns:
        X (numpy.ndarray) : array of x values of CDF (sorted data)
        
        F (numpy.ndarray) : array of CDF values for each X value or cumulative 
            exceedence probability, in [0,1].
    """
    n_bins = len(data)
    X = np.sort(data)
    F = np.array(range(n_bins))/float(n_bins)
    return X,F

def Kolmogorov_Smirnov(uncond, cond, n_bins=10000):
    # changed function name for PEP 8 style
    warnings.warn("Kolmogorov_Smirnov is deprecated, please use "+\
                  "util.komogorov_smirnov instead", DeprecationWarning)
    return kolmogorov_smirnov(uncond, cond, n_bins=10000)

def kolmogorov_smirnov(uncond, cond, n_bins=10000):
    """ 
    Calculate the Kolmogorov-Smirnov statistic between two datasets by first 
    computing their empirical CDFs
    
    Arguments:
        uncond (array_like) : data for creating the unconditional CDF.
        cond (array_like) : data for creating the conditional CDF
        n_bins (int) : number of bins for both CDFs, note if n_bins > length
            of either dataset then CDF values are interpolated by numpy
        
    Returns: 
        KS (float) : Kolmogorov-Smirnov statistic, i.e. absolute max distance
            between uncond and cond CDFs
    """
    # create unconditional CDF (F_Uc)
    H,X = np.histogram(uncond, bins=n_bins, normed=True)
    dx = X[1] - X[0]
    F_Uc = np.cumsum(H)*dx    
    # create conditional CDF (F_C)
    H,X = np.histogram(cond, bins=n_bins, normed=True)
    dx = X[1] - X[0]
    F_C = np.cumsum(H)*dx
    # Calc max absolulte divergence
    KS = np.max(np.abs(F_Uc - F_C)) 
    return KS

def remove_all_optimization_sims_of_other_stage(work_directory, stage):
    """
    Track number of simulation directories not tracked by a specific stage
    and recursively delete them and their contents. This was created to avoid
    having nutracked simulations in an optimizer working directory for example
    when an optimization method was interupted before data was saved to a meta
    data file.
    
    Arguments:
        work_directory (str) : Directory to look for Optimization metadata 
            json files and simulation directories to keep or remove.         
        stage (str) : Optimization stage that will not have its simulation
            data deleted. All other stages if any are found in metadata files
            will have their associated simulation directories deleted.        
    Returns:
        None
    """
    from .optimizer import OptimizationResult # avoid circular import

    try:
        result = OptimizationResult(work_directory,stage=stage)    
        tracked_dirs = []    
        for f in result.metadata_json_paths[stage]:
            with open(f) as fh:
                json_data = json.load(fh)
                tracked_dirs.extend(json_data.get('sim_dirs'))
        count = 0
        for d in os.listdir(result.working_dir):
            path = os.path.join(result.working_dir, d)
            if path in tracked_dirs:
                continue      
            elif os.path.isdir(path) and '_archived' not in path:
                count+=1
                for dirpath, dirnames, filenames in os.walk(path,\
                                                                topdown=False):
                    shutil.rmtree(dirpath, ignore_errors=True)                    

    # if no json file in working dir for given stage, delete any other sim dirs 
    except:
        count = 0
        for d in os.listdir(work_directory):
            path = os.path.join(work_directory, d)
            if os.path.isdir(path) and '_archived' not in path:
                count+=1
                for dirpath, dirnames, filenames in os.walk(path,\
                                                                topdown=False):
                    shutil.rmtree(dirpath, ignore_errors=True)                    

    print('deleted {} simulations that were either not tracked by a JSON file'\
          .format(count) + ' or were not part of {} optimization stage'\
          .format(stage))

  
def delete_files(work_directory, file_name=''):
    """
    Recursively delete all files of a certain name from multiple PRMS 
    simulations that are within a given directory. Can be useful to removw 
    large files that are no longer needed. For example initial condition 
    output files are often large and not always used, similarly animation, 
    data, control, ... files may no longer be needed. 

    Arguments:
        work_directory (str) : path to directory with simulations.
        file_name (str) : Name of the PRMS input or output file(s) to be 
            removed, default = '' empty string- nothing will be deleted.             

    Example: 
        e.g. if you have several simulation directories:

	>>> "test/results/intcp:-26.50_slope:0.49", 
	    "test/results/intcp:-11.68_slope:0.54", 
	    "test/results/intcp:-4.70_slope:0.51", 
	    "test/results/intcp:-35.39_slope:0.39", 
	    "test/results/intcp:-20.91_slope:0.41"

        each of these contains an '/inputs' folder with a duplicate data 
        file that you would like to delete. In this case, delete all 
        data files like so:

        >>> work_dir = 'test/results/'
        >>> delete_files(work_dir, file_name='data')
		    
    Returns:
        None     
    """
    for dirpath, dirnames, filenames in os.walk(work_directory, topdown=False):
        paths = (os.path.join(dirpath, filename) for filename in filenames\
                if filename == file_name)
        for path in paths:
            os.remove(path)        

def load_statvar(statvar_file):
    """
    Read the statvar file and load into a datetime indexed
    Pandas dataframe object

    Arguments:
        statvar_file (str): statvar file path
    Returns:
        (pandas.DataFrame) Pandas DataFrame of PRMS variables date indexed
            from statvar file
    """
    # make list of statistical output variables for df header
    column_list = ['index',
                   'year',
                   'month',
                   'day',
                   'hh',
                   'mm',
                   'sec']

    # append to header list the variables present in the file
    with open(statvar_file, 'r') as inf:
        for idx, l in enumerate(inf):
            # first line is always number of stat variables
            if idx == 0:
                n_statvars = int(l)
            elif idx <= n_statvars and idx != 0:
                column_list.append(l.rstrip().replace(' ', '_'))
            else:
                break

    # arguments for read_csv function
    missing_value = -999
    skiprows = n_statvars+1
    df = pd.read_csv(
        statvar_file, delim_whitespace=True, skiprows=skiprows,
        header=-1, na_values=[missing_value]
    )

    # apply correct header names using metadata retrieved from file
    df.columns = column_list
    date = pd.Series(
        pd.to_datetime(df.year*10000+df.month*100+df.day, format='%Y%m%d'),
        index=df.index
    )

    # make the df index the datetime for the time series data
    df.index = pd.to_datetime(date)

    # drop unneeded columns
    df.drop(['index', 'year', 'month', 'day', 'hh', 'mm', 'sec'],
            axis=1, inplace=True)

    # name dataframe axes (index,columns)
    df.columns.name = 'statistical_variables'
    df.index.name = 'date'

    return df


def load_data_file(data_file):
    # changed function name for PEP 8 style
    warnings.warn("load_data_file is deprecated, please use "+\
                  "util.load_data instead", DeprecationWarning)
    return load_data(data_file)

def load_data(data_file):
    """
    Read the data file and load into a datetime indexed Pandas dataframe object.
    
    Arguments: 
	    data_file (str): data file path 
    Returns:
	    df (pandas.DataFrame): Pandas dataframe of input time series data 
	        from data file with datetime index
    """
    # valid input time series that can be put into a data file
    valid_input_variables = ('gate_ht',
                             'humidity',
                             'lake_elev',
                             'pan_evap',
                             'precip',
                             'rain_day',
                             'runoff',
                             'snowdepth',
                             'solrad',
                             'tmax',
                             'tmin',
                             'wind_speed')
    # starting list of names for header in dataframe
    column_list = ['year',
                   'month',
                   'day',
                   'hh',
                   'mm',
                   'sec']
    # append to header list the variables present in the file
    with open(data_file, 'r') as inf:
        for idx, l in enumerate(inf):

            # first line always string identifier of the file- may use later
            if idx == 0:
                data_head = l.rstrip()

            elif l.startswith('/'):  # comment lines
                continue

            # header lines with name and number of input variables
            if l.startswith(valid_input_variables):
                # split line into list, first element name and
                # second number of columns
                h = l.split()

                # more than one input time series of a particular variable
                if int(h[1]) > 1:
                    for el in range(int(h[1])):
                        tmp = '{var_name}_{var_ind}'.format(var_name=h[0],
                                                            var_ind=el+1)
                        column_list.append(tmp)
                elif int(h[1]) == 1:
                    column_list.append(h[0])
            # end of header info and begin time series input data
            if l.startswith('#'):
                skip_line = idx+1
                break

    # read data file into pandas dataframe object with correct header names
    missing_value = -999  # missing data representation
    df = pd.read_csv(data_file, header=-1, skiprows=skip_line,
                     delim_whitespace=True, na_values=[missing_value])

    # apply correct header names using metadata retrieved from file
    df.columns = column_list

    # create date column
    date = pd.Series(
        pd.to_datetime(df.year*10000+df.month*100+df.day, format='%Y%m%d'),
        index=df.index
    )

    df.index = pd.to_datetime(date)  # make the df index the datetime

    # drop unneeded columns
    df.drop(['year', 'month', 'day', 'hh', 'mm', 'sec'], axis=1, inplace=True)
    df.columns.name = 'input variables'
    df.index.name = 'date'  # name dataframe axes (index,columns)
    return df


def nash_sutcliffe(observed, modeled):
    """
    Calculates the Nash-Sutcliffe Goodness-of-fit

    Arguments:
        observed (numpy.ndarray): historic observational data

        modeled (numpy.ndarray): model output with matching time index
    """
    numerator = sum((observed - modeled)**2)
    denominator = sum((observed - np.mean(observed))**2)

    return 1 - (numerator/denominator)

def percent_bias(observed, modeled):
    """
    Calculates percent bias 
    
    Arguments:
        observed (numpy.ndarray): historic observational data

        modeled (numpy.ndarray): model output with matching time index
    """
    return 100 * ( sum( modeled - observed ) / sum( observed ) )


def rmse(observed, modeled):
    """
    Calculates root mean squared error
    
    Arguments:
        observed (numpy.ndarray): historic observational data

        modeled (numpy.ndarray): model output with matching time index
    """    
    return np.sqrt( sum((observed - modeled)**2) / len(observed) )



