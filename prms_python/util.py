"""
Utilities for working with PRMS data or other functionality that aren't
appropriate to put elsewhere at this time.
"""

import numpy as np
import pandas as pd
import os

def delete_out_files(work_directory, file_name=''):
    """
    Delete all output files of a certain name from PRMS simulations,
    can be useful since files can be large and may not be being used.
    For example initial condition output files are often large and not
    always used, alternatively animation files may no longer be needed.

    Arguments:
        work_directory (str): path to directory with simulation outputs
            two directories above where the actual prms_ic.out files exist.
        file_name (str) = Name of the PRMS output file(s) to be removed, 
            default='' empty string- nothing will be deleted.             

            e.g. if you have several simulation directories:

			"test/results/intcp:-26.50_slope:0.49", 
			"test/results/intcp:-11.68_slope:0.54", 
			"test/results/intcp:-4.70_slope:0.51", 
			"test/results/intcp:-35.39_slope:0.39", 
			"test/results/intcp:-20.91_slope:0.41"

            each of these contains an '/outputs' folder with a prms_ic.out 
            file that you would like to delete. In this case, delete all ic 
            files like so:

            >>> work_dir = 'test/results/'
            >>> delete_ic_files(work_dir, file_name='prms_ic.out')
		    
    Returns:
        None     
    """
    for fd in os.listdir(work_directory):
        if os.path.isdir(os.path.join(work_directory,fd)):
            try:
                os.remove(os.path.join(work_directory, fd, 'outputs', file_name))
            except: # file might not exist
                continue


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
    """
    Read the data file and load into a datetime indexed Pandas dataframe object
    Arguments: 
	data_file (string): data file path 
    Returns:
	df (pandas.DataFrame): Pandas dataframe of input time series data from data file with datetime index
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



