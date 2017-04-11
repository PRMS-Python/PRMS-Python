"""
Utilities for working with PRMS data or other functionality that aren't
appropriate to put elsewhere at this time.
"""
import os
import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from parameters import Parameters

OPJ = os.path.join


def plot_params(params, nrows, which='all', out_dir=None, xlabel=None,\
                ylabel=None, cbar_label=None, title=None, mpl_style=None):
    """
    Plot PRMS parameters as time series or 2D spatial grid depending on 
    parameter dimension. The PRMS parameter file is assumed to represent 
    a model that was set up on a uniform rectangular grid with the spatial 
    index of HRUs starting in the upper left corner and moving left to 
    right across columns and down rows. 
    
    Args:
        params (prms_python.Parameters): An instance of Parameters that 
            corresponds with the PRMS parameter file to plot. 
        nrows (int): The number of rows in the PRMS model grid for plotting spatial
            parameters. Function will only work for rectangular gridded models
            with HRU indices starting in the upper left cell moving left to right
            across columns and down across rows.
    
    Kwargs:
        which (str): name of PRMS parameter to plot or 'all'. If 'all' then
            the function will print 3 multipage pdfs: one for nhru dimensional
            parameters, one for nhru by monthly parameters, one for other parameters
            of length > 1, and one html file containing single valued parameters
        out_dir (str): path to an output dir, default: current directory
        xlabel (str): x label for plot(s)
        ylabel (str): y label for plot(s)
        cbar_label (str): label for colorbar on spatial plot(s)
        title (str): plot title
        mpl_style (str, list): name or list of names of matplotlib style sheets
            to use for plot(s)

    Returns: None
    """
    
    if not isinstance(params, Parameters):
        raise TypeError('params must be instance of Parameters, not '\
                       + str(type(params)))
    if not out_dir:
        out_dir = os.getcwd()
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
        
    nhru = params.dimensions['nhru']
    ncols = nhru//nrows
    
    if not mpl_style:
        mpl_style = 'classic'
    plt.style.use(mpl_style)

    # make pdfs and html of all parameters seperated in 4 files based on dimension
    if which == 'all':
        ## spatial parameters with dimension of length nhru
        p_names = [param['name'] for param in params.base_params if\
                   param['length'] == nhru and len(param['dimnames'])==1 ]
        with PdfPages(OPJ(out_dir,'nhru_param_maps.pdf')) as pdf:
            for p in p_names:
                try:
                    plt.figure()
                    ax = plt.gca()
                    im = ax.imshow(params['{}'.format(p)].reshape(nrows,ncols), origin='upper')
                    # origin upper- assumes indices of parameters starts in upper left
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax)
                    ax.set_title('{}'.format(p))
                    ax.tick_params(left='off', bottom='off', labelleft='off', labelbottom='off')
                    pdf.savefig()
                    plt.close()
                except:
                    print('{param} parameter failed to plot'.format(param=p))
                    
        ## monthly spatial parameters (on plot per month)
        p_names = [param['name'] for param in params.base_params if\
                   param['dimnames'][0] == 'nhru' and len(param['dimnames'])==2 ]
        with PdfPages(OPJ(out_dir,'nhru_by_nmonths_param_maps.pdf')) as pdf:
            for p in p_names:
                try:
                    for i in range(12): #month
                        plt.figure()
                        ax = plt.gca()
                        im = ax.imshow(params['{}'.format(p)][i].reshape(nrows,ncols), origin='upper')
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        plt.colorbar(im, cax=cax)
                        ax.set_title('{} {}'.format(p, calendar.month_name[i+1]))
                        ax.tick_params(left='off', bottom='off', labelleft='off', labelbottom='off')
                        pdf.savefig()
                        plt.close()
                except:
                    print('{param} for {month} failed to plot'.\
                          format(param=p, month=calendar.month_name[i+1]))
                    
        ## non spatial parameters with dimension length > 1 to be plotted as time series
        p_names = [param['name'] for param in params.base_params if\
                   ( 1 < param['length'] <= 366 )\
                   and param['dimnames'][0] != 'nhru' ]
        with PdfPages(OPJ(out_dir,'non_spatial_param_plots.pdf')) as pdf:
            for p in p_names:
                try:
                    param_dict = [param for param in params.base_params if param['name'] == p][0]
                    plt.plot(np.arange(1,param_dict['length']+1,1), params[p])
                    plt.xlabel(param_dict['dimnames'][0])
                    plt.ylabel(p)
                    pdf.savefig()
                    plt.close()
                except:
                    print('{param} parameter failed to plot'.format(param=p))

        ## html table of parameters with dimension length = 1 
        p_names = [param['name'] for param in params.base_params if param['length'] == 1]
        df = pd.DataFrame()
        df.index.name = 'parameter'
        for p in p_names:
            df.set_value(p, 'value', params[p])
        df.to_html(OPJ(out_dir,'single_valued_params.html'))
    ################################################################                
    # plot single parameter, in case of nhru by monthly param, 
    # save multi-page pdf 
    else:
        param_name = which
        try:
            params[which]
        except:
            print('{param} is not a valid PRMS parameter'.format(param=param_name))
            return
        
        param_dict = [param for param in params.base_params if param['name'] == param_name][0]
        
        # labels for single plots
        if not cbar_label:
            cbar_label = param_name
        if not title:
            title = ''
                   
        # if parameter is not spatial, one dimensional, with length greater than one, plot as line
        if param_dict['ndims'] == 1 and ( 1 < param_dict['length'] <= 366 )\
                                    and param_dict['dimnames'][0] != 'nhru':
            if not xlabel:
                xlabel = param_dict['dimnames'][0]
            if not ylabel:
                ylabel = param_name
            plt.plot(np.arange(1,param_dict['length']+1,1), params[param_name], **plot_kwargs)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
        # if spatial and one dimensional, plot
        elif param_dict['ndims'] == 1 and param_dict['length'] == params.dimensions['nhru']:
            if not xlabel:
                xlabel = ''
            if not ylabel:
                ylabel = ''
            plt.figure()
            ax = plt.gca()
            im = ax.imshow(params[param_name].reshape(nrows,ncols), origin='upper')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax,label=cbar_label)
            ax.tick_params(left='off', bottom='off', labelleft='off', labelbottom='off')
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
        # spatial monthly parameter
        elif param_dict['dimnames'][0] == 'nhru' and param_dict['dimnames'][1] == 'nmonths'\
             and param_dict['ndims'] == 2: 
            if not xlabel:
                xlabel = ''
            if not ylabel:
                ylabel = ''
            file_name = '{}.pdf'.format(param_name)
            with PdfPages(OPJ(out_dir, file_name)) as pdf:
                for i in range(12): #month
                    plt.figure()
                    ax = plt.gca()
                    im = ax.imshow(params['{}'.format(param_name)][i].reshape(nrows,ncols), origin='upper')
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax)
                    ax.set_title('{} {}'.format(param_name, calendar.month_name[i+1]))
                    ax.set_ylabel(ylabel)
                    ax.set_xlabel(xlabel)
                    ax.tick_params(left='off', bottom='off', labelleft='off', labelbottom='off')
                    pdf.savefig()
                    plt.close()
        else:
            val = params[param_name]
            print('{param} is single valued with value: {v}'.format(param=param_name, v=val))          


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



