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
import numpy as np
import pandas as pd
import os

class Data(object):
    """
    Object that reads the PRMS data file and loads it into a date-time indexed
    DataFrame for data management, analysis and visualization. The ``modify`` 
    method allow for inplace modification of one or more time series inputs in the 
    data file based on a user defined function. The ``write`` method reformats the 
    Dataframe to PRMS text format and writes a new data file to disk.  Here is an
    example of loading a data file, modifying the temperature inputs ('tmin' and 
    'tmax') by adding two degrees to each element, and rewritting the modified 
    data to disk: 

    >>> d = Data('example_data_file')
    >>> def f(x):
            return x + 2
    >>> d.modify(f,['tmax','tmin'])
    >>> d.write('example_modified_data_file')

    d is a Data instance of the example data file, calling 

    >>> d.DataFrame 

    shows the datetime indexed DataFrame of the input data that is created when a 
    ``Data`` object is initiated. We then pass the function f(x) to d.modify 
    along with a Python list of input variable/s that we want to modify, this 
    modifies d.DataFrame in place. Printing the metadata attribute of the data
    object,

    >>> d.metadata

    will show the names of the variables in the data file in case you forget which
    you would like to modify. Last we call d.write with an output path to write 
    the modified data variables to disk in PRMS text format. 
    """

    ## data file constant attributes
    date_header = ['year',
               'month',
               'day',
               'hh',
               'mm',
               'sec']

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

    na_rep = -999

    def __init__(self, base_file):

        self.base_file = base_file
        self.metadata = self.__load_metadata()
        self.data_frame = self.__load_data()

    def __load_metadata(self):

        ## starting list for variable names in data file
        input_data_names = []

        ## open data file and read header information
        with open(self.base_file, 'r') as inf:
            for idx,l in enumerate(inf):
                if idx == 0: ## first line always string identifier of the file- may use later
                    data_head = l.rstrip()
                elif l.startswith('/'): ## comment lines
                    continue
                if l.startswith(Data.valid_input_variables): 
                    h = l.split() ## split line into list, first element name and second number of columns
                    if int(h[1]) > 1: ## more than one input time series of a particular variable
                        for el in range(int(h[1])):
                            tmp = '{var_name} {var_ind}'.format(var_name = h[0], var_ind = el+1)
                            input_data_names.append(tmp)
                    elif int(h[1]) == 1:
                        input_data_names.append(h[0])
                if l.startswith('#'): ## end of header info and begin time series input data
                    data_startline = idx+1 ## 0 indexed line of first data entry
                    break

        return dict([('data_startline',data_startline), ('data_variables',input_data_names)])

    def __load_data(self):

        df = pd.read_csv(self.base_file, header = -1, skiprows = self.metadata['data_startline'],
                         delim_whitespace = True, na_values = [Data.na_rep]) ## read data file
        df.columns = Data.date_header + self.metadata['data_variables']
        date = pd.Series(pd.to_datetime(df.year * 10000 + df.month * 100 +\
			 df.day, format = '%Y%m%d'), index = df.index)
        df.index = pd.to_datetime(date) ## assign datetime index
        df.drop(Data.date_header, axis = 1, inplace = True) ## unneeded columns
        df.columns.name = 'input variables' ; df.index.name = 'date'
        return df

    def modify(self, func, vars_to_adjust):
        for v in vars_to_adjust:
            self.data_frame[v] = self.data_frame[v].apply(func)

    def write(self, out_path):

        ## reconstruct PRMS data file format, don't overwrite date-indexed
	df = self.data_frame[self.metadata['data_variables']]
        df['year'] = self.data_frame.index.year
        df['month'] = self.data_frame.index.month
        df['day'] = self.data_frame.index.day
        df['hh'] = df['mm'] = df['sec'] = 0
        df = df[Data.date_header + self.metadata['data_variables']]
        with open(out_path,'w') as outf:
            with open(self.base_file) as data:
                for idx, line in enumerate(data):
                    if idx == self.metadata['data_startline']:
                        df.to_csv(outf, sep=' ', header=None,\
						index=False, na_rep=Data.na_rep)
                        break
                    outf.write(line) # write line by line the header lines from base file


