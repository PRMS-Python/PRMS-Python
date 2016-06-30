"""
Description to come
"""
import numpy as np
import pandas as pd
import os

class Data(object):
    """
    PRMS data object to read, write and modify time
    series input to PRMS that are located in the PRMS data file
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

    def __init__(self, base_file):
        self.base_file = base_file
        self.metadata = self.__load_metadata()
        self.data_frame = self.__load_data()

    def __load_metadata(self):
        """
        """
        ## valid input time series that can be put into a data file

        #### starting list of names for header in dataframe
        input_data_names = []
        ## append to header list the variables present in the file
        with open(self.base_file, 'r') as inf:
            for idx,l in enumerate(inf):
                if idx == 0: ## first line always string identifier of the file- may use later
                    data_head = l.rstrip()
                elif l.startswith('/'): ## comment lines
                    continue
                if l.startswith(Data.valid_input_variables): ## header lines with name and number of input variables
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
        missing_value = -999 ## missing data representation
        df = pd.read_csv(self.base_file, header = -1, skiprows = self.metadata['data_startline'],
                         delim_whitespace = True, na_values = [missing_value]) ## read file
        df.columns = Data.date_header + self.metadata['data_variables']
        date = pd.Series(pd.to_datetime(df.year * 10000 + df.month * 100                                        + df.day, format = '%Y%m%d'),                                         index = df.index)
        df.index = pd.to_datetime(date)
        df.drop(Data.date_header, axis = 1, inplace = True) ## unneeded columns
        df.columns.name = 'input variables' ; df.index.name = 'date'
        return df

    def adjust(self, func, vars_to_adjust):
        for v in vars_to_adjust:
            self.data_frame[v] = self.data_frame[v].apply(func)

    def write(self, out_path):
        ## reconstruct original datafile format
        self.data_frame['year'] = self.data_frame.index.year
        self.data_frame['month'] = self.data_frame.index.month
        self.data_frame['day'] = self.data_frame.index.day
        self.data_frame['hh'] = self.data_frame['mm'] = self.data_frame['sec'] = 0
        self.data_frame = self.data_frame[Data.date_header + self.metadata['data_variables']]
        with open(out_path,'w') as outf:
            with open(self.base_file) as data:
                for idx, line in enumerate(data):
                    if idx == self.metadata['data_startline']:
                        self.data_frame.to_csv(outf, sep=' ', header=None, index=False, na_rep=-999)
                        break
                    outf.write(line) # write line by line the header lines from original

