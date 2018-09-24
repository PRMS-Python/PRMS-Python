# -*- coding: utf-8 -*-
'''
data.py -- holds ``Data`` class for standard PRMS climate input data.
'''

import pandas as pd
from shutil import copyfile

class Data(object):
    """
    Object to access or create a PRMS data file with ability to load/assign it to a 
    date-time indexed pandas.DataFrame for data management, analysis and visualization. 
    It can be used to build a new PRMS data file from user defined metadata and a 
    ``pandas.DataFrame`` of PRMS datetime-indexed climatic forcing and observation 
    variables.

    The class properties ``metadata`` and ``data_frame`` can be later assigned if no
    ``base_file`` is given on initialization, allowing for the creation of PRMS climatic
    forcing file in a Python environment.

    Keyword Arguments:
        base_file (str, optional): path to standard PRMS data file 
        na_rep (int, optional): how to represent missing values default = -999

    Attributes:
        date_header (list): date and time header for PRMS data file 
        valid_input_variables (tuple): valid hydro-climate variables for PRMS data file

    Note:
        If using the ``Data`` class to create a new data file, it is up to the user
        to ensure that the metadata and :class:`pandas.DataFrame` assigned are correct 
        and compatible.  

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

    def __init__(self, base_file=None, na_rep=-999):
        self.base_file = base_file
        self.na_rep = na_rep
        self._metadata = None 
        self._data_frame = None 

    @property
    def metadata(self):
        """
        :obj:`dict`:A property that gets and sets the header information from 
        a standard PRMS climate input data file held in a Python dictionary. As 
        a property it can be assigned directly to overwrite or create a new PRMS 
        data file. As such the user is in control and must supply the correct 
        syntax for PRMS standard data files, e.g. text lines before header should 
        begin with "//". Here is an example of the information gathered and held in 
        this attribute:

        Example:
            >>> data.metadata
                {
                 'data_startline' : 6,
                 'data_variables' : ['runoff 1', 'runoff 2', 'tmin', 'tmax', 'ppt']
                 'text_before_header' : "Title of data file \\n //some comments\\nrunoff 2
                                         \\ntmin 1\\ntmax 1\\nppt 1\\nrunoff 2\\ntmin 1
                                         \\ntmax 1\\nppt 1\\n
                                         ########################################\\n"
                } 
    
        Note:
            When assigning or creating a new data file, the ``Data.write`` method will
            assign the appropriate date header that follows the line of number signs "#".

        Raises:
            ValueError: if data in metadata is accessed before data is assigned, 
                e.g. if accessed to write a PRMS data file from a ``Data`` instance 
                that was initialized without a valid PRMS data file.
            TypeError: if an object that is not a Python dictionary is assigned.
        
        """
        # to avoid overwriting pre-assigned data, check if already exists
        if isinstance(self._metadata, dict):
            return self._metadata
        elif not self.base_file:
            raise ValueError('No data file was given on initialization')
            
        ## starting list for variable names in data file
        input_data_names = []
        text_before_header = str()
        ## open data file and read header information
        with open(self.base_file, 'r') as inf:
            for idx,l in enumerate(inf):
                text_before_header+=l
                if idx == 0: ## first line always string identifier of the file- may use later
                    data_head = l.rstrip()
                elif l.startswith('/'): ## comment lines
                    continue
                if l.startswith(Data.valid_input_variables): 
                    h = l.split() ## split, first name and second number of columns
                    if int(h[1]) > 1: ## more than one input time series of a particular variable
                        for el in range(int(h[1])):
                            tmp = '{var_name} {var_ind}'.format(var_name = h[0], var_ind = el+1)
                            input_data_names.append(tmp)
                    elif int(h[1]) == 1:
                        input_data_names.append(h[0])
                if l.startswith('#'): ## end of header info and begin time series input data
                    data_startline = idx+1 ## 0 indexed line of first data entry
                    break
                    
        self._metadata = dict([('data_startline',data_startline),
                               ('data_variables',input_data_names),
                               ('text_before_header',text_before_header)])
        return self._metadata
    
    @metadata.setter
    def metadata(self, dic):
        if not isinstance(dic, dict):
            raise TypeError('Must assign a Python dictionary for new Data object/file metadata')
        self._metadata = dic
    
    @property
    def data_frame(self):
        """
        A property that gets and sets the climatic forcing data for a standard PRMS 
        climate input data file as a :class:`pandas.DataFrame`.

        Example:
            d is a Data instance, calling 
    
            >>> d.data_frame 
                input variables	runoff 1 runoff 2 runoff 3 precip tmax tmin
                date						
                1996-12-27	0.54	1.6	NaN	0.0	46	32.0
                1996-12-28	0.65	1.6	NaN	0.0	45	24.0
                1996-12-29	0.80	1.6	NaN	0.0	44	28.0
                1996-12-30	0.90	1.6	NaN	0.0	51	33.0
                1996-12-31	1.00	1.7	NaN	0.0	47	32.0 
        
            shows the date-indexed ``pd.DataFrame`` of the input data that is created 
            when a ``Data`` object is initiated if given a valid ``base_file``, i.e. 
            file path to a PRMS climate data file. 

        Raises:
            ValueError: if attribute is accessed before either assigning a PRMS data
                file on ``Data`` initialization or not assigning a compatabile 
                date-indexed ``pandas.DataFrame`` of hydro-climate variables. 
            TypeError: if a data type other than ``pandas.DataFrame`` is assigned. 
        """
        if not self._metadata:
            self.metadata
        elif not isinstance(self._data_frame, pd.DataFrame) and self.base_file == None:
            raise ValueError('No data base_file given on initialization, '\
                             'therefore you must assign a DataFrame'\
                              +' before accessing the .data_frame attribute!')
        # to avoid overwriting pre-assigned data
        elif isinstance(self._data_frame, pd.DataFrame):
            return self._data_frame
        
        df = pd.read_csv(self.base_file, header = -1, skiprows = self.metadata['data_startline'],
                         delim_whitespace = True, na_values = [self.na_rep]) ## read data file
        df.columns = Data.date_header + self.metadata['data_variables']
        date = pd.Series(pd.to_datetime(df.year * 10000 + df.month * 100 +\
             df.day, format = '%Y%m%d'), index = df.index)
        df.index = pd.to_datetime(date) ## assign datetime index
        df.drop(Data.date_header, axis = 1, inplace = True) ## unneeded columns
        df.columns.name = 'input variables' ; df.index.name = 'date'
        self._data_frame = df
        return self._data_frame
    
    @data_frame.setter
    def data_frame(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Must assign a Pandas.DataFrame object for PRMS data input")
        self._data_frame = df
        
    def modify(self, func, vars_to_adjust):
        """
        Apply a user defined function to one or more variable(s) in the data file.

        The ``modify`` method allows for inplace modification of one or more 
        time series inputs in the data file based on a user defined function.  
        
        Arguments:
            func (function): function to apply to each variable in vars_to_adjust
            vars_to_adjust (list or tuple): collection of variable names to apply func to.

        Returns: 
            None

        Example:
            Here is an example of loading a data file, modifying the temperature inputs 
            (*tmin* and *tmax*) by adding two degrees to each element, and rewritting the 
            modified data to disk,

            >>> d = Data('path_to_data_file')
            >>> def f(x):
                    return x + 2
            >>> d.modify(f,['tmax','tmin'])
            >>> d.write('data_temp_plus_2')
        """

        if not isinstance(self._data_frame, pd.DataFrame):
            self.data_frame # will raise ValueError from data_frame property
        for v in vars_to_adjust:
            self._data_frame[v] = self._data_frame[v].apply(func)

    def write(self, out_path):
        """
        Writes the current state of the ``Data`` to PRMS text format
        utilizing the ``Data.metadata`` and ``Data.data_frame`` instance
        properties. If ``Data.data_frame`` was never accessed or assigned
        new values then this method simply copies the original PRMS
        data file to ``out_path``.

        Arguments:
            out_path (str): full path to save or copy the current PRMS data 
                in PRMS text format.

        Returns:
            None

        Raises:
            ValueError: if the ``write`` method is called without assigning either 
                an initial data (``base_file``) path or assigning correct ``metadata`` 
                and ``data_frame`` properties. 

        """
        # if file data was never accessed- unchanged
        if not isinstance(self._data_frame, pd.DataFrame): 
            if self.base_file:
                copyfile(self.base_file, out_path)
            else: # if data not from original file and dataframe never assigned
                raise ValueError('No data base_file was given and'\
                                +' no data was assigned!')
                
        ## reconstruct PRMS data file format, don't overwrite date-indexed
        else:
            df = self._data_frame[self.metadata['data_variables']]
            df['year'] = self._data_frame.index.year
            df['month'] = self._data_frame.index.month
            df['day'] = self._data_frame.index.day
            df['hh'] = df['mm'] = df['sec'] = 0
            df = df[Data.date_header + self._metadata['data_variables']]
            with open(out_path,'w') as outf: # write comment header then data
                outf.write(self._metadata['text_before_header'])
                df.to_csv(outf, sep=' ', header=None,\
                          index=False, na_rep=self.na_rep)

