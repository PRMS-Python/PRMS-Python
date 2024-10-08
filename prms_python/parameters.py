# -*- coding: utf-8 -*-
'''
parameters.py -- holds ``Parameter`` class with multiple functionality for 
the standard PRMS parameters input file.
'''

import datetime, calendar
import io, os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import OrderedDict

OPJ = os.path.join

class Parameters(object):
    '''
    Disk-based representation of a PRMS parameter file. 
    
    For the sake of memory efficiency, we only load parameters from 
    ``base_file`` that get modified through item assignment or accessed directly. 
    Internally, a reference is kept to only previously accessed parameter data, 
    so when ``write`` is called most copying is from ``base_file`` directly. 
    When parameters are accessed or modified using the dictionary-like syntax, 
    a ``np.ndarray`` representation of the parameter is returned. As a result 
    ``numpy`` mathematical rules including efficient vectorization of math applied 
    to arrays can be applied to modify parameters directly. The ``Parameter`` 
    objects user methods allow for visualization of most PRMS parameters, function 
    based modification of parameters, and a write function that writes the data 
    back to PRMS text format.  

    Arguments:
        base_file (str): path to PRMS parameters file 

    Attributes:
        base_file (str): path to PRMS parameters file 
        base_file_reader (file): file handle of PRMS parameters file
        dimensions (:obj:`collections.OrderedDict`): dictionary with 
            parameter dimensions as defined in parameters file loaded on
            initialization
        base_params (list of dicts): list of dictionaries of parameter
            metadata loaded on initialization e.g. name, dimension(s), data 
            type, length of data array, and lines where data starts and ends 
            in file
        param_arrays (dict): dictionary with parameteter names as keys and
            ``numpy.array`` and ``numpy.ndarray`` representations of parameter
            values as keys. Initially empty, uses getter and setter functions.

    Example:
        >>> p = Parameters('path/to/a/parameter/file')
        >>> p['jh_coef'] = p['jh_coef']*1.1
        >>> p.write('example_modified_params')
    
        will read parameter information from the params file to check that
        *jh_coef* is present in the parameter file, read the lines corresponding
        to *jh_coef* data and assign the new value as requested. Calling
        the ``write`` method next will copy all parameters except *jh_coef*
        to the new parameter file and append the newly modified *jh_coef*
        to the end of the new file from the modified values stored in the
        parameter instance ``p``. 
    '''

    def __init__(self, base_file):
        self.base_file = base_file
        self.base_file_reader = open(base_file)
        self.dimensions, self.base_params = self.__read_base(base_file)
        self.param_arrays = dict()

    def write(self, out_name):
        """
        Writes current state of ``Parameters`` to disk in PRMS text format

        To reduce memory usage the ``write`` method copies parameters
        from the initial ``base_file`` parameter file for all parameters
        that were never modified. 

        Arguments:
            out_name (str): path to write ``Parameters`` data to PRMS text
                format.

        Returns:
            None
        """
        data_type_dic = {'1': 'int', 
                         '2': 'float'} # retain PRMS data types

        with open(self.base_file, 'r') as base_file:
            with open(out_name, 'w') as out_file:
                # write metadata
                out_file.write('File Auto-generated by PRMS-Python\n')
                out_file.write(datetime.datetime.now().isoformat() + '\n')

                # # write dimensions
                out_file.write('** Dimensions **\n')

                # write parameters; pre-sorted by data start line on read
                name_is_next = False
                params_start = False
                write_params_lines = False
                for l in base_file:

                    if not params_start and l.strip() == '** Parameters **':
                        out_file.write('** Parameters **\n')
                        params_start = True

                    elif l.strip() == '####':
                        name_is_next = True

                    elif name_is_next:
                        name = l.strip().split()[0] 
                        if name not in self.param_arrays:
                            out_file.write('####\n')
                            out_file.write(name + '\n')
                            name_is_next = False
                            write_params_lines = True
                        else:
                            write_params_lines = False
                            name_is_next = False

                    elif write_params_lines:
                        out_file.write(l.strip() + '\n')

                # write all parameters that had been accessed and/or modified
                for param, new_arr in self.param_arrays.items():

                    out_file.write('####\n')

                    param_info = [el for el in self.base_params
                                  if el['name'] == param].pop()

                    out_file.write(str(param_info['name']) + '\n')
                    out_file.write(str(param_info['ndims']) + '\n')
                    for dimname in param_info['dimnames']:
                        out_file.write(dimname + '\n')
                    out_file.write(str(param_info['length']) + '\n')
                    out_file.write(str(param_info['vartype']) + '\n')
                    out_file.writelines([str(a) + '\n'
                                         for a in new_arr.flatten().\
                                        astype(data_type_dic[param_info\
                                        ['vartype']])])

    def plot(self, nrows, which='all', out_dir=None, xlabel=None,\
                    ylabel=None, cbar_label=None, title=None, mpl_style=None,
                    na_val=-99999):
        """
        Versatile method that plots most parameters in a standard PRMS parameter
        file assuming the PRMS model was built on a uniform spatial grid. 
        
        Plots parameters as line plots for series or 2D spatial grid depending on 
        parameter dimension. The PRMS parameter file is assumed to hold parameters 
        for a model that was set up on a uniform rectangular grid with the spatial 
        index of HRUs starting in the upper left corner and moving left to 
        right across columns and down rows. Default function is to print four
        files, each with plots of varying parameter dimensions as explained 
        under Kwargs ``which`` and more detailed explanation in the example
        `Jupyter notebook <https://github.com/PRMS-Python/PRMS-Python/blob/master/notebooks/param_examples.ipynb>`_.
        
        Arguments:
            nrows (int): The number of rows in the PRMS model grid for plotting 
                spatial parameters. Will only work correctly for rectangular gridded models 
                with HRU indices starting in the upper left cell moving left to right 
                across columns and down across rows.
        
        Keyword Arguments:
            which (str): name of PRMS parameter to plot or 'all'. If 'all' then
                the function will print 3 multipage pdfs, one for nhru 
                dimensional parameters, one for nhru by monthly parameters, one 
                for other parameters of length > 1, and one html file containing
                single valued parameters.
            out_dir (str): path to an output dir, default current directory
            xlabel (str): x label for plot(s)
            ylabel (str): y label for plot(s)
            cbar_label (str): label for colorbar on spatial plot(s)
            title (str): plot title
            mpl_style (str, list): name or list of names of matplotlib style sheets to 
                use for plot(s).
            na_val (float, int): default -99999. Value to mask in plots.
    
        Returns: 
            None

        Examples:
            If the plot method is called with the keyword argument ``which`` set
            to a parameter that has length one, i.e. single valued it will simply
            print out the value e.g.:

            >>> p = Parameters('path/to/parameters')
            >>> p.plot(nrows=10, which='radj_sppt')
                radj_sppt is single valued with value: 0.4924942352224324

            The default action is particularly useful which makes four multi-page
            pdfs of most PRMS parameters where each file contains parameters
            of different dimensions e.g.:

            >>> p.plot(nrows=10, which='all', mpl_style='ggplot')

            will produce the following four files named by parameters of certain
            dimensions:

            >>> import os
            >>> os.listdir(os.getcwd()) # list files in current directory
                nhru_param_maps.pdf    
                nhru_by_nmonths_param_maps.pdf
                non_spatial_param_plots.pdf  
                single_valued_params.html
    
        """
        params = self
    
        if not isinstance(params, Parameters):
            raise TypeError('params must be instance of Parameters, not '\
                           + str(type(params)))
        if not out_dir:
            out_dir = os.getcwd()
        
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
            
        nhru = params.dimensions['nhru']
        ncols = nhru // nrows
        
        if not mpl_style:
            mpl_style = 'classic'
        plt.style.use(mpl_style)
    
        # make pdfs and html of all parameters seperated in 4 files based on dimension
        if which == 'all':
            ## spatial parameters with dimension of length nhru
            p_names = [param['name'] for param in params.base_params if\
                       param['length'] == nhru and len(param['dimnames'])==1]
            with PdfPages(OPJ(out_dir,'nhru_param_maps.pdf')) as pdf:
                for p in p_names:
                    try:
                        plt.figure()
                        ax = plt.gca()

                        d = np.where(params[p] == na_val, np.nan, params[p])
                        im = ax.imshow(d.reshape(nrows,ncols), origin='upper')
                        # origin upper- assumes indices of parameters starts in upper left
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        plt.colorbar(im, cax=cax)
                        ax.set_title('{}'.format(p))
                        ax.tick_params(left='off', bottom='off', labelleft='off',labelbottom='off')
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
                            
                            d = params['{}'.format(p)][i].reshape(nrows, ncols)
                            d = np.where(d == na_val, np.nan, d)
                            im = ax.imshow(d, origin='upper')
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
                        plt.plot(np.arange(1, param_dict['length']+1, 1), params[p])
                        plt.xlabel(param_dict['dimnames'][0])
                        plt.ylabel(p)
                        plt.xlim(0.5, param_dict['length']+0.5)
                        pdf.savefig()
                        plt.close()
                    except:
                        print('{param} parameter failed to plot'.format(param=p))
    
            ## html table of parameters with dimension length = 1 
            p_names = [param['name'] for param in params.base_params if param['length'] == 1]
            df = pd.DataFrame()
            df.index.name = 'parameter'
            for p in p_names:
                df.at[p, 'value'] = params[p]
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
                plt.plot(np.arange(1, param_dict['length']+1,1), params[param_name])
                plt.xlim(0.5, param_dict['length']+0.5)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(title)
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

    def __read_base(self, base_file):
        "Read base file returning 2-tuple of dimension and params dict"

        params_startline, dimensions = self.__make_dimensions_dict(base_file)
        base_params = self.__make_parameter_dict(base_file, params_startline)

        return (dimensions, base_params)

    def __make_dimensions_dict(self, base_file):
        """
        Extract dimensions and each dimension length. Runs before
        __make_parameter_dict.
        """
        ret = OrderedDict()

        dim_name = ''
        dim_len = 0
        # finished = False
        found_dim_start = False
        # while not finished:
        for idx, l in enumerate(self.base_file_reader):

            if l.strip() == '** Dimensions **':  # start of dimensions
                found_dim_start = True

            elif '#' in l:  # comments
                pass

            elif l.strip() == '** Parameters **':  # start of parameters
                dimlines = idx
                # finished = True
                break

            elif found_dim_start:

                if dim_name == '':
                    dim_name = l.strip()
                else:
                    dim_len = int(l)
                    ret.update({dim_name: dim_len})
                    dim_name = ''

        return (dimlines, ret)

    def __make_parameter_dict(self, base_file, params_startline=0):
        ret = []

        name = ''
        ndims = 0
        dimnames = []
        length = 0
        vartype = ''

        dimnames_read = 0
        data_startline = 0

        for idx, l in enumerate(self.base_file_reader):

            if '#' in l:
                # we have a comment; the next lines will be new
                # parameter metadata. No data for the first time through, so
                # we don't want to append an metadata blob with empty values
                if name:
                    ret.append(
                        dict(
                            name=name,
                            ndims=ndims,
                            dimnames=dimnames,
                            length=length,
                            vartype=vartype,
                            data_startline=data_startline
                        )
                    )

                    name = ''
                    ndims = 0
                    dimnames = []
                    length = 0
                    vartype = ''
                    dimnames_read = 0

            elif not name:
                name = l.strip().split()[0] # in case old format with integer after name

            elif not ndims:
                ndims = int(l.strip())

            elif not (dimnames_read == ndims):
                dimnames.append(l.strip())
                dimnames_read += 1

            elif not length:
                length = int(l.strip())

            elif not vartype:
                vartype = l.strip()
                # advance one from current position and account for starting
                # to count from zero
                data_startline = params_startline + idx + 2

        # need to append one more time since iteration will have stopped after
        # last line
        ret.append(
            dict(
                name=name,
                ndims=ndims,
                dimnames=dimnames,
                length=length,
                vartype=vartype,
                data_startline=data_startline
            )
        )

        return ret

    def __getitem__(self, key):
        """
        Look up a parameter by its name.

        Raises:
            KeyError if parameter name is not valid
        """
        def load_parameter_array(param_metadata):

            startline = param_metadata['data_startline']
            endline = startline + param_metadata['length'] + 1

            param_slice = itertools.islice(
                    io.open(self.base_file, 'rb'), startline, endline
                )

            arr = np.genfromtxt(param_slice)

            if param_metadata['ndims'] > 1:
                dimsizes = [
                    self.dimensions[d] for d in param_metadata['dimnames']
                ]
                dimsizes.reverse()
                arr = arr.reshape(dimsizes)

            return arr

        if key in self.param_arrays:
            return self.param_arrays[key]

        else:
            try:
                param_metadata = [
                    el for el in self.base_params if el['name'] == key
                ].pop()

            except IndexError:
                raise KeyError(key)

            arr = load_parameter_array(param_metadata)

            # cache the value for future access (but maybe shouldn't?)
            self.param_arrays.update({key: arr})

            return arr

    def __setitem__(self, key, value):

        if key in self.param_arrays:
            cur_arr = self.param_arrays[key]
            if not value.shape == cur_arr.shape:
                raise ValueError('New array does not match existing')

            self.param_arrays[key] = value

def modify_params(params_in, params_out, param_mods=None):
    '''
    Given a parameter file in and a dictionary of param_mods, write modified
    parameters to params_out.


    Arguments:
        params_in (str): location on disk of the base parameter file
        params_out (str): location on disk where the modified parameters will 
            be written

    Keyword Arguments:
        param_mods (dict): param name-keyed, param modification function-valued

    Returns:
        None

    Example:
        Below we modify the monthly *jh_coef* parameter by increasing it 10% 
        for every month,
    
            >>> params_in = 'models/lbcd/parameters'
            >>> params_out = 'scenarios/jh_coef_1.1/params'
            >>> scale_10pct = lambda x: x * 1.1
            >>> modify_params(params_in, params_out, {'jh_coef': scale_10pct})
    
        So param_mods is a dictionary of with keys being parameter names and
        values a function that operates on a single value. Currently we only
        accept functions that operate without reference to any other 
        parameters. The function will be applied to every cell, month, or 
        cascade routing rule for which the parameter is defined.
    '''
    p_in = Parameters(params_in)

    for k in param_mods:
        p_in[k] = param_mods[k](p_in[k])

    p_in.write(params_out)
