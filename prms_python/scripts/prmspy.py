import click
import io
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re

from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

from ..scenario import ScenarioSeries
from ..util import load_data, load_statvar, nash_sutcliffe


@click.group()
def prmspy():
    "access PRMS-Python functionality from the command line"
    click.echo('\n*** Welcome to PRMS-Python! ***\n')


@prmspy.command()
@click.argument('base_data_dir', nargs=1)
@click.option('--params', '-p', nargs=1, type=str, multiple=True,
              help='list of parameters to edit')
@click.option('--scale_vals', '-s', nargs=1, type=str, multiple=True,
              help='list of scaling values; must be divisible by len(params), '
              'e.g. "[0.8, 0.9, 1.0, 1.1, 1.2]"'
              )
@click.option('--output-dir', '-o', nargs=1, type=str,
              help='directory where scenario data should be written')
@click.option('--title', '-t', nargs=1, type=str,
              help='title for this parameter scaling experiment')
@click.option('--description', '-d', nargs=1, type=str,
              help='description for this parameter scaling experiment')
@click.option('--run-prms', is_flag=True,
              help='run PRMS after creating scenario input data')
@click.option('--prms-exec', '-e', default='prms',
              help='PRMS executable to be used, eg prmsV4')
@click.option('--nproc', '-n', default=None, type=int,
              help='number of processors to use')
@click.option('--analyze-output', is_flag=True,
              help='create analysis of model output in the form of a '
                   '"Nash-Sutcliffe Matrix"')
def param_scale_sim(base_data_dir,
                    params,
                    scale_vals,
                    output_dir,
                    title,
                    description,
                    run_prms,
                    prms_exec,
                    nproc,
                    analyze_output):
    'Provide params and scaling values; run PRMS scenarios'

    # scale_vals must be divisible by the number of params
    # valid_input = len(scale_vals) % len(params) == 0
    print(scale_vals)
    scale_vals = [eval(el) for el in scale_vals]

    # if not valid_input:
        # raise IOError('The length of scale_vals list is not divisible '
                      # 'by the length of the parameter list')

    if not output_dir:
        output_dir =\
            re.sub(r'\/|\\\\', '-', base_data_dir) + \
            '-modified-params-' + datetime.now().isoformat()

        os.mkdir(output_dir)

    s = ScenarioSeries(
        base_data_dir, output_dir, title=title, description=description
    )

    def _scale_fun(val):
        def scale(x):
            return x * val

        return scale

    # assign scaling values to parameter names
    # dimensionality correctness guaranteed from valid_input check
    pval_len = len(scale_vals)/len(params)

    def _slice(idx):
        return slice(pval_len*idx, pval_len*(idx+1))

    param_val_dict = {
        param: zip(itertools.repeat(param), scale_vals[idx])
        for idx, param in enumerate(params)
    }
    # param_val_dict = {
        # param: zip(itertools.repeat(param, pval_len), scale_vals[_slice(idx)])
        # for idx, param in enumerate(params)
    # }
    # for every combination we need to iterate over every value of
    # each param we have
    combinations = itertools.product(*param_val_dict.values())

    # now build the scenario_list with re-callable titles
    scenario_list = []
    for combo in combinations:

        scenario_def = {
            'title': _build_title(combo)
        }
        scenario_def.update(
            {
                param: _scale_fun(val)
                for param, val in combo
            }
        )

        scenario_list.append(scenario_def)

    s.build(scenario_list)

    if run_prms:
        s.run(prms_exec=prms_exec, nproc=nproc)

    if analyze_output:
        print('\n** Sorry, analyze_output has not yet been implemented! **\n')


@prmspy.command()
@click.argument('data_dir', nargs=1)
@click.argument('output_pdf_path', nargs=1)
def nash_sutcliffe_matrix(data_dir, output_pdf_path):
    'Save a PDF of the Nash-Sutcliffe created from <data_dir> to <output_pdf_path>'

    observed = load_data(
        os.path.join(data_dir, 'base_inputs', 'data')
    ).runoff_1

    series_metadata = json.loads(
        io.open(os.path.join(data_dir, 'series_metadata.json')).read()
    )
    modeled_flows = {

        title: load_statvar(
            os.path.join(data_dir, uu, 'outputs', 'statvar.dat')
        ).basin_cfs_1

        for uu, title in series_metadata['uuid_title_map'].items()
    }

    titles = list(modeled_flows.keys())

    params = [pair.split(':')[0] for pair in titles[0].split('|')]

    if len(params) > 2:
        print('This command is only supported for two covarying parameters!')
        exit(1)

    p1_vals = sorted({
        float(pair[0].split(':')[1])
        for pair in map(lambda t: t.split('|'), titles)
    })

    p2_vals = sorted({
        float(pair[1].split(':')[1])
        for pair in map(lambda t: t.split('|'), titles)
    })

    print(p1_vals)

    nash_sutcliffe_mat = np.zeros((len(p1_vals), len(p2_vals)))

    for p1_idx, p1_val in enumerate(p1_vals):
        for p2_idx, p2_val in enumerate(p2_vals):

            coord = (p1_idx, p2_idx)
            title = '{p1}:{p1val:.3f}|{p2}:{p2val:.3f}'.format(
                p1=params[0], p2=params[1], p1val=p1_val, p2val=p2_val
            )
            modeled = modeled_flows[title]

            nash_sutcliffe_mat[coord] = nash_sutcliffe(observed, modeled)

    with PdfPages(output_pdf_path) as pdf:

        fig, ax = plt.subplots()

        cax = ax.matshow(nash_sutcliffe_mat, cmap='viridis')

        ytix = p1_vals
        xtix = p2_vals
        plt.yticks(range(len(ytix)), ytix)
        plt.xticks(range(len(xtix)), xtix)

        ax.xaxis.set_ticks_position('bottom')

        plt.ylabel(params[0].replace('"', ''))
        plt.xlabel(params[1].replace('"', ''))

        for i, j in itertools.product(range(len(ytix)), range(len(xtix))):
            plt.text(j, i, '%.2f' % nash_sutcliffe_mat[i, j],
                     horizontalalignment='center',
                     color='w'
                     if nash_sutcliffe_mat[i, j] <
                     np.mean(nash_sutcliffe_mat.flatten())
                     else 'k')

        plt.title('Nash-Sutcliffe Matrix')

        plt.grid(b=False)
        fig.colorbar(cax)

        pdf.savefig()
        plt.close()


def _build_title(combo):
    '''
    Given a list of tuples with parameter/scale_val pairs, build the title
    for the parameterization. For example,
    >>> combo = [('snow_adj', 0.8), ('rad_trncf', 0.9), ('jh_coef', 1.1)]
    >>> assert _build_title(combo) ==\
            '"snow_adj":0.800|"rad_trncf":0.900|"jh_coef":1.100'
    '''

    return '|'.join('"{0}":{1:.3f}'.format(param, val) for param, val in combo)
