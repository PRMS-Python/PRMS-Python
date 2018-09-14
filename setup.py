from prms_python import __version__
from setuptools import setup

requires = [
    'click == 6.6',
    'numpy >= 1.11.1',
    'pandas >= 0.18.1',
    'matplotlib >= 1.5.1'
]

tests_require = []

classifiers = [
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Environment :: Console',
    'Development Status :: 4 - Beta',
    'Topic :: Scientific/Engineering',
    'Intended Audience :: Science/Research'
]

setup(
    name='prms-python',
    description='A Python package with tools for the PRMS hydrologic model.',
    long_description='''
PRMS-Python provides a Python interface to PRMS data files and manages PRMS simulations. This module aims to improve the efficiency of PRMS simulations while providing "pythonic" tools to do scenario-based PRMS simulations. By "scenario-based" we mean testing model hypotheses associated with model inputs, outputs, and model structure. For example, parameter sensitivity analysis, where each "scenario" is an iterative perturbation of one or many parameters. Another example "scenario-based" modeling exercise would be climate scenario modeling: what will happen to modeled outputs if the input meteorological data were to change?
    ''',
    author='John Volk and Matthew Turner',
    author_email='jmvolk@unr.edu',
    license='BSD3',
    version=__version__,
    url='https://github.com/JohnVolk/PRMS-Python',
    platforms=['Windows','Linux','Mac OS X'],
    classifiers=classifiers,
    packages=['prms_python', 'prms_python.scripts', 'test'],
    install_requires=requires,
    tests_require=tests_require,
    package_data={'prms_python': ['models/lbcd/*']},
    include_package_data=True,
    entry_points='''
        [console_scripts]
        prmspy=prms_python.scripts.prmspy:prmspy
    '''
)
