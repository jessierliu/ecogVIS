# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

# Get the long description from the README file
with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='ecogvis',
    version='1.0.1',
    description='Timeseries visualizer for Electrocorticography (ECoG) signals stored in NWB files.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Luiz Tauffer and Ben Dichter',
    email='ben.dichter@gmail.com',
    url='https://github.com/catalystneuro/ecogVIS',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['PyQt5', 'matplotlib', 'cycler', 'scipy', 'numpy',
                      'h5py', 'pyqtgraph', 'pandas', 'pynwb>=1.1.2', 'PyYAML',
                      'nwbext_ecog==0.7.2', 'ndx-spectrum', 'ndx-survey-data',
                      'tqdm', 'pyopengl', 'process_nwb', 'tdt'],
    entry_points={
        'console_scripts': ['ecogvis=ecogvis.ecogvis:cmd_line_shortcut'],
    }
)
