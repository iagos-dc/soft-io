#!/usr/bin/env python

from setuptools import setup

setup(
    name='softio',
    version='2.0',
    description='Soft-IO',
    author='Pawel Wolff',
    author_email='pawel.wolff@aero.obs-mip.fr',
    packages=[
        'common',
        'dataviz', 
        'fpsim', 
        'softio', 
    ],
    install_requires=[
        'pandas',
        'xarray>=0.19',
        'numpy',
        'tqdm',
        'scipy',
        'hvplot',
        'dask',
        'toolz',
        'holoviews',
        'geoviews', 
        'cartopy', 
    ],
)
