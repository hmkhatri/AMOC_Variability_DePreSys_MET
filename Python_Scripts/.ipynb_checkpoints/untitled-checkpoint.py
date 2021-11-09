# load libraries
import numpy as np
import scipy as sc
import xarray as xr
import dask.distributed
from dask.distributed import Client
from dask import delayed
from dask import compute

import warnings
warnings.filterwarnings('ignore')

var_list = ['psl', 'ua', 'va', 'sfcWind', 'tas', 'pr', 'evspsbl', 'tauu', 'tauv','clt',
            'hfds', 'mlotst', 'tos', 'sos', 'zos']