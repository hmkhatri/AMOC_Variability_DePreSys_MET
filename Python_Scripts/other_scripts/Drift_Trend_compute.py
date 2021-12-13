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

ppdir="/home/users/hkhatri/DePreSys4_Data/Data_Drift_Removal/"

var_list = ['psl', 'ua', 'va', 'sfcWind', 'tas', 'pr', 'evspsbl', 'tauu', 'tauv','clt',
           'hfds', 'mlotst', 'tos', 'sos', 'zos', 'diaptr']

for var in var_list:
    
    ds = xr.open_dataset(ppdir + 'Drift_' + var + '.nc')
    
    # use polyfit to compute slope and incercept 
    p1 = ds.polyfit(dim='year', deg=1)
    
    # use polyval to compute linear trend data    
    for var2 in list(ds.keys()):
        
        var2_name = var2 + '_trend'
        p1_name = var2 + '_polyfit_coefficients'
        
        p1[var2_name] = xr.polyval(ds.year, p1[p1_name])

    # Save trends in fields
    save_file = ppdir + "Linear_Trend_" + var + ".nc"
    p1.to_netcdf(save_file)

    print("File saved for var = ", var)
        
        