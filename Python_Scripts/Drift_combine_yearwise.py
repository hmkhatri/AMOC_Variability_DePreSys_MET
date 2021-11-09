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

var_list = ['psl', 'ua', 'va', 'sfcWind', 'tas', 'pr', 'evspsbl', 'tauu', 'tauv','clt', 'hfds', 'mlotst', 'tos', 'sos', 'zos']

ppdir = "/home/users/hkhatri/DePreSys4_Data/Data_Drift_Removal/Temp/"
save_path = "/home/users/hkhatri/DePreSys4_Data/Data_Drift_Removal/"

for var in var_list:
    
    ds = []
    
    for year in range(1960, 2017, 1):
        
        d = xr.open_dataset(ppdir + "Drift_" + var + "_" + str(year) + ".nc")
        ds.append(d)
        
    ds = xr.concat(ds, dim='start_year')

    ds = ds.assign(start_year = np.arange(1960, 2017, 1))

    ds = ds.chunk({'start_year': 1})

    print("Data read complete")
    
    ds_save = ds.sum('start_year') 
    ds_save = ds_save / (len(ds.start_year))
    
    ## -------- Save File ----------- ## 
    save_file = save_path +"Drift_" + var + ".nc"
    ds_save.to_netcdf(save_file)

    print("File saved for var = ", var) 
    
    ds.close()
    ds_save.close()
                         
                        
