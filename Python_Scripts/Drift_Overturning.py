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


### ------------- Read Data ------------

ppdir="/home/users/hkhatri/DePreSys4_Data/Data_Consolidated/"

save_path="/home/users/hkhatri/DePreSys4_Data/Data_Drift_Removal/"

# variable list to keep in the dataset
var_list = ['hfbasin_atlantic', 'hfbasinpmdiff_atlantic', 'hfovgyre_atlantic', 'hfovovrt_atlantic', 'sophtadv_atlantic', 
            'sltbasin_atlantic', 'sltbasinpmdiff_atlantic', 'sltovgyre_atlantic', 'sltovovrt_atlantic', 'sopstadv_atlantic',
            'zomsfatl', 'zosalatl','zosrfatl']

ds = []

# Loop over year to combine indivual year files
for year in range(1960, 2017):
    
    d = xr.open_dataset(ppdir + str(year) + "_diaptr.nc", chunks={'r':1})
    d = d.get(var_list)
    ds.append(d)
    
ds = xr.concat(ds, dim='start_year')

ds = ds.assign(start_year = np.arange(1960, 2017))

ds = ds.chunk({'start_year': 1})

## --------- Parallel Computations with dask ------------ ##

def processDataset(ds1):
    
    tmp = (ds1.sum('r').groupby('time_centered.year').mean('time_counter'))
    
    tmp = tmp.assign(year = np.arange(1, 11)) # to have same year values to compute mean
    
    return tmp

ds_save = []

for j in range(0,len(ds.start_year)):
    
    tmp = delayed(processDataset)(ds.isel(start_year=j, time_counter=slice(2,122)))
    # slice in time_counter to remove months of incomplete years
    
    ds_save.append(tmp)
    
ds_save = delayed(sum)(ds_save)

compute(ds_save)

ds_save = ds_save / (len(ds.start_year) * len(ds.r))
ds_save = ds_save.drop('start_year')

## -------- Save File ----------- ## 
save_file = save_path +"Drift_diaptr.nc"
ds_save.to_netcdf(save_file)
    