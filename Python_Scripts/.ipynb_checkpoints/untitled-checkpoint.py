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

ppdir="/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/"

save_path="/home/users/hkhatri/DePreSys4_Data/Data_Drift_Removal/"

var_list = ['hfds', 'mlotst', 'tos', 'sos', 'zos']

var = var_list[0]

# Loop over year to combine indivual year files

ds = []

for year in range(1960, 2017, 50):
    
    ds1 = []

    for i in range(0,2):
        
        var_path = "s" + str(year) +"-r" + str(i+1) + "i1p1f2/Omon/" + var + "/gn/files/d20200417/"
        
        d = xr.open_mfdataset(ppdir + var_path + "*.nc")
        d = d.drop(['vertices_latitude', 'vertices_longitude', 'time_bnds'])
        d = d.isel(i=slice(749,1199), j = slice(699, 1149))
        
        ds1.append(d)
        
    ds1 = xr.concat(ds1, dim='r')
    
    ds.append(ds1)
    
ds = xr.concat(ds, dim='start_year')

ds = ds.assign(start_year = np.arange(1960, 2017, 50))

ds = ds.chunk({'start_year': 1})

print("Data read complete")

## --------- Parallel Computations with dask ------------ ##

def processDataset(ds1):
    
    tmp = (ds1.sum('r').groupby('time.year').mean('time'))
    
    tmp = tmp.assign(year = np.arange(1, 11)) # to have same year values to compute mean
    
    return tmp

ds_save = []

for j in range(0,len(ds.start_year)):
   
    print("Year running = ", ds.start_year.values[j])
 
    tmp = delayed(processDataset)(ds.isel(start_year=j, time=slice(2,122)))
    # slice in time_counter to remove months of incomplete years
    
    ds_save.append(tmp)
    
ds_save = delayed(sum)(ds_save)

ds_save = ds_save.compute()

ds_save = ds_save / (len(ds.start_year) * len(ds.r))
ds_save = ds_save.drop('start_year')

print("Computations Complete")

## -------- Save File ----------- ## 
save_file = save_path +"Drift_" + var + ".nc"
ds_save.to_netcdf(save_file)

print("File saved")  


                  
            