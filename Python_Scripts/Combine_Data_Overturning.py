# load libraries
import numpy as np
import xarray as xr
import dask.distributed
from dask.distributed import Client

# Read ensemble data for one year

ppdir="/home/users/hkhatri/DePreSys4_Data/Ensemble_Data/2008-az256/"

save_path="/home/users/hkhatri/DePreSys4_Data/Ensemble_Data/Data_Consolidated/" 

ds = []

for i in range(0,10):
    
    d = xr.open_mfdataset(ppdir + "r" + str(i+1) + "/onm/*.nc", concat_dim='time_counter')
    ds.append(d)
    
ds = xr.concat(ds, dim='r')

ds['time'] = ds['time_centered'].astype("datetime64[ns]") # convert cftime to convenient form

save_file = save_path + "2008_diaptr.nc"
ds_save = ds.load()
ds_save.to_netcdf(save_file)
