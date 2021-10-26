# load libraries
import numpy as np
import xarray as xr
import dask.distributed
from dask.distributed import Client

import warnings
warnings.filterwarnings('ignore')

# Read ensemble data for one year

# year = 2008; code = "az256"
# year = 2010; code = "bi795"
# year = 1960; code = "av640"
# year = 1966; code = "ax118"

ppdir="/home/users/hkhatri/DePreSys4_Data/Ensemble_Data/"

save_path="/home/users/hkhatri/DePreSys4_Data/Ensemble_Data/Data_Consolidated/" 

for year in range(2012, 2017, 2):

    ds = []

    print("Year Running - ", year)

    for i in range(0,10):
    
        d = xr.open_mfdataset(ppdir + str(year) + "/r" + str(i+1) + "/onm/*diaptr.nc")
        ds.append(d)
    
    ds = xr.concat(ds, dim='r')

    ds['time'] = ds['time_centered'].astype("datetime64[ns]") # convert cftime to convenient form

    save_file = save_path + str(year) + "_diaptr.nc"
    ds_save = ds.load()
    ds_save.to_netcdf(save_file)
