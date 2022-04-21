# load libraries

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import numpy as np
import xarray as xr
import dask.distributed
from dask.distributed import Client

import warnings
warnings.filterwarnings('ignore')

# Paths and assign variables to save

ppdir="/Volumes/Seagate_Hemant2/Work/Ocean/2021_Overturning_Subpolar_Atlantic/Data_DePreSys4/Ensemble_Data/"

save_path="/Users/hemantkhatri/OneDrive - The University of Liverpool/Work_Subpolar_Atlantic/Data_Consolidated/"

var_list = ['hfbasin_atlantic', 'hfbasinpmdiff_atlantic', 'hfovgyre_atlantic', 'hfovovrt_atlantic',
            'sophtadv_atlantic', 'sltbasin_atlantic', 'sltbasinpmdiff_atlantic', 'sltovgyre_atlantic',
            'sltovovrt_atlantic', 'sopstadv_atlantic', 'zomsfatl', 'zosalatl','zosrfatl','zotematl']

# Loop for reading and combining data

ds_final = []

for year in range(1961, 2017, 1):

    ds = []

    print("Year Running - ", year)

    # First combine data for each year using concat
    for i in range(0,10):

        d = xr.open_mfdataset(ppdir + str(year) + "/r" + str(i+1) + "/onm/*diaptr.nc", combine='nested', concat_dim='time_counter')
        d = d.get(var_list)
        ds.append(d)

    ds = xr.concat(ds, dim='r')

    save_file = save_path + str(year) + "_diaptr.nc"
    ds_save = ds.load()
    ds_save.to_netcdf(save_file)

    #ds_final.append(ds)

# Combine data for all year using concat

#ds_final = xr.concat(ds_final, dim='start_year')
#ds_final = ds_final.assign(start_year = np.arange(1960, 2017, 1))
#ds_final = ds_final.chunk({'start_year': 1})

#save_file = save_path + "_diaptr.nc"
#ds_final.to_netcdf(save_file)
