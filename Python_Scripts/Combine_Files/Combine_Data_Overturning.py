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

""" This is for available overturning diagnostics in DePreSys dataset 

ppdir="/home/users/hkhatri/DePreSys4_Data/Ensemble_Data/"

save_path="/home/users/hkhatri/DePreSys4_Data/Data_Consolidated/" 

for year in range(1961, 2016, 2):

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
    
"""

ppdir = "/gws/nopw/j04/snapdragon/hkhatri/Data_sigma/"
save_path = "/home/users/hkhatri/DePreSys4_Data/Data_Consolidated/Temp/" 

year1, year2 = (1960, 2017)

for r in range(9, 10, 1):

    ds1 = []
    
    for year in range(year1, year2):
        
        d1 = xr.open_dataset(ppdir + "Overturning_Heat_Transport/Overturning_Heat_Transport_" + 
                            str(year) + "_r" + str(r+1) + ".nc", chunks = {'time':1}, decode_times= False)
        
        d2 = xr.open_dataset(ppdir + "Overturning_Ekman/Overturning_Ekman_" + 
                            str(year) + "_r" + str(r+1) + ".nc", chunks = {'time':1}, decode_times= False)
        
        d = xr.merge([d1, d2])
        
        ds1.append(d.drop('time'))
        
    ds1 = xr.concat(ds1, dim='start_year')
    
    ds1 = ds1.assign(start_year = np.arange(year1, year2, 1))
    
    print('Data read complete for r = ', r+1)
    
    ds1 = ds1.astype(np.float32).compute()
    
    save_file = save_path + "Overturning_Heat_Transport_r" + str(r+1) + ".nc"
    
    ds1.to_netcdf(save_file)
    
    print('File saved successfully')
    
    