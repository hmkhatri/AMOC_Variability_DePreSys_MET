## ---------------------------- ##

# The script computes model drift for disngotics using the method described in https://doi.org/10.5194/gmd-9-3751-2016. Here, we compute the mean over years Nov, 1970 - Mar, 2017 for all hindcasts while mainting the information about the lead year in hindcasts. 
# For example, hindcasts started in Nov, 1970 and Nov, 1971 both have Nov-Dec 1971. However, these cannot be added together in the mean calculation because hindcasts have run for different amount of simulation time before reaching Nov, 1971. 
# To get correct model drift and model climatology, compute mean over months that are in the same year as well as in the same simulations month time.

## ----------------------------- ##

# load libraries
import numpy as np
import scipy as sc
import xarray as xr
import dask.distributed
from dask.distributed import Client
from dask import delayed
from dask import compute
from dask import persist

import warnings
warnings.filterwarnings('ignore')

### ------ Function to sum over selected years ----------

def processDataset(ds1, year1, year2, lead_year):
    
    ds_save = []
    
    for year in range(year1, year2):
        
        # Extract relevant DJF months data and mean over the season
        ds2 = ds1.sel(start_year = year - lead_year).isel(time=slice(12*lead_year, np.minimum(12 + 12*lead_year, len(ds1['time']))))    
        ds_save.append(ds2)
        
    return ds_save

### ------------- Main computations ------------

ppdir="/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/"

save_path="/home/users/hkhatri/DePreSys4_Data/Data_Drift_Removal/Drift_1970_2016_Method_DCPP/"

var_list = ['hfds', 'tos', 'sos', 'mlotst'] #, 'zos'] # ocean vars

#var_list = ['psl', 'ua', 'va', 'sfcWind', 'tas', 'pr', 'evspsbl', 'tauu', 'tauv','clt'] # atmosphere vars

year1, year2 = (1970, 2017) # range over to compute average using DCPP 2016 paper -> 1970 - 2016
            

for var in var_list:
    
    ds = []
    
    for year in range(year1-10, year2, 1):
        
        ds1 = []
        
        for r in range(0,10):
            
            # Read data for each hindcast for every ensemble member
            
            var_path = "s" + str(year) +"-r" + str(r+1) + "i1p1f2/Omon/" + var + "/gn/files/d20200417/" # for ocean vars
            #var_path = "s" + str(year) +"-r" + str(r+1) + "i1p1f2/Amon/" + var + "/gn/files/d20200417/" # for atmosphere vars
            
            d = xr.open_mfdataset(ppdir + var_path + "*.nc")
            
            if ( var == 'ua' or var == 'va'):
                d = d.sel(plev = 85000.)
    
            ds1.append(d)
            
        ds1 = xr.concat(ds1, dim='r')
        ds1['time_val'] = ds1['time']
        
        # drop time coordinate as different time values create an issue in concat operation
        
        # for ocean vars only
        ds1 = ds1.drop(['vertices_latitude', 'vertices_longitude', 'time_bnds', 'time'])
        ds1 = ds1.isel(i=slice(749,1199), j = slice(699, 1149)) 
        
        # for atmosphere vars only
        #ds1 = ds1.drop(['lon_bnds', 'lat_bnds', 'time_bnds', 'time'])
        
        ds.append(ds1)
        
    ds = xr.concat(ds, dim='start_year')
    ds = ds.assign(start_year = np.arange(year1-10, year2, 1))
    ds = ds.chunk({'time':1, 'r':1, 'j':50, 'i':50}) #  for ocean vars
    
    print("Data read complete var = ", var)
    
    # loop over lead year and compute mean values
    for lead_year in range (0,11):
    
        print("Lead Year running = ", lead_year)

        ds_save = processDataset(ds, year1, year2, lead_year)
    
        ds_save = sum(ds_save) / (year2 - year1)

        ds_save = ds_save.compute()
    
        save_file = save_path +"Drift_" + var + "_Lead_Year_" + str(int(lead_year+1)) + ".nc"
        ds_save.to_netcdf(save_file)

