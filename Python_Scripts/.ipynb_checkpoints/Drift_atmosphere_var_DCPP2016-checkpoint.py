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
        #ds2 = ds1.sel(start_year = year - lead_year).isel(time=slice(1 + 12*lead_year, 4 + 12*lead_year)).mean('time')    

        # Extract data relavant start year and sum over all hindcasts
        ds2 = ds1.sel(start_year = year - lead_year).isel(time=slice(12*lead_year, np.minimum(12 + 12*lead_year, len(ds1['time']))))
        ds_save.append(ds2)

    return ds_save

### ------------- Main computations ------------

ppdir="/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/"

# For DJF season only
#save_path="/home/users/hkhatri/DePreSys4_Data/Data_Drift_Removal/Drift_2016_DCPP/"

# For monthly drift
save_path="/home/users/hkhatri/DePreSys4_Data/Data_Drift_Removal/Drift_1970_2016_Method_DCPP/"

#var_list = ['va', 'ua'] 
var_list = ['psl', 'sfcWind', 'tas', 'pr', 'evspsbl', 'tauu', 'tauv','clt']

year1, year2 = (1979, 2017) # range over to compute average using DCPP 2016 paper

for var in var_list:
    
    for r in range(0,10):
       
        print("Var = ", var, "; Ensemble = ", r)

        ds = []

        for year in range(year1-10, year2, 1):
            
            # Read data for each hindcast for every ensemble member
            
            var_path = "s" + str(year) +"-r" + str(r+1) + "i1p1f2/Amon/" + var + "/gn/files/d20200417/"
            
            d = xr.open_mfdataset(ppdir + var_path + "*.nc")
            d = d.drop('time') # drop time coordinate as different time values create an issue in concat operation
            
            if ( var == 'ua' or var == 'va'):
                d = d.sel(plev = 85000.)
            
            ds.append(d)
            
        # combine data for hindcasts
        ds = xr.concat(ds, dim='start_year')
        ds = ds.drop(['lon_bnds', 'lat_bnds', 'time_bnds'])
        
        ds = ds.assign(start_year = np.arange(year1-10, year2, 1))
        ds = ds.chunk({'start_year':1, 'time':1})
        
        print("Data read complete")
        
        # loop over lead year and compute mean values
        for lead_year in range (0,11):
    
            #print("Lead Year running = ", lead_year)

            ds_save = processDataset(ds, year1, year2, lead_year)
    
            ds_save = sum(ds_save) / (year2 - year1)

            ds_save = ds_save.compute()
    
            save_file = save_path +"Drift_" + var + "_r" + str(r+1)+ "_Lead_Year_" + str(int(lead_year+1)) + ".nc"
            ds_save.to_netcdf(save_file)

            print("File saved, Lead Year = ", lead_year + 1)
