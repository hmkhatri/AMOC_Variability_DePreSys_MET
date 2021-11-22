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

save_path="/home/users/hkhatri/DePreSys4_Data/Data_Drift_Removal/Drift_2016_DCPP/"

# variable list to keep in the dataset
var_list = ['hfbasin_atlantic', 'hfbasinpmdiff_atlantic', 'hfovgyre_atlantic', 'hfovovrt_atlantic', 'sophtadv_atlantic', 
            'sltbasin_atlantic', 'sltbasinpmdiff_atlantic', 'sltovgyre_atlantic', 'sltovovrt_atlantic', 'sopstadv_atlantic',
            'zomsfatl', 'zosalatl','zosrfatl', 'zotematl']

ds = []

# Loop over year to combine indivual year files
for year in range(1960, 2017, 1):
    
    d = xr.open_dataset(ppdir + str(year) + "_diaptr.nc", chunks={'r':1})
    d = d.get(var_list)
    ds.append(d)
    
ds = xr.concat(ds, dim='start_year')

ds = ds.assign(start_year = np.arange(1960, 2017, 1))

ds = ds.chunk({'start_year': 1})

print("Data read complete")


## ------------- Doug's method (DCPP 2016 Appendix) ---------

# Consider winter seasons DJF in the time period 1970 - 2016. We compute average over these seasonal mean values while retaining the
# lead year information. For example, for 1st DJF - consider hindcasts 1970 - 2016, for 2nd DJF consider hindcasts 1969- 2015 etc.
# Compute the mean for all ensembles separately and substract this mean to obtain anomaly trend.

year1, year2 = (1979, 2017) # DJF (1979) to DJF (2016)  

def processDataset(ds1, year1, year2, lead_year):
    
    ds_save = []
    
    for year in range(year1, year2):
        
        # Extract relevant DJF months data and mean over the season
        ds1 = ds.sel(start_year = year - lead_year).isel(time_counter=slice(1 + 12*lead_year, 4 + 12*lead_year)).mean('time_counter')
        
        ds_save.append(ds1)
        
    return ds_save

for lead_year in range (0,11):
    
    print("Lead Year running = ", lead_year)

    ds_save = processDataset(ds, year1, year2, lead_year)
    
    ds_save = sum(ds_save) / (year2 - year1)

    ds_save = ds_save.compute()
    
    save_file = save_path +"Drift_diaptr_Lead_Year_" + str(int(lead_year+1)) + ".nc"
    ds_save.to_netcdf(save_file)

    print("File saved")    
    

# Old method required for linear trend computations

## --------- Parallel Computations with dask ------------ ##

"""
def processDataset(ds1):
    
    tmp = (ds1.sum('r').groupby('time_centered.year').mean('time_counter'))
    
    tmp = tmp.assign(year = np.arange(1, 11)) # to have same year values to compute mean
    
    return tmp

ds_save = []

for j in range(0,len(ds.start_year)):
   
    print("Year running = ", ds.start_year.values[j])
 
    tmp = delayed(processDataset)(ds.isel(start_year=j, time_counter=slice(2,122)))
    # slice in time_counter to remove months of incomplete years
    
    ds_save.append(tmp)
    
ds_save = delayed(sum)(ds_save)

ds_save = ds_save.compute()

ds_save = ds_save / (len(ds.start_year) * len(ds.r))
ds_save = ds_save.drop('start_year')

print("Computations Complete")

## -------- Save File ----------- ## 
save_file = save_path +"Drift_diaptr.nc"
ds_save.to_netcdf(save_file)

print("File saved")    
"""
