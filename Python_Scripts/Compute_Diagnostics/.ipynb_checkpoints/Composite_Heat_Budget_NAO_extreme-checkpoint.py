"""
Author: Hemant Khatri
Email: hkhatri@liverpool.ac.uk

This script can be used for computing composite mean of anomalies in heat budget terms for ensemble members that has very strong or weak NAO conditions in a winter. These composites would help us evalute the role of surface forcing vs heat flux convergence in setting the heat content anomalies in the different ocean regions. 
"""

# ------- load libraries ------------ 
import numpy as np
import xarray as xr
from xgcm import Grid
import gc
from tornado import gen
import os
from dask.diagnostics import ProgressBar

import warnings
warnings.filterwarnings('ignore')

### ------ Functions for computations ----------

def select_subset(dataset):
    
    """Select subset of dataset in xr.open_mfdataset command
    """
    dataset = dataset.isel(i=slice(749,1199), j = slice(699, 1149)) # indices range
    dataset = dataset.drop(['vertices_latitude', 'vertices_longitude', 
                            'time_bnds']) # drop variables 
    
    return dataset

### ------ Main calculations ------------------

data_dir1 = "/gws/nopw/j04/snapdragon/hkhatri/Data_Heat_Budget/"
data_dir2 = "/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/"

data_drift_dir = "/gws/nopw/j04/snapdragon/hkhatri/Data_Drift/"

ppdir_NAO="/home/users/hkhatri/DePreSys4_Data/Data_Anomaly_Compute/NAO/"

save_path="/home/users/hkhatri/DePreSys4_Data/Data_Composite/time_series/NAO/Heat_Budget/"

year1, year2 = (1960, 2017)

# NAO seasonal data -> identify high/low NAO periods
ds_NAO = xr.open_dataset(ppdir_NAO + "NAO_SLP_Anomaly.nc")
ds_NAO = ds_NAO.isel(r=slice(0,4)) # diagnostics for all ensembles are not complete yet

NAO = ds_NAO['NAO']
tim = ds_NAO['time_val'].isel(start_year=0).drop('start_year')
NAO = NAO.assign_coords(time=tim)
NAO = NAO.chunk({'start_year':-1, 'r':1, 'time':1})

NAO = NAO.isel(time=slice(1,len(NAO.time)-1)) # get rid of first Nov and last Mar for seasonal avg
NAO_season = NAO.resample(time='QS-DEC').mean('time')

NAO_cut = 2.5 # based on plot for individual NAO values
tim_ind = 12 # could be any index (0, 4, 8, ... are for DJF)

ind_NAOp = xr.where(NAO_season.isel(time=tim_ind) >= NAO_cut, 1, 0)
ind_NAOn = xr.where(NAO_season.isel(time=tim_ind) <= -NAO_cut, 1, 0)

#case = 'NAOp' 
case = 'NAOn'

if (case == 'NAOp'):
    count_NAO = ind_NAOp
elif (case == 'NAOn'):
    count_NAO = ind_NAOn
else:
    print("Choose a valid case")


# read drift model data (for now considering only 4 ensembles as diagnostics for all runs are not yet complete)

ds_drift = []

print("Running Heat Budget composite for - ", case, "and time index - ", tim_ind) 

for r in range(0,4):
    
    ds1 = []
    
    for lead_year in range(0, 11):
        
        d1 = xr.open_dataset(data_drift_dir + "Heat_Budget/drift_budget_terms/Drift_Heat_Budget_r" + str(r+1) + 
                             "_Lead_Year_" + str(lead_year + 1) + ".nc", decode_times= False)
        
        d2 = xr.open_dataset(data_drift_dir + "hfds/Drift_hfds_r" + str(r+1) + "_Lead_Year_" +
                             str(lead_year + 1) + ".nc", decode_times= False)
        
        d = xr.merge([d1, d2])
        d = d.assign(time = np.arange(lead_year*12, 12*lead_year + 
                                      np.minimum(12, len(d['time'])), 1))
        
        ds1.append(d)
        
    ds1 = xr.concat(ds1, dim='time')
    
    ds_drift.append(ds1.drop('time'))
    
ds_drift = xr.concat(ds_drift, dim='r')
ds_drift = ds_drift.chunk({'time':1})

print("Reading Drift Data Complete")

# read full data (for now considering only 4 ensembles as diagnostics for all runs are not yet complete)

ds = []

count = 0

for r in range(0,4):
        
    for year in range(year1, year2, 1):
        
        if(count_NAO.isel(r=r).sel(start_year=year) == 1):
            
            count = count + 1
            
            var_path = data_dir2 + "s" + str(year) +"-r" + str(r+1) + "i1p1f2/Omon/hfds/gn/latest/*.nc"
            
            with xr.open_mfdataset(var_path, preprocess=select_subset, chunks={'time':1}, engine='netcdf4') as ds_hfds:
                
                ds_hfds = ds_hfds
                
            ds_data = xr.open_dataset(data_dir1 + "Heat_Budget_"+ str(year) + "_r" + str(r+1) + ".nc", chunks={'time':1})
            
            ds_data = xr.merge([ds_data, ds_hfds['hfds']]) 
            
            ds1 = ds_data.drop('time') - ds_drift.isel(r=r)
            
            ds.append(ds1)
            
ds = xr.concat(ds, dim='comp')
ds = ds.assign_coords(time=tim)
    
print("Composite Data read complete")
print("Total cases = ", count)

# compute omposite mean

with ProgressBar():
    comp_save = ds.mean('comp').compute()

save_file = save_path + "Composite_" + case + "_Heat_Budget_tim_ind_" + str(tim_ind) + ".nc"
comp_save.to_netcdf(save_file)
    
print("Data saved successfully")