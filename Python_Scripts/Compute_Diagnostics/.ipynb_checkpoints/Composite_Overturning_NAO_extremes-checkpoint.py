"""
Author: Hemant Khatri
Email: hkhatri@liverpool.ac.uk

This script can be used for computing composite mean of anomalies in overturning circulation and meridional heat transport for ensemble members that has very strong or weak NAO conditions in a winter. 
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

### ------ Main calculations ------------------

data_dir =  "/gws/nopw/j04/snapdragon/hkhatri/Data_Consolidated/Overturning_Heat_Salt_Transport_Baro_Decompose/"

data_drift_dir = "/gws/nopw/j04/snapdragon/hkhatri/Data_Drift/"

ppdir_NAO="/home/users/hkhatri/DePreSys4_Data/Data_Anomaly_Compute/NAO/"

save_path="/gws/nopw/j04/snapdragon/hkhatri/Data_Composite/NAO_hpa/" 

year1, year2 = (1960, 2017)

# NAO seasonal data -> identify high/low NAO periods
ds_NAO = xr.open_dataset(ppdir_NAO + "NAO_SLP_Anomaly_new.nc")
#ds_NAO = ds_NAO.isel(r=slice(0,4)) # diagnostics for all ensembles are not complete yet

# NAO = ds_NAO['NAO'] # for normalised NAO indices
NAO = (ds_NAO['P_south'] - ds_NAO['P_north']) # for NAO indices in pa

tim = ds_NAO['time_val'].isel(start_year=0).drop('start_year')
NAO = NAO.assign_coords(time=tim)
#NAO = NAO.chunk({'start_year':-1, 'r':1, 'time':1})

NAO = NAO.isel(time=slice(1,len(NAO.time)-1)) # get rid of first Nov and last Mar for seasonal avg
NAO_season = NAO.resample(time='QS-DEC').mean('time')

# NAO_cut = 2.5 # based on plot for individual normalised NAO values
NAO_cut = 1300. # based on plot for individual NAO values in pa

case_list = ['NAOp', 'NAOn'] 
#case = 'NAOn'

# read drift model data

ds_drift = []

for r in range(0,10):
    
    ds1 = []
    
    for lead_year in range(0, 11):
        
        d = xr.open_dataset(data_drift_dir + "psi_sigma/Drift_Overturning_r" + str(r+1) + "_Lead_Year_" +
                            str(lead_year + 1) + ".nc", decode_times= False)
        
        d = d.assign(time = np.arange(lead_year*12, 12*lead_year + 
                                      np.minimum(12, len(d['time'])), 1))
        
        ds1.append(d)
        
    ds1 = xr.concat(ds1, dim='time')
    
    ds_drift.append(ds1.drop('time'))
    
ds_drift = xr.concat(ds_drift, dim='r')
    
ds_drift = ds_drift.chunk({'time':1})

print("Reading Drift Data Complete")

for case in case_list:
    
    ds = []

    for tim_ind in range(4,13,4):

        ind_NAOp = xr.where(NAO_season.isel(time=tim_ind) >= NAO_cut, 1, 0)
        ind_NAOn = xr.where(NAO_season.isel(time=tim_ind) <= -NAO_cut, 1, 0)

        if (case == 'NAOp'):
            count_NAO = ind_NAOp
        elif (case == 'NAOn'):
            count_NAO = ind_NAOn
        else:
            print("Choose a valid case")

        for r in range(0,10):

            d = xr.open_dataset(data_dir + "Overturning_Heat_Salt_Transport_r" + str(r+1) + ".nc", chunks={'start_year':1})

            for year in range(year1, year2, 1):

                if(count_NAO.isel(r=r).sel(start_year=year) == 1):

                    ds1 = (d.sel(start_year=year).drop(['start_year']) - 
                           ds_drift.isel(r=r).drop(['Overturning_max_z', 'Overturning_max_sigma']))

                    # compute anomaly in overturning maximum
                    ds1['Overturning_max_z'] = ((d['Overturning_z'].sel(start_year=year).drop(['start_year'])).max(dim='lev') - 
                                                (ds_drift['Overturning_max_z'].isel(r=r)))

                    ds1['Overturning_max_sigma'] = ((d['Overturning_sigma'].sel(start_year=year).drop(['start_year'])).max(dim='sigma0') -
                                                    (ds_drift['Overturning_max_sigma'].isel(r=r)))

                    # to get mean densities and layer thicknesses
                    ds1['Depth_sigma'] = d['Depth_sigma'].sel(start_year=year).drop(['start_year'])
                    ds1['Density_z'] = d['Density_z'].sel(start_year=year).drop(['start_year'])

                    ds.append(ds1.isel(time = slice((int(tim_ind/4)-1)*12, (int(tim_ind/4) + 7)*12 + 5)))

    ds = xr.concat(ds, dim='comp')

    ds = ds.drop('latitude')
    ds['latitude'] = ds_drift['latitude'].isel(r=0, time=0) # get actual latitude values for overturning

    print("Composite Data read complete")
    print("Total cases = ", len(ds['comp']), " - case ", case)

    comp_save = ds.astype(np.float32).compute()

    save_file = save_path + "Composite_" + case + "_Overturning_MHT.nc"
    comp_save.to_netcdf(save_file)
    print("Data saved successfully")
