## --------------------------------------------------------------------
# This script for saving anomaly data for high and low NAO phase periods. 
# This is specifically for 3D vars.

# check Drift_thetao_r2_Lead_Year_1.nc -> probably corrupted
## --------------------------------------------------------------------

# load libraries
import numpy as np
import xarray as xr
import gc
import dask.distributed
import os, psutil, sys
from tornado import gen
import dask
from tornado import gen

from dask_mpi import initialize
initialize()

import warnings
warnings.filterwarnings('ignore')

from dask.distributed import Client, performance_report
client = Client()

# -------- functions define -----------
def select_subset(d1):
    
    d1 = d1.isel(i=slice(749,1199), j = slice(699, 1149))
    d1 = d1.drop(['vertices_latitude', 'vertices_longitude', 'time_bnds', 'lev_bnds'])
    
    return d1

async def stop(dask_scheduler):
    await gen.sleep(0.1)
    await dask_scheduler.close()
    loop = dask_scheduler.loop
    loop.add_callback(loop.stop)

# -------- Define paths and read data --------------

ppdir="/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/"

ppdir_NAO="/home/users/hkhatri/DePreSys4_Data/Data_Anomaly_Compute/NAO/"

ppdir_drift="/gws/nopw/j04/snapdragon/hkhatri/Data_Drift/"
    
save_path="/gws/nopw/j04/snapdragon/hkhatri/Data_Composite/NAO/"

year1, year2 = (1960, 2017)

var_list = ['thetao'] #['mlotst', 'tos', 'sos', 'hfds'] # ocean vars

# --------- NAO seasonal data -> identify high/low NAO periods -----------
ds_NAO = xr.open_dataset(ppdir_NAO + "NAO_SLP_Anomaly.nc")

NAO = ds_NAO['NAO']
tim = ds_NAO['time_val'].isel(start_year=0).drop('start_year')
NAO = NAO.assign_coords(time=tim)
NAO = NAO.chunk({'start_year':-1, 'r':1, 'time':1})

NAO = NAO.isel(time=slice(1,len(NAO.time)-1)) # get rid of first Nov and last Mar for seasonal avg
NAO_season = NAO.resample(time='QS-DEC').mean('time')

NAO_cut = 2.5 # based on plot for individual NAO values
tim_ind = 4 # could be any index (0, 4, 8, ... are for DJF)

ind_NAOp = xr.where(NAO_season.isel(time=tim_ind) >= NAO_cut, 1, 0)
ind_NAOn = xr.where(NAO_season.isel(time=tim_ind) <= -NAO_cut, 1, 0)

case = 'NAOp' 
#case = 'NAOn'

if (case == 'NAOp'):
    count_NAO = ind_NAOp
elif (case == 'NAOn'):
    count_NAO = ind_NAOn
else:
    print("Choose a valid case")
    
# ----------- Read data and save relevant values ----------- 

r_range = [0, 2, 3, 5, 6, 7, 8, 9] 

for var in var_list:
    
    print("Running var = ", var)

    # Read drift data
    ds_drift = []

    for r in r_range: #range (0,10):

        ds1 = []
        for lead_year in range(0,11):

            d = xr.open_dataset(ppdir_drift + var + "/Drift_"+ var + "_r" + 
                                str(r+1) +"_Lead_Year_" + str(lead_year+1) + ".nc",
                                chunks={'lev':1, 'time':1})
            d = d.assign(time = np.arange(lead_year*12, 12*lead_year + 
                                          np.minimum(12, len(d['time'])), 1))
            ds1.append(d)
                
        ds1 = xr.concat(ds1, dim='time')
        ds_drift.append(ds1)

    ds_drift = xr.concat(ds_drift, dim='r')
    ds_drift = ds_drift.assign(r = r_range)
    ds_drift = ds_drift.drop('time')
    #ds_drift = ds_drift.chunk({'time':12, 'j':50, 'i':50})

    print("Drift Data read complete")
    
    # Read full data to compute anomaly
    ds = []
    ds_d = []
    
    for r in r_range: #range(0,10):
        
        for year in range(year1, year2, 1):
            
            if(count_NAO.isel(r=r).sel(start_year=year) == 1):
                
                var_path = (ppdir + "s" + str(year) +"-r" + str(r+1) + 
                            "i1p1f2/Omon/" + var + "/gn/latest/*.nc")
                
                with xr.open_mfdataset(var_path, parallel=True, preprocess=select_subset, 
                                       chunks={'lev':1, 'time':1},
                                       decode_times=False, engine='netcdf4') as d:
                    d = d

                # compute anomaly
                d = d[var].drop('time') #- ds_drift[var].sel(r=r)
                
                d1 = ds_drift[var].sel(r=r)
                
                ds.append(d)
                ds_d.append(d1)
                
    ds = xr.concat(ds, dim='comp')
    ds = ds.assign_coords(time=tim)
    
    ds_d = xr.concat(ds_d, dim='comp')
    
    print("Composite Data read complete")
    
    for lev in range(0,5):
        
        # compute anomaly for selected levels
        ds1 = (ds.isel(lev = slice(lev*15, lev*15 + 15)) -
               ds_d.isel(lev = slice(lev*15, lev*15 + 15)))
    
        ds_season = ds1.mean('comp').compute()
    
        comp_save = xr.Dataset()
        comp_save[var] = ds_season.astype(np.float32)
        save_file = (save_path + "Composite_" + case + "_" + var + '_tim_ind_' 
                     + str(tim_ind) + "_lev_" + str(lev) + ".nc")
        comp_save.to_netcdf(save_file)
    
        print("Data saved successfully for levels:", lev*15, " - ",  lev*15 + 15)
    
print('Closing cluster')
client.run_on_scheduler(stop, wait=False)
