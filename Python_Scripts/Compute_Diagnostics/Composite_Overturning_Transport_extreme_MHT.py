"""
Author: Hemant Khatri
Email: hkhatri@liverpool.ac.uk

This script can be used for computing composite mean of anomalies in overturning circulation and meridional heat transport for ensemble members that have positive or negative anomalies in MHT (or overturning). 
"""

# ------- load libraries ------------ 
import numpy as np
import xarray as xr
from xgcm import Grid
import gc
from tornado import gen
import os

import warnings
warnings.filterwarnings('ignore')

from dask_mpi import initialize
initialize()

from dask.distributed import Client
client = Client()

### ------ Main calculations ------------------

data_dir =  "/gws/nopw/j04/snapdragon/hkhatri/DePreSys4/Data_Consolidated/Overturning_Heat_Salt_Transport_Baro_Decompose/"

data_drift_dir = "/gws/nopw/j04/snapdragon/hkhatri/DePreSys4/Data_Drift/"

ppdir_NAO="/home/users/hkhatri/DePreSys4_Data/Data_Anomaly_Compute/NAO/"

save_path="/gws/nopw/j04/snapdragon/hkhatri/DePreSys4/Data_Composite/Overturning_MHT/"

year1, year2 = (1960, 2017)

# --- read nao data to get time-values ----
ds_NAO = xr.open_dataset(ppdir_NAO + "NAO_SLP_Anomaly_new.nc")
tim = ds_NAO['time_val'].isel(start_year=0).drop('start_year')
tim = tim.astype("datetime64[ns]")

ds_cut = xr.open_dataset(save_path + "MHT_Overturning_456yr.nc")

ensemble = 'Psi' # choose ensemble-selection criteria
#ensemble = 'MHT'

if (ensemble == 'MHT'):
    case_list = ['MHTp', 'MHTn']
    MHT_std_cut = 1. * ds_cut['MHT_mean'].std(['start_year', 'r'])  # outside 1-sigma / 2-sigma based on mean MHT 4-6 years of simulations
    MHT_mean_cut = ds_cut['MHT_mean'].mean(['start_year', 'r'])

elif (ensemble == 'Psi'):
    case_list = ['Psin'] #, 'Psin']
    Psi_std_cut = 1. * ds_cut['Psimax_mean'].std(['start_year', 'r'])  # outside 1-sigma / 2-sigma based on mean MHT 4-6 years of simulations
    Psi_mean_cut = ds_cut['Psimax_mean'].mean(['start_year', 'r'])

else:
    print("Use a valid ensemble criteria")

# ---- read drift model data ------

ds_drift = []

for r in range(0,10):
    
    ds1 = []
    
    for lead_year in range(0, 11):
        
        d = xr.open_dataset(data_drift_dir + "psi_sigma/Drift_Overturning_r" + str(r+1) + "_Lead_Year_" +
                            str(lead_year + 1) + ".nc", decode_times= False, chunks={'time':1})
        
        d = d.assign(time = np.arange(lead_year*12, 12*lead_year + 
                                      np.minimum(12, len(d['time'])), 1))
        
        ds1.append(d)
        
    ds1 = xr.concat(ds1, dim='time')
    
    ds_drift.append(ds1.drop('time'))
    
ds_drift = xr.concat(ds_drift, dim='r')

ds_drift = ds_drift.drop('j_c')
ds_drift = ds_drift.assign_coords(j_c=ds_drift['latitude'].isel(time=0, r=0))
ds_drift = ds_drift.chunk({'time':1})

print("Reading Drift Data Complete")

# ---- Create composites ------------
for case in case_list:
    
    ds = []

    if (ensemble == 'MHT'):
        ind_p = xr.where(ds_cut['MHT_mean'] > MHT_mean_cut + MHT_std_cut, 1, 0)
        ind_n = xr.where(ds_cut['MHT_mean'] < MHT_mean_cut - MHT_std_cut, 1, 0)

        if (case == 'MHTp'):
            count_case = ind_p
        elif (case == 'MHTn'):
            count_case = ind_n
        
    elif (ensemble == 'Psi'):
        ind_p = xr.where(ds_cut['Psimax_mean'] > Psi_mean_cut + Psi_std_cut, 1, 0)
        ind_n = xr.where(ds_cut['Psimax_mean'] < Psi_mean_cut - Psi_std_cut, 1, 0)

        if (case == 'Psip'):
            count_case = ind_p
        elif (case == 'Psin'):
            count_case = ind_n

    for r in range(0,10):
        
        d = xr.open_dataset(data_dir + "Overturning_Heat_Salt_Transport_r" + str(r+1) + ".nc", 
                            chunks={'start_year':1, 'time':1})

        for year in range(year1, year2, 1):

            if(count_case.isel(r=r).sel(start_year=year) == 1):

                ds1 = d.sel(start_year=year).drop(['start_year'])

                ds1 = ds1.assign_coords(j_c=ds1['latitude'])

                ds1['Overturning_max_sigma'] = (ds1['Overturning_sigma'] 
                                                - ds1['Overturning_sigma_barotropic']).max(dim='sigma0')
                ds1['Overturning_max_z'] = (ds1['Overturning_z'] 
                                            - ds1['Overturning_z_barotropic']).max(dim='lev')

                ds1 = (ds1 - ds_drift.isel(r=r))

                ds.append(ds1)

    ds = xr.concat(ds, dim='comp')
    ds = ds.drop('latitude')

    ds = ds.assign_coords(time=tim)

    print("Composite Data read complete")
    print("Total cases = ", len(ds['comp']), " - case ", case)

    for i in range(85, len(ds['comp'])):

        comp_save = ds.isel(comp=i)
        comp_save = comp_save.astype(np.float32).compute()
        save_file = (save_path + ensemble + "/Composite_Overturning_MHT/Composite_" 
                     + case + "_Overturning_MHT_comp_" + str(i+1) + ".nc")
        comp_save.to_netcdf(save_file)

    #comp_save = ds.astype(np.float32).compute()

    #save_file = save_path + "Composite_" + case + "_Overturning_MHT.nc"
    #comp_save.to_netcdf(save_file)
    print("Data saved successfully")



                
        
