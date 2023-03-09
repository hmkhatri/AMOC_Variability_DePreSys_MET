"""
Author: Hemant Khatri
Email: hkhatri@liverpool.ac.uk

This script can be used for computing composite mean of anomalies in ocean heat content, convergence and surface flux for ensemble members that have positive or negative anomalies in MHT (or overturning). 
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

### ------ Functions for computations ----------

def select_subset(dataset):
    
    """Select subset of dataset in xr.open_mfdataset command
    """
    dataset = dataset.isel(i=slice(749,1199), j = slice(699, 1149)) # indices range
    dataset = dataset.drop(['vertices_latitude', 'vertices_longitude', 
                            'time_bnds']) # drop variables 
    
    return dataset

### ------ Main calculations ------------------

data_dir1 = "/gws/nopw/j04/snapdragon/hkhatri/DePreSys4/Data_Heat_Budget/"
data_dir2 = "/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/"

data_drift_dir = "/gws/nopw/j04/snapdragon/hkhatri/DePreSys4/Data_Drift/"

ppdir_NAO="/home/users/hkhatri/DePreSys4_Data/Data_Anomaly_Compute/NAO/"

save_path="/gws/nopw/j04/snapdragon/hkhatri/DePreSys4/Data_Composite/Overturning_MHT/"

year1, year2 = (1960, 2017)

# --- read nao data to get time-values ----
ds_NAO = xr.open_dataset(ppdir_NAO + "NAO_SLP_Anomaly_new.nc")
tim = ds_NAO['time_val'].isel(start_year=0).drop('start_year')
tim = tim.astype("datetime64[ns]")

ds_cut = xr.open_dataset(save_path + "MHT_Overturning_456yr.nc")

#ensemble = 'Psi' # choose ensemble-selection criteria
ensemble = 'MHT'

if (ensemble == 'MHT'):
    case_list = ['MHTp', 'MHTn']
    MHT_std_cut = 1. * ds_cut['MHT_mean'].std(['start_year', 'r'])  # outside 1-sigma / 2-sigma based on mean MHT 4-6 years of simulations
    MHT_mean_cut = ds_cut['MHT_mean'].mean(['start_year', 'r'])

elif (ensemble == 'Psi'):
    case_list = ['Psip', 'Psin']
    Psi_std_cut = 1. * ds_cut['Psimax_mean'].std(['start_year', 'r'])  # outside 1-sigma / 2-sigma based on mean MHT 4-6 years of simulations
    Psi_mean_cut = ds_cut['Psimax_mean'].mean(['start_year', 'r'])

else:
    print("Use a valid ensemble criteria")

# -- read drift model data --- 

ds_drift = [] 

for r in range(0,10):
    
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

        for year in range(year1, year2, 1):

            if(count_case.isel(r=r).sel(start_year=year) == 1):

                var_path = data_dir2 + "s" + str(year) +"-r" + str(r+1) + "i1p1f2/Omon/hfds/gn/latest/*.nc"
                
                with xr.open_mfdataset(var_path, preprocess=select_subset, chunks={'time':1}, engine='netcdf4') as ds_hfds:
                
                    ds_hfds = ds_hfds

                ds_data = xr.open_dataset(data_dir1 + "Heat_Budget_"+ str(year) + "_r" + str(r+1) + ".nc", chunks={'time':1})
            
                ds_data = xr.merge([ds_data, ds_hfds['hfds']]) 

                ds1 = ds_data.drop('time') - ds_drift.isel(r=r)
                
                ds.append(ds1)

    ds = xr.concat(ds, dim='comp')
    
    print("Composite Data read complete")
    print("Total cases = ", len(ds['comp']), " - case ", case)

    # --- Data save ---------

    comp_save = ds.astype(np.float32)

    ds_save = xr.Dataset()

    ds_save['Heat_Divergence_200'] = (comp_save['Heat_Divergence_Horizontal_200'] +
                                      comp_save['Heat_Divergence_Vertical_200'])
    ds_save['Heat_Divergence_200'].attrs['units'] = "Watt/m^2"
    ds_save['Heat_Divergence_200'].attrs['long_name'] = "Div(u.theta) integrated between 0 and 210.18 m"
    
    ds_save['Heat_Content_200'] = comp_save['Heat_Content_200']
    ds_save['Heat_Content_200'].attrs['units'] = "Joules/m^2"
    ds_save['Heat_Content_200'].attrs['long_name'] = "Heat Content integrated between 0 and 210.18 m"
    
    ds_save['Heat_Divergence'] = (comp_save['Heat_Divergence_Horizontal_200'] +
                                  comp_save['Heat_Divergence_Horizontal_1300'] +
                                  comp_save['Heat_Divergence_Vertical_200'] +
                                  comp_save['Heat_Divergence_Vertical_1300'])
    
    ds_save['Heat_Divergence'].attrs['units'] = "Watt/m^2"
    ds_save['Heat_Divergence'].attrs['long_name'] = "Div(u.theta) integrated between 0 and 1325.67 m"
    
    ds_save['Heat_Content_1300'] = comp_save['Heat_Content_200'] + comp_save['Heat_Content_1300']
    
    ds_save['Heat_Content_1300'].attrs['units'] = "Joules/m^2"
    ds_save['Heat_Content_1300'].attrs['long_name'] = "Heat Content integrated between 0 and 1325.67 m"
    
    ds_save['Heat_Content'] = comp_save['Heat_Content']
    ds_save['Heat_Content'].attrs['units'] = "Joules/m^2"
    ds_save['Heat_Content'].attrs['long_name'] = "Heat Content integrated over full depth"

    for i in range(0, len(ds_save['comp'])):

        com_save = ds_save.isel(comp=i)
        com_save = com_save.astype(np.float32).compute()
        save_file = (save_path + ensemble + "/Composite_Heat_Budget/Composite_" 
                     + case + "_Heat_Budget_comp_" + str(i+1) + ".nc")
        com_save.to_netcdf(save_file)

    print("Data saved successfully")




