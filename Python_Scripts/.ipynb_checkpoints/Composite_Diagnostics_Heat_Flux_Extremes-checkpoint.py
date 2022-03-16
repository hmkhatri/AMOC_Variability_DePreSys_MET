## --------------------------------------------------------------------
# This script for saving anomaly data for high and low surface heat flux anomaly periods. 
# This is averaged over all members to save storage.
## --------------------------------------------------------------------

# load libraries
import numpy as np
import scipy as sc
import xarray as xr
from dask.distributed import Client, LocalCluster
from dask import delayed
from dask import compute
from dask.diagnostics import ProgressBar

import warnings
warnings.filterwarnings('ignore')

#from dask_mpi import initialize

#initialize()
#client = Client()

# -------- Define paths and read data --------------

ppdir="/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/"

ppdir_NAO="/home/users/hkhatri/DePreSys4_Data/Data_Anomaly_Compute/NAO/"

ppdir_drift="/home/users/hkhatri/DePreSys4_Data/Data_Drift_Removal/Drift_1970_2016_Method_DCPP/"
    
save_path="/home/users/hkhatri/DePreSys4_Data/Data_Composite/time_series/hfds_Irminger/"

year1, year2 = (1960, 2017)

var_list = ['mlotst', 'tos', 'hfds'] # sos] ocean vars
#var_list =  ['tauu', 'tauv'] # ['pr', 'evspsbl'] atmosphere vars

# --------- hfds seasonal data -> identify high/low hfds periods -----------
ds_NAO = xr.open_dataset(ppdir_NAO + "NAO_SLP_Anomaly.nc")

NAO = ds_NAO['NAO']
tim = ds_NAO['time_val'].isel(start_year=0).drop('start_year')
NAO = NAO.assign_coords(time=tim)
NAO = NAO.chunk({'start_year':-1, 'r':1, 'time':1})

NAO = NAO.isel(time=slice(1,len(NAO.time)-1)) # get rid of first Nov and last Mar for seasonal avg
NAO_season = NAO.resample(time='QS-DEC').mean('time')

ds_hfds = []
    
for i in range(0,10):
    
    d = xr.open_dataset(ppdir_NAO + "../Area_mean_timeseries/Anomaly_hfds_r" + str(i+1) + ".nc")
        
    ds_hfds.append(d)
    
ds_hfds = xr.concat(ds_hfds, dim='r')

ds_hfds = ds_hfds.assign_coords(time=tim)
ds_hfds = ds_hfds.assign(start_year = np.arange(year1, year2, 1))

# compute seasonal averages
HF_Irm = ds_hfds['Irminger_Sea'].isel(time=slice(1,len(ds_hfds.time)-1)) # get rid of first Nov and last Mar for seasonal avg
HF_Irm_season = HF_Irm.resample(time='QS-DEC').mean('time')

HF_Lab = ds_hfds['Labrador_Sea'].isel(time=slice(1,len(ds_hfds.time)-1)) # get rid of first Nov and last Mar for seasonal avg
HF_Lab_season = HF_Lab.resample(time='QS-DEC').mean('time')

HF_cut = 50. # based on plot for individual NAO values
tim_ind = 8 # could be any index (0, 4, 8, ... are for DJF)

ind_HFp = xr.where(HF_Irm_season.isel(time=tim_ind) >= HF_cut, 1, 0)
ind_HFn = xr.where(HF_Irm_season.isel(time=tim_ind) <= -HF_cut, 1, 0)

case = 'HFp' 
#case = 'HFn'

if (case == 'HFp'):
    count_HF = ind_HFp
    print('Total members = ', count_HF.sum().values)
elif (case == 'HFn'):
    count_HF = ind_HFn
    print('Total members = ', count_HF.sum().values)
else:
    print("Choose a valid case")
    
# ----------- Read data and save relevant values ----------- 

for var in var_list:
    
    print("Running var = ", var)

    # Read drift data
    ds_drift = []

    for r in range (0,10):

        ds1 = []
        for lead_year in range(0,11):

            d = xr.open_dataset(ppdir_drift + var + "/Drift_"+ var + "_r" + 
                                str(r+1) +"_Lead_Year_" + str(lead_year+1) + ".nc")
            d = d.assign(time = np.arange(lead_year*12, 12*lead_year + 
                                          np.minimum(12, len(d['time'])), 1))
            ds1.append(d)
                
        ds1 = xr.concat(ds1, dim='time')
        ds_drift.append(ds1)

    ds_drift = xr.concat(ds_drift, dim='r')
    ds_drift = ds_drift.drop('time')
    #ds_drift = ds_drift.chunk({'time':12, 'j':50, 'i':50})

    print("Drift Data read complete")
    
    # Read full data to compute anomaly
    ds = []
    
    for r in range(0,10):
        
        for year in range(year1, year2, 1):
            
            if(count_HF.isel(r=r).sel(start_year=year) == 1):
                
                var_path = "s" + str(year) +"-r" + str(r+1) + "i1p1f2/Omon/" + var + "/gn/latest/"
                #var_path = "s" + str(year) +"-r" + str(r+1) + "i1p1f2/Amon/" + var + "/gn/latest/"
                
                d = xr.open_mfdataset(ppdir + var_path + "*.nc")
                #d = d.chunk({'time':12, 'j':50, 'i':50})
                d = d[var].drop('time') - ds_drift[var].isel(r=r)
                
                ds.append(d)
                
    ds = xr.concat(ds, dim='comp')
    ds = ds.assign_coords(time=tim)
    
    print("Composite Data read complete")
    
    # Option two
    with ProgressBar():
        ds_season = ds.mean('comp').compute()
    
    comp_save = xr.Dataset()
    comp_save[var] = ds_season
    save_file = save_path + "Composite_" + case + "_" + var + '_tim_ind_' + str(tim_ind) + ".nc"
    comp_save.to_netcdf(save_file)
    
    print("Data saved successfully")

