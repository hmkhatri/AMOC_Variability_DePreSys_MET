## ------------------------------------ ##

# The script is for computing anomalies for ocean variables for all hindcasts. The code reads through the full dataset and then computes the seasonal means. The mean model drift pattern is then subtracted for all lead years. The area-weighted mean anomalies are computed for different regions as defined in Mask_Regions notebook and saved in separate nc files for further analysis. 

## ------------------------------------ ##

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

# Function to compute area-weighted mean
def Compute_area_weighted_mean(ds, area, mask, mask_val): 
    
    tmp = ds.where(mask == mask_val)
    dA = area.where(mask == mask_val)
    
    tmp1 = (tmp * dA).sum(['i', 'j']) / dA.sum(['i', 'j']) 
    
    return tmp1

# Paths and varnames

var_list = ['hfds', 'tos', 'sos'] #, 'mlotst', 'zos']

region_list = ['Labrador_Sea', 'Irminger_Sea', 'Iceland_Basin', 'North_East_Region', 'South_West_Region', 
               'South_East_Region', 'North_Atlantic'] # mask value 0-5 for these regions, see details in the main loop 

ppdir="/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/"

ppdir_drift="/home/users/hkhatri/DePreSys4_Data/Data_Drift_Removal/Drift_2016_DCPP/"

save_path="/home/users/hkhatri/DePreSys4_Data/Data_Anomaly_Compute/"

year1, year2 = (1960, 2017)

# Read grid mask files for averaging 
ds_mask = xr.open_dataset('/home/users/hkhatri/DePreSys4_Data/Data_Consolidated/Mask_Regions.nc')
ds_area = xr.open_dataset('/home/users/hkhatri/DePreSys4_Data/Data_Consolidated/Area.nc')
ds_mask['area'] = ds_area['area_t'].rename({'x':'i', 'y':'j'})

# Main Loop for computations

for var in var_list:
    
    # First read data for the mean model drift 
    ds_drift = []
    
    for lead_year in range(0, 11):

        ds1 =[]
        for r in range(1,11):

            d = xr.open_dataset(ppdir_drift + "Drift_" + var + "_r" + str(r) + "_Lead_Year_" + str(lead_year + 1) + ".nc", decode_times= False)
            ds1.append(d)

        ds1 = xr.concat(ds1, dim='ensemble')
        ds_drift.append(ds1)

    ds_drift = xr.concat(ds_drift, dim='lead_year')
    
    # Read data for all hindcasts 
    for r in range(0,10):
       
        print("Var = ", var, "; Ensemble = ", r)

        ds = []
        
        for year in range(year1, year2, 1):
            
            var_path = "s" + str(year) +"-r" + str(r+1) + "i1p1f2/Omon/" + var + "/gn/files/d20200417/"
            
            d = xr.open_mfdataset(ppdir + var_path + "*.nc")
            d['time_val'] = d['time'] # dropping time coordinate is required; otherwise xarray concat does not work properly
            d = d.drop(['time', 'vertices_latitude', 'vertices_longitude', 'time_bnds'])
            
            ds.append(d)
            
        # combine data for hindcasts
        ds = xr.concat(ds, dim='start_year')
        ds = ds.isel(i=slice(749,1199), j = slice(699, 1149))
        ds = ds.chunk({'start_year':1})
        
        # Compute seasonal means and then isolate DJF season
        ds = ds.assign_coords(time=ds['time_val'].isel(start_year=0)) # this is just to get seasonal averaging work

        ds_resam = ds.resample(time='QS-DEC').mean('time')
        ds_resam = ds_resam.sel(time = ds_resam['time.season'] == 'DJF')

        # again drop time and change names to be consistent with ds_drift
        ds_resam = ds_resam.drop('time')
        ds_resam = ds_resam.rename({'time':'lead_year'})

        ds_resam = ds_resam.transpose('start_year','lead_year','j','i')
        
        # compute anomaly
        ds_anom = ds_resam - ds_drift.isel(ensemble = r)

        # compute area-weighted mean anomlies for regions
        mask_val = 0.
        anom_save = xr.Dataset()
        
        for mask_region in region_list:
            
            if(mask_region == 'North_Atlantic'):
                area_weigh_val = Compute_area_weighted_mean(ds_anom[var], ds_mask['area'], ds_mask['mask_North_Atl'], 0.)
            else:
                area_weigh_val = Compute_area_weighted_mean(ds_anom[var], ds_mask['area'], ds_mask['mask_regions'], mask_val)
                mask_val = mask_val + 1
                
            anom_save[mask_region] = area_weigh_val.compute()
        
        # Save file for each ensemble
        save_file = save_path +"Anomaly_" + var + "_r" + str(r+1) + ".nc"
        anom_save.to_netcdf(save_file)

        print("File saved")
