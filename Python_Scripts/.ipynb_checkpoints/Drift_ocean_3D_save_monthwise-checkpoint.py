# ------------------------------------------------ #
# Script for drift calculation for 3D ocean vars using the method of Dune et al. (2016). 
# Drift is calculated for years 1979-2016, so it can be directly comapred against OSRA5 and EN4.
# Drift data is saved for each month and each ensemble member separately.
# ------------------------------------------------ #

# load libraries
import numpy as np
import scipy as sc
import xarray as xr
import gc

import warnings
warnings.filterwarnings('ignore')

from dask_mpi import initialize
initialize()

from dask.distributed import Client, performance_report
client = Client()

### ------ Function to sum over selected years and saving chunks ----------

def processDataset(ds1, year1, year2, lead_year):
    
    ds_drift = []
    
    for year in range(year1, year2):
        
        # Extract data relavant start year and sum over all hindcasts
        ds2 = ds1.sel(start_year = year - lead_year)
        ds2 = ds2.isel(time=slice(12*lead_year, np.minimum(12 + 12*lead_year, len(ds1['time']))))
        
        ds_drift.append(ds2.drop('start_year'))
        
    ds_drift = xr.concat(ds_drift, dim='hindcast')
        
    return ds_drift

### ------------- Main computations ------------

ppdir="/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/"

# for monthly drift
save_path="/gws/nopw/j04/snapdragon/hkhatri/Data_Drift/" # this is for 3D vars
#save_path="/home/users/hkhatri/DePreSys4_Data/Data_Drift_Removal/"

var_list = ['vo'] # 'thetao', 'uo'

year1, year2 = (1979, 2017) # range over to compute average using DCPP 2016 paper

for var in var_list:
    
    for r in range(0,1):
       
        print("Var = ", var, "; Ensemble = ", r)

        ds = []

        for year in range(year1-10, year2, 1):
            
            # Read data for each hindcast for every ensemble member
            
            var_path = "s" + str(year) +"-r" + str(r+1) + "i1p1f2/Omon/" + var + "/gn/latest/"
            
            d = xr.open_mfdataset(ppdir + var_path + "*.nc", parallel=True, decode_times=False)
            
            # drop time coordinate as different time values create an issue in concat operation
            d = d.drop(['time', 'vertices_latitude', 'vertices_longitude', 'time_bnds', 'lev_bnds'])
            
            d = d.isel(i=slice(749,1199), j=slice(699, 1149))
            
            ds.append(d)
            
        # combine data for hindcasts
        ds = xr.concat(ds, dim='start_year')
        
        ds = ds.assign(start_year = np.arange(year1-10, year2, 1))
        
        #ds = ds.chunk({'start_year':1, 'lev':1})
        
        print("Data read complete")
        
        # loop over lead year and compute mean values
        for lead_year in range(0,11):
    
            #print("Lead Year running = ", lead_year)

            ds_var = processDataset(ds, year1, year2, lead_year)
            
            #ds_var = (ds_var.mean('hindcast'))
            
            #with performance_report(filename="dask-report.html"):
            
            for month in range(0,len(ds_var.time)):
                
                ds_save = ds_var.isel(time=month)
                ds_save = ds_save.copy() # required, so dask does not load all data before isel operation
                
                #with performance_report(filename="dask-report.html"):
                ds_save = ds_save.mean('hindcast').persist()
            
                save_file = (save_path +"Drift_" + var + "_r" + str(r+1)+ 
                             "_Month_" + str(lead_year*12 + month + 1) + ".nc")
                ds_save.to_netcdf(save_file)

                print("File saved for Month = ", lead_year*12 + month + 1)
                
                ds_save.close()
            
            ds_var.close()
            
        ds.close()