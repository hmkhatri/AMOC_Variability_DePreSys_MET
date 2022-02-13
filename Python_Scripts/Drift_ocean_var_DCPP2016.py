# load libraries
import numpy as np
import scipy as sc
import xarray as xr
import dask.distributed
from dask.distributed import Client, performance_report
from dask import delayed
from dask import compute
from dask import persist

import warnings
warnings.filterwarnings('ignore')

from dask_mpi import initialize
initialize()
client = Client()

import dask
import distributed
dask.config.set({"distributed.comm.timeouts.tcp": "50s"})

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

# for DJF season only
#save_path="/home/users/hkhatri/DePreSys4_Data/Data_Drift_Removal/Drift_2016_DCPP/"

# for monthly drift
#save_path="/home/users/hkhatri/DePreSys4_Data/Data_Drift_Removal/Drift_1970_2016_Method_DCPP/"

save_path="/gws/nopw/j04/snapdragon/hkhatri/Data_Drift/" # this is for 3D vars

#var_list = ['hfds', 'tos', 'sos'] #, 'mlotst', 'zos']
var_list = ['thetao']

year1, year2 = (1979, 2017) # range over to compute average using DCPP 2016 paper

for var in var_list:
    
    for r in range(1,10):
       
        print("Var = ", var, "; Ensemble = ", r)

        ds = []

        for year in range(year1-10, year2, 1):
            
            # Read data for each hindcast for every ensemble member
            
            var_path = "s" + str(year) +"-r" + str(r+1) + "i1p1f2/Omon/" + var + "/gn/latest/"
            
            d = xr.open_mfdataset(ppdir + var_path + "*.nc", parallel=True, decode_times=False)
            
            # drop time coordinate as different time values create an issue in concat operation
            d = d.drop(['time', 'vertices_latitude', 'vertices_longitude', 'time_bnds'])
            if(var=='thetao'):
                d = d.drop('lev_bnds')
                
            d = d.isel(i=slice(749,1199), j = slice(699, 1149))
            
            ds.append(d)
            
        # combine data for hindcasts
        ds = xr.concat(ds, dim='start_year')
        #ds = ds.drop(['vertices_latitude', 'vertices_longitude', 'time_bnds'])
        #ds = ds.isel(i=slice(749,1199), j = slice(699, 1149))
        
        ds = ds.assign(start_year = np.arange(year1-10, year2, 1))
        #ds = ds.chunk({'start_year':1, 'j':50, 'i':50})
        
        #if(var=='thetao'):
        #    ds = ds.chunk({'start_year':1, 'time':12, 'lev':5})
        #else:
        #    ds = ds.chunk({'start_year':1, 'time':12})
        
        print("Data read complete")
        
        # loop over lead year and compute mean values
        for lead_year in range (0,11):
    
            #print("Lead Year running = ", lead_year)

            ds_save = xr.map_blocks(processDataset, ds, year1, year2, lead_year)
    
            ds_save = sum(ds_save) / (year2 - year1)

            #with performance_report(filename="dask-report.html"):
            ds_save = ds_save.compute()
    
            save_file = (save_path +"Drift_" + var + "_r" + str(r+1)+ 
                         "_Lead_Year_" + str(int(lead_year+1)) + ".nc")
            ds_save.to_netcdf(save_file)

            print("File saved")
            
        ds.close()

client.close()
