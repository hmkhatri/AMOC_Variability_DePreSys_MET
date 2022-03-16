# load libraries
import numpy as np
import scipy as sc
import xarray as xr
import gc
import dask.distributed
import os, psutil
#from dask.distributed import Client

#from mpi4py import MPI
import dask
#import distributed

from dask_mpi import initialize
initialize()

import warnings
warnings.filterwarnings('ignore')

from dask.distributed import Client, performance_report
client = Client()

### ------ Function to sum over selected years ----------

def processDataset(ds1, y1, y2, lead_yr):
    
    #ds_save = []
    
    #for yr in range(y1, y2):
        
        # Extract relevant DJF months data and mean over the season
        #ds2 = ds1.sel(start_year = year - lead_year).isel(time=slice(1 + 12*lead_year, 4 + 12*lead_year)).mean('time')
        
        # Extract data relavant start year and sum over all hindcasts
        #ds2 = ds1.sel(start_year = year - lead_year).isel(time=slice(12*lead_year, np.minimum(12 + 12*lead_year, len(ds1['time']))))
        
     #   ind_sy = yr - lead_yr - y1 + 10
     #   ind_tim1 = 12*lead_yr
     #   ind_tim2 = np.minimum(12 + 12*lead_yr, len(ds1['time']))
     #   ds2 = ds1.isel(start_year = ind_sy, time=slice(ind_tim1, ind_tim2))
        
     #   ds_save.append(ds2)
        
    #ds_save = xr.concat(ds_save, dim='hindcast')
    
    # wihtout using append loop
    ind_tim1 = 12*lead_yr
    ind_tim2 = np.minimum(12 + 12*lead_yr, len(ds1['time']))
    
    ds2 = ds1.isel(start_year = slice(y1-lead_yr-y1+10, y2-lead_yr-y1+10))
    ds2 = ds2.isel(time=slice(ind_tim1, ind_tim2))
    # isel -> slice(0,10) will isolate data for 0, 1,2, ...9 indices
    # sel -> slice(1960, 1970) will isolate data for 1960, 1961, .... 1970 years
        
    return ds2

def select_subset(d1):
    
    d1 = d1.isel(i=slice(749,1199), j = slice(699, 1149))
    d1 = d1.drop(['vertices_latitude', 'vertices_longitude', 'time_bnds', 'lev_bnds'])
    
    return d1

### ------------- Main computations ------------

ppdir="/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/"

# for DJF season only
#save_path="/home/users/hkhatri/DePreSys4_Data/Data_Drift_Removal/Drift_2016_DCPP/"

# for monthly drift
#save_path="/home/users/hkhatri/DePreSys4_Data/Data_Drift_Removal/Drift_1970_2016_Method_DCPP/"
save_path="/gws/nopw/j04/snapdragon/hkhatri/Data_Drift/" # this is for 3D vars

#var_list = ['hfds'] #, 'tos', 'sos'] #, 'mlotst', 'zos']
var_list = ['thetao'] #['thetao', 'vo', 'uo']

year1, year2 = (1979, 2017) # range over to compute average using DCPP 2016 paper

dropvars = ['vertices_latitude', 'vertices_longitude', 'time_bnds', 'lev_bnds']

for var in var_list:
    
    for r in range(4,5,1):
       
        print("Var = ", var, "; Ensemble = ", r)

        ds = []

        for year in range(year1-10, year2, 1):
            
            # Read data for each hindcast for every ensemble member
            
            var_path = (ppdir + "s" + str(year) +"-r" + str(r+1) + 
                        "i1p1f2/Omon/" + var + "/gn/latest/*.nc")
            
            #d = xr.open_mfdataset(var_path, parallel=True, preprocess=select_subset,
            #                      decode_times=False, engine='netcdf4')
            # netdcf4 engine seems to be faster than h5netcdf in reading files
            
            with xr.open_mfdataset(var_path, parallel=True, preprocess=select_subset, 
                                   chunks={'lev':1, 'time':1},
                                   decode_times=False, engine='netcdf4') as d:
                d = d
            
            # drop time coordinate as different time values create an issue in concat operation
            d = d.drop('time')
            #if(var=='thetao'):
            #d = d.drop('lev_bnds')
            #d = d.isel(i=slice(749,1199), j = slice(699, 1149))
            
            ds.append(d)
            
        # combine data for hindcasts
        ds = xr.concat(ds, dim='start_year')
        
        #ds = ds.isel(i=slice(749,1199), j = slice(699, 1149))
        #ds = ds.isel(i=slice(749,849), j = slice(699, 799)) # just for a check
        #ds = ds.isel(lev=0)
        
        #ds = ds.assign(start_year = np.arange(year1-10, year2, 1))
        # ignore assign -> seems to break the mpi code
        
        #if(var=='thetao'):
        #ds = ds.chunk({'start_year':1, 'lev':10})
        #else:
        #ds = ds.chunk({'start_year':1, 'time':1, 'j':50})
        
        print("Data read complete")
        
        # loop over lead year and compute mean values
        for lead_year in range(0,11):
    
            #print("Lead Year running = ", lead_year)

            ds_var = processDataset(ds[var], year1, year2, lead_year)
    
            #ds_var = ds_var.mean('hindcast')
            #ds_var = sum(ds_var)/(year2 - year1)
            
            #print("Lead Year = ", lead_year, "; Shape of array is ", ds_var.shape)
            #ds_var = ds_var.persist()
            
            ds_var = ds_var.mean('start_year')

            #with performance_report(filename="dask-report.html"):
            #ds_save = ds_var.compute()
            
            ds_save = xr.Dataset()
            
            with performance_report(filename="memory-mpi.html"):
                ds_save[var] = ds_var.compute() #persist()
    
            save_file = (save_path +"Drift_" + var + "_r" + str(r+1)+ 
                         "_Lead_Year_" + str(int(lead_year+1)) + ".nc")
            ds_save.to_netcdf(save_file)

            print("File saved for Lear Year = ", lead_year+1)
            
            client.cancel([ds_var, ds_save])
            ds_save.close()
            ds_var.close()
            
            process = psutil.Process(os.getpid())
            print("Memory usage in GB = ", process.memory_info().rss/1e9)
            
            #gc.collect()
        
        client.cancel([ds])
        ds.close()
        del ds
        
        print("Completed r = ", r+1)
        
client.close()
