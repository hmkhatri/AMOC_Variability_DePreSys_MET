# load libraries
import numpy as np
import xarray as xr
import dask.distributed
from dask.distributed import Client

import warnings
warnings.filterwarnings('ignore')

ppdir="/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/"

save_path="/home/users/hkhatri/DePreSys4_Data/Ensemble_Data/Data_Consolidated/"

var_list = ['volo', 'hfds', 'mlotst', 'tos', 'sos', 'zos']

for year in range (2008, 2011, 2):
    
    #ds = []
    
    print("Year Running - ", year)
    
    for var in var_list:
        
        ds = []

        for i in range(0,10):
    
        #for var in var_list:
        
            var_path = "s" + str(year) +"-r" + str(i+1) + "i1p1f2/Omon/" + var + "/gn/files/d20200417/"
    
            d = xr.open_mfdataset(ppdir + var_path + "*.nc")
        
            ds.append(d)
        
        #ds1 = xr.merge(ds1)
    
        #ds.append(ds1)
    
        ds = xr.concat(ds, dim='r')

        ds = ds.drop(['vertices_latitude', 'vertices_longitude'])
        ds = ds.isel(i=slice(749,1199), j = slice(699, 1149)) 

        print("Data reading completed -> Moving to write variable: ", var)

        save_file = save_path + str(year) + "_" + "var.nc"
        #ds_save = ds.persist()
        ds_save.to_netcdf(save_file)
