# load libraries
import numpy as np
import xarray as xr
import dask.distributed
from dask.distributed import Client

import warnings
warnings.filterwarnings('ignore')

ppdir="/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/"

save_path="/home/users/hkhatri/DePreSys4_Data/Ensemble_Data/Data_Consolidated/"

var_list = ['tauu', 'tauv', 'psl', 'sfcWind', 'tas', 'pr', 'evspsbl', 'clt', 'ua', 'va']

for year in range (2008, 2011, 2):
    
    #ds = []
    
    print("Year Running - ", year)
    
    for var in var_list:
        
        ds = []

        for i in range(0,10):
    
        #for var in var_list:
        
            var_path = "s" + str(year) +"-r" + str(i+1) + "i1p1f2/Amon/" + var + "/gn/files/d20200417/"
    
            d = xr.open_mfdataset(ppdir + var_path + "*.nc")
        
            ds.append(d)
        
        #ds1 = xr.merge(ds1)
    
        #ds.append(ds1)
    
        ds = xr.concat(ds, dim='r')

        if ( var == 'ua' or var == 'va'):
            ds = ds.sel(lon=slice(260., 360.), lat = slice(0., 80.), plev = 85000.)
        else:
            ds = ds.sel(lon=slice(260., 360.), lat = slice(0., 80.))

        print("Data reading completed -> Moving to write variable: ", var)

        save_file = save_path + str(year) + "_" + var + ".nc"
        #ds = ds.persist()
        ds.to_netcdf(save_file)
