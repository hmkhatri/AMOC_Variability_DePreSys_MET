"""
Author: Hemant Khatri
Email: hkhatri@liverpool.ac.uk

This script computes Ekamn component of the overturning circulation in the North Atlantic as a function of latitude. 
1. Read zonal surface wind stress data.
2. Compute overturning by integrating wind stress in the zonal direction and dividing by -(f x rho_0).
3. Finally, data is saved in netcdf format.

This script makes use of xarray, dask, xgcm libraries for computations and working with netcdf files. 
The code can work in serial mode as well as in parallel (see below for details). 

For parallelization, daks-mpi (http://mpi.dask.org/en/latest/) is used to initialize multiple workers on a dask client. This is to ensure that dask is aware of multiple cores assigned through slurm batch submission job. 

Instead of dask-mpi, one could also use dask jobqueue (http://jobqueue.dask.org/en/latest/), which is very effective for interactive dask session on jupyter notebook or ipython console.

To run in serial, use "python file.py"
To run in parallel, use "mpirun -np NUM_Cores python file.py"
"""

# ------- load libraries ------------ 
import numpy as np
import xarray as xr
from xgcm import Grid
from cmip6_preprocessing.regionmask import merged_mask
import regionmask
import gc
from tornado import gen
import os

import warnings
warnings.filterwarnings('ignore')

### ------ Functions for computations ----------

def select_subset(dataset):
    
    """Select subset of dataset in xr.open_mfdataset command
    """
    
    dataset = dataset.sel(lat=slice(0.,85.)) # coord range to choose
    dataset = dataset.drop(['lon_bnds', 'lat_bnds', 'time_bnds']) # drop variables 
    
    return dataset

def compute_overturning_ekman(data_tau, lat, dx, Omega = 7.27e-5, Rho_0 = 1000., long_name=None): 
    
    """Compute Ekman component of overturning circulation 
    Parameters
    ----------
    data_tau : xarray DataArray - Zonal wind stress on the ocean surface
    lat : xarray DataArray - latitudes
    dx : xarray DataArray - Zonal grid spacing
    Omega : Constant - Angular frequency of the Earth
    Rho_0 : Constant - Reference density
    
    Returns
    -------
    overturning : xarray DataArray - overturning circulation
    """
    
    int_tau = (data_tau * dx).sum('lon')
    f = 2 * OMEGA * np.sin(lat * np.pi/180.)
    overturning =  - int_tau / (f * RHO_0)
    overturning.attrs['units'] = "m^3/s"
    overturning.attrs['long_name'] = long_name
    
    return overturning


### ------------- Main computations ------------

data_dir="/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/"
save_path="/gws/nopw/j04/snapdragon/hkhatri/Data_sigma/Overturning_Ekman/"

year1, year2 = (1960, 2017) # range of years for reading data 

RAD_EARTH, RHO_0 = (6.387e6, 1035.0) # constants
OMEGA = 2 * np.pi / (24*3600.)

basins = regionmask.defined_regions.natural_earth_v4_1_0.ocean_basins_50 # north atlantic mask

for r in range(1,10):
    
    for year in range(year1, year2, 1):
       
        print("Running: Ensemble = ", r+1, ", Hindcast Year = ", year)
        
        var_path = (data_dir + "s" + str(year) +"-r" + str(r+1) + 
                    "i1p1f2/Amon/" + "tauu" + "/gn/latest/*.nc")
        
        with xr.open_mfdataset(var_path, preprocess=select_subset, 
                               chunks={'time':1}, engine='netcdf4') as ds:
            ds = ds 

        print("Data read complete")
        
        # create mask and compute zonal grid spacing
        mask_NA = merged_mask(basins, ds, lon_name="lon", lat_name="lat")
        
        dx = (np.mean(ds['lon'].diff('lon')) * np.cos(ds['lat'] * np.pi / 180.) 
              * (2 * np.pi * RAD_EARTH / 360.))
        dx = dx.where(mask_NA==0).compute()
        
        # compute overturning and save data
        ds_save = xr.Dataset()
        
        ds_save['Overturning_Ekman'] = compute_overturning_ekman(ds['tauu'], ds['lat'], dx, Omega = OMEGA, 
                                                                 Rho_0 = RHO_0, long_name = "Overturning Ekman component")
        
        save_file_path = (save_path + "Overturning_Ekman_"+ str(year) + "_r" + str(r+1) + ".nc")
        
        ds_save = ds_save.compute() # compute before saving
        ds_save.to_netcdf(save_file_path)
    
        print("Data saved succefully")
        
        ds_save.close()
        ds.close()
        
        
        