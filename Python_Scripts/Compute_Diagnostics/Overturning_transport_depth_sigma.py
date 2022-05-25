"""
Author: Hemant Khatri
Email: hkhatri@liverpool.ac.uk

This script computes zonal and meridional transport at fixed density levels from velocity data at depth-levels. 
1. Potential densities are computed using monthly Temparature and Salinity. 
2. xgcm transform funtionality is used for computing transport at fixed density levels.
3. Depths of these density isolines are also computed. 
4. Finally, data is saved in netcdf format.

This script makes use of xarray, dask, xgcm libraries for computations and working with netcdf files. 
The code can work in serial mode as well as in parallel (see below for details). 

For parallelization, daks-mpi (http://mpi.dask.org/en/latest/) is used to initialize multiple workers on a dask client. This is to ensure that dask is aware of multiple cores assigned through slurm batch submission job. 
With dask, chunksizes are very important. Generally, it is good to have chunksize of 10-100 MB. 
For this script, it is recomended to use chunks={'time':1, 'j':45, 'lev':-1}, for which computations are found to be very efficient.

Instead of dask-mpi, one could also use dask jobqueue (http://jobqueue.dask.org/en/latest/), which is very effective for interactive dask session on jupyter notebook or ipython console.

To run in serial, use "python file.py"
To run in parallel, use "mpirun -np NUM_Cores python file.py"
"""

# ------- load libraries ------------ 
import numpy as np
import xarray as xr
import cf_xarray
import gsw as gsw
from xgcm import Grid
import gc
from tornado import gen
import os

import warnings
warnings.filterwarnings('ignore')

# -- commands in the next four lines are for using dask with "mpirun -np num_tasks python file.py" command ---- 
# ------ comment these if running as a serial code -------
#from dask_mpi import initialize
#initialize()

#from dask.distributed import Client, performance_report
#client = Client() # don't use processes=True/False inside (), it will get stuck
#print(client)

#os.environ["MALLOC_MMAP_MAX_"]=str(40960) # to reduce memory clutter. This is temparory, no permanent solution yet.
#os.environ["MALLOC_MMAP_THRESHOLD_"]=str(16384)
# see https://github.com/pydata/xarray/issues/2186, https://github.com/dask/dask/issues/3530

# ------------------------------------------------------------------------------

### ------ Functions for computations ----------

def select_subset(dataset):
    
    """Select subset of dataset in xr.open_mfdataset command
    """
    dataset = dataset.isel(i=slice(749,1199), j = slice(699, 1149)) # indices range
    dataset = dataset.drop(['vertices_latitude', 'vertices_longitude', 
                            'time_bnds']) # drop variables 
    
    return dataset

def compute_density(data_salinity, data_temperature):

    """Compute potential density using salinity and potential temperature data 
    Parameters
    ----------
    data_salinity : xarray DataArray for salinity data
    data_temperature : xarray DataArray for potential temperature data
    
    Returns
    -------
    pot_density : xarray DataArray for potential density data
    """

    pot_density = gsw.density.sigma0(data_salinity, data_temperature)

    return pot_density

def density_levels(density_min=0., density_max=35.):
    
    """Define density levels
    Parameters
    ----------
    density_min : float value for minimum density
    density_max : float value for maximum density
    
    Returns
    -------
    sigma_levels : numpy array for density levels
    """

    density_rang1 = np.arange(density_min, 20., 2.0)
    density_rang2 = np.arange(20., 23.1, 1.)
    density_rang3 = np.arange(23.2, 26., 0.2)
    density_rang4 = np.arange(26.1, 28., 0.1)
    density_rang5 = np.arange(28.0, 28.8, 0.2)
    density_rang6 = np.arange(29.0, density_max, 1.)
    
    sigma_levels = np.concatenate((density_rang1 ,density_rang2, density_rang3, density_rang4, 
                                   density_rang5, density_rang6))
    
    return sigma_levels

def transport_z(ds_vel, z_lev, grid, assign_name='transport'): 
    
    """Compute volume transport per unit horizontal length 
    Parameters
    ----------
    ds_vel : xarray DataArray - velocity data
    z_lev : xarray DataArray - outer z levels
    grid : xgcm Grid object
    assign_name : name for the volume transport
    
    Returns
    -------
    transport : xarray DataArray for volume transport
    """
    
    thickness = grid.diff(z_lev, 'Z') # compute vertical grid spacing
    transport =  ds_vel * thickness # velocity x vertical grid spacing
    transport = transport.fillna(0.).rename(assign_name)
    
    return transport
    
def transport_sigma(pot_density, transport, density_levels, grid, dim=None, method='linear'):
    
    """Compute volume transport in density layers using xgcm.transform method 
    Parameters
    ----------
    pot_density : xarray DataArray - potential density
    transport : xarray DataArray - volume transport on z-levels
    density_levels : numpy array - Target density levels at which one wants to compute volume transport 
    grid : xgcm Grid object
    dim : dimension for interpolation to have ensure pot_density and transport on the same grid
    method : transform method out of 'linear' or 'conservative'
    
    Returns
    -------
    transport_sigma : xarray DataArray - volume transport at denisty levels
    """
    
    if(dim != None): # interpolate the density to transport data grid
        sigma_interp = grid.interp(pot_density, [dim], boundary='extend')
    else:
        sigma_interp = pot_density

    sigma_interp = sigma_interp.rename('sigma0')
    transport_sigma = grid.transform(transport, 'Z', density_levels,
                                     target_data=sigma_interp, method=method)
    transport_sigma.sigma0.attrs['units'] = "kg/m^3"
    transport_sigma.sigma0.attrs['long_name'] = "Potential density with reference to ocean surface - 1000."
    
    return transport_sigma

def save_data(data_var, save_location, short_name, long_name=None, var_units=None):
    """Save final diagnostics in netcdf format
    Parameters
    ----------
    data_var : xarray DataArray
    save_location : string - location for saving data including filename
    short_name : string - name for the variable
    long_name : string - long name for the variable
    var_units : string - dimensional units
    """
    
    data_save = data_var.astype(np.float32).to_dataset(name=short_name)
    data_save[short_name].attrs['units'] = var_units
    data_save[short_name].attrs['long_name'] = long_name
    
    data_save = data_save.compute() # compute before saving
    data_save.to_netcdf(save_location)
    
    print("Data saved succefully - ", short_name)
    
    data_save.close()

async def stop(dask_scheduler):
    """To close all workers after job completion
    """ 
    await gen.sleep(0.1)
    await dask_scheduler.close()
    loop = dask_scheduler.loop
    loop.add_callback(loop.stop)

### ------------- Main computations ------------

# first define paths for reading data and saving final diagnostics 
data_dir="/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/"
save_path="/gws/nopw/j04/snapdragon/hkhatri/Data_sigma/Transport_sigma/Temp/"

# read grid information and masking data
ds_grid = xr.open_dataset("/home/users/hkhatri/DePreSys4_Data/Data_Consolidated/Ocean_Area_Updated.nc")
ds_mask = xr.open_dataset("/home/users/hkhatri/DePreSys4_Data/Data_Consolidated/Mask_UV_grid.nc")

year1, year2 = (2002, 2017) # range of years for reading data 
var_list = ['thetao', 'so', 'vo', 'uo'] # list of variables to be read from model output

# get sigma levels
sigma_min, sigma_max = (15., 31.1) 
target_sigma_levels = density_levels(density_min=sigma_min, density_max=sigma_max)

# Loop for going through multiple ensemble and hindcast members for computations
for r in range(1,2):
    
    for year in range(year1, year2, 1):
        
        print("Running: Ensemble = ", r+1, ", Hindcast Year = ", year)
        
        ds = []

        for var in var_list:

            var_path = (data_dir + "s" + str(year) +"-r" + str(r+1) + 
                        "i1p1f2/Omon/" + var + "/gn/latest/*.nc")

            with xr.open_mfdataset(var_path, parallel=True, preprocess=select_subset, 
                                   chunks={'time':1, 'j':45}, engine='netcdf4') as d:
                d = d 
            # chunksize is decided such a way that 
            # inter-worker communication is minimum and computations are efficient
                
            # renaming coords is required because variables are on different grid points (common in C-grid)
            # later xgcm.grid function is used to fix locations of these vars on model grid
            if(var == 'vo'):
                d = d.rename({'j':'j_c', 'longitude':'longitude_v', 'latitude':'latitude_v'})
            elif(var == 'uo'):
                d = d.rename({'i':'i_c', 'longitude':'longitude_u', 'latitude':'latitude_u'})

            ds.append(d)

        ds = xr.merge(ds) # merge to get a single dataset
        #ds = xr.merge([ds, ds_grid['dx_v'].rename({'x':'i', 'yv':'j_c'}), 
        #               ds_grid['dy_u'].rename({'xu':'i_c', 'y':'j'})]) 
        
        print("Data read complete")
        
        # ---------------------- Computations (points 1-3) ------------------------- #
        # create outer depth levels (required for transforming to density levels)
        level_outer_data = (cf_xarray.bounds_to_vertices(ds['lev_bnds'].isel(time=0),
                                                         'bnds').load().data)
        ds = ds.assign_coords({'level_outer': level_outer_data})
        
        # create grid object such that xgcm is aware of locations of velocity and tracer grid points
        grid = Grid(ds, coords={'Z': {'center': 'lev', 'outer': 'level_outer'},
                                'X': {'center': 'i', 'right': 'i_c'},
                                'Y': {'center': 'j', 'right': 'j_c'}}, periodic=[],)
        
        # compute density (xr.apply_ufunc ensures that non-xarray operations are efficient with dask
        sigma = xr.apply_ufunc(compute_density, ds['so'], ds['thetao'], dask='parallelized',
                               output_dtypes=[ds['thetao'].dtype])
        sigma = sigma.rename('sigma0').persist()
        
        # compute zonal and meridional transport
        Zonal_Transport = transport_z(ds['uo'], ds['level_outer'], grid, 
                                      assign_name='Zonal_Transport')
        Meridional_Transport = transport_z(ds['vo'], ds['level_outer'], grid, 
                                           assign_name='Meridional_Transport')
        
        # compute zonal/meridional transport on sigma levels and depth of density isolines
        Zonal_Transport_sigma = transport_sigma(sigma, Zonal_Transport, target_sigma_levels, 
                                                grid=grid, dim='X', method='conservative')
        
        Meridional_Transport_sigma = transport_sigma(sigma, Meridional_Transport, target_sigma_levels,
                                                     grid=grid, dim='Y',method='conservative')
        
        depth = xr.ones_like(sigma) * ds['lev'] # get 4D var for depth
        depth_sigma = transport_sigma(sigma, depth, target_sigma_levels,
                                      grid=grid, dim=None, method='linear')
        
        # -------------------- Save final diagnostics (point 4) ------------------------- #
        save_file = (save_path + "Zonal_Transport_sigma_"+ str(year) + "_r" + str(r+1) + ".nc")  
        save_data(Zonal_Transport_sigma, save_file, "Zonal_Transport_sigma", 
                  long_name="Zonal velocity x layer thickness", var_units="m^2/s")
        
        save_file = (save_path + "Meridional_Transport_sigma_"+ str(year) + "_r" + str(r+1) + ".nc")
        save_data(Meridional_Transport_sigma, save_file, "Meridional_Transport_sigma", 
                  long_name="Meridional velocity x layer thickness", var_units="m^2/s")
        
        save_file = (save_path + "Depth_sigma_"+ str(year) + "_r" + str(r+1) + ".nc")
        save_data(depth_sigma, save_file, "Depth_sigma", 
                  long_name="Depth of density layer", var_units="m")
        
        ds.close()
        
        #client.cancel([sigma, Zonal_Transport, Meridional_Transport, depth,
        #              Zonal_Transport_sigma, Meridional_Transport_sigma, depth_sigma])
        
        #client.run(gc.collect)
        
print('Closing cluster')
#client.run_on_scheduler(stop, wait=False)     
        
        
        
