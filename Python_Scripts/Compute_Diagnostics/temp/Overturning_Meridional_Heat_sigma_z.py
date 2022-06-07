"""
Author: Hemant Khatri
Email: hkhatri@liverpool.ac.uk

This script computes overturning circulation at different density and depth levels. 
1. Read u, v data on z-levels and transport data on density-levels.
2. Compute overturning by integrating transport in the zonal direction as well as in the direction of increasing depth (or density).
3. Compute Meridional heat transport in sigma layers as well as at z-levels.
4. Finally, data is saved in netcdf format.

This script makes use of xarray, dask, xgcm libraries for computations and working with netcdf files. 
The code can work in serial mode as well as in parallel (see below for details). 

With dask, chunksizes are very important. Generally, it is good to have chunksize of 10-100 MB. 
For this script, it is recomended to use chunks={'time':1, 'lev':12}, chunks={'time':1, 'sigma0':12}.
However, the most efficient chunking varies from dataset to dataset. Some manual testing is required to find the most suitable chunking method.

For parallelization, daks-mpi (http://mpi.dask.org/en/latest/) is used to initialize multiple workers on a dask client. This is to ensure that dask is aware of multiple cores assigned through slurm batch submission job. 

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
#client = Client()

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

def Compute_Heat_transport(Field, Velocity, grid, dim, const_multi = 1.):
    
    """Compute transport of field along velocity component 
    Parameters
    ----------
    Field : xarray DataArray - tracer field
    Velocity : xarray DataArray - velocity along any cartesian direction
    grid : xgcm Grid object
    dim : strig - dimension name
    const_multi : constant - multiplier
    
    Returns
    -------
    Transport : xarray DataArray for volume transport
    """
    
    Field_vel = grid.interp(Field, [dim], boundary='extend') # interpolate field to velocity grid
    
    Transport = Field_vel * Velocity * const_multi
    
    return Transport

def Compute_Overturning(Transport, dx = 1., dim_v='Z', dim_x = 'X', long_name=None):
    
    """Compute Overturning circulation using meridional velocity x thickness data
    Parameters
    ----------
    Transport : xarray DataArray - meridional velocity x layer thickness
    dx : xarray DataArray - Zonal grid spacing
    dim_v : Vertical dimension name
    dim_x : Zonal dimension name
    long_name : string - long name for the variable
    
    Returns
    -------
    overturning : xarray DataArray - overturning circulation
    """
    
    overturning = (Transport * dx).sum(dim=dim_x).cumsum(dim=dim_v)
    overturning.attrs['units'] = "m^3/s"
    overturning.attrs['long_name'] = long_name
    
    return overturning

def Meridional_Heat_Transport(Transport, Heat, dx = 1., dim_x = 'X', long_name=None):
    
    """Compute Meriidonal Heat Transport corresponding to meridional overturning circulation
    Parameters
    ----------
    Transport : xarray DataArray - meridional velocity x layer thickness
    Heat : xarray DataArray - Heat Content
    dx : xarray DataArray - Zonal grid spacing
    dim_x : Zonal dimension name
    long_name : string - long name for the variable
    
    Returns
    -------
    Meridional_Heat_Transport : xarray DataArray - meridional heat transport due to overturning circulation
    """
    
    Transport_Meridional = (Transport * dx).sum(dim=dim_x) # first get net meridional volume transport
    Meridional_Heat = (Heat * dx).sum(dim=dim_x) / dx.sum(dim=dim_x) # zonal mean of heat content
    
    Meridional_Heat_Transport = Transport_Meridional * Meridional_Heat
    
    Meridional_Heat_Transport.attrs['units'] = "Joules/s"
    Meridional_Heat_Transport.attrs['long_name'] = long_name
    
    return Meridional_Heat_Transport

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
save_path="/gws/nopw/j04/snapdragon/hkhatri/Data_sigma/Overturning_Heat_Transport/"

# read grid information and masking data
ds_grid = xr.open_dataset("/home/users/hkhatri/DePreSys4_Data/Data_Consolidated/Ocean_Area_Updated.nc")
ds_mask = xr.open_dataset("/home/users/hkhatri/DePreSys4_Data/Data_Consolidated/Mask_UV_grid.nc")

year1, year2 = (1960, 1990) # range of years for reading data 
var_list = ['vo', 'uo', 'thetao', 'so'] # list of variables to be read from model output

datadir_sigma ="/gws/nopw/j04/snapdragon/hkhatri/Data_sigma/Transport_sigma/Temp/"
var_sigma_list = ['Zonal_Transport_sigma_', 'Meridional_Transport_sigma_', 'Depth_sigma_']

rho_cp = 4.09 * 1.e6 # constant from Williams et al. 2015

# get sigma levels
sigma_min, sigma_max = (15., 31.1) 
target_sigma_levels = density_levels(density_min=sigma_min, density_max=sigma_max)

# Loop for going through multiple ensemble and hindcast members for computations
for r in range(9,10):
    
    for year in range(year1, year2, 1):
        
        print("Running: Ensemble = ", r+1, ", Hindcast Year = ", year)
        
        ds = []

        for var in var_list:

            var_path = (data_dir + "s" + str(year) +"-r" + str(r+1) + 
                        "i1p1f2/Omon/" + var + "/gn/latest/*.nc")

            with xr.open_mfdataset(var_path, preprocess=select_subset, 
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
        ds = xr.merge([ds, ds_grid['dx_v'].rename({'x':'i', 'yv':'j_c'}), 
                       ds_grid['dy_u'].rename({'xu':'i_c', 'y':'j'})]) 
        
        """
        # Read transport data on density-levels
        ds_sigma = []

        for var in var_sigma_list:
            
            var_path = datadir_sigma + var + str(year) +"_r" + str(r+1) + ".nc"
            with xr.open_mfdataset(var_path, parallel=True, chunks={'time':1, 'sigma0':12}, 
                                   engine='netcdf4') as d:
                d = d

            if(var == 'Depth_sigma_'):
                d = d.rename({'sigma0':'sigma0_bnds'})

            ds_sigma.append(d)

        ds_sigma = xr.merge(ds_sigma)
        
        ds = xr.merge([ds, ds_sigma])
        
        ds = ds.chunk({'j':45, 'j_c':45})
        """
        
        print("Data read complete")

        # ---------------------- Computations (point 2-3) ------------------------- #
        # create outer depth levels (required for transforming to density levels)
        level_outer_data = (cf_xarray.bounds_to_vertices(ds['lev_bnds'].isel(time=0).chunk({'lev':-1}),
                                                         'bnds').load().data)
        ds = ds.assign_coords({'level_outer': level_outer_data})
        
        # create grid object such that xgcm is aware of locations of velocity and tracer grid points
        grid = Grid(ds, coords={'Z': {'center': 'lev', 'outer': 'level_outer'},
                                'X': {'center': 'i', 'right': 'i_c'},
                                'Y': {'center': 'j', 'right': 'j_c'}}, periodic=[],)

        # compute meridional volume and heat transport on z-levels
        Meridional_Transport = transport_z(ds['vo'], ds['level_outer'], grid, 
                                           assign_name='Meridional_Transport')
        
        Heat_Transport = Compute_Heat_transport(ds['thetao'], Meridional_Transport, grid = grid, 
                                                dim = 'Y', const_multi = rho_cp)
        
        # compute meridional volume and heat transport on sigma-levels
        # conserve vertically integrated volume transport and heat transport
        sigma = xr.apply_ufunc(compute_density, ds['so'], ds['thetao'], dask='parallelized',
                               output_dtypes=[ds['thetao'].dtype])
        sigma = sigma.rename('sigma0') #.persist()
        
        Meridional_Transport_sigma = transport_sigma(sigma, Meridional_Transport, target_sigma_levels,
                                                     grid=grid, dim='Y',method='conservative')
        
        Heat_Transport_sigma = transport_sigma(sigma, Heat_Transport, target_sigma_levels,
                                               grid=grid, dim='Y',method='conservative')
        
        Meridional_Transport_sigma = Meridional_Transport_sigma #.persist()
        Heat_Transport_sigma = Heat_Transport_sigma #.persist()
        
        # Overturning computations
        dx_v = ds['dx_v'].where(ds_mask['mask_North_Atl_v'] == 0.).compute() # dx mask for North Atlantic
        
        ds_save = xr.Dataset()
        
        ds_save['latitude'] = ds['latitude_v'].where(ds_mask['mask_North_Atl_v']).mean('i').compute()
        
        ds_save['Overturning_z'] = Compute_Overturning(Meridional_Transport, dx = dx_v, dim_v = 'lev', 
                                                       dim_x = 'i', long_name="Overturning circulation vs depth")
        ds_save['Overturning_z'] = ds_save['Overturning_z'] - ds_save['Overturning_z'].isel(lev=-1) # to integrate from top to bottom
        
        ds_save['Overturning_sigma'] = Compute_Overturning(Meridional_Transport_sigma, dx = dx_v, dim_v = 'sigma0', 
                                                           dim_x = 'i', long_name="Overturning circulation vs sigma0")
        
        # Meridional Heat Transport computations
        ds_save['MHT_sigma'] = (Heat_Transport_sigma * dx_v).sum(dim='i')
        ds_save['MHT_sigma'].attrs['units'] = "Joules/s"
        ds_save['MHT_sigma'].attrs['long_name'] = "Meridional heat transport"
        
        heat_content_sigma = (Heat_Transport_sigma / Meridional_Transport_sigma)
        ds_save['MHT_overturning_sigma'] = Meridional_Heat_Transport(Meridional_Transport_sigma, heat_content_sigma, 
                                                                     dx = dx_v, dim_x = 'i',
                                                                     long_name="Meridional heat transport due to overturning circulation")
        
        ds_save['MHT_z'] = (Heat_Transport * dx_v).sum(dim='i')
        ds_save['MHT_z'].attrs['units'] = "Joules/s"
        ds_save['MHT_z'].attrs['long_name'] = "Meridional heat transport"
        
        heat_content_z = (Heat_Transport / Meridional_Transport)
        ds_save['MHT_overturning_z'] = Meridional_Heat_Transport(Meridional_Transport, heat_content_z, dx = dx_v, dim_x = 'i', 
                                                                 long_name="Meridional heat transport due to overturning circulation")
        
        # --------------------- Save data (point 4) -------------------------- #
        save_file_path = (save_path + "Overturning_Heat_Transport_"+ str(year) + "_r" + str(r+1) + ".nc")
        
        ds_save = ds_save.astype(np.float32).compute() # compute before saving
        ds_save.to_netcdf(save_file_path)
    
        print("Data saved succefully")
        
        ds_save.close()
        ds.close()
        
        #client.run(gc.collect)
        
print('Closing cluster')
#client.run_on_scheduler(stop, wait=False)
        
