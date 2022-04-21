"""
Author: Hemant Khatri
Email: hkhatri@liverpool.ac.uk

This script computes overturning circulation at different density and depth levels. 
1. Read u, v data on z-levels and transport data on density-levels.
2. Compute overturning by integrating transport in the zonal direction as well as in the direction of increasing depth (or density).
3. Finally, data is saved in netcdf format.

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
from xgcm import Grid
import gc
from tornado import gen
import os

import warnings
warnings.filterwarnings('ignore')

# -- commands in the next four lines are for using dask with "mpirun -np num_tasks python file.py" command ---- 
# ------ comment these if running as a serial code -------
from dask_mpi import initialize
initialize()

from dask.distributed import Client, performance_report
client = Client()

os.environ["MALLOC_MMAP_MAX_"]=str(40960) # to reduce memory clutter. This is temparory, no permanent solution yet.
os.environ["MALLOC_MMAP_THRESHOLD_"]=str(16384)
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

def get_latitude_masks(lat_val, yc, grid): # adopted from ecco_v4_py
    
    """Compute maskW/S which grabs vector field grid cells along specified latitude band 
    Parameters
    ----------
    lat_val : float - latitude at which to compute mask
    yc : xarray DataArray - Contains latitude values at cell centers
    grid : xgcm Grid object
    Returns
    -------
    maskWedge, maskSedge : xarray DataArray
        contains masks of latitude band at grid cell west and south grid edges
    """
    # Compute difference in X, Y direction.
    # multiply by 1 so that "True" -> 1, 2nd arg to "where" puts False -> 0
    ones = xr.ones_like(yc)
    maskC = ones.where(yc >= lat_val, 0)

    maskWedge = grid.diff(maskC, ['X'], boundary='fill')
    maskSedge = grid.diff(maskC, ['Y'], boundary='fill')

    return maskWedge, maskSedge

def meridional_overturning(ds_var, transport_u, transport_v, latitude, lat_vals, 
                           dx, dy, grid, dim='z'):
    
    """Compute meridional overturning at specified latitudes
    Parameters
    ----------
    ds_var : xarray Dataarray - To save overturning data
    transport_u : xarray DataArray - Zonal transport
    transport_v : xarray DataArray - Meridional transport
    latitude : xarray DataArray - latitude values at cell centers 
    lat_vals : Numpy array for range of lattiudes
    grid : xgcm Grid object
    dx : xarray DataArray - Zonal grid spacing at meridional velocity grid
    dy : xarray DataArray - Meridional grid spacing at zonal velocity grid
    dim : vertical dimension name
    Returns
    -------
    ds_var : xarray DataArray - meridional overturning at lat_vals
    """
    
    transport_u = (transport_u * dy).persist()
    transport_v = (transport_v * dx).persist()
    
    for lat in lat_vals:

        lat_maskW, lat_maskS = get_latitude_masks(lat, latitude, grid)
        
        lat_trsp_x = (transport_u * lat_maskW).sum(dim=['i_c','j'])
        lat_trsp_y = (transport_v * lat_maskS).sum(dim=['i','j_c'])
        
        ds_var.loc[{'lat':lat}] = lat_trsp_x + lat_trsp_y
        
    ds_var = ds_var.cumsum(dim=dim)
                    
    return ds_var

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
save_path="/gws/nopw/j04/snapdragon/hkhatri/Data_sigma/Overturning_Atlantic/"

# read grid information and masking data
ds_grid = xr.open_dataset("/home/users/hkhatri/DePreSys4_Data/Data_Consolidated/Ocean_Area_Updated.nc")
ds_mask = xr.open_dataset("/home/users/hkhatri/DePreSys4_Data/Data_Consolidated/Mask_UV_grid.nc")

year1, year2 = (1967, 1970) # range of years for reading data 
var_list = ['vo', 'uo'] # list of variables to be read from model output

datadir_sigma ="/gws/nopw/j04/snapdragon/hkhatri/Data_sigma/Transport_sigma/Temp/"
var_sigma_list = ['Zonal_Transport_sigma_', 'Meridional_Transport_sigma_', 'Depth_sigma_']

lat_range = np.arange(3., 70., 1.) # lat values at which overturning is computed

# Loop for going through multiple ensemble and hindcast members for computations
for r in range(0,1):
    
    for year in range(year1, year2, 1):
        
        print("Running: Ensemble = ", r+1, ", Hindcast Year = ", year)
        
        ds = []

        for var in var_list:

            var_path = (data_dir + "s" + str(year) +"-r" + str(r+1) + 
                        "i1p1f2/Omon/" + var + "/gn/latest/*.nc")

            with xr.open_mfdataset(var_path, parallel=True, preprocess=select_subset, 
                                   chunks={'time':1, 'lev':12}, engine='netcdf4') as d:
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
        
        #ds = ds.chunk({'j':45, 'j_c':45})
        
        print("Data read complete")

        # ---------------------- Computations (point 1) ------------------------- #
        # create outer depth levels (required for transforming to density levels)
        level_outer_data = (cf_xarray.bounds_to_vertices(ds['lev_bnds'].isel(time=0).chunk({'lev':-1}),
                                                         'bnds').load().data)
        ds = ds.assign_coords({'level_outer': level_outer_data})
        
        # create grid object such that xgcm is aware of locations of velocity and tracer grid points
        grid = Grid(ds, coords={'Z': {'center': 'lev', 'outer': 'level_outer'},
                                'X': {'center': 'i', 'right': 'i_c'},
                                'Y': {'center': 'j', 'right': 'j_c'}}, periodic=[],)

        # compute zonal and meridional transport on z-levels
        Zonal_Transport = transport_z(ds['uo'], ds['level_outer'], grid, 
                                      assign_name='Zonal_Transport')
        Meridional_Transport = transport_z(ds['vo'], ds['level_outer'], grid, 
                                           assign_name='Meridional_Transport')
        
        #Zonal_Transport = Zonal_Transport.persist()
        #Meridional_Transport = Meridional_Transport.persist()
        
        #--------------------- Overturning computations (point 2) ----------------- #
        ds_save = xr.Dataset()
        
        # first create xarray dataarray for overturning vars
        tmp_array_sigma = np.zeros((len(ds['time']), len(ds['sigma0']), len(lat_range)))
        tmp_array_z = np.zeros((len(ds['time']), len(ds['lev']), len(lat_range)))
        
        ds_save['Overturning_sigma'] = xr.DataArray(data=tmp_array_sigma.copy(), 
                                                    coords={'time':ds['time'], 'sigma0': ds['sigma0'],
                                                            'lat': lat_range}, dims=['time', 'sigma0','lat'])
        
        ds_save['Overturning_z'] = xr.DataArray(data=tmp_array_z.copy(), 
                                                coords={'time':ds['time'],'lev': ds['lev'],
                                                        'lat': lat_range}, dims=['time', 'lev','lat'])

        dy_u = ds['dy_u'].where(ds_mask['mask_North_Atl_u'] == 0.).compute() # apply North Atlantic mask in dx, dy
        dx_v = ds['dx_v'].where(ds_mask['mask_North_Atl_v'] == 0.).compute() # thus, mask will be applied to v*dx and u*dy
        
        ds_save['Overturning_sigma'] = meridional_overturning(ds_save['Overturning_sigma'], 
                                                              ds['Zonal_Transport_sigma'], ds['Meridional_Transport_sigma'], ds['latitude'], 
                                                              lat_range, dx_v, dy_u, grid, dim='sigma0')
        print("Overturning-sigma calculation completed")
        
        ds_save['Overturning_z'] = meridional_overturning(ds_save['Overturning_z'], Zonal_Transport,
                                                          Meridional_Transport, ds['latitude'], lat_range,
                                                          dx_v, dy_u, grid, dim='lev')
        ds_save['Overturning_z'] = ds_save['Overturning_z'] - ds_save['Overturning_z'].isel(lev=-1) # to integrate from top to bottom
        print("Overturning-z calculation completed")
        
        # --------------------- Save data (point 3) -------------------------- #
        save_file_path = (save_path + "Overturning_"+ str(year) + "_r" + str(r+1) + ".nc")
        
        ds_save['Overturning_sigma'].attrs['units'] = "m^3/s"
        ds_save['Overturning_z'].attrs['units'] = "m^3/s"
        
        ds_save['Overturning_sigma'].attrs['long_name'] = "Overturning circulation vs sigma0"
        ds_save['Overturning_z'].attrs['long_name'] = "Overturning circulation vs depth"
        
        ds_save = ds_save.compute() # compute before saving
        ds_save.to_netcdf(save_file_path)
    
        print("Data saved succefully")
        
        ds_save.close()
        ds.close()
        
        #client.run(gc.collect)
        
print('Closing cluster')
client.run_on_scheduler(stop, wait=False)
        