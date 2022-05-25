"""
Author: Hemant Khatri
Email: hkhatri@liverpool.ac.uk

This script computes heat budget terms using temperature and velocity data. 
1. Heat content and gradients of heat tranport terms are computed at depth levels.
2. These terms are integrated in different depth bands (0-200 m, 200-1300 m and so on) to save storage space.
3. Finally, data is saved in netcdf format.
4. Surface heat flux data, which is already available as a model output diagnostic, is not saved with these terms to save storage space.

This script makes use of xarray, dask, xgcm libraries for computations and working with netcdf files. 
The code can work in serial mode as well as in parallel (see below for details). 

For parallelization, daks-mpi (http://mpi.dask.org/en/latest/) is used to initialize multiple workers on a dask client. This is to ensure that dask is aware of multiple cores assigned through slurm batch submission job. 
With dask, chunksizes are very important. Generally, it is good to have chunksize of 10-100 MB. 
For this script, it is recomended to use chunks={'time':1, 'j':45}, for which computations are found to be very efficient.

Instead of dask-mpi, one could also use dask jobqueue (http://jobqueue.dask.org/en/latest/), which is very effective for interactive dask session on jupyter notebook or ipython console.

To run in serial, use "python file.py"
To run in parallel, use "mpirun -np NUM_Cores python file.py"
"""

# ------- load libraries ------------ 
import numpy as np
import xarray as xr
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
                            'time_bnds', 'lev_bnds']) # drop variables 
    
    return dataset

def Compute_Heat_transport(Field, Velocity, grid, dim, delta = 1., area = 1., const_multi = 1.):
    
    """Compute divergence of advective transport of field along velocity component 
    Parameters
    ----------
    Field : xarray DataArray - tracer field
    Velocity : xarray DataArray - velocity along any cartesian direction
    grid : xgcm Grid object
    dim : strig - dimension name
    delta : xarray DataArray - grid spacing across Velocity direction
    area : xarray DataArray - cell area at tracer points
    const_multi : constant - multiplier
    
    Returns
    -------
    Transport : xarray DataArray for divergence of advective transport
    """
    
    Field_vel = grid.interp(Field, [dim], boundary='extend') # interpolate field to velocity grid
    Transport = grid.diff(Field_vel * Velocity * delta, [dim], boundary='fill') / area
    
    Transport = Transport * const_multi
    
    return Transport

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
save_path="/gws/nopw/j04/snapdragon/hkhatri/Data_Heat_Budget/"

# read grid information and masking data
ds_grid = xr.open_dataset("/home/users/hkhatri/DePreSys4_Data/Data_Consolidated/Ocean_Area_Updated.nc")

year1, year2 = (1995, 2017) # range of years for reading data 
var_list = ['thetao', 'vo', 'uo', 'wo'] # list of variables to be read from model output

rho_cp = 4.09 * 1.e6 # constant from Williams et al. 2015

# Loop for going through multiple ensemble and hindcast members for computations
for r in range(8,10):
    
    for year in range(year1, year2, 1):
        
        print("Running: Ensemble = ", r+1, ", Hindcast Year = ", year)
        
        ds = []

        for var in var_list:

            var_path = (data_dir + "s" + str(year) +"-r" + str(r+1) + 
                        "i1p1f2/Omon/" + var + "/gn/latest/*.nc")

            with xr.open_mfdataset(var_path, preprocess=select_subset, # parallel=True,
                                   chunks={'time':1}, engine='netcdf4') as d:
                d = d 
            # chunksize is decided such a way that 
            # inter-worker communication is minimum and computations are efficient
                
            # renaming coords is required because variables are on different grid points (common in C-grid)
            # later xgcm.grid function is used to fix locations of these vars on model grid
            if(var == 'vo'):
                d = d.rename({'j':'j_c', 'longitude':'longitude_v', 'latitude':'latitude_v'})
            elif(var == 'uo'):
                d = d.rename({'i':'i_c', 'longitude':'longitude_u', 'latitude':'latitude_u'})
            elif(var == 'wo'):
                d = d.rename({'lev':'lev_w'})

            ds.append(d)

        ds = xr.merge(ds) # merge to get a single dataset
        
        ds = xr.merge([ds, ds_grid.rename({'y':'j', 'x':'i', 'yv':'j_c', 'xu':'i_c', 
                                           'deptht':'lev', 'depthw':'lev_w'})])
        
        print("Data Read Complete")
        
        # ---------------------- Computations (points 1) ------------------------- #
        
        # first create grid object
        grid = Grid(ds, coords={'Z': {'center': 'lev', 'right': 'lev_w'},
                                'X': {'center': 'i', 'right': 'i_c'},
                                'Y': {'center': 'j', 'right': 'j_c'}}, periodic=[],)
        
        # Heat content 
        Q = (rho_cp * ds['thetao'] * ds['dz_t'])
        
        # Divergence of Heat transport
        HT_zon = Compute_Heat_transport(ds.thetao, ds.uo, grid = grid, dim = 'X', 
                                        delta = ds.dy_u, area=ds.area_t, const_multi = rho_cp)
        HT_mer = Compute_Heat_transport(ds.thetao, ds.vo, grid = grid, dim = 'Y', 
                                        delta = ds.dx_v, area=ds.area_t, const_multi = rho_cp)
        HT_ver = Compute_Heat_transport(ds.thetao, ds.wo, grid = grid, dim = 'Z', 
                                        delta = 1., area=ds.dz_t, const_multi = rho_cp)
        
        # Integrate in depth bands (0-200, 200-1300, 0 - full depth)
        HT_horizonal = ((HT_zon + HT_mer) * ds['dz_t']) #.persist()
        HT_vertical = (HT_ver * ds['dz_t']) #.persist()
        #Q = Q.persist()
        
        ds_save = xr.Dataset()
        
        # Heat content components
        ds_save['Heat_Content_200'] = Q.isel(lev=slice(0,31)).sum('lev')
        ds_save['Heat_Content_200'].attrs['units'] = "Joules/m^2"
        ds_save['Heat_Content_200'].attrs['long_name'] = "Heat Content integrated between 0 and 210.18 m"
        
        ds_save['Heat_Content_1300'] = Q.isel(lev=slice(31,49)).sum('lev')
        ds_save['Heat_Content_1300'].attrs['units'] = "Joules/m^2"
        ds_save['Heat_Content_1300'].attrs['long_name'] = "Heat Content integrated between 210.18 and 1325.67 m"
        
        ds_save['Heat_Content'] = Q.sum('lev')
        ds_save['Heat_Content'].attrs['units'] = "Joules/m^2"
        ds_save['Heat_Content'].attrs['long_name'] = "Heat Content integrated over full depth"
        
        # Divergence of heat transport
        ds_save['Heat_Divergence_Horizontal_200'] = HT_horizonal.isel(lev=slice(0,31)).sum('lev')
        ds_save['Heat_Divergence_Horizontal_200'].attrs['units'] = "Watt/m^2"
        ds_save['Heat_Divergence_Horizontal_200'].attrs['long_name'] = "Horizontal Div(u.theta) integrated between 0 and 210.18 m"
        
        ds_save['Heat_Divergence_Horizontal_1300'] = HT_horizonal.isel(lev=slice(31,49)).sum('lev')
        ds_save['Heat_Divergence_Horizontal_1300'].attrs['units'] = "Watt/m^2"
        ds_save['Heat_Divergence_Horizontal_1300'].attrs['long_name'] = "Horizontal Div(u.theta) integrated between 210.18 and 1325.67 m"
        
        ds_save['Heat_Divergence_Horizontal'] = HT_horizonal.sum('lev')
        ds_save['Heat_Divergence_Horizontal'].attrs['units'] = "Watt/m^2"
        ds_save['Heat_Divergence_Horizontal'].attrs['long_name'] = "Horizontal Div(u.theta) integrated over full depth"
        
        ds_save['Heat_Divergence_Vertical_200'] = HT_vertical.isel(lev=slice(0,31)).sum('lev')
        ds_save['Heat_Divergence_Vertical_200'].attrs['units'] = "Watt/m^2"
        ds_save['Heat_Divergence_Vertical_200'].attrs['long_name'] = "Horizontal Div(u.theta) integrated between 0 and 210.18 m"
        
        ds_save['Heat_Divergence_Vertical_1300'] = HT_vertical.isel(lev=slice(31,49)).sum('lev')
        ds_save['Heat_Divergence_Vertical_1300'].attrs['units'] = "Watt/m^2"
        ds_save['Heat_Divergence_Vertical_1300'].attrs['long_name'] = "Horizontal Div(u.theta) integrated between 210.18 and 1325.67 m"
        
        ds_save['Heat_Divergence_Vertical'] = HT_vertical.sum('lev')
        ds_save['Heat_Divergence_Vertical'].attrs['units'] = "Watt/m^2"
        ds_save['Heat_Divergence_Vertical'].attrs['long_name'] = "Horizontal Div(u.theta) integrated over full depth"
        
        # Save data in netcdf format
        save_file_path = (save_path + "Heat_Budget_"+ str(year) + "_r" + str(r+1) + ".nc")
        
        ds_save = ds_save.astype(np.float32).compute()
        
        ds_save.to_netcdf(save_file_path)
        
        print("Data saved succefully")
        
        ds_save.close()
        ds.close()
        
        #client.run(gc.collect)
        
#print('Closing cluster')
#client.run_on_scheduler(stop, wait=False)
