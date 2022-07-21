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

#from dask_gateway import Gateway
#g = Gateway()
#options = g.cluster_options()
#options.environment = {"MALLOC_TRIM_THRESHOLD_": "0"}
#cluster = g.new_cluster(options)

#from dask_mpi import initialize
#initialize()

#from dask.distributed import Client
#client = Client()

#os.environ["MALLOC_MMAP_MAX_"]=str(40960) # to reduce memory clutter. This is temparory, no permanent solution yet.
#os.environ["MALLOC_MMAP_THRESHOLD_"]=str(16384)

#os.environ["MALLOC_TRIM_THRESHOLD_"]=str(0)

# see https://github.com/pydata/xarray/issues/2186, https://github.com/dask/dask/issues/3530

#from dask_jobqueue import SLURMCluster
#from dask.distributed import Client

#cluster = SLURMCluster(queue='par-single', cores=16, memory="64GB", walltime="20:00:00")
#cluster.adapt(minimum=2, maximum=4) # number of nodes
#client = Client(cluster)
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

def compute_barotropic(ds_vel, Field, grid, dim=None, dz = 1., dx = 1., dim_v='Z', dim_x = 'X'):
    
    """Compute zonal and depth mean velocity, and tracer field
    Parameters
    ----------
    ds_vel : xarray DataArray - velocity data
    Field : xarray DataArray/Datset - tracer field data
    grid : xgcm Grid object
    dim : dimension for interpolation to have tracel and velocity on the same grid
    dz : xarray DataArray - grid cell thickness
    dx : xarray DataArray - Zonal grid spacing
    dim_v : Vertical dimension name
    dim_x : Zonal dimension name

    Returns
    -------
    transport : xarray DataArray for volume transport
    """
    
    if(dim != None): # interpolate Field to velocity data grid
        
        Field_baro = xr.Dataset()
        for var in list(Field.keys()):
            Field_baro[var] = grid.interp(Field[var], [dim], boundary='extend')
    
    vel_baro = ((ds_vel * dz * dx).sum(dim=[dim_v, dim_x])
                / (dz * dx).sum(dim=[dim_v, dim_x]))
    
    Field_baro = ((Field_baro * dz * dx).sum(dim=[dim_v, dim_x])
                  / (dz * dx).sum(dim=[dim_v, dim_x]))
    
    return vel_baro, Field_baro

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
                                     target_data=sigma_interp, method=method, mask_edges=False)
    transport_sigma.sigma0.attrs['units'] = "kg/m^3"
    transport_sigma.sigma0.attrs['long_name'] = "Potential density with reference to ocean surface - 1000."

    return transport_sigma

def Compute_Tracer_transport(Field, Velocity, grid, dim, const_multi = 1.):

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

def Meridional_Tracer_Transport_Overturning(Transport, Field, dx = 1., dim_x = 'X', dimen = None, long_name = None):

    """Compute Meriidonal Heat Transport corresponding to meridional overturning circulation
    Parameters
    ----------
    Transport : xarray DataArray - meridional velocity x layer thickness
    Field : xarray DataArray - Tracer Field
    dx : xarray DataArray - Zonal grid spacing
    dim_x : Zonal dimension name
    dimen : string - dimentional units for output
    long_name : string - long name for output

    Returns
    -------
    Meridional_Tracer_Transport : xarray DataArray - meridional tracer transport due to overturning circulation
    """

    Transport_Meridional = (Transport * dx).sum(dim=dim_x) # first get net meridional volume transport
    Meridional_Tracer = (Field * dx).sum(dim=dim_x) / (Field * dx / Field).sum(dim=dim_x) # zonal mean of tracer content
    # Field /Field is used in denominator, so only wet grid points are considered

    Meridional_Tracer_Transport = Transport_Meridional * Meridional_Tracer

    Meridional_Tracer_Transport.attrs['units'] = dimen
    Meridional_Tracer_Transport.attrs['long_name'] = long_name

    return Meridional_Tracer_Transport

def Zonal_Mean_Depth(pot_density, depth, density_levels, grid, dx = 1., dim=None, dim_x = 'X', method='linear'):
    
    """Compute depth of density layers using xgcm.transform method
    Parameters
    ----------
    pot_density : xarray DataArray - potential density
    depth : xarray DataArray - depth at on z-levels
    density_levels : numpy array - Target density levels at which one wants to compute volume transport
    grid : xgcm Grid object
    dx : xarray DataArray - Zonal grid spacing
    dim : dimension for interpolation to have ensure pot_density and transport on the same grid
    dim_x : Zonal dimension name
    method : transform method out of 'linear' or 'conservative'

    Returns
    -------
    depth_sigma : xarray DataArray - depth levels at denisty levels
    density_depth : xarray DataArray - Mean density at fixed depth levels
    """

    if(dim != None): # interpolate the density to transport data grid
        sigma_interp = grid.interp(pot_density, [dim], boundary='extend')
    else:
        sigma_interp = pot_density

    sigma_interp = sigma_interp.rename('sigma0')
    depth_sigma = grid.transform(depth, 'Z', density_levels,
                                 target_data=sigma_interp, method=method)

    depth_sigma = (depth_sigma * dx).sum(dim=dim_x) / (depth_sigma * dx / depth_sigma).sum(dim=dim_x) # zonal mean

    depth_sigma.sigma0.attrs['units'] = "kg/m^3"
    depth_sigma.sigma0.attrs['long_name'] = "Potential density with reference to ocean surface - 1000."

    density_depth = (sigma_interp * dx).sum(dim=dim_x) / (sigma_interp * dx / 
                                                          sigma_interp).sum(dim=dim_x) # zonal mean

    return depth_sigma, density_depth

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
#save_path="/gws/nopw/j04/snapdragon/hkhatri/Data_sigma/Overturning_Heat_Transport/"
save_path="/gws/nopw/j04/snapdragon/hkhatri/Data_sigma/Overturning_Heat_Salt_Transport/"

# read grid information and masking data
ds_grid = xr.open_dataset("/home/users/hkhatri/DePreSys4_Data/Data_Consolidated/Ocean_Area_Updated.nc")
ds_mask = xr.open_dataset("/home/users/hkhatri/DePreSys4_Data/Data_Consolidated/Mask_UV_grid.nc")

year1, year2 = (1960, 2017) # range of years for reading data
var_list = ['vo', 'thetao', 'so'] # list of variables to be read from model output

datadir_sigma ="/gws/nopw/j04/snapdragon/hkhatri/Data_sigma/Transport_sigma/Temp/"
var_sigma_list = ['Zonal_Transport_sigma_', 'Meridional_Transport_sigma_', 'Depth_sigma_']

rho_cp = 4.09 * 1.e6 # constant from Williams et al. 2015
S_ref = 35. # Reference salinity in psu

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
        
        cell_dz = grid.diff(ds['level_outer'], 'Z')
        cell_dz = (cell_dz * ds['vo'] / ds['vo']).fillna(0.) # remove values for in-land grid cells

        dx_v = ds['dx_v'].where(ds_mask['mask_North_Atl_v'] == 0.).compute() # dx mask for North Atlantic

        # Compute barotropic zonal-mean components of velocity and tracer fields
        [v_baro, tracer_baro] = compute_barotropic(ds['vo'], ds.get(['thetao', 'so']), grid, dim = 'Y', dz = cell_dz, 
                                                   dx = dx_v, dim_v='lev', dim_x = 'i')

        # compute meridional volume and heat transport on z-levels
        Meridional_Transport = transport_z(ds['vo'], ds['level_outer'], grid,
                                           assign_name='Meridional_Transport')

        Heat_Transport = Compute_Tracer_transport(ds['thetao'], Meridional_Transport, grid = grid,
                                                dim = 'Y', const_multi = rho_cp)

        Salt_Transport = Compute_Tracer_transport(ds['so'], Meridional_Transport, grid = grid,
                                                dim = 'Y', const_multi = 1./S_ref)

        # compute thcknesses, meridional volume and heat transport on sigma-levels
        # conserve vertically integrated volume transport and heat transport
        sigma = xr.apply_ufunc(compute_density, ds['so'], ds['thetao'], dask='parallelized',
                               output_dtypes=[ds['thetao'].dtype])
        sigma = sigma.rename('sigma0') #.persist()

        Meridional_Transport_sigma = transport_sigma(sigma, Meridional_Transport, target_sigma_levels,
                                                     grid=grid, dim='Y',method='conservative')

        Heat_Transport_sigma = transport_sigma(sigma, Heat_Transport, target_sigma_levels,
                                               grid=grid, dim='Y',method='conservative')

        Salt_Transport_sigma = transport_sigma(sigma, Salt_Transport, target_sigma_levels,
                                               grid=grid, dim='Y',method='conservative')
        
        Thickness_sigma = transport_sigma(sigma, cell_dz, target_sigma_levels,
                                          grid=grid, dim='Y',method='conservative')

        # get tracer values on density layers using linear interpolation
        thetao_vel = grid.interp(ds['thetao'], 'Y', boundary='extend') # interpolate to velocity grid
        so_vel = grid.interp(ds['so'], 'Y', boundary='extend') # interpolate to velocity grid
        
        thetao_sigma = transport_sigma(sigma, thetao_vel, (target_sigma_levels[1:] + target_sigma_levels[:-1])*0.5,
                                       grid=grid, dim='Y', method='linear') # for overturning heat transport

        so_sigma = transport_sigma(sigma, so_vel, (target_sigma_levels[1:] + target_sigma_levels[:-1])*0.5,
                                   grid=grid, dim='Y', method='linear') # for overturning FW transport
        
        thetao_sigma = thetao_sigma.drop('sigma0') # to make the code faster
        so_sigma = so_sigma.drop('sigma0')

        #Meridional_Transport_sigma = Meridional_Transport_sigma.persist()
        #Heat_Transport_sigma = Heat_Transport_sigma.persist()

        # ------------------------------------------------- #
        # Overturning computations
        # ------------------------------------------------- #
        ds_save = xr.Dataset()

        ds_save['latitude'] = ds['latitude_v'].where(ds_mask['mask_North_Atl_v']).mean('i').compute()
        
        with xr.set_options(keep_attrs=True):
            ds_save['Overturning_z'] = Compute_Overturning(Meridional_Transport, dx = dx_v, dim_v = 'lev',
                                                           dim_x = 'i', long_name="Overturning circulation vs depth")
            
            Meridional_Transport_baro = cell_dz * v_baro
            ds_save['Overturning_z_barotropic'] = Compute_Overturning(Meridional_Transport_baro, dx = dx_v, dim_v = 'lev', dim_x = 'i', 
                                                                      long_name="Overturning circulation vs depth - Barotropic Component")

        with xr.set_options(keep_attrs=True):
            ds_save['Overturning_sigma'] = Compute_Overturning(Meridional_Transport_sigma, dx = dx_v, dim_v = 'sigma0',
                                                               dim_x = 'i', long_name="Overturning circulation vs sigma0")
            
            Meridional_Transport_sigma_baro = Thickness_sigma * v_baro
            ds_save['Overturning_sigma_barotropic'] = Compute_Overturning(Meridional_Transport_sigma_baro, dx = dx_v, dim_v = 'sigma0', dim_x = 'i', 
                                                                          long_name="Overturning circulation vs sigma0 - Barotropic Component")

        # ------------------------------------------------- #
        # Meridional Heat Transport computations
        # ------------------------------------------------- #
        ds_save['MHT_sigma'] = (Heat_Transport_sigma * dx_v).sum(dim='i')
        ds_save['MHT_sigma'].attrs['units'] = "Joules/s"
        ds_save['MHT_sigma'].attrs['long_name'] = "Meridional heat transport"
        
        ds_save['MHT_sigma_baro'] = (Thickness_sigma * v_baro * tracer_baro['thetao'] * rho_cp * dx_v).sum(dim='i')
        ds_save['MHT_sigma_baro'].attrs['units'] = "Joules/s"
        ds_save['MHT_sigma_baro'].attrs['long_name'] = "Meridional heat transport - Barotropic v and Barotropic theta"

        #heat_content_sigma = (Heat_Transport_sigma / Meridional_Transport_sigma)
        
        # Ideally method above (in comments) should work fine for computing heat content in density layers.
        # However, in some cases, denominator could be very small and 0./0. situation could give large errors.
        # Thus, we interpolate thetao to sigma levels instead.

        
        with xr.set_options(keep_attrs=True):
            heat_content_sigma = (thetao_sigma - tracer_baro['thetao']) * rho_cp
            transport = Meridional_Transport_sigma - Thickness_sigma * v_baro
            ds_save['MHT_overturning_sigma'] = Meridional_Tracer_Transport_Overturning(transport, heat_content_sigma,
                                                                           dx = dx_v, dim_x = 'i', dimen = "Joules/s",
                                                                           long_name = "Meridional heat transport due to overturning circulation - Baroclinic v and Baroclinic theta")

            heat_content_sigma = (thetao_sigma - tracer_baro['thetao']) * rho_cp
            transport = Thickness_sigma * v_baro
            ds_save['MHT_overturning_sigma_baro_v'] = Meridional_Tracer_Transport_Overturning(transport, heat_content_sigma,
                                                                                       dx = dx_v, dim_x = 'i', dimen = "Joules/s",
                                                                                       long_name = "Meridional heat transport due to overturning circulation - Barotropic v and Baroclinic theta")

            heat_content_sigma = tracer_baro['thetao'] * rho_cp
            transport = Meridional_Transport_sigma - Thickness_sigma * v_baro
            ds_save['MHT_overturning_sigma_baro_theta'] = Meridional_Tracer_Transport_Overturning(transport, heat_content_sigma,
                                                                                       dx = dx_v, dim_x = 'i', dimen = "Joules/s",
                                                                                       long_name = "Meridional heat transport due to overturning circulation - Baroclinic v and Barotropic theta")

        ds_save['MHT_z'] = (Heat_Transport * dx_v).sum(dim='i')
        ds_save['MHT_z'].attrs['units'] = "Joules/s"
        ds_save['MHT_z'].attrs['long_name'] = "Meridional heat transport"
        
        ds_save['MHT_z_baro'] = (cell_dz * v_baro * tracer_baro['thetao'] * rho_cp * dx_v).sum(dim='i')
        ds_save['MHT_z_baro'].attrs['units'] = "Joules/s"
        ds_save['MHT_z_baro'].attrs['long_name'] = "Meridional heat transport - Barotropic v and Barotropic theta"

        with xr.set_options(keep_attrs=True):
            heat_content_z = (thetao_vel - tracer_baro['thetao']) * rho_cp
            transport = Meridional_Transport - cell_dz * v_baro
            ds_save['MHT_overturning_z'] = Meridional_Tracer_Transport_Overturning(transport, heat_content_z,
                                                                                       dx = dx_v, dim_x = 'i', dimen = "Joules/s",
                                                                                       long_name = "Meridional heat transport due to overturning circulation - Baroclinic v and Baroclinic theta")

            heat_content_z = (thetao_vel - tracer_baro['thetao']) * rho_cp
            transport = cell_dz * v_baro
            ds_save['MHT_overturning_z_baro_v'] = Meridional_Tracer_Transport_Overturning(transport, heat_content_z,
                                                                                       dx = dx_v, dim_x = 'i', dimen = "Joules/s",
                                                                                       long_name = "Meridional heat transport due to overturning circulation - Barotropic v and Baroclinic theta")

            heat_content_z = tracer_baro['thetao'] * rho_cp
            transport = Meridional_Transport - cell_dz * v_baro
            ds_save['MHT_overturning_z_baro_theta'] = Meridional_Tracer_Transport_Overturning(transport, heat_content_z,
                                                                                       dx = dx_v, dim_x = 'i', dimen = "Joules/s",
                                                                                       long_name = "Meridional heat transport due to overturning circulation - Baroclinic v and Barotropic theta")
        
        # ------------------------------------------------- #
        # Meridional Salt/Freshwater Transport computations
        # ------------------------------------------------- #
        ds_save['MFT_sigma'] = - (Salt_Transport_sigma * dx_v).sum(dim='i')
        ds_save['MFT_sigma'].attrs['units'] = "m^3/s"
        ds_save['MFT_sigma'].attrs['long_name'] = "Meridional freshwater transport"
        
        ds_save['MFT_sigma_baro'] = - (Thickness_sigma * v_baro * tracer_baro['so'] * dx_v / S_ref).sum(dim='i')
        ds_save['MFT_sigma_baro'].attrs['units'] = "m^3/s"
        ds_save['MFT_sigma_baro'].attrs['long_name'] = "Meridional freshwater transport - Barotropic v and Barotropic salt"

        with xr.set_options(keep_attrs=True):
            
            salt_content_sigma =  - (so_sigma - tracer_baro['so']) / S_ref
            transport = Meridional_Transport_sigma - Thickness_sigma * v_baro
            ds_save['MFT_overturning_sigma'] = Meridional_Tracer_Transport_Overturning(transport, salt_content_sigma,
                                                                           dx = dx_v, dim_x = 'i', dimen = "m^3/s",
                                                                           long_name = "Meridional freshwater transport due to overturning circulation - Baroclinic v and Baroclinic salt")

            salt_content_sigma = - (so_sigma - tracer_baro['so']) / S_ref
            transport = Thickness_sigma * v_baro
            ds_save['MFT_overturning_sigma_baro_v'] = Meridional_Tracer_Transport_Overturning(transport, salt_content_sigma,
                                                                                       dx = dx_v, dim_x = 'i', dimen = "m^3/s",
                                                                                       long_name = "Meridional freshwater transport due to overturning circulation - Barotropic v and Baroclinic salt")

            salt_content_sigma = - tracer_baro['so'] / S_ref
            transport = Meridional_Transport_sigma - Thickness_sigma * v_baro
            ds_save['MFT_overturning_sigma_baro_so'] = Meridional_Tracer_Transport_Overturning(transport, salt_content_sigma,
                                                                                       dx = dx_v, dim_x = 'i', dimen = "m^3/s",
                                                                                       long_name = "Meridional freshwater transport due to overturning circulation - Baroclinic v and Barotropic salt")
            
        ds_save['MFT_z'] = - (Salt_Transport * dx_v).sum(dim='i')
        ds_save['MFT_z'].attrs['units'] = "m^3/s"
        ds_save['MFT_z'].attrs['long_name'] = "Meridional freshwater transport"
        
        ds_save['MFT_z_baro'] = - (cell_dz * v_baro * tracer_baro['so'] * dx_v / S_ref).sum(dim='i')
        ds_save['MFT_z_baro'].attrs['units'] = "m^3/s"
        ds_save['MFT_z_baro'].attrs['long_name'] = "Meridional freshwater transport - Barotropic v and Barotropic salt"

        with xr.set_options(keep_attrs=True):
            salt_content_z =  - (so_vel - tracer_baro['so']) / S_ref
            transport = Meridional_Transport - cell_dz * v_baro
            ds_save['MFT_overturning_z'] = Meridional_Tracer_Transport_Overturning(transport, salt_content_z,
                                                                                       dx = dx_v, dim_x = 'i', dimen = "m^3/s",
                                                                                       long_name = "Meridional freshwater transport due to overturning circulation - Baroclinic v and Baroclinic salt")

            salt_content_z =  - (so_vel - tracer_baro['so']) / S_ref
            transport = cell_dz * v_baro
            ds_save['MFT_overturning_z_baro_v'] = Meridional_Tracer_Transport_Overturning(transport, salt_content_z,
                                                                                       dx = dx_v, dim_x = 'i', dimen = "m^3/s",
                                                                                       long_name = "Meridional freshwater transport due to overturning circulation - Barotropic v and Baroclinic salt")

            salt_content_z =  - tracer_baro['so'] / S_ref
            transport = Meridional_Transport - cell_dz * v_baro
            ds_save['MFT_overturning_z_baro_so'] = Meridional_Tracer_Transport_Overturning(transport, salt_content_z,
                                                                                       dx = dx_v, dim_x = 'i', dimen = "m^3/s",
                                                                                       long_name = "Meridional freshwater transport due to overturning circulation - Baroclinic v and Barotropic salt")

        # ------------------------------------------------------------------------------------------------ #
        # --------------------- zonal mean densities at depth levels and mean depths of denisty layers ------
        # ------------------------------------------------------------------------------------------------ #
        depth = ds['lev'] * ds['vo'] / ds['vo'] # to make z a array of the same shape as vo
        [depth_sigma, density_z] = Zonal_Mean_Depth(sigma, depth, target_sigma_levels,
                                                    grid=grid, dx = dx_v, dim='Y', dim_x = 'i', method='linear')
        
        with xr.set_options(keep_attrs=True):
            ds_save['Depth_sigma'] = depth_sigma.rename({'sigma0':'sigma0_outer'})
            ds_save['Density_z'] = density_z
            
        ds_save['Depth_sigma'].attrs['units'] = "m"
        ds_save['Depth_sigma'].attrs['long_name'] = "Depth of density layer"

        ds_save['Density_z'].attrs['long_name'] = "Zonal mean density at depth levels"
        
        ds_save = ds_save.assign_coords({'lev_outer': level_outer_data})
        
        ds_save = ds_save.transpose('time','sigma0', 'sigma0_outer', 'lev', 'lev_outer','j_c')

        # --------------------- Save data (point 4) -------------------------- #
        #save_file_path = (save_path + "Overturning_Heat_Transport_"+ str(year) + "_r" + str(r+1) + ".nc")
        save_file_path = (save_path + "Overturning_Heat_Salt_Transport_"+ str(year) + "_r" + str(r+1) + ".nc")

        ds_save = ds_save.astype(np.float32).compute() # compute before saving
        ds_save.to_netcdf(save_file_path)

        print("Data saved succefully")

        ds_save.close()
        ds.close()

        #client.run(gc.collect)

print('Closing cluster')
#client.run_on_scheduler(stop, wait=False)
