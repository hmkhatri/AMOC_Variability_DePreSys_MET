"""
Author: Hemant Khatri
Email: hkhatri@liverpool.ac.uk

This script computes gyre heat transport at depth levels.
1. Read u, v, T data on z-level.
2. Compute Meridional heat transport due to gyres at z-levels (decompose into velocity anomaly and temperature anomaly components).
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


def Meridional_Tracer_Transport_Gyre(Transport, Field, grid, const_multi = 1., dx = 1., dz = 1., dim_x = 'X', dimen = None, long_name = None):

    """Compute Meridional Heat Transport corresponding to meridional overturning circulation
    Parameters
    ----------
    Transport : xarray DataArray - meridional velocity x layer thickness
    Field : xarray DataArray - Tracer Field
    grid : xgcm Grid object
    const_multi : constant - multiplier
    dx : xarray DataArray - Zonal grid spacing
    dz : xarray DataArray - Layer thicknesses
    dim_x : Zonal dimension name
    dimen : string - dimentional units for output
    long_name : string - long name for output

    Returns
    -------
    Meridional_Tracer_Transport : xarray DataArray - meridional tracer transport due to overturning circulation
    """

    Field_vel = Field * const_multi # correct once the code is finsihed running

    Transport_Meridional = (Transport -
                            (Transport * dx).sum(dim=dim_x) / ((dx * dz / dz).sum(dim=dim_x) + 1.e-10)) # transport anomaly from zonal-mean
    Meridional_Tracer = Field - (Field * dz * dx).sum(dim=dim_x) / ((dz * dx).sum(dim=dim_x) + 1.e-10) # anomaly from the zonal mean of tracer content
    # Field /Field is used in denominator, so only wet grid points are considered
    # + 1.e-10 is to avoid 0./0. situations

    Meridional_Tracer_Transport = (Transport_Meridional * Meridional_Tracer * dx).sum(dim=dim_x)

    Meridional_Tracer_Transport.attrs['units'] = dimen
    Meridional_Tracer_Transport.attrs['long_name'] = long_name

    return Meridional_Tracer_Transport

### ------------- Main computations ------------

# first define paths for reading data and saving final diagnostics
data_dir="/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/"

ppdir_drift="/gws/nopw/j04/snapdragon/hkhatri/Data_Drift/"

ppdir_NAO="/home/users/hkhatri/DePreSys4_Data/Data_Anomaly_Compute/NAO/"

save_path="/gws/nopw/j04/snapdragon/hkhatri/Data_sigma/Heat_Salt_Gyre_Decompose/"

# read grid information and masking data
ds_grid = xr.open_dataset("/home/users/hkhatri/DePreSys4_Data/Data_Consolidated/Ocean_Area_Updated.nc")
ds_mask = xr.open_dataset("/home/users/hkhatri/DePreSys4_Data/Data_Consolidated/Mask_UV_grid.nc")

# --------- NAO seasonal data -> identify high/low NAO periods -----------
#case = 'NAOp' 
case = 'NAOn'

ds_NAO = xr.open_dataset(ppdir_NAO + "NAO_SLP_Anomaly_new.nc")

# NAO = ds_NAO['NAO'] # for normalised NAO indices
NAO = (ds_NAO['P_south'] - ds_NAO['P_north']) # for NAO indices in pa

tim = ds_NAO['time_val'].isel(start_year=0).drop('start_year')
NAO = NAO.assign_coords(time=tim)
#NAO = NAO.chunk({'start_year':-1, 'r':1, 'time':1})

NAO = NAO.isel(time=slice(1,len(NAO.time)-1)) # get rid of first Nov and last Mar for seasonal avg
NAO_season = NAO.resample(time='QS-DEC').mean('time')

# NAO_cut = 2.5 # based on plot for individual normalised NAO values
NAO_cut = 1300. # based on plot for individual NAO values in pa

tim1 = ds_NAO['time_val'].isel(start_year=0,time=slice(0,101)).drop('start_year')
tim1 = tim1.astype("datetime64[ns]")


year1, year2 = (1960, 2017) # range of years for reading data
var_list = ['vo', 'thetao', 'so'] # list of variables to be read from model output

rho_cp = 4.09 * 1.e6 # constant from Williams et al. 2015
S_ref = 35. # Reference salinity in psu

# ------------- Read model drift data ----------------- 
ds_drift = []

for var in var_list:
    
    print("Running var = ", var)

    ds_drift1 = []

    for r in range (0,10):

        ds1 = []
        for lead_year in range(0,11):

            d = xr.open_dataset(ppdir_drift + var + "/Drift_"+ var + "_r" + 
                                str(r+1) +"_Lead_Year_" + str(lead_year+1) + ".nc",
                                chunks={'time':1, 'j':45}, engine='netcdf4')
            d = d.assign(time = np.arange(lead_year*12, 12*lead_year +  np.minimum(12, len(d['time'])), 1))
            ds1.append(d)

        ds1 = xr.concat(ds1, dim='time')

        if(var == 'vo'):
            ds1 = ds1.rename({'j':'j_c', 'longitude':'longitude_v', 'latitude':'latitude_v'})
            
        ds_drift1.append(ds1)

    ds_drift1 = xr.concat(ds_drift1, dim='r')
    ds_drift1 = ds_drift1.drop('time')

    ds_drift.append(ds_drift1)

ds_drift = xr.merge(ds_drift)

print("Drift Data read complete")

# ------------ Loop for going through multiple ensemble and hindcast members for computations ---------------
ds = []
ds_drf = []

for tim_ind in range(4,13,4):
        
    print("Running composite for - ", case, "and time index - ", tim_ind)
    
    ind_NAOp = xr.where(NAO_season.isel(time=tim_ind) >= NAO_cut, 1, 0)
    ind_NAOn = xr.where(NAO_season.isel(time=tim_ind) <= -NAO_cut, 1, 0)

    if (case == 'NAOp'):
        count_NAO = ind_NAOp
    elif (case == 'NAOn'):
        count_NAO = ind_NAOn
    else:
        print("Choose a valid case")
            
    for r in range(0,10):
        
        for year in range(year1, year2, 1):

            if(count_NAO.isel(r=r).sel(start_year=year) == 1):

                ds1 = []
                ds1_drf = []

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
        
                    ds1.append(d[var])

                ds1 = xr.merge(ds1) # merge to get a single dataset

                ds.append(ds1.isel(time = slice((int(tim_ind/4)-1)*12, (int(tim_ind/4) + 7)*12 + 5)).drop('time'))

                ds_drf.append(ds_drift.isel(r=r).isel(time = slice((int(tim_ind/4)-1)*12, (int(tim_ind/4) + 7)*12 + 5)))

ds = xr.concat(ds, dim='comp')
ds_drf = xr.concat(ds_drf, dim='comp')

ds_anom = ds - ds_drf

ds = xr.merge([ds, ds_grid['dx_v'].rename({'x':'i', 'yv':'j_c'}),
               ds_grid['dy_u'].rename({'xu':'i_c', 'y':'j'})])

print("Composite Data read complete")
print("Total cases = ", len(ds['comp']), " - case ", case)

# ---------------------- Computations (point 2-3) ------------------------- #

level_outer_data = (cf_xarray.bounds_to_vertices(d['lev_bnds'].isel(time=0).chunk({'lev':-1}),
                                                 'bnds').load().data)
ds = ds.assign_coords({'level_outer': level_outer_data})

# create grid object such that xgcm is aware of locations of velocity and tracer grid points
grid = Grid(ds, coords={'Z': {'center': 'lev', 'outer': 'level_outer'},
                        'X': {'center': 'i', 'right': 'i_c'},
                        'Y': {'center': 'j', 'right': 'j_c'}}, periodic=[],)

dz = grid.diff(ds['level_outer'], 'Z')
cell_dz = xr.ones_like(ds['vo'].isel(time=0, comp=0)) * dz
cell_dz = cell_dz * (ds['vo'].isel(time=0, comp=0) / ds['vo'].isel(time=0, comp=0)).fillna(0.) # remove values for in-land grid cells

dx_v = ds['dx_v'].where(ds_mask['mask_North_Atl_v'] == 0.).compute() # dx mask for North Atlantic

for i in range(56, 60):

    # Compute barotropic zonal-mean components of velocity and tracer fields
    [v_baro, tracer_baro] = compute_barotropic(ds_anom['vo'].isel(comp=i), ds_anom.get(['thetao', 'so']).isel(comp=i), grid, dim = 'Y', dz = cell_dz, 
                                               dx = dx_v, dim_v='lev', dim_x = 'i')
    
    [v_baro_drf, tracer_baro_drf] = compute_barotropic(ds_drf['vo'].isel(comp=i), ds_drf.get(['thetao', 'so']).isel(comp=i), grid, dim = 'Y', dz = cell_dz, 
                                                       dx = dx_v, dim_v='lev', dim_x = 'i')
    
    # compute meridional volume and heat transport on z-levels
    Meridional_Transport = transport_z(ds_anom['vo'].isel(comp=i), ds['level_outer'], grid, 
                                       assign_name='Meridional_Transport')
    
    Meridional_Transport_drf = transport_z(ds_drf['vo'].isel(comp=i), ds['level_outer'], grid, 
                                           assign_name='Meridional_Transport')
    
    ds_save = xr.Dataset()
    
    ds_save['latitude'] = ds['latitude_v'].where(ds_mask['mask_North_Atl_v']).mean('i').compute()
    
    with xr.set_options(keep_attrs=True):
    
        transport = Meridional_Transport - cell_dz * v_baro
        transport_drf = Meridional_Transport_drf - cell_dz * v_baro_drf
    
        heat = grid.interp(ds_anom['thetao'].isel(comp=i), 'Y', boundary='extend') - tracer_baro['thetao']
        heat_drf = grid.interp(ds_drf['thetao'].isel(comp=i), 'Y', boundary='extend') - tracer_baro_drf['thetao']
    
        ds_save['MHT_Gyre_Tanom_Vmean'] = Meridional_Tracer_Transport_Gyre(transport_drf, heat, grid, const_multi = rho_cp, dx = dx_v, 
                                                                           dz = cell_dz, dim_x = 'i', dimen = "Joules/s", 
                                                                           long_name = "Meridional heat transport Gyre - Temp anomalies and mean velocity")
    
        ds_save['MHT_Gyre_Tmean_Vanom'] = Meridional_Tracer_Transport_Gyre(transport, heat_drf, grid, const_multi = rho_cp, dx = dx_v, 
                                                                           dz = cell_dz, dim_x = 'i', dimen = "Joules/s", 
                                                                           long_name = "Meridional heat transport Gyre - Mean Temp and velocity anomalies")
    
        salt = grid.interp(ds_anom['so'].isel(comp=i), 'Y', boundary='extend') - tracer_baro['so']
        salt_drf = grid.interp(ds_drf['so'].isel(comp=i), 'Y', boundary='extend') - tracer_baro_drf['so']
    
        ds_save['MFT_Gyre_Sanom_Vmean'] = Meridional_Tracer_Transport_Gyre(transport_drf, salt, grid, const_multi = 1./S_ref, dx = dx_v, 
                                                                           dz = cell_dz, dim_x = 'i', dimen = "m^3/s", 
                                                                           long_name = "Meridional salt transport Gyre - Salt anomalies and mean velocity")
    
        ds_save['MFT_Gyre_Smean_Vanom'] = Meridional_Tracer_Transport_Gyre(transport, salt_drf, grid, const_multi = 1./S_ref, dx = dx_v, 
                                                                           dz = cell_dz, dim_x = 'i', dimen = "m^3/s", 
                                                                           long_name = "Meridional salt transport Gyre - Mean Salt and velocity anomalies")
    
    # -------- Save data -----------
    #ds_save = ds_save.transpose('comp', 'time', 'lev', 'j_c', 'j', 'i')
    
    save_file_path = (save_path + "Gyre_Heat_Salt_Transport_"+ case + "_comp_" + str(i) + ".nc")
    
    ds_save = ds_save.astype(np.float32).compute() # compute before saving
    ds_save.to_netcdf(save_file_path)
    
    print("Data saved succefully - member no = ", i)

ds_save.close()
ds.close()
ds_drift.close()
    












