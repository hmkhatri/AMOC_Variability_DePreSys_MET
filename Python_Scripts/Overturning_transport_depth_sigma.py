## --------------------------------------------------------------------
# This scripts compute meridional transport at fixed sigma levels 
# Potential densities are computed using monthly T and S at wrt p = 1 bar. 
# xgcm transform funtionality is used for computing transform at specific density levels
# Moreoever, depths of these density isolines are also computed.
## --------------------------------------------------------------------

# ------- load libraries ------------ 
import numpy as np
import scipy as sc
import xarray as xr
import dask.distributed
import cf_xarray
import gsw as gsw
from xgcm import Grid

import warnings
warnings.filterwarnings('ignore')

from dask_mpi import initialize
initialize()

from dask.distributed import Client, performance_report
client = Client()

### ------ Function to compute potential density and transform to density layers ----------

def pdens(S,theta):

    pot_dens = gsw.density.sigma0(S, theta)

    return pot_dens

### ------------- Main computations ------------

ppdir="/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/"

# for monthly drift
save_path="/gws/nopw/j04/snapdragon/hkhatri/Data_sigma/Transport_sigma/"

year1, year2 = (1960, 1961)
var_list = ['thetao', 'so', 'vo']

for r in range(0,10):
    
    for year in range(year1, year2, 1):
        
        print("Running: Ensemle = ", r+1, ", Year = ", year)
        
        ds = []
        
        for var in var_list:
        
            var_path = "s" + str(year) +"-r" + str(r+1) + "i1p1f2/Omon/" + var + "/gn/latest/"
        
            d = xr.open_mfdataset(ppdir + var_path + "*.nc", parallel=True)
            
            d = d.isel(i=slice(749,1199), j = slice(699, 1149))
            
            d = d.drop(['vertices_latitude', 'vertices_longitude', 'time_bnds', 'i', 'j'])
            
            if(var == 'vo'):
                d = d.rename({'j':'j_c', 'longitude':'longitude_v', 'latitude':'latitude_v'})
                
            ds.append(d)
            
        ds = xr.merge(ds)
        
        # create outer depth levels
        level_outer_data = (cf_xarray.bounds_to_vertices(ds.lev_bnds.isel(time=0),
                                                         'bnds').load().data)
        ds = ds.assign_coords({'level_outer': level_outer_data})
        
        ds = ds.drop('lev_bnds')
        
        ds = ds.chunk({'time':1})
        
        print("Data read complete")
        
        # create grid using xgcm
        grid = Grid(ds, coords={'Z': {'center': 'lev', 'outer': 'level_outer'},
                                'Y': {'center': 'j', 'right': 'j_c'}}, periodic=[],)
        
        # compute meridional transport
        thickness = grid.diff(ds.level_outer, 'Z')
        v_transport =  ds.vo * thickness
        v_transport = v_transport.fillna(0.).rename('v_transport')
        
        # compute potential density
        sigma = xr.apply_ufunc(pdens, ds.so, ds.thetao, dask='parallelized',
                               output_dtypes=[ds.thetao.dtype])

        ds['sigma0'] = grid.interp(sigma, ['Y'], boundary='extend')
        
        # define sigma levels for transform
        a = np.arange(15., 20., 2.0)
        b = np.arange(20., 23.1, 1.)
        c = np.arange(23.2, 26., 0.2)
        d = np.arange(26.1, 28., 0.1)
        e = np.arange(28.0, 28.8, 0.2)
        f = np.arange(29.0, 31.1, 1.)
        
        # compute transport and depth at sigma levels
        target_sigma_levels = np.concatenate((a ,b, c, d, e, f))
        
        v_transport_sigma = grid.transform(v_transport, 'Z', target_sigma_levels,
                                           target_data=ds.sigma0, method='conservative')
        
        depth = xr.ones_like(v_transport) * v_transport['lev']
        
        depth_sigma = grid.transform(depth, 'Z', target_sigma_levels,
                                     target_data=ds.sigma0, method='linear')
        
        # save data
        
        ds_save = xr.Dataset()
        
        ds_save['v_transport_sigma'] = v_transport_sigma.astype(np.float32)
        ds_save['depth_sigma'] =(depth_sigma.rename({'sigma0':
                                                    'sigma0_bnds'})).astype(np.float32)

        ds_save = ds_save.transpose('time','sigma0','sigma0_bnds','j_c','i')
        
        save_file = save_path + "Transport_sigma_"+ str(year) + "_r" + str(r+1) + ".nc"
                     
        delayed_obj = ds_save.to_netcdf(save_file, compute = False)
        
        delayed_obj.persist()
        
        print("Computations completed and file saved succesfully")
        
        ds_save.close()
        ds.close()
        
client.close()
        
        
        