## --------------------------------------------------------------------
# This scripts compute meridional transport at fixed sigma levels 
# Potential densities are computed using monthly T and S at wrt p = 1 bar. 
# xgcm transform funtionality is used for computing transform at specific density levels
# Moreoever, depths of these density isolines are also computed.
#
# very slow with dask-mpi. Effectively no speed up from serial code.
# probably because chunks need to separated in (x, y, time) and there is a lot of inter-worker communication.
## --------------------------------------------------------------------

# ------- load libraries ------------ 
import numpy as np
import scipy as sc
import xarray as xr
import dask.distributed
import cf_xarray
import gsw as gsw
from xgcm import Grid
import gc
from tornado import gen

import warnings
warnings.filterwarnings('ignore')

import dask
from dask_mpi import initialize
initialize()

from dask.distributed import Client, performance_report
client = Client()
#print(client)

### ------ Function to compute potential density and transform to density layers ----------

def pdens(S,theta):

    pot_dens = gsw.density.sigma0(S, theta)

    return pot_dens

def select_subset(d1):
    
    d1 = d1.isel(i=slice(749,1199), j = slice(699, 1149))
    d1 = d1.drop(['vertices_latitude', 'vertices_longitude', 'time_bnds'])
    
    return d1

async def stop(dask_scheduler):
    await gen.sleep(0.1)
    await dask_scheduler.close()
    loop = dask_scheduler.loop
    loop.add_callback(loop.stop)

### ------------- Main computations ------------

ppdir="/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/"

# for monthly drift
save_path="/gws/nopw/j04/snapdragon/hkhatri/Data_sigma/Transport_sigma/Temp/"

year1, year2 = (1962, 1963)
var_list = ['thetao', 'so', 'vo']

# define sigma levels for transform
a = np.arange(15., 20., 2.0)
b = np.arange(20., 23.1, 1.)
c = np.arange(23.2, 26., 0.2)
d = np.arange(26.1, 28., 0.1)
e = np.arange(28.0, 28.8, 0.2)
f = np.arange(29.0, 31.1, 1.)

# compute transport and depth at sigma levels
target_sigma_levels = np.concatenate((a ,b, c, d, e, f))

for r in range(3,10):
    
    for year in range(year1, year2, 1):
        
        print("Running: Ensemble = ", r+1, ", Year = ", year)
        
        ds = []

        for var in var_list:

            var_path = (ppdir + "s" + str(year) +"-r" + str(r+1) + 
                        "i1p1f2/Omon/" + var + "/gn/latest/*.nc")

            #d = xr.open_mfdataset(var_path, parallel=True) #chunks={'time':1, 'j':90}, 
            with xr.open_mfdataset(var_path, parallel=True, preprocess=select_subset, 
                                   chunks={'time':1, 'j':45}, engine='netcdf4') as d:
                d = d # don't change chunks as it takes only 1 min for computations 

            #d = d.isel(i=slice(749,1199), j = slice(699, 1149)) # try with smaller set
            #d = d.drop(['vertices_latitude', 'vertices_longitude', 'time_bnds', 'i', 'j'])

            if(var == 'vo'):
                d = d.rename({'j':'j_c', 'longitude':'longitude_v', 'latitude':'latitude_v'})

            ds.append(d)

        ds = xr.merge(ds)

        # create outer depth levels
        #level_outer_data = (cf_xarray.bounds_to_vertices(ds.lev_bnds.isel(time=0),
        #                                                 'bnds').load().data)
        #level_outer_data = (cf_xarray.bounds_to_vertices(ds.lev_bnds,
        #                                                'bnds').load().data)
        #ds = ds.assign_coords({'level_outer': level_outer_data})

        #ds = ds.drop('lev_bnds')

        #ds = ds.chunk({'time':1})
        #ds = ds.chunk({'time':1, 'i':225, 'j':225, 'j_c':225})

        print("Data read complete")
            
        # put a loop for months
        for mon in range(0,5):
            
            # extract individual time steps
            #ds1 = ds.isel(time = mon)
            ds1 = ds.isel(time = slice(mon*25, mon*25+25))
            
            # create outer depth levels
            level_outer_data = (cf_xarray.bounds_to_vertices(ds.lev_bnds.isel(time=0),'bnds').load().data)
            ds1 = ds1.assign_coords({'level_outer': level_outer_data})
            
            # create grid using xgcm
            grid = Grid(ds1, coords={'Z': {'center': 'lev', 'outer': 'level_outer'},
                                     'Y': {'center': 'j', 'right': 'j_c'}}, periodic=[],)

            # compute meridional transport
            thickness = grid.diff(ds1.level_outer, 'Z')
            v_transport =  ds1.vo * thickness
            v_transport = v_transport.fillna(0.).rename('v_transport')

            # compute potential density
            sigma = xr.apply_ufunc(pdens, ds1.so, ds1.thetao, dask='parallelized',
                                   output_dtypes=[ds1.thetao.dtype])

            ds1['sigma0'] = grid.interp(sigma, ['Y'], boundary='extend')

            v_transport_sigma = grid.transform(v_transport, 'Z', target_sigma_levels,
                                               target_data=ds1.sigma0, method='conservative')

            depth = xr.ones_like(v_transport) * v_transport['lev']

            depth_sigma = grid.transform(depth, 'Z', target_sigma_levels,
                                         target_data=ds1.sigma0, method='linear')

            # save data

            ds_save = xr.Dataset()

            ds_save['v_transport_sigma'] = v_transport_sigma.astype(np.float32)
            ds_save['depth_sigma'] =(depth_sigma.rename({'sigma0':
                                                        'sigma0_bnds'})).astype(np.float32)

            #ds_save = ds_save.transpose('sigma0','sigma0_bnds','j_c','i')
            ds_save = ds_save.transpose('time', 'sigma0','sigma0_bnds','j_c','i')

            ds_save = ds_save.compute() #client.persist(ds_save).results()

            save_file = (save_path + "Transport_sigma_"+ str(year) + "_r" + str(r+1) 
                         + "_time_" + str(mon) + ".nc")

            ds_save.to_netcdf(save_file) #, compute = False)

            # note - persist() does not seem to work, only metadata is saved

            print("Computations completed and file saved succesfully")

            del depth_sigma, depth, v_transport_sigma, v_transport
            ds_save.close()
            ds1.close()
            
            #client.cancel([depth_sigma, depth, v_transport_sigma, v_transport])
            #client.cancel([ds1, ds_save])
            #client.run(gc.collect) # this doesn't help
            
        ds.close()
            
        #client.cancel(ds)
        
#client.close()
print('Closing cluster')
client.run_on_scheduler(stop, wait=False)
        
        
        
