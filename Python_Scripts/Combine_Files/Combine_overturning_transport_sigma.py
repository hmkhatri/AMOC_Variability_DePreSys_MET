# --------------------------------------------------------- #
# Regrid transport data on density levels to 0.5 degree grid to save space
# Additionally, ocmpute overturning in sigma space and save
# --------------------------------------------------------- #

# ------- load libraries ------------ 
import numpy as np
import scipy as sc
import xarray as xr
import dask.distributed
import xesmf as xe
from xgcm import Grid
import gc

import warnings
warnings.filterwarnings('ignore')

import dask

from dask_mpi import initialize
initialize()

from dask.distributed import Client, performance_report
client = Client()

# -------- functions --------------
async def stop(dask_scheduler):
    await gen.sleep(0.1)
    await dask_scheduler.close()
    loop = dask_scheduler.loop
    loop.add_callback(loop.stop)
    
# -------- Main Code ----------------

# read paths
ppdir="/gws/nopw/j04/snapdragon/hkhatri/Data_sigma/Transport_sigma/Temp/"
save_path="/gws/nopw/j04/snapdragon/hkhatri/Data_sigma/"

ds_mask = xr.open_dataset("/home/users/hkhatri/DePreSys4_Data/Data_Consolidated/Mask_V_grid.nc")
ds_mask = ds_mask.rename({'j':'j_c'})

ds_grid = xr.open_dataset("/home/users/hkhatri/DePreSys4_Data/Data_Consolidated/Ocean_Area_Updated.nc")

path = "/home/users/hkhatri/DePreSys4_Data/Test_Data/ONM_Monthly/"
ds_T = xr.open_dataset(path + "nemo_av830o_1m_19810301-19810401_grid-T.nc")
ds_U = xr.open_dataset(path + "nemo_av830o_1m_19810301-19810401_grid-U.nc")


# grid for regridded data"

ds_test = xr.open_dataset(ppdir+"Transport_sigma_1960_r1_time_0.nc") # just for creating model grid

grid_model = xr.Dataset()

grid_model['lat'] = xr.DataArray(ds_test['latitude_v'].values, dims=['y','x']) # these are v grid
grid_model['lon'] = xr.DataArray(ds_test['longitude_v'].values, dims=['y','x']) # these are v grid

grid_model['lat_b'] = xr.DataArray(ds_U['nav_lat'].isel(x=slice(749,1200), y = slice(700, 1151)).values, 
                                   dims=['yq','xq']) # these must be on T grid
grid_model['lon_b'] = xr.DataArray(ds_U['nav_lon'].isel(x=slice(749,1200), y = slice(700, 1151)).values, 
                                   dims=['yq','xq']) # these must be on U grid

ds_out = xr.Dataset({"lat": (["lat"], np.arange(5., 72., 0.5)),
                     "lon": (["lon"], np.arange(-95., 5., 0.5)),
                    "lat_b": (["lat_b"], np.arange(4.75, 72., 0.5)),
                    "lon_b": (["lon_b"], np.arange(-95.25, 5.25, 0.5)),})

regridder = xe.Regridder(grid_model, ds_out, "conservative_normed")

# apply regridding on full data
year1, year2 = (1962, 1963)

for year in range(year1, year2, 1):
    
    for r in range(1,11):
        
        print("Started year = ", year, ", Ensemble = ", r) 
        
        file_path = ppdir+"Transport_sigma_" + str(year) + "_r" + str(r) + "_*.nc"
        with xr.open_mfdataset(file_path, combine="nested", concat_dim='time',
                              chunks={'time':1,'sigma0':12,'sigma0_bnds':12}, parallel=True) as ds:
            ds = ds # don't change chunks as it takes only 1 min for computations now

        ds = ds.sortby(ds.time)
        
        ds = xr.merge([ds, ds_mask, ds_grid['dx_v'].rename({'x':'i', 'yv':'j_c'})])

        ds = ds.drop(['nav_lat', 'nav_lon'])
        
        print("Data Read Complete")
        
        # regrid data
        tmp = ds.rename({'latitude_v':'lat', 'longitude_v':'lon'}).drop(['mask_North_Atl', 'dx_v'])
        ds1 = regridder(tmp)
        
        # compute overturning 
        dx = ds['dx_v']
        psi = ((ds.v_transport_sigma.where((ds.mask_North_Atl==0.)) * dx)).sum(dim='i').cumsum(dim='sigma0')
        
        lat = ds['latitude_v'].where(ds.mask_North_Atl==0.).mean('i')
        psi = psi.assign_coords(j_c = lat)
        
        ds1['psi_sigma'] = psi
        
        ds1 = ds1.astype(np.float32).compute()
        
        save_file = (save_path + "Transport_sigma/Regridded/Transport_sigma_"+ 
                     str(year) + "_r" + str(r) + ".nc")
        ds1.to_netcdf(save_file)
        
        print("Data saved successfully")
        
        ds.close()
        ds1.close()
        
        gc.collect()
        
#client.close()

print('Closing cluster')
client.run_on_scheduler(stop, wait=False)
        

