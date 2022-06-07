## --------------------------------------------------------------------
# we compute NAO indices using monthly sea-level pressure (SLP) anomaly timeseries. We first substract the mean SLP data from 
# SLP values to correct for the model drift. Then, we compute NAO indices by computing SLP anomaly (normalised) differences between 
# Azores (36N-40N, 28W-20W) and Iceland (63N-70N, 25W-16W) regions (see Dunstone et al. 2016, Nature GeoSci.).
## --------------------------------------------------------------------

# ------- load libraries ------------ 
import numpy as np
import scipy as sc
import xarray as xr
import dask.distributed
from dask.distributed import Client

#from dask_mpi import initialize
#initialize()

#from dask.distributed import Client, performance_report
#client = Client()

# -------- Define paths and read data --------------

ppdir="/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/"

save_path="/home/users/hkhatri/DePreSys4_Data/Data_Anomaly_Compute/NAO/"

ppdir_drift="/gws/nopw/j04/snapdragon/hkhatri/Data_Drift/"

year1, year2 = (1960, 2017)

# Model drift data
ds_drift = []

for r in range (0,10):
    
    ds1 = []
    for lead_year in range(0,11):
    
        d = xr.open_dataset(ppdir_drift + "psl/Drift_psl_r" + str(r+1) +"_Lead_Year_" + str(lead_year+1) + ".nc")
        d = d.assign(time = np.arange(lead_year*12, 12*lead_year + np.minimum(12, len(d['time'])), 1))
        ds1.append(d)
    ds1 = xr.concat(ds1, dim='time')
    ds_drift.append(ds1)
    
ds_drift = xr.concat(ds_drift, dim='r')
ds_drift = ds_drift.drop('time')

# Read full data
ds = []
for year in range(year1, year2, 1):
    ds1 = []
    for r in range(0,10):
        var_path = "s" + str(year) +"-r" + str(r+1) + "i1p1f2/Amon/psl/gn/files/d20200417/"
        d = xr.open_mfdataset(ppdir + var_path + "*.nc")
        ds1.append(d)
    
    ds1 = xr.concat(ds1, dim='r')
    
    ds1['time_val'] = ds1['time']  
    ds1 = ds1.drop('time') # drop time coord to avoid issue with concat
    ds.append(ds1)

ds = xr.concat(ds, dim='start_year')
ds = ds.assign(start_year = np.arange(year1, year2, 1))

print('Data read complete')

# --------- Compute NAO indices and save data ----------

psl_anom = ds['psl'] - ds_drift['psl']
#psl_anom = ds['psl'] # without removing drift

save_file = save_path +"NAO_SLP_Anomaly_new.nc"
#save_file = save_path +"NAO_SLP_Raw.nc" # without removing drift


# grid cell areas for area-integration
RAD_EARTH = 6.387e6
ds['dx'] = np.mean(ds['lon'].diff('lon')) * np.cos(ds['lat'] * np.pi / 180.) * (2 * np.pi * RAD_EARTH / 360.)
ds['dy'] = np.mean(ds['lat'].diff('lat')) * (2 * np.pi * RAD_EARTH / 360.)

dA = ds['dx'] * ds['dy']
dA, tmp = xr.broadcast(dA, ds['psl'].isel(r=0, time=0, start_year=0))

# Compute NAO

P_south = ((psl_anom.sel(lat = slice(36., 40.), lon = slice(332., 340.)) * 
            dA.sel(lat = slice(36., 40.), lon = slice(332., 340.))).sum(['lat','lon']) / 
           dA.sel(lat = slice(36., 40.), lon = slice(332., 340.)).sum(['lat','lon']))

P_north = ((psl_anom.sel(lat = slice(63., 70.), lon = slice(335., 344.)) * 
            dA.sel(lat = slice(63., 70.), lon = slice(335., 344.))).sum(['lat','lon']) / 
           dA.sel(lat = slice(63., 70.), lon = slice(335., 344.)).sum(['lat','lon']))

NAO = (P_south - P_south.mean('time')) / P_south.std('time') - (P_north - P_north.mean('time')) / P_north.std('time')

# save data
NAO_save = xr.Dataset()

NAO_save['NAO'] = NAO.compute()
NAO_save['P_south'] = P_south.compute()
NAO_save['P_north'] = P_north.compute()
NAO_save['time_val'] = ds['time_val']

NAO_save.to_netcdf(save_file)