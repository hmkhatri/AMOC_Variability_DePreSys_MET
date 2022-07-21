"""
Author: Hemant Khatri
Email: hkhatri@liverpool.ac.uk

The script is used for computing composite of tracer field (Depth-Lon structure) corrsponding to extreme NAO conditions. Latitude limits are provided and the code compuptes mean over latitudes to give data on Depth-Lon grid. The final output is saved in netcdf format. 

"""

# ------- load libraries ------------
import numpy as np
import xarray as xr
import scipy.stats as sc
import dask.distributed

import warnings
warnings.filterwarnings("ignore")


### ------ Functions for computations ----------

def select_subset(dataset):
    
    """Select subset of dataset in xr.open_mfdataset command
    """
    dataset = dataset.isel(i=slice(749,1199), j = slice(699, 1149)) # indices range
    dataset = dataset.drop(['vertices_latitude', 'vertices_longitude', 
                            'time_bnds']) # drop variables 
    
    return dataset

### ------ Main calculations ------------------

# -------- Define paths and read data --------------

ppdir="/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/"

ppdir_NAO="/home/users/hkhatri/DePreSys4_Data/Data_Anomaly_Compute/NAO/"

ppdir_drift="/gws/nopw/j04/snapdragon/hkhatri/Data_Drift/"
    
#save_path="/home/users/hkhatri/DePreSys4_Data/Data_Composite/time_series/"
save_path="/gws/nopw/j04/snapdragon/hkhatri/Data_Composite/NAO_hpa/" 

year1, year2 = (1960, 2017)

var_list = ['thetao'] #, 'so']

case = 'NAOp' 
#case = 'NAOn'

# -------------- Define lat-lon bands for tracer profiles ----------

ds_grid = xr.open_dataset("/home/users/hkhatri/DePreSys4_Data/Data_Consolidated/Ocean_Area_Updated.nc")

# i-j indices are for 50N - 55N and 70W - 2W
dA = ds_grid.get(['area_t', 'dy_t']).isel(y=slice(910-699, 955-699+1), 
                                          x=slice(860-749, 1150-749+1)).rename({'y':'j', 'x':'i'})

lat_lim = [50., 55.] # lat bands to average over
lon_lim = [-70., -2.] # lon bands 

# --------- NAO seasonal data -> identify high/low NAO periods -----------
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

# ----------- Create Composite of the tracer field --------------
for var in var_list:
    
    print("Running var = ", var)

    # Read drift data
    ds_drift = []

    for r in range (0,10):

        ds1 = []
        for lead_year in range(0,11):

            d = xr.open_dataset(ppdir_drift + var + "/Drift_"+ var + "_r" + 
                                str(r+1) +"_Lead_Year_" + str(lead_year+1) + ".nc",
                                chunks={'time':12}, engine='netcdf4')
            d = d.assign(time = np.arange(lead_year*12, 12*lead_year +  np.minimum(12, len(d['time'])), 1))
            ds1.append(d)

        ds1 = xr.concat(ds1, dim='time')
        ds_drift.append(ds1)

    ds_drift = xr.concat(ds_drift, dim='r')
    ds_drift = ds_drift.drop('time')

    print("Drift Data read complete")
    
    ds = []
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
                    
                    # this is for ocean vars
                    var_path = ppdir + "s" + str(year) +"-r" + str(r+1) + "i1p1f2/Omon/" + var + "/gn/latest/*.nc"
                    with xr.open_mfdataset(var_path, preprocess=select_subset, chunks={'time':12}, engine='netcdf4') as d:
                        d = d
                    
                    # i-j indices are for 50N - 55N and 70W - 2W
                    ds1 = (d[var].sel(j=slice(910, 955), i=slice(860,1150)).drop('time') - 
                           ds_drift[var].sel(j=slice(910, 955), i=slice(860,1150)).isel(r=r))

                    ds.append(ds1.isel(time = slice((int(tim_ind/4)-1)*12, (int(tim_ind/4) + 7)*12 + 5)))
                
    ds = xr.concat(ds, dim='comp')
    ds = xr.merge([ds, dA])
    
    print("Composite Data read complete")
    print("Total cases = ", len(ds['comp']), " - case ", case)
    
    # ----- Compute average over chosen latitude limits -------- 
    
    dy = ds['dy_t'].where((ds.nav_lat >= lat_lim[0]) & (ds.nav_lat <= lat_lim[1]) 
                          & (ds.nav_lon >= lon_lim[0]) & (ds.nav_lon <= lon_lim[1])).compute()

    dy = ds1.isel(time=0) * dy / ds1.isel(time=0) # to get only wet grid points
    
    lon = ((ds['nav_lon'] * ds['dy_t'].where((ds.nav_lat >= lat_lim[0]) & (ds.nav_lat <= lat_lim[1]))).sum('j') /
           (ds['dy_t'].where((ds.nav_lat >= lat_lim[0]) & (ds.nav_lat <= lat_lim[1]))).sum('j'))
    
    ds_mean = xr.Dataset()
    
    ds_mean[var] = ((ds[var] * dy).sum('j') / dy.sum('j')) # mean over chosen lat bands
    
    ds_mean['longitude'] = lon
    
    save_file_path = (save_path + "Composite_"+ case + "_" + var + "_Depth_Lon.nc")
    ds_mean = ds_mean.astype(np.float32).compute()
    ds_mean.to_netcdf(save_file_path)
    
    print("Data saved succefully")

    ds_mean.close()
    ds.close()

