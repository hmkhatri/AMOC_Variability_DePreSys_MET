## --------------------------------------------------------------------
# This script for saving anomaly data for high and low NAO phase periods. 
# There are two options
# Option one - Save data for all members for the specfic season with extreme NAO indices 
# Option two - Save data time-series for members having extreme NAO indices (averaged for all members)
# Newest code - save timeseries data for all selected members 
## --------------------------------------------------------------------

# load libraries
import numpy as np
import scipy as sc
import xarray as xr
from distributed import Client
from dask import delayed
from dask import compute
#from dask.diagnostics import ProgressBar

import warnings
warnings.filterwarnings('ignore')

#from dask_mpi import initialize

#initialize()
#client = Client()

#os.environ["MALLOC_MMAP_MAX_"]=str(40960) # to reduce memory clutter. This is temparory, no permanent solution yet.
#os.environ["MALLOC_MMAP_THRESHOLD_"]=str(16384)
#os.environ["MALLOC_TRIM_THRESHOLD_"]=str(0)

### ------ Functions for computations ----------

def select_subset(dataset):
    
    """Select subset of dataset in xr.open_mfdataset command
    """
    dataset = dataset.isel(i=slice(749,1199), j = slice(699, 1149)) # indices range
    dataset = dataset.drop(['vertices_latitude', 'vertices_longitude', 
                            'time_bnds', 'lev_bnds']) # drop variables 
    
    return dataset

### ------ Main calculations ------------------

# -------- Define paths and read data --------------

ppdir="/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/"

ppdir_NAO="/home/users/hkhatri/DePreSys4_Data/Data_Anomaly_Compute/NAO/"

ppdir_drift="/gws/nopw/j04/snapdragon/hkhatri/Data_Drift/"
    
#save_path="/home/users/hkhatri/DePreSys4_Data/Data_Composite/time_series/"
save_path="/gws/nopw/j04/snapdragon/hkhatri/Data_Composite/NAO_hpa/" 

year1, year2 = (1960, 2017)

var_list = ['vo'] # ocean vars

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

case = 'NAOp' 
#case = 'NAOn'

for var in var_list:
    
    print("Running var = ", var)

    # Read drift data
    ds_drift = []

    for r in range (0,10):

        ds1 = []
        for lead_year in range(0,11):

            d = xr.open_dataset(ppdir_drift + var + "/Drift_"+ var + "_r" + 
                                str(r+1) +"_Lead_Year_" + str(lead_year+1) + ".nc",
                                chunks={'time':1}, engine='netcdf4')
            d = d.assign(time = np.arange(lead_year*12, 12*lead_year +  np.minimum(12, len(d['time'])), 1))
            ds1.append(d)

        ds1 = xr.concat(ds1, dim='time')
        ds_drift.append(ds1)

    ds_drift = xr.concat(ds_drift, dim='r')
    ds_drift = ds_drift.drop('time')
    #ds_drift = ds_drift.chunk({'time':1})

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
                    with xr.open_mfdataset(var_path, preprocess=select_subset, chunks={'time':1}, engine='netcdf4') as d:
                        d = d
                    
                    d = d[var].drop('time') - ds_drift[var].isel(r=r)

                    ds.append(d.isel(time = slice((int(tim_ind/4)-1)*12, (int(tim_ind/4) + 7)*12 + 5)))
                
    ds = xr.concat(ds, dim='comp')
    
    print("Composite Data read complete")
    print("Total cases = ", len(ds['comp']), " - case ", case)

    # --------- Compute barotropic gyre transport anomaly ----------- #

    ds_grid = xr.open_dataset("/home/users/hkhatri/DePreSys4_Data/Data_Consolidated/Ocean_Area_Updated.nc")

    ds_mask = xr.open_dataset("/home/users/hkhatri/DePreSys4_Data/Data_Consolidated/Mask_UV_grid.nc")

    ds1 = xr.merge([ds.rename({'j':'j_c'}), ds_grid.rename({'y':'j', 'x':'i', 'yv':'j_c', 'xu':'i_c', 'deptht':'lev'}).get(['dx_v', 'dz_t']),
                   ds_mask.drop(['nav_lat', 'nav_lon'])])

    vo_intz = (ds1['vo'] * ds1['dz_t']).sum('lev')

    dx = ds1['dx_v'].where(ds1['mask_North_Atl_v'] == 0.)

    psi = (vo_intz * dx).cumsum(dim = 'i')
    psi.attrs['units'] = "m^3/s"

    comp_save = psi.to_dataset(name='Psi_Gyre')
    comp_save = comp_save.rename({'j_c':'j'})
    
    comp_save = (comp_save.sel(j=slice(780, 1100),i=slice(810,1170))).astype(np.float32).compute()
    
    save_file = save_path + "Composite_" + case + "_Gyre.nc"
    comp_save.to_netcdf(save_file)
    print("Data saved successfully")
