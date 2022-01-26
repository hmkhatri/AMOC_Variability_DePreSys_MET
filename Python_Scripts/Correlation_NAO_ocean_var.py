## --------------------------------------------------------------------
# This notebook computes correlation between seasonal NAO anomalies and ocean surface variable anomalies across all hindcasts.
# The spatial patterns of these correlations can then be averaged over all ensemble members but lead-year information is retained.# In summary, we corrlation pattern (time,x,y) and time is number of seasons in each hindcast.
# In addition, standard deviation is saved as well.
## --------------------------------------------------------------------

# load libraries
import numpy as np
import scipy as sc
import xarray as xr
from dask.distributed import Client, LocalCluster
from dask import delayed
from dask import compute
from dask.diagnostics import ProgressBar

#from dask_mpi import initialize

#initialize()

#if __name__ == '__main__':
     
    #initialize()
    #client = Client()
    #print(client)
    
# ------- Functions -----
def corr_NAO(NAO_ind, ds, dim='time'):

    #ds_season = ds.resample(time='QS-DEC').mean('time')
    corr = xr.corr(NAO_ind, ds, dim=dim)

    return corr
    
def resample_season(ds):
        
    ds_season = ds.resample(time='QS-DEC').mean('time')
        
    return ds_season


# -------- Define paths and read data --------------

ppdir="/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/"

ppdir_NAO="/home/users/hkhatri/DePreSys4_Data/Data_Anomaly_Compute/"

ppdir_drift="/home/users/hkhatri/DePreSys4_Data/Data_Drift_Removal/Drift_1970_2016_Method_DCPP/"
    
save_path="/home/users/hkhatri/DePreSys4_Data/Data_Correlation/"

year1, year2 = (1960, 2017)

var_list = ['mlotst'] #'tos', 'sos', 'hfds'

# NAO data and create seasonal data
ds_NAO = xr.open_dataset(ppdir_NAO + "NAO_SLP_Anomaly.nc")

NAO = ds_NAO['NAO']
tim = ds_NAO['time_val'].isel(start_year=0).drop('start_year')
NAO = NAO.assign_coords(time=tim)
NAO = NAO.chunk({'start_year':-1, 'r':1, 'time':1})

NAO = NAO.isel(time=slice(1,len(NAO.time)-1)) # get rid of first Nov and last Mar for better seasonal avg
NAO = NAO.resample(time='QS-DEC').mean('time')
    
for var in var_list:

    # Read drift data
    ds_drift = []

    for r in range (0,10):

        ds1 = []
        for lead_year in range(0,11):

            d = xr.open_dataset(ppdir_drift + var + "/Drift_"+ var + "_r" + 
                                str(r+1) +"_Lead_Year_" + str(lead_year+1) + ".nc")
            d = d.assign(time = np.arange(lead_year*12, 12*lead_year + np.minimum(12, len(d['time'])), 1))
            ds1.append(d)
                
        ds1 = xr.concat(ds1, dim='time')
        ds_drift.append(ds1)

    ds_drift = xr.concat(ds_drift, dim='r')
    ds_drift = ds_drift.drop('time')
    ds_drift = ds_drift.chunk({'r':1, 'time':1, 'j':50, 'i':50})

    print("Drift Data read complete")

    # Read full data
    for r in range(2,10):

        print("Var = ", var, "; Ensemble = ", r)

        ds = []

        for year in range(year1, year2, 1):

            var_path = "s" + str(year) +"-r" + str(r+1) + "i1p1f2/Omon/" + var + "/gn/latest/"

            d = xr.open_mfdataset(ppdir + var_path + "*.nc")
            d = d.drop(['time', 'vertices_latitude', 'vertices_longitude', 'time_bnds'])

            ds.append(d)

        # combine data for hindcasts
        ds = xr.concat(ds, dim='start_year')
        ds = ds.isel(i=slice(749,1199), j = slice(699, 1149))
        ds = ds.chunk({'start_year':-1, 'time':1, 'j':50, 'i':50})
        print("Data read complete")

        # seasonal means and correlations
        tmp_var = ds[var] - ds_drift[var].isel(r=r)

        tmp_var = tmp_var.assign_coords(time=tim) # required for seasonal calculations
        tmp_var = tmp_var.isel(time=slice(1,len(ds.time)-1))
        #tmp_var = tmp_var.resample(time='QS-DEC').mean('time')
        #tmp_var = tmp_var.chunk({'start_year':-1, 'time':-1, 'j':25, 'i':25})
            
        ds_season = delayed(resample_season)(tmp_var)

        corr = delayed(corr_NAO)(NAO.isel(r=r), ds_season, dim='start_year')
            
        #with ProgressBar():
        corr = corr.compute()

        # Save data
        corr_save = xr.Dataset()

        corr_save[var] = corr
        corr_save[var].attrs['standard_name'] = ["Corelations with seasonal mean NAO indices"]
        save_file = save_path + "Correlation_NAO_" + var + "_r" + str(r+1) + ".nc"
        corr_save.to_netcdf(save_file)

        print("Correlation data saved succefully")
