"""
Author: Hemant Khatri
Email: hkhatri@liverpool.ac.uk

This script can be used to compute standard error and confidence intervals using bootstrapping method (with replacement). For this, scipy.stats.bootstrap function is used.
The script writes the output in a netcdf file that confidence intervals and standard error.

"""

# ------- load libraries ------------
import numpy as np
import xarray as xr
import scipy.stats as sc

import warnings
warnings.filterwarnings("ignore")

### ------ Functions for computations ----------

def data_bootstrap(data, stat = np.mean, cf_lev = 0.95, num_sample = 1000):
    
    """Compute bootstrap confidence intervals and standard error for data along axis =0
    Parameters
    ----------
    data : xarray DataArray for data
    stat : statisctic required for bootstrapping function, e.g. np.mean, np.std
    cf_lev : confidence level
    num_sample : Number of bootstrap samples to generate

    Returns
    -------
    bootstrap_ci : object contains float or ndarray of
        bootstrap_ci.confidence_interval : confidence intervals
        bootstrap_ci.standard_error : standard error
    """
    
    data = (data,)
    
    bootstrap_ci = sc.bootstrap(data, stat, confidence_level= cf_lev, vectorized=True, axis=0, n_resamples=num_sample,
                                random_state=1, method='BCa')
    
    return bootstrap_ci

### ------------- Main computations ------------

ppdir = "/gws/nopw/j04/snapdragon/hkhatri/Data_Composite/NAO_hpa/"
save_path = "/gws/nopw/j04/snapdragon/hkhatri/Data_Composite/NAO_hpa/Bootstrap_Confidence/"

var_list = ['mlotst', 'tos', 'hfds', 'Heat_Budget_new']

case_list = ['NAOp', 'NAOn']
tim_ind = 4 

# get days in month from actual time values
ds_NAO = xr.open_dataset("/home/users/hkhatri/DePreSys4_Data/Data_Anomaly_Compute/NAO/NAO_SLP_Anomaly_new.nc")
tim1 = ds_NAO['time_val'].isel(start_year=0,time=slice(0,101)).drop('start_year')
tim1 = tim1.astype("datetime64[ns]")

# Loop over vars

for case in case_list:
    
    print("Working for case – ", case)
    
    for var in var_list:
        
        ds = xr.open_mfdataset(ppdir + "Composite_" + case + "_" + var + ".nc", chunks={'time':1})
        
        # compute DJFM mean when NAO extremes occur
        days = tim1.dt.days_in_month.isel(time=slice(4*(tim_ind-1) + 1, 4*(tim_ind-1) + 1 + 4))
        data_var = (ds_NAOp[var].isel(time=slice(4*(tim_ind-1) + 1, 4*(tim_ind-1) + 1 + 4)) * days).sum('time')
        data_var = data_var / days.sum('time')
        
        print(var, " in progress – ", "DJFM days = ", days.sum('time').values) 
        
        # compute bootstrap confidence intervals
        bootstrap_ci = data_bootstrap(data_var, stat = np.mean, cf_lev = 0.95, num_sample = 1000)
        
        ds_save = xr.Dataset()
        ds_save[var] = data_var.mean('comp')
        ds_save['standard_error'] = xr.DataArray(data = bootstrap_ci.standard_error, dims=['j', 'i'])
        ds_save['confidence_lower'] = xr.DataArray(data = bootstrap_ci.confidence_interval[0], dims=['j', 'i'])
        ds_save['confidence_upper'] = xr.DataArray(data = bootstrap_ci.confidence_interval[1], dims=['j', 'i'])
        
        ds_save['latitude'] = ds['latitude']
        ds_save['longitude'] = ds['longitude']
        
        save_file_path = (save_path + "Bootstrap_"+ case + "_" + var + ".nc")
        ds_save = ds_save.astype(np.float32).compute()
        ds_save.to_netcdf(save_file_path)

        print("Data saved succefully")

        ds_save.close()
        ds.close()
        
        
        
        