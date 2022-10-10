"""
Author: Hemant Khatri
Email: hkhatri@liverpool.ac.uk

This script can be used to compute standard error and confidence intervals using bootstrapping method (with replacement). 
For this, scipy.stats.bootstrap function is used.
The output is saved in a netcdf file that contains confidence intervals and standard error.

This script is computes confidence intervals for 6-month averaged anomlies in oceanic diagnostics. 
Timeseries of anomalies in NDJFMA and MJJASO months are evaluated for confidence intervals.

"""

# ------- load libraries ------------
import numpy as np
import xarray as xr
import xskillscore as xs
import scipy.stats as sc

import warnings
warnings.filterwarnings("ignore")


def annaul_mean_data(ds, var_name, num_days, method = 'mean'):
    
    """Compute annual mean of data for bootstrapping
    Means are computed for NDJFMA and MJJASO months in year = -1, 0, 1, 2, 3, 4, 5, 6, 7
    Parameters
    ----------
    ds : xarray Dataset for data variables
    var_name : list of avariables for computing 6-month means
    num_days : Number of days in months
    method : Options - compute 'mean', 'integrate', 'difference' over time 
    
    Returns
    -------
    ds_annual : Dataset containting 6-month means
    """
    
    ds_annual = xr.Dataset()
    
    for var1 in var_name:
        
        data_var1 = []
        
        data_ref = ds[var1].isel(time = slice(0, 6))
        days = num_days.dt.days_in_month.isel(time = slice(0, 6))
        data_ref = ((data_ref * days).sum('time')/ days.sum('time'))
        
        for i in range(0,16): # len(time) = 101 months, so we have 16 6-month invervals


            days = num_days.dt.days_in_month.isel(time = slice(6*i, 6*i + 6))
            data_var = ds[var1].isel(time = slice(6*i, 6*i + 6))
            
            if(method == 'mean'):
                data_var = ((data_var * days).sum('time')/ days.sum('time')) - data_ref # remove first 6-month for better ref.
            elif(method == 'integrate'):
                data_var = ((data_var * days).sum('time') * 3600. * 24.)
            elif(method == 'difference'):
                data_var = (data_var.isel(time=-1) - data_var.isel(time=0))
            else:
                print("Method is not valid")
            
            data_var1.append(data_var)
            
        ds_annual[var1] = xr.concat(data_var1, dim='year')
    
    ds_annual = ds_annual.chunk({'year':-1})
        
    return ds_annual
            

def data_bootstrap(data, cf_lev = 0.95, num_sample = 1000):
    
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
    
    bootstrap_ci = sc.bootstrap(data, statistic=np.mean, confidence_level=cf_lev, vectorized=True, axis=0, n_resamples=num_sample,
                                random_state=1, method='BCa')
    
    return bootstrap_ci

### ------------- Main computations ------------

ppdir = "/gws/nopw/j04/snapdragon/hkhatri/Data_Composite/NAO_hpa/"
save_path = "/gws/nopw/j04/snapdragon/hkhatri/Data_Composite/NAO_hpa/Bootstrap_Confidence/"

var_list = ['tos', 'hfds']

case_list = ['NAOp', 'NAOn']
cf_lev = 0.8 # confidence level
num_sample = 1000 # bootstrap samples to create

# get days in month from actual time values
ds_NAO = xr.open_dataset("/home/users/hkhatri/DePreSys4_Data/Data_Anomaly_Compute/NAO/NAO_SLP_Anomaly_new.nc")
tim1 = ds_NAO['time_val'].isel(start_year=0,time=slice(0,101)).drop('start_year')
tim1 = tim1.astype("datetime64[ns]")

# Loop over vars

for case in case_list:
    
    print("Working for case – ", case)
    
    for var in var_list:
        
        ds = xr.open_dataset(ppdir + "Composite_" + case + "_" + var + ".nc", chunks={'time':101})
        
        #ds = ds - ds.isel(time=0) # to remove time=0 signal for better interpretation
        
        var_name = [var]
        
        ds_annual = annaul_mean_data(ds, var_name, tim1, method = 'mean')
        
        
        # compute bootstrap confidece intervals
        ds_save = xr.Dataset()
        
        sde_var1 = []
        cfd_up_var1 = []
        cfd_low_var1 = []
        dim_list = ['j', 'i']
        
        for yr in range(0, len(ds_annual['year'])):
            
            data_var = ds_annual[var].isel(year=yr).compute()
            
            bootstrap_ci = data_bootstrap(data_var, cf_lev = cf_lev, num_sample = num_sample)
                
            sde = xr.DataArray(data = bootstrap_ci.standard_error, dims=dim_list)
            sde_var1.append(sde)
                
            cfd_up = xr.DataArray(data = bootstrap_ci.confidence_interval[1], dims=dim_list)
            cfd_up_var1.append(cfd_up)
                
            cfd_low = xr.DataArray(data = bootstrap_ci.confidence_interval[0], dims=dim_list) 
            cfd_low_var1.append(cfd_low)
            
        ds_save[var] = ds_annual[var].mean('comp')
        ds_save[var + '_standard_error'] = xr.concat(sde_var1, dim='year') 
        ds_save[var + '_confidence_lower'] = xr.concat(cfd_low_var1, dim='year') 
        ds_save[var + '_confidence_upper'] = xr.concat(cfd_up_var1, dim='year')
        
        # save data
        ds_save = ds_save.assign(year = np.arange(-1, 7, 0.5))
        ds_save['year'].attrs['long_name'] = "6-month means - NDJFMA, MJJASO, .."
        
        save_file_path = (save_path + "Bootstrap_"+ case + "_" + var + "_6month.nc")
        ds_save = ds_save.astype(np.float32).compute()
        ds_save.to_netcdf(save_file_path)

        print("Data saved succefully")

        ds_save.close()
        ds_annual.close()
        ds.close()
