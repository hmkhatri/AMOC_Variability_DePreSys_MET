"""
Author: Hemant Khatri
Email: hkhatri@liverpool.ac.uk

This script can be used to compute standard error and confidence intervals using bootstrapping method (with replacement). For this, scipy.stats.bootstrap function is used. The script writes the output in a netcdf file that contains confidence intervals and standard error.

This script is for overturning and meridional heat/freshwater transport diagnostics.

"""

# ------- load libraries ------------
import numpy as np
import xarray as xr
import scipy.stats as sc
import glob

import warnings
warnings.filterwarnings("ignore")

### ------ Functions for computations ----------

def Moving_mean(data, time_val, month_window):
    
    """Computes moving window using monthly-mean data
    Parameters
    ----------
    data : xarray DataArray for data
    time_val : xarray DataArray for time values
    month_window : int number of months for averaging
    
    Returns
    -------
    data_mean : xarray DataArray for smoothed data
    """
    
    for i in range(0, month_window):
        if(i == 0):
            day_count = time_val.dt.days_in_month.isel(time=slice(i, len(data['time']) - month_window + i)) # day count
            days = time_val.dt.days_in_month.isel(time=slice(i, len(data['time']) - month_window + i))
            data_mean = data.isel(time=slice(i, len(data['time']) - month_window + i)) * days

        else:
            day_count = day_count + time_val.dt.days_in_month.isel(time=slice(i, len(data['time']) - month_window + i)) # day_count
            days = time_val.dt.days_in_month.isel(time=slice(i, len(data['time']) - month_window + i))
            data_mean = data_mean + data.isel(time=slice(i, len(data['time']) - month_window + i)) * days

        
    data_mean = (data_mean / day_count).assign_coords(time=time_val.isel(time=slice(int(np.floor(month_window/2)), len(data['time']) 
                                                                                    - month_window + int(np.floor(month_window/2)))))
    
    return data_mean


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

ppdir = "/gws/nopw/j04/snapdragon/hkhatri/Data_sigma/Heat_Salt_Gyre_Decompose/"
#save_path = "/gws/nopw/j04/snapdragon/hkhatri/Data_Composite/NAO_hpa/Bootstrap_Confidence/"
save_path = "/home/users/hkhatri/tmp_data/"

ppdir_NAO = "/home/users/hkhatri/DePreSys4_Data/Data_Anomaly_Compute/NAO/"
ds_NAO = xr.open_dataset(ppdir_NAO + "NAO_SLP_Anomaly_new.nc")

tim1 = ds_NAO['time_val'].isel(start_year=0,time=slice(0,101)).drop('start_year')
tim1 = tim1.astype("datetime64[ns]")

win_month = 6 # number of months for averaging
cf_lev = 0.95 # confidence level
num_sample = 1000 # bootstrap samples to create

rho_cp = 4.09 * 1.e6 # constant from Williams et al. 2015 (forgot to multiply in the original code)
S_ref = 35. # Reference salinity in psu

case_list = ['NAOp', 'NAOn']

for case in case_list:

    print("Case running - ", case)
    
    var_list = ['MHT_Gyre_Tanom_Vmean', 'MHT_Gyre_Tmean_Vanom', 
                'MFT_Gyre_Sanom_Vmean', 'MFT_Gyre_Smean_Vanom']

    # ----------------------------
    # Read datafiles
    # ----------------------------

    files = glob.glob(ppdir + "*" + case + "*.nc")

    ds = []
    
    for file in files:

        d = xr.open_dataset(file, chunks={'time':1})

        ds.append(d)

    ds = xr.concat(ds, dim='comp')

    ds1 = ds.get(var_list).sum('lev')

    ds1['MHT_Gyre_Tanom_Vmean'] = ds1['MHT_Gyre_Tanom_Vmean'] * rho_cp
    ds1['MHT_Gyre_Tmean_Vanom'] = ds1['MHT_Gyre_Tmean_Vanom'] * rho_cp
    ds1['MFT_Gyre_Sanom_Vmean'] = ds1['MFT_Gyre_Sanom_Vmean'] / S_ref
    ds1['MFT_Gyre_Smean_Vanom'] = ds1['MFT_Gyre_Smean_Vanom'] / S_ref

    ds1 = (ds1.isel(j_c=slice(160,303))) # get values between 40N - 60N

    # ----------------------------
    # Compute moving means
    # ----------------------------
    
    ds_mean = xr.Dataset()

    for var1 in var_list:
    
        ds_mean[var1] = Moving_mean(ds1[var1], tim1, win_month)
        
    ds_mean = ds_mean.transpose('comp', 'time', 'j_c')
    
    print("Data read complete")

    # ----------------------------
    # Bootstraping 
    # ----------------------------
    
    ds_save = xr.Dataset()
    ds_save['latitude'] = ds['latitude'].isel(j_c=slice(160,303), comp=0)

    for var1 in var_list:
        
        data_var = ds_mean[var1].compute()
        bootstrap_ci = data_bootstrap(data_var.compute(), cf_lev = cf_lev, num_sample = num_sample)
        
        dim_list = list(ds_mean[var1].dims[1:])
        
        ds_save[var1] = data_var.mean('comp')
        ds_save[var1 + '_standard_error'] = xr.DataArray(data = bootstrap_ci.standard_error, dims=dim_list)
        ds_save[var1 + '_confidence_lower'] = xr.DataArray(data = bootstrap_ci.confidence_interval[0], dims=dim_list)
        ds_save[var1 + '_confidence_upper'] = xr.DataArray(data = bootstrap_ci.confidence_interval[1], dims=dim_list)

    save_file_path = (save_path + "Bootstrap_"+ case + "_Gyre_MHT_timeseries.nc")
    ds_save = ds_save.astype(np.float32).compute()
    ds_save.to_netcdf(save_file_path)

    print("Data saved succefully")

    ds_save.close()
    ds_mean.close()
    ds.close()




