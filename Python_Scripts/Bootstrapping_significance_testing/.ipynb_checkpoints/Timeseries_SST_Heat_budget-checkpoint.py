"""
Author: Hemant Khatri
Email: hkhatri@liverpool.ac.uk

This script can be used to compute standard error and confidence intervals using bootstrapping method (with replacement). For this, scipy.stats.bootstrap function is used.
The script writes the output in a netcdf file that contains confidence intervals and standard error.

"""

# ------- load libraries ------------
import numpy as np
import xarray as xr
import scipy.stats as sc

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
            data_mean = data.isel(time=slice(i, len(data['time']) - month_window + i)).drop('time') * days

        else:
            day_count = day_count + time_val.dt.days_in_month.isel(time=slice(i, len(data['time']) - month_window + i)) # day_count
            days = time_val.dt.days_in_month.isel(time=slice(i, len(data['time']) - month_window + i))
            data_mean = data_mean + data.isel(time=slice(i, len(data['time']) - month_window + i)).drop('time') * days

        
    data_mean = (data_mean / day_count).assign_coords(time=time_val.isel(time=slice(int(np.floor(month_window/2)), len(data['time']) 
                                                                                    - month_window + int(np.floor(month_window/2)))))
    
    return data_mean

def Data_time_tendency(data, time_val, month_window):
    """Computes time-derivative using a timeseries
    Parameters
    ----------
    data : xarray DataArray for data
    time_val : xarray DataArray for time values
    month_window : int number of months for computing time derivative
    
    Returns
    -------
    data_dt : xarray DataArray for d(data) / dt
    """
    
    for i in range(0, win_month):
        if(i == 0):
            day_count = time_val.dt.days_in_month.isel(time=slice(i, len(data['time']) - month_window + i)) * 0.5 # day count
        elif(i < win_month-1):
            day_count = day_count + time_val.dt.days_in_month.isel(time=slice(i, len(data['time']) - month_window + i)) 
        else:
            day_count = day_count + time_val.dt.days_in_month.isel(time=slice(i, len(data['time']) - month_window + i)) * 0.5
            
    data_dt = ((-data.isel(time = slice(0, len(data['time']) - month_window)).drop('time') + 
                data.isel(time = slice(month_window, len(data['time']))).drop('time')) / (24. * 3600. * day_count))
    
    data_dt = data_dt.assign_coords(time=time_val.isel(time=slice(int(np.floor(month_window/2)), 
                                                                  len(data['time']) - month_window + int(np.floor(month_window/2)))))
    
    return data_dt


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

ds_grid = xr.open_dataset("/home/users/hkhatri/DePreSys4_Data/Data_Consolidated/Ocean_Area_Updated.nc")
dA = ds_grid['area_t'].isel(y=slice(780-699, 1100-699+1), x=slice(810-749,1170-749+1)).rename({'y':'j', 'x':'i'})

ppdir_NAO = "/home/users/hkhatri/DePreSys4_Data/Data_Anomaly_Compute/NAO/"
ds_NAO = xr.open_dataset(ppdir_NAO + "NAO_SLP_Anomaly_new.nc")

tim1 = ds_NAO['time_val'].isel(start_year=0,time=slice(0,101)).drop('start_year')
tim1 = tim1.astype("datetime64[ns]")

name_list = ['tos', 'hfds', 'mlotst', 'Heat_Budget']

var_list = ['tos', 'hfds', 'mlotst', 'Heat_Divergence', 'Heat_Content_1300', 'Heat_Divergence_200', 'Heat_Content_200']

case_list = ['NAOp', 'NAOn']

win_month = 6 # number of months for averaging
cf_lev = 0.95 # confidence level
num_sample = 10000 # bootstrap samples to create

lat_lim = [45., 60.]
lon_lim = [-50., -20.]

for case in case_list:
    
    # ----------------------------
    # Read datafiles
    # ----------------------------
    
    ds = []
    
    for file_name in name_list:
        
        d = xr.open_mfdataset(ppdir + "Composite_" + case + "_" + file_name + ".nc", chunks={'time':1})
        
        ds.append(d)

        ds.append(d)
            
    ds = xr.merge(ds)
    ds = ds.assign_coords(time=tim1)
    
    # ----------------------------
    # Compute moving means
    # ----------------------------
    
    dA_NA = dA.where((ds['tos'].isel(time=0,comp=0) < 100.) & (dA.nav_lat >= lat_lim[0]) &
                     (dA.nav_lat <= lat_lim[1]) & (dA.nav_lon >= lon_lim[0]) & (dA.nav_lon <= lon_lim[1])).compute() # Region to consider
    
    ds_mean = xr.Dataset()
    
    for var1 in var_list:
        
        if((var1 == 'tos') or (var1 == 'mlotst')):
            data = (ds[var1] * dA_NA).sum(['i', 'j']) / (dA_NA).sum(['i', 'j'])
        else:
            data = (ds[var1] * dA_NA).sum(['i', 'j']) 
        
        if((var1 == 'Heat_Content_1300') or (var1 == 'Heat_Content_200')):
            ds_mean[var1] = Data_time_tendency(data, tim1, win_month)
        else:
            ds_mean[var1] = Moving_mean(data, tim1, win_month)
        
    ds_mean = ds_mean.transpose('comp', 'time')
    
    print("Data read complete")
    
    # ----------------------------
    # Bootstraping 
    # ----------------------------
    
    ds_save = xr.Dataset()
    
    for var1 in var_list:
        data_var = ds_mean[var1].compute()
        bootstrap_ci = data_bootstrap(data_var.compute(), cf_lev = cf_lev, num_sample = num_sample)
        
        dim_list = ['time'] 
        
        ds_save[var1] = data_var.mean('comp')
        ds_save[var1 + '_standard_error'] = xr.DataArray(data = bootstrap_ci.standard_error, dims=dim_list)
        ds_save[var1 + '_confidence_lower'] = xr.DataArray(data = bootstrap_ci.confidence_interval[0], dims=dim_list)
        ds_save[var1 + '_confidence_upper'] = xr.DataArray(data = bootstrap_ci.confidence_interval[1], dims=dim_list)
        
    
    ds_save.attrs['description'] = ("Bootstrapping standard errors and confirdence for area-mean and area-integrated diagnostics. " + 
                                   "Confidence interval is " + str(cf_lev*100) + "%. " + "Chosen Region is: latitudes " + str(lat_lim[0])
                                   + "N – " + str(lat_lim[1]) + "N and longitudes " + str(-lon_lim[0]) + "W – " + str(-lon_lim[1]) + "W.")
    save_file_path = (save_path + "Bootstrap_"+ case + "_Diagnostics_Area_Integrated.nc")
    ds_save = ds_save.astype(np.float32).compute()
    ds_save.to_netcdf(save_file_path)

    print("Data saved succefully")

    ds_save.close()
    ds_mean.close()
    ds.close()
