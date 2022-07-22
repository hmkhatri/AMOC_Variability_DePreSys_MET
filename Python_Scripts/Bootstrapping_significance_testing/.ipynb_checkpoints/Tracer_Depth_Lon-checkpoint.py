"""
Author: Hemant Khatri
Email: hkhatri@liverpool.ac.uk

The script is used for computing composite of tracer field (Depth-Lon structure) corrsponding to extreme NAO conditions.

This script can be used to compute standard error and confidence intervals using bootstrapping method (with replacement). For this, scipy.stats.bootstrap function is used. The script writes the output in a netcdf file that contains confidence intervals and standard error.

"""

# ------- load libraries ------------
import numpy as np
import xarray as xr
import scipy.stats as sc

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

def annaul_mean_data(ds, var_name, num_days, method = 'mean'):
    
    """Compute annual mean of data for bootstrapping
    Means are computed for year = -1, 0, 1, 2, 3-4, 5-6
    Parameters
    ----------
    ds : xarray Dataset for data variables
    var_name : list of avariables for computing annual means
    num_days : Number of days in months
    method : Options - compute 'mean', 'integrate', 'difference' over time 
    
    Returns
    -------
    ds_annual : Dataset containting annual means
    """
    
    ds_annual = xr.Dataset()
    
    for var1 in var_name:
        
        data_var1 = []
        
        ind_correct = 0
        for i in range(0,6):

            if (i<=3):
                days = num_days.dt.days_in_month.isel(time = slice(12*i + 2, 12*i + 2 + 12))
                data_var = ds[var1].isel(time = slice(12*i + 2, 12*i + 2 + 12))
            else:
                days = num_days.dt.days_in_month.isel(time = slice(12*(i + ind_correct) + 2, 12*(i + ind_correct + 1) + 2 + 12))
                data_var = ds[var1].isel(time = slice(12*(i + ind_correct) + 2, 12*(i + ind_correct + 1) + 2 + 12))
                ind_correct = ind_correct + 1

            if(method == 'mean'):
                data_var = ((data_var * days).sum('time')/ days.sum('time'))
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
    
    bootstrap_ci = sc.bootstrap(data, statistic=np.mean, confidence_level=cf_lev, vectorized=True, axis=0,
                                n_resamples=num_sample, random_state=1, method='BCa')
    
    return bootstrap_ci

### ------ Main calculations ------------------

# -------- Define paths and read data --------------

ppdir = "/gws/nopw/j04/snapdragon/hkhatri/Data_Composite/NAO_hpa/"
save_path = "/gws/nopw/j04/snapdragon/hkhatri/Data_Composite/NAO_hpa/Bootstrap_Confidence/"


var_list = ['thetao', 'Heat_Budget_new']

case_list = ['NAOp', 'NAOn']
cf_lev = 0.8 # confidence level
num_sample = 1000 # bootstrap samples to create

# get days in month from actual time values
ds_NAO = xr.open_dataset("/home/users/hkhatri/DePreSys4_Data/Data_Anomaly_Compute/NAO/NAO_SLP_Anomaly_new.nc")
tim1 = ds_NAO['time_val'].isel(start_year=0,time=slice(0,101)).drop('start_year')
tim1 = tim1.astype("datetime64[ns]")

# ----------- Create Composite of the tracer field --------------
# Loop over vars

for case in case_list:
    
    print("Working for case – ", case)
    
    ds = []
    
    for var in var_list:
     
        if(var == 'Heat_Budget_new'):
            d = xr.open_dataset(ppdir + "Composite_" + case + "_" + var + ".nc", chunks={'time':101, 'j':107})
            d['Heat_Content_1300_full'] = d['Heat_Content'] - d['Heat_Content_1300']
            ds.append(d.get(['Heat_Content_1300', 'Heat_Content_1300_full']).drop(['latitude', 'longitude']))
        else:
            d = xr.open_dataset(ppdir + "Composite_" + case + "_" + var + "_Depth_Lon.nc", chunks={'time':101})
            ds.append(d.rename({'i': 'im'})) # this is because the datafiles are of not same size in 'i'
        
    ds = xr.merge(ds)
    
    ds = ds - ds.isel(time=0) # to remove time=0 signal for better interpretation
    
    var_name = ['thetao', 'Heat_Content_1300', 'Heat_Content_1300_full']
    ds_annual = annaul_mean_data(ds, var_name, tim1, method = 'mean')
    
    ds_save = xr.Dataset()
    
    # Compute bootstrap confidence intervals
    
    for var in var_name:
        
        sde_var = []
        cfd_up_var = []
        cfd_low_var = []
        
        if(var == 'thetao'):
            dim_list = ['lev', 'im']
        else:
            dim_list = ['j', 'i']
        
        for yr in range(0, len(ds_annual['year'])):
            
            data_var = ds_annual[var].isel(year=yr).compute()
            bootstrap_ci = data_bootstrap(data_var, cf_lev = cf_lev, num_sample = num_sample)

            sde = xr.DataArray(data = bootstrap_ci.standard_error, dims=dim_list)
            sde_var.append(sde)

            cfd_up = xr.DataArray(data = bootstrap_ci.confidence_interval[1], dims=dim_list)
            cfd_up_var.append(cfd_up)

            cfd_low = xr.DataArray(data = bootstrap_ci.confidence_interval[0], dims=dim_list) 
            cfd_low_var.append(cfd_low)

        ds_save[var] = ds_annual[var].mean('comp')
        ds_save[var + '_standard_error'] = xr.concat(sde_var, dim='year') 
        ds_save[var + '_confidence_lower'] = xr.concat(cfd_low_var, dim='year') 
        ds_save[var + '_confidence_upper'] = xr.concat(cfd_up_var, dim='year')
    
    ds_save['longitude'] = ds['longitude']
    
    save_file_path = (save_path + "Bootstrap_"+ case + "_" + var_list[0] + "_Depth_Lon.nc")
    ds_save = ds_save.astype(np.float32).compute()
    ds_save.to_netcdf(save_file_path)

    print("Data saved succefully")

    ds_save.close()
    ds_annual.close()
