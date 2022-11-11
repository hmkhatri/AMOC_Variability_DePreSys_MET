"""
Author: Hemant Khatri
Email: hkhatri@liverpool.ac.uk

This script can be used to compute standard error and confidence intervals using bootstrapping method (with replacement). For this, scipy.stats.bootstrap function is used.
The script writes the output in a netcdf file that contains confidence intervals and standard error.

Update - xskillscore implementation is added to work with large datasets and dask. Output contains the sample mean and standard error from bootstrap samples.
standrd error = standard deviation (sample mean data of bootstrap samples)

"""

# ------- load libraries ------------
import numpy as np
import xarray as xr
import xskillscore as xs
import scipy.stats as sc

import warnings
warnings.filterwarnings("ignore")

#from dask_mpi import initialize
#initialize()

#from dask.distributed import Client
#client = Client()

#os.environ["MALLOC_MMAP_MAX_"]=str(40960) # to reduce memory clutter. This is temparory, no permanent solution yet.
#os.environ["MALLOC_MMAP_THRESHOLD_"]=str(16384)

### ------ Functions for computations ----------

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
    
    bootstrap_ci = sc.bootstrap(data, statistic=np.mean, confidence_level=cf_lev, vectorized=True, axis=0, n_resamples=num_sample,
                                random_state=1, method='BCa')
    
    return bootstrap_ci

def data_bootstrap_xskill(data, dim_iter = 'time', num_sample = 1000):
    
    """Compute bootstrap standard error for data along an axis
    Parameters
    ----------
    data : xarray DataArray for data
    dim_iter : dimension name for creating iterations
    num_sample : Number of bootstrap samples to generate

    Returns
    -------
    standard_error : From bootstrap samples
    """
    
    data_resample = xs.resample_iterations(data, num_sample, dim_iter, replace=True)
    
    standard_error = (data_resample.mean(dim = dim_iter)).std(dim = 'iteration')
    
    return standard_error

### ------------- Main computations ------------

ppdir = "/gws/nopw/j04/snapdragon/hkhatri/Data_Composite/NAO_hpa/"
save_path = "/gws/nopw/j04/snapdragon/hkhatri/Data_Composite/NAO_hpa/Bootstrap_Confidence/"
#save_path = "/gws/nopw/j04/snapdragon/hkhatri/Data_Composite/NAO_hpa/Bootstrap_xskillscore/"

#var_list = ['tos', 'hfds', 'Heat_Budget_new'] #, 'Overturning']
#var_list = ['Heat_Budget_new']
#var_list = ['Overturning_MHT']
var_list = ['Gyre']

case_list = ['NAOp', 'NAOn']
cf_lev = 0.9 # confidence level
num_sample = 1000 # bootstrap samples to create

# get days in month from actual time values
ds_NAO = xr.open_dataset("/home/users/hkhatri/DePreSys4_Data/Data_Anomaly_Compute/NAO/NAO_SLP_Anomaly_new.nc")
tim1 = ds_NAO['time_val'].isel(start_year=0,time=slice(0,101)).drop('start_year')
tim1 = tim1.astype("datetime64[ns]")

# Loop over vars

for case in case_list:
    
    print("Working for case – ", case)
    
    for var in var_list:
        
        ds = xr.open_dataset(ppdir + "Composite_" + case + "_" + var + ".nc", chunks={'time':101, 'j':107})
        
        # compute annual means
        if(var == 'Heat_Budget_new'):
            var_name = ['Heat_Divergence', 'Heat_Content_1300', 'Heat_Content']
            ds_annual = annaul_mean_data(ds, var_name, tim1, method = 'difference')
            
            ds_annual['Heat_Content_1300_full'] = ds_annual['Heat_Content'] - ds_annual['Heat_Content_1300']
            var_name = ['Heat_Divergence', 'Heat_Content_1300', 'Heat_Content', 'Heat_Content_1300_full']
            
        elif(var == 'tos'):
            var_name = [var]
            ds_annual = annaul_mean_data(ds, var_name, tim1, method = 'mean')

        elif(var == 'Gyre'):
            var_name = ['Psi_Gyre']
            ds_annual = annaul_mean_data(ds, var_name, tim1, method = 'mean')
            
        elif(var == 'hfds'):
            var_name = [var]
            ds_annual = annaul_mean_data(ds, var_name, tim1, method = 'integrate')
        
        elif(var == 'Overturning_MHT'):
            ds1 = xr.Dataset()
            ds1['Overturning_z'] = ds['Overturning_z'] - ds['Overturning_z_barotropic']
            ds1['Overturning_sigma'] = ds['Overturning_sigma'] - ds['Overturning_sigma_barotropic']
            ds1['Depth_sigma'] = ds['Depth_sigma']
            ds1['Density_z'] = ds['Density_z']
    
            var_name = ['Overturning_z', 'Overturning_sigma', 'Depth_sigma', 'Density_z']
            ds_annual = annaul_mean_data(ds1, var_name, tim1, method = 'mean')
        
        # compute bootstrap confidece intervals
        ds_save = xr.Dataset()
        
        if(var == 'Overturning_MHT'):
            
            ds_save['Depth_sigma'] = ds_annual['Depth_sigma'].mean('comp')
            ds_save['Density_z'] = ds_annual['Density_z'].mean('comp')
            
            ds_save['latitude'] = ds['latitude']
            
            var_name = ['Overturning_z', 'Overturning_sigma']
        
        # -------- Using scipy library -------------------
        for var1 in var_name:
            
            sde_var1 = []
            cfd_up_var1 = []
            cfd_low_var1 = []
            
            if((var1 == 'Overturning_sigma') or (var1 == 'MHT_sigma')):
                dim_list = ['sigma0', 'j_c']
            elif((var1 == 'Overturning_z') or (var1 == 'MHT_z')):
                dim_list = ['lev', 'j_c']
            else:
                dim_list = ['j', 'i']
                
            #print(dim_list)
            
            for yr in range(0, len(ds_annual['year'])):
                
                data_var = ds_annual[var1].isel(year=yr).compute()
                bootstrap_ci = data_bootstrap(data_var, cf_lev = cf_lev, num_sample = num_sample)
                
                sde = xr.DataArray(data = bootstrap_ci.standard_error, dims=dim_list)
                sde_var1.append(sde)
                
                cfd_up = xr.DataArray(data = bootstrap_ci.confidence_interval[1], dims=dim_list)
                cfd_up_var1.append(cfd_up)
                
                cfd_low = xr.DataArray(data = bootstrap_ci.confidence_interval[0], dims=dim_list) 
                cfd_low_var1.append(cfd_low)
                
            ds_save[var1] = ds_annual[var1].mean('comp')
            ds_save[var1 + '_standard_error'] = xr.concat(sde_var1, dim='year') 
            ds_save[var1 + '_confidence_lower'] = xr.concat(cfd_low_var1, dim='year') 
            ds_save[var1 + '_confidence_upper'] = xr.concat(cfd_up_var1, dim='year')
            
        #if(var == 'Overturning_MHT'):
        #    ds_save['latitude'] = ds['latitude']
            
        save_file_path = (save_path + "Bootstrap_"+ case + "_" + var + "_annual.nc")
        ds_save = ds_save.astype(np.float32).compute()
        ds_save.to_netcdf(save_file_path)

        print("Data saved succefully")

        ds_save.close()
        ds_annual.close()
        ds.close()
        
        
        """
        # -------- Using xskillscore library (not working properly - too slow)-------------------
        for var1 in var_name:
            
            ds_save[var1] = ds_annual[var1].mean('comp')
            
            se = data_bootstrap_xskill(ds_annual[var1].compute(), dim_iter = 'comp', num_sample = num_sample)
            
            ds_save[var1 + '_standard_error'] = se
            
        if(var == 'Overturning'):
            ds_save['latitude'] = ds['latitude']
            
        save_file_path = (save_path + "Bootstrap_"+ case + "_" + var + "_annual.nc")
        ds_save = ds_save.astype(np.float32).compute()
        ds_save.to_netcdf(save_file_path)

        print("Data saved succefully")

        ds_save.close()
        ds_annual.close()
        ds.close()
        """
            
