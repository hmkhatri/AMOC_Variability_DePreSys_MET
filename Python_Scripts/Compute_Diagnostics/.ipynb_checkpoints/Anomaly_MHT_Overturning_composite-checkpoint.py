## --------------------------------------------------------------------
# We compute anomalies in meridional heat transport (MHT) and overturning at 26.55 at 50N across all simualtions (after correcting for model drift).
# These anomalies the time-mean over years 4-6 in simualtions
# These anomalies will then be used for creating composites using members that have the same sign of MHT and overturning anomalies.
## --------------------------------------------------------------------

# ------- load libraries ------------ 
import numpy as np
import scipy as sc
import xarray as xr

# ------ Functions ----------------- 
def mean_data(ds, year1, year2):

    """Compute time-mean between year1 and year2
    Parameters
    ----------
    ds : xarray DataArray
    year : year for averaging

    Returns
    -------
    ds_mean : xarray DataArray for time-mean
    ds_mean_anom : xarray DAtaArray for time-mean with respect to the time-mean of the full simulation
    """

    days = ds.time.dt.daysinmonth
    
    tmp_mean = (ds * days).sum('time') / days.sum('time')
    
    tmp = ds.sel(time = ds['time.year'] >= year1) 
    tmp = ds.sel(time = ds['time.year'] <= year2)

    days = tmp.time.dt.daysinmonth
    ds_mean = (tmp * days).sum('time') / days.sum('time') 
    
    ds_mean_anom = ds_mean - tmp_mean

    return ds_mean, ds_mean_anom

# -------- Define paths and read data --------------

data_dir =  "/gws/nopw/j04/snapdragon/hkhatri/DePreSys4/Data_Consolidated/Overturning_Heat_Salt_Transport_Baro_Decompose/"

data_drift_dir = "/gws/nopw/j04/snapdragon/hkhatri/DePreSys4/Data_Drift/"

ppdir_NAO="/home/users/hkhatri/DePreSys4_Data/Data_Anomaly_Compute/NAO/"

save_path="/gws/nopw/j04/snapdragon/hkhatri/DePreSys4/Data_Composite/Overturning_MHT/"

ds_NAO = xr.open_dataset(ppdir_NAO + "NAO_SLP_Anomaly_new.nc")
tim = ds_NAO['time_val'].isel(start_year=0).drop('start_year')

# --------- Read drift data ---------------

ds_drift = []

for r in range(0,10):
    
    ds1 = []
    
    for lead_year in range(0, 11):
        
        d = xr.open_dataset(data_drift_dir + "psi_sigma/Drift_Overturning_r" + str(r+1) + "_Lead_Year_" +
                            str(lead_year + 1) + ".nc", decode_times=False, chunks={'j_c':50})
        
        d = d.assign(time = np.arange(lead_year*12, 12*lead_year + 
                                      np.minimum(12, len(d['time'])), 1))
        
        ds1.append(d)
        
    ds1 = xr.concat(ds1, dim='time')
    
    ds_drift.append(ds1.drop('time'))
    
ds_drift = xr.concat(ds_drift, dim='r')
    
ds_drift = ds_drift.drop('j_c')
ds_drift = ds_drift.assign_coords(j_c=ds_drift['latitude'].isel(time=0, r=0))

ds_drift = ds_drift.isel(j_c=225, lat = 89) # to get values closest to 50N. j_c = 49.95, lat = 49.72

ds_drift = ds_drift.chunk({'time':-1})

print("Reading Drift Data Complete")

# -------- Read main data ----------------

ds = []

for r in range(0,10):

    d = xr.open_dataset(data_dir + "Overturning_Heat_Salt_Transport_r" + str(r+1) + ".nc", chunks={'start_year':1, 'j_c':50})

    ds.append(d)

ds = xr.concat(ds, dim='r')

ds = ds.drop('j_c')

ds = ds.assign_coords(j_c=ds['latitude'].isel(start_year=0, r=0).drop('start_year'))
ds = ds.isel(j_c=225, lat = 89) # to get values closest to 50N. j_c = 49.95, lat = 49.72

ds['Overturning_max_sigma'] = (ds['Overturning_sigma'] - ds['Overturning_sigma_barotropic']).max(dim='sigma0')


# ----------- Main computations --------------

ds1 = (ds - ds_drift).assign_coords(time=tim) # get anomaly by removing model drift

year1, year2 = (1964, 1966) # get 3-yr averages (1964-1966 is in the middle of the simulations)

ds_save = xr.Dataset()

# time-mean for MHT 
MHT = (ds1['MHT_z'] - ds1['MHT_z_baro']).sum('lev')

MHT_mean, MHT_mean_anom = mean_data(MHT, year1, year2)

ds_save['MHT_mean'] = MHT_mean
ds_save['MHT_mean'].attrs['units'] = "Joules/s"
ds_save['MHT_mean'].attrs['long_name'] = "3-yr mean of MHT - MHT Drift (Jan 1964 to Dec 1966 if simualtion starts in Nov 1960)"

ds_save['MHT_mean_anom'] = MHT_mean_anom
ds_save['MHT_mean_anom'].attrs['units'] = "Joules/s"
ds_save['MHT_mean_anom'].attrs['long_name'] = ("3-yr mean of MHT - MHT Drift (Jan 1964 to Dec 1966 if simualtion starts in Nov 1960)" + 
                                               "wrt to the time-mean of each simulation")

# time-mean for overturning at 27.55 kg/m^3 
psi = (ds1['Overturning_sigma'].sel(sigma0=27.55, method='nearest') - ds1['Overturning_sigma_barotropic'].sel(sigma0=27.55, method='nearest'))

psi_mean, psi_mean_anom = mean_data(psi, year1, year2)

ds_save['Psi_mean'] = psi_mean
ds_save['Psi_mean'].attrs['units'] = "m^3/s"
ds_save['Psi_mean'].attrs['long_name'] = "3-yr mean of Psi - Psi Drift at sigma=27.55 (Jan 1964 to Dec 1966 if simualtion starts in Nov 1960)"

ds_save['Psi_mean_anom'] = psi_mean_anom
ds_save['Psi_mean_anom'].attrs['units'] = "m^3/s"
ds_save['Psi_mean_anom'].attrs['long_name'] = ("3-yr mean of Psi - Psi Drift at sigma=27.55 (Jan 1964 to Dec 1966 if simualtion starts in Nov 1960)" +
                                               "wrt to the time-mean of each simulation")

# time-mean for overturning max
psi = ds1['Overturning_max_sigma']

psi_mean, psi_mean_anom = mean_data(psi, year1, year2)

ds_save['Psimax_mean'] = psi_mean
ds_save['Psimax_mean'].attrs['units'] = "m^3/s"
ds_save['Psimax_mean'].attrs['long_name'] = "3-yr mean of Psimax - Psimax Drift at sigma=27.55 (Jan 1964 to Dec 1966 if simualtion starts in Nov 1960)"

ds_save['Psimax_mean_anom'] = psi_mean_anom
ds_save['Psimax_mean_anom'].attrs['units'] = "m^3/s"
ds_save['Psimax_mean_anom'].attrs['long_name'] = ("3-yr mean of Psimax - Psimax Drift at sigma=27.55 (Jan 1964 to Dec 1966 if simualtion starts in Nov 1960)" +
                                               "wrt to the time-mean of each simulation")
                            

# save data
save_file = save_path +"MHT_Overturning_456yr.nc"

ds_save = ds_save.astype(np.float32).compute()

ds_save.to_netcdf(save_file)