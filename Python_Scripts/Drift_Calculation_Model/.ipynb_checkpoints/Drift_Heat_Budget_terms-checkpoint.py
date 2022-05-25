"""
Author: Hemant Khatri
Email: hkhatri@liverpool.ac.uk

This script computes model drift in Heat budget terms.
For this, method from Dune et al. (2016) is adopted. 

"""

# ------- load libraries ------------ 
import numpy as np
import xarray as xr
from xgcm import Grid
import gc

import warnings
warnings.filterwarnings('ignore')

# ------- Functions -----------------

def processDataset(ds1, y1, y2, lead_yr):
    
    """Computes model drift by averaging over hindcasts
    while mainitaining the information about lead years
    
    Parameters
    ----------
    ds1 : xarray Dataset / DataArray for computing drift
    y1, y2 : interger years for which data is averaged 
    lead_yr : integer - Lead year information
    
    Returns
    -------
    ds2 : xarray Dataset / DataArray corresponding to model drift data
    """
    
    ind_tim1 = 12*lead_yr
    ind_tim2 = np.minimum(12 + 12*lead_yr, len(ds1['time']))
    
    ds2 = ds1.isel(start_year = slice(y1-lead_yr-y1+10, y2-lead_yr-y1+10))
    ds2 = ds2.isel(time=slice(ind_tim1, ind_tim2))
    # isel -> slice(0,10) will isolate data for 0, 1,2, ...9 indices
    # sel -> slice(1960, 1970) will isolate data for 1960, 1961, .... 1970 years
        
    return ds2

# ----------- Main computations -----------------
data_dir = "/gws/nopw/j04/snapdragon/hkhatri/Data_Heat_Budget/"
#save_path = "/gws/nopw/j04/snapdragon/hkhatri/Data_Drift/Heat_Budget/drift_budget_terms/"
save_path = "/home/users/hkhatri/DePreSys4_Data/Data_Drift_Removal/Drift_1970_2016_Method_DCPP/"

year1, year2 = (1979, 2017) # range over to compute average using DCPP 2016 paper

for r in range(0,9):
    
    print("Var = Heat Budget diagnostics", "; Ensemble = ", r)
    
    ds = []
    
    # loop for reading files for all years
    for year in range(year1-10, year2, 1):
        
        file_path = (data_dir + "Heat_Budget_" + str(year) +"_r" + str(r+1) + ".nc")
        
        d = xr.open_dataset(file_path, chunks={'time':1}, decode_times=False, engine='netcdf4')
        
        ds.append(d.drop('time'))
        
    # combine data for hindcasts
    ds = xr.concat(ds, dim='start_year')
    
    print("Data read complete")
    
    # loop over lead year and compute mean values
    for lead_year in range(0,11):
        
        ds_var = processDataset(ds, year1, year2, lead_year)
        
        ds_var = ds_var.mean('start_year').compute()
        
        save_file = (save_path + "Drift_Heat_Budget_r" + str(r+1) + 
                     "_Lead_Year_" + str(int(lead_year+1)) + ".nc")
        ds_var.to_netcdf(save_file)

        print("File saved for Lear Year = ", lead_year+1)