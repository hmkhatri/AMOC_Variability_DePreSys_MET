## ---------------------------- ##

# The script computes model drift for disngotics using the method described in https://doi.org/10.5194/gmd-9-3751-2016. Here, we compute the mean over years Nov, 1970 - Mar, 2017 for all hindcasts while mainting the information about the lead year in hindcasts. 
# For example, hindcasts started in Nov, 1970 and Nov, 1971 both have Nov-Dec 1971. However, these cannot be added together in the mean calculation because hindcasts have run for different amount of simulation time before reaching Nov, 1971. 
# To get correct model drift and model climatology, compute mean over months that are in the same year as well as in the same simulations month time.

## ----------------------------- ##

# load libraries
import numpy as np
import scipy as sc
import xarray as xr
import dask.distributed
from dask.distributed import Client
from dask import delayed
from dask import compute
from dask import persist

import warnings
warnings.filterwarnings('ignore')

### ------------- Main computations ------------

ppdir="/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/"

save_path="/home/users/hkhatri/DePreSys4_Data/Data_Drift_Removal/Drift_1970_2016_Method_DCPP/"

var_list = ['hfds', 'tos', 'sos'] #, 'mlotst', 'zos']

year1, year2 = (1970, 2017) # range over to compute average using DCPP 2016 paper

