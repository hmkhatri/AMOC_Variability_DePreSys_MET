## ------------------------------------------------------- ##
# This is a test script to see if the nc files are read in time or not. Some scripts on jasmin cluster seem to
# be very inefficient and this test script can be used to identify the bugs in the reading process.

# load libraries
import numpy as np
import scipy as sc
import xarray as xr
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
import time

# Read data

ppdir="/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/"

year1, year2 = (1980, 2017)

var_list = ['sos'] #['mlotst']

for var in var_list:

    for r in range(0,1):

        print("Ensemble = ", r)

        for year in range(year1, year2, 1):

            # Read T/S data for each hindcast for every ensemble member
            
            print("Year Running = ", year)
            
            start = time.time()

            var_path = "s" + str(year) +"-r" + str(r+1) + "i1p1f2/Omon/" + var + "/gn/files/d20200417/"
            
            if(year % 2 == 0): 
                d1 = xr.open_mfdataset(ppdir + var_path + "*.nc")
                
            else:
                d1 = xr.open_mfdataset(ppdir + var_path + "*.nc", decode_times=False)
                d1 = xr.decode_cf(d1)

            #d = d.drop('time') # drop time coordinate as different time values create an issue in concat operation
            print("months = ", len(d1['time']))
                  
            end = time.time()
            
            print("Runtime of the year = ", str(end - start), " sec")

