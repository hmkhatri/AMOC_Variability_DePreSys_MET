# load libraries
import numpy as np
import scipy as sc
import xarray as xr
import gc
import dask.distributed

from dask_mpi import initialize
initialize()

import warnings
warnings.filterwarnings('ignore')

from dask.distributed import Client, performance_report
client = Client()

### ------ Function to sum over selected years and saving chunks ----------

def processDataset(ds1, year1, year2, lead_year):
    
    ds_save = []
    
    for year in range(year1, year2):
        
        # Extract data relavant start year and sum over all hindcasts
        ds2 = ds1.sel(start_year = year - lead_year)
        ds2 = ds2.isel(time=slice(12*lead_year, np.minimum(12 + 12*lead_year, len(ds1['time']))))
        
        ds_save.append(ds2.drop('start_year'))
        
    ds_save = xr.concat(ds_save, dim='hindcast')
        
    return ds_save

def split_by_chunks(dataset):
    chunk_slices = {}
    for dim, chunks in dataset.chunks.items():
        slices = []
        start = 0
        for chunk in chunks:
            if start >= dataset.sizes[dim]:
                break
            stop = start + chunk
            slices.append(slice(start, stop))
            start = stop
        chunk_slices[dim] = slices
    for slices in itertools.product(*chunk_slices.values()):
        selection = dict(zip(chunk_slices.keys(), slices))
        yield dataset[selection]
        
def create_filepath(prefix='filename', root_path=".", i=0):
    filepath = f'{root_path}/{prefix}_chunk_{i}.nc'
    return filepath

### ------------- Main computations ------------

ppdir="/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/"

# for monthly drift
save_path="/gws/nopw/j04/snapdragon/hkhatri/Data_Drift/" # this is for 3D vars

#var_list = ['hfds'] #, 'tos', 'sos'] #, 'mlotst', 'zos']
var_list = ['thetao']

year1, year2 = (1979, 2017) # range over to compute average using DCPP 2016 paper

for var in var_list:
    
    for r in range(4,5):
       
        print("Var = ", var, "; Ensemble = ", r)

        ds = []

        for year in range(year1-10, year2, 1):
            
            # Read data for each hindcast for every ensemble member
            
            var_path = "s" + str(year) +"-r" + str(r+1) + "i1p1f2/Omon/" + var + "/gn/latest/"
            
            d = xr.open_mfdataset(ppdir + var_path + "*.nc", decode_times=False)
            
            # drop time coordinate as different time values create an issue in concat operation
            d = d.drop(['time', 'vertices_latitude', 'vertices_longitude', 'time_bnds', 'lev_bnds'])
            
            d = d.isel(i=slice(749,1199), j=slice(699, 1149))
            
            ds.append(d)
            
            d.close()
            
        # combine data for hindcasts
        ds = xr.concat(ds, dim='start_year')
        
        ds = ds.assign(start_year = np.arange(year1-10, year2, 1))
        
        ds = ds.chunk({'start_year':1, 'time':12, 'lev':5})
        
        print("Data read complete")
        
        # loop over lead year and compute mean values
        for lead_year in range(0,1):
    
            #print("Lead Year running = ", lead_year)

            ds_var = processDataset(ds, year1, year2, lead_year)
            
            ds_var = (ds_var.mean('hindcast'))
            
            #with performance_report(filename="dask-report.html"):
            ds_save = ds_var.persist()
            
            print("Data persist successful")
            
            # split data into chunks
            datasets = list(split_by_chunks(ds_save))
            
            filename = ("Drift_" + var + "_r" + str(r+1)+ "_Lead_Year_" + str(lead_year+1))
            
            paths = []
            # filenames for individual chunk files
            for i in range(0,len(datasets)):
                tmp = create_filepath(prefix=filename, root_path=save_path, i=i+1)
                paths.append(tmp)
            
            xr.save_mfdataset(datasets=datasets, paths=paths)
            
            print("File saved for Lear Year = ", lead_year+1)
            
            ds_var.close()
            ds_save.close()
            gc.collect()
            
        ds.close()

#client.close()

