import xarray as xr
import numpy as np
import dask
import gc
import os, psutil

from dask_mpi import initialize
initialize()

from dask.distributed import Client, performance_report
client = Client()
print(client)

### ----------------------------------------------- ###
# First generate dataset (this is one-time step, won't be required after the first time)
#(nt,ny,nx)=(360,100,100)

#dummy=xr.DataArray(data=np.random.randn(nt,ny,nx),dims=['t','y','x'])

#ds = dummy.to_dataset(name = 'data')
#ds['data_snapshot'] = dummy.isel(t=0)

#ds = ds.astype(np.float32) 

#ds.to_netcdf('/home/users/hkhatri/Git_Repo/test.nc')
### ----------------------------------------------- ###


# Read data for testing

ds = []
ds1 = []
for ensemble in range(0,100):
    
    with xr.open_dataset('/home/users/hkhatri/Git_Repo/test.nc', chunks={'t':90}) as d:
        d = d
    
    ds.append(d)
    
    ds1.append(d.isel(t=0))
    
ds = xr.concat(ds, dim='r')
ds1 = xr.concat(ds1, dim='r')

print("Data read complete")

#with performance_report(filename="memory-mpi.html"):
#    tmp = ds['data'].mean('r')
#    ds1_mean = tmp.compute()
    
#with performance_report(filename="memory-mpi1.html"):
#    tmp = ds['data'].mean('t')
#    ds1_mean = tmp.compute()
    
#with performance_report(filename="memory-serial.html"):
#tmp = ds['data'].mean('r')
#ds1_mean = tmp.compute()

print("Data computations complete")

with performance_report(filename="memory-mpi-loop.html"):
    
    for r in range(0,len(ds['r'])):
        tmp = ds['data'].isel(r=r).mean('t')
        ds1_mean = tmp.compute()
        
        print("Data computations complete for r = ", r+1)
    
        process = psutil.Process(os.getpid())
        print("Memory usage in MB = ", process.memory_info().rss /1e6)
    
        client.cancel(ds1_mean)






