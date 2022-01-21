# load libraries
import numpy as np
import scipy as sc
import xarray as xr
import gsw as gsw
#from dask_jobqueue import SLURMCluster
from dask.distributed import Client, LocalCluster
from dask import delayed
from dask import compute
from dask import persist

import warnings
warnings.filterwarnings('ignore')

from dask_mpi import initialize

#cluster = SLURMCluster(cores=8,memory="16GB")
#cluster.adapt(minimum=0, maximum=8)
#client = Client(cluster)

if __name__ == '__main__':
    
    #cluster = LocalCluster()
    
    initialize()
    client = Client()
    print(client)

    ### ------ Function to compute potential density and sum over selected years ----------

    def pdens(S,theta):

        pot_dens = gsw.density.sigma0(S, theta)

        return pot_dens

    def processDataset(ds1, year1, year2, lead_year):

        ds_save = []

        for year in range(year1, year2):

            # Extract data relavant start year and sum over all hindcasts
            ds2 = ds1.sel(start_year = year - lead_year).isel(time=slice(12*lead_year, np.minimum(12 + 12*lead_year, len(ds1['time']))))

            ds_save.append(ds2)

        return ds_save

    ### ------------- Main computations ------------

    ppdir="/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/"

    # for monthly drift
    save_path="/home/users/hkhatri/DePreSys4_Data/Data_Drift_Removal/Drift_1970_2016_Method_DCPP/"

    var_list = ['sigma0_surface'] 

    year1, year2 = (1979, 1990) # range over to compute average using DCPP 2016 paper

    for var in var_list:

        for r in range(0,1):

            print("Var = ", var, "; Ensemble = ", r)

            ds = []

            for year in range(year1-10, year2, 1):

                # Read T/S data for each hindcast for every ensemble member

                var_path = "s" + str(year) +"-r" + str(r+1) + "i1p1f2/Omon/" + "tos" + "/gn/files/d20200417/"
                d1 = xr.open_mfdataset(ppdir + var_path + "*.nc")

                var_path = "s" + str(year) +"-r" + str(r+1) + "i1p1f2/Omon/" + "sos" + "/gn/files/d20200417/" 
                d2 = xr.open_mfdataset(ppdir + var_path + "*.nc")

                d = xr.merge([d1.tos, d2.sos])

                d = d.drop('time') # drop time coordinate as different time values create an issue in concat operation

                ds.append(d)

            # combine data for hindcasts
            ds = xr.concat(ds, dim='start_year')
            #ds = ds.drop(['vertices_latitude', 'vertices_longitude', 'time_bnds'])
            ds = ds.isel(i=slice(749,1199), j = slice(699, 1149))

            ds = ds.assign(start_year = np.arange(year1-10, year2, 1))
            ds = ds.chunk({'start_year':-1, 'time':-1, 'j':50, 'i':50})

            print("Data read complete")

            #rho = xr.apply_ufunc(pdens, ds.sos, ds.tos, dask='parallelized',
            #        output_dtypes=[ds.tos.dtype])
            #rho = pdens(ds.sos, ds.tos)
            rho = delayed(pdens)(ds.sos, ds.tos)

            # loop over lead year and compute mean values
            for lead_year in range (0,11):

                #print("Lead Year running = ", lead_year) 

                #sigma0 = processDataset(rho, year1, year2, lead_year)
                sigma0 = delayed(processDataset)(rho, year1, year2, lead_year)

                #sigma0 = sum(sigma0) / (year2 - year1)
                sigma0 = delayed(sum)(sigma0) / (year2 - year1)

                sigma0 = sigma0.compute() #scheduler="processes")
                #sigma0 = compute(sigma0) #, scheduler="processes")

                ds_save = xr.Dataset()
                ds_save['sigma0'] = sigma0.drop('start_year')
                ds_save.sigma0.attrs['standard_name'] = "Potential Density wrt zero pressure - 1000"
                ds_save.sigma0.attrs['units'] = "kg/m3"

                save_file = save_path +"Drift_" + var + "_r" + str(r+1)+ "_Lead_Year_" + str(int(lead_year+1)) + ".nc"
                ds_save.to_netcdf(save_file)

                print("File saved for Lead Year = ", lead_year + 1)

#client.close()
