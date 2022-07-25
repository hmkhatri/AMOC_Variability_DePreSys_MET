## Description of Python Scripts

| Filename | Description |
| --- | --- |
| [Drift_ocean_var_DCPP2016.py](./Drift_Calculation_Model/Drift_ocean_var_DCPP2016.py) | To compute model drift in ocean diagnostics and save drift data in nc files |
| [Drift_atmosphere_var_DCPP2016.py](./Drift_Calculation_Model/Drift_atmosphere_var_DCPP2016.py) | To compute model drift in atmospheric diagnostics and save drift data in nc files |
| [Drift_ocean_derived_var_DCPP2016.py](./Drift_Calculation_Model/Drift_ocean_derived_var_DCPP2016.py) | To compute model drift in potential density computed using T/S ocean diagnostics and save drift data in nc files |
| [Drift_Overturning.py](./Drift_Calculation_Model/Drift_Overturning.py) | To compute model drift in overturning and heat/salt transport diagnotics |
| [Drift_Overturning_MHT_sigma_z.py](./Drift_Calculation_Model/Drift_Overturning_MHT_sigma_z.py) | To drift in overturning, northward heat/salt transport diagnostics |
| [Drift_Heat_Budget_terms.py](./Drift_Calculation_Model/Drift_Heat_Budget_terms.py) | Drift in heat budget diagnostics |
| | |
| [Correlation_NAO_ocean_var.py](./Compute_Diagnostics/Correlation_NAO_ocean_var.py) | To compute corretion between seasonal NAO indices and anomalies in sea-surface diagnostics |
| [Anomaly_DCPP2016.py](./Compute_Diagnostics/Anomaly_DCPP2016.py) | To compute time series of domain-mean anomalies by removing model drift in different diagnostics | 
| [Anomaly_SLP_NAO_compute.py](./Compute_Diagnostics/Anomaly_SLP_NAO_compute.py) | To compute time series of NAO indices using sea-level pressure anomalies |
| | |
| [Overturning_transport_depth_sigma.py](./Compute_Diagnostics/Overturning_transport_depth_sigma.py) | To compute meridional tranpsort in density layers and depth of density layers |
| [Overturning_Meridional_Heat_sigma_z.py](./Compute_Diagnostics/Overturning_Meridional_Heat_sigma_z.py) | To compute meridional overutrning, northward heat/salt transport in density layers and at depth levels |
| [Overturning_Ekman.py](./Compute_Diagnostics/Overturning_Ekman.py) | To compute total meridional transfort from Ekman component |
| | | 
| [Heat_budget.py](./Compute_Diagnostics/Heat_budget.py) | To compute heat convergence and heat content terms |
| [Heat_budget_Drift_Data.py](./Compute_Diagnostics/Heat_budget_Drift_Data.py) | To compute heat convergence and heat content terms using model drift data |
| | |
| [Combine_Data_Atmosphere_var.py](./Combine_Files/Combine_Data_Atmosphere_var.py) | To save different atmospheric diagnostics in the same file |
| [Combine_Data_Ocean_Surface_grid_T.py](./Combine_Files/Combine_Data_Ocean_Surface_grid_T.py) | To save different ocean diagnostics in the same file |
| [Combine_Data_Overturning.py](./Combine_Files/Combine_Data_Overturning.py) | To save overturning and heat/salt transport diagnostics in the same file |
| [Combine_overturning_transport_sigma.py](./Combine_Files/Combine_overturning_transport_sigma.py) | To regrid transport, thickness in density layers to coarse 2D grid and save combined data |
| | |
| [Composite_Diagnostics_NAO_phases.py](./Compute_Diagnostics/Composite_Diagnostics_NAO_phases.py) | To save composites of ocean/atmosphere vars for selected members based on NAO extreme values |
| [Composite_Overturning_NAO_extremes.py](./Compute_Diagnostics/Composite_Overturning_NAO_extremes.py) | To save composites of overturning-related diagnostics for selected members based on NAO extreme values |
| [Composite_Heat_Budget_NAO_extreme.py](./Compute_Diagnostics/Composite_Heat_Budget_NAO_extreme.py) | To save composites of heat budget diagnostics for selected members based on NAO extreme values |
| | |
| [Seasonal_Surface_Diag_NAO.py](./Bootstrapping_significance_testing/Seasonal_Surface_Diag_NAO.py) | To compute bootstrap confidence intervals for seasonal-mean ocean surface diagnostics corresponding to NAO extremes |
| [Annual_Heat_Overturning.py](./Bootstrapping_significance_testing/Annual_Heat_Overturning.py) | To compute boot
strap confidence intervals for annual-mean diagnostics (spatial pattern) |
| [Tracer_Depth_Lon.py](./Bootstrapping_significance_testing/Tracer_Depth_Lon.py) | To compute boot
strap confidence intervals for annual-mean diagnostics (depth-lon structure) |
| [Timeseries_SST_Heat_budget.py](./Bootstrapping_significance_testing/Timeseries_SST_Heat_budget.py) | To compute boot strap confidence intervals for area-integrated diagnostics (timeseries) |
| [Timeseries_Overturning_MHT.py](./Bootstrapping_significance_testing/Timeseries_Overturning_MHT.py) | To compute boot strap confidence intervals for overturning diagnostics (timeseries) |
| | |
| [sbatch_job_submit](./sbatch_job_submit) | To submit python jobs on slurm cluster | 
