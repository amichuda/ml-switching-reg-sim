# %%
#!%load_ext autoreload


# %%
#!%autoreload 2

import os
import sys

os.chdir("/home/michuda/Dissertation")
sys.path.append("/home/michuda/Dissertation")
from UgandaUber.Estimation.modules.monte_carlo import UberMonteCarlo
import numpy as np
from UgandaUber.Estimation.modules.utils import create_list_covariance_matrices


# %%
umc = UberMonteCarlo()

#! Regimes = 2

N_within = 5
n_jobs = 7

weights = [0.1, 0.7]

# Default data construction parameters
construct_kwds = {
    "y_sd": [1,1],
    "drought_mean": [0, 0],
    "drought_cov": [1,1],
    "beta0": [20,35],
    "beta1": [-1,-2],
    "y_name": "y",
    "weight": 0.1,
    "reg_ready": False,
    "output_true_beta": True,
    "output_sigma" : True
}


# %%
#! Change STN so each regime changes at same rate

y_sd_changes_all = np.append(
    np.linspace(0.25,4,2)[:,np.newaxis], 
    np.linspace(0.5,8,2)[:,np.newaxis], 
    axis=1
    )

dl_ysd = umc.change_param(n_jobs_within = n_jobs, 
                      N_within = N_within, 
                      param_dict={'y_sd' : y_sd_changes_all, 'weight' : weights},
                      construct_dict = construct_kwds)

umc.simulate(dl_ysd, n_jobs=n_jobs, directory = "test", overwrite=True, return_estimator=True)

# %%

# # Get matrix of covariate shocks
drought_shocks = create_list_covariance_matrices()

dl_drought_shock = umc.change_param(n_jobs_within = n_jobs, 
                      N_within = N_within, 
                      param_dict={'drought_cov' : drought_shocks, 'weight' : weights},
                      construct_dict = construct_kwds)

umc.simulate(dl_drought_shock, n_jobs=n_jobs, directory = "drought_shocks_two_weights_new3", overwrite=True)

# %%

# #! Regimes = 3


# Default data construction parameters
construct_kwds_3 = {
    "y_sd": [1,1,1],
    "drought_mean": [0, 0,0],
    "drought_cov": [1,1,1],
    "beta0": [10,20,35],
    "beta1": [-1,-2,-3],
    "y_name": "y",
    "weight": 0.1,
    "reg_ready": False,
    "output_true_beta": True,
    'output_sigma' : True
}

umc_3 = UberMonteCarlo(regimes=3)


y_sd_changes_3 = np.append(
    np.append(
    np.linspace(0.25,4,10)[:,np.newaxis], 
    np.linspace(0.5,8,10)[:,np.newaxis], 
    axis=1
    ), np.linspace(0.75,12,10)[:, np.newaxis],
    axis=1)

dl_ysd = umc_3.change_param(n_jobs_within = n_jobs, 
                      N_within = N_within, 
                      param_dict={'y_sd' : y_sd_changes_3, 'weight' : weights},
                      construct_dict = construct_kwds_3)

umc_3.simulate(dl_ysd, 
               n_jobs=n_jobs, 
               directory = "y_sd_sym_3_two_weights_new3", overwrite=True)

# # %%
# #! Drought shocks with 3

drought_shocks = create_list_covariance_matrices(size=3)

dl_drought_shock_3 = umc_3.change_param(n_jobs_within = n_jobs, 
                      N_within = N_within, 
                      param_dict={'drought_cov' : drought_shocks, 'weight' : weights},
                      construct_dict = construct_kwds_3)

umc_3.simulate(dl_drought_shock_3, 
               n_jobs=n_jobs, 
               directory = 'drought_shocks_3_two_weights_new3', overwrite=True)


# #!! Regime Similarity 2, 3
# # %%
# Beta 1 difference
umc = UberMonteCarlo()

beta0_diff = np.append(
    np.zeros(10)[:,np.newaxis], 
    np.linspace(0,2,10)[:, np.newaxis], 
    axis=1
    )

dl_beta0 = umc.change_param(n_jobs_within = n_jobs, 
                      N_within = N_within, 
                      param_dict={'beta0' : beta0_diff, 'weight' : weights},
                      construct_dict = construct_kwds)

umc.simulate(dl_beta0, 
             n_jobs=n_jobs, 
             directory = 'beta0_diff_two_weights3', overwrite=True)
#%%
#! Now beta1, regimes=2
beta1_diff = np.append(
    np.zeros(10)[:,np.newaxis], 
    np.linspace(-2,2,10)[:, np.newaxis], 
    axis=1
    )

# Based on Michael's rec
beta1_diff = np.append(
    np.zeros(10)[:,np.newaxis], 
    np.linspace(0,-2,10)[:, np.newaxis], 
    axis=1
    )

dl_beta1 = umc.change_param(n_jobs_within = n_jobs, 
                      N_within = N_within, 
                      param_dict={'beta1' : beta1_diff, 'weight' : weights},
                      construct_dict = construct_kwds)

umc.simulate(dl_beta1, n_jobs=n_jobs, directory='beta1_diff_two_weights_new3', overwrite=True)

# # %%
# #! 3 regimes

# # Keep variance the same but add regime

umc_3 = UberMonteCarlo(regimes=3)

# Now append that to beta0_diff

beta0_diff_3 = np.append(
    beta0_diff, 
    2*beta0_diff[:,1][:, np.newaxis], 
    axis=1
    )

# Based on Michael's Rec
beta1_diff_3 = np.append(
    beta1_diff, 
    np.linspace(0,2,10)[:, np.newaxis], 
    axis=1
    )

# # Now run simulation

# # beta0

dl_beta0_3 = umc_3.change_param(n_jobs_within = n_jobs, 
                      N_within = N_within, 
                      param_dict={'beta0' : beta0_diff_3, 'weight' : weights},
                      construct_dict = construct_kwds_3)

umc_3.simulate(dl_beta0_3, 
               n_jobs=n_jobs, 
               directory = 'beta0_diff_3_two_weights3', overwrite=True)

# # %%

dl_beta1_3 = umc_3.change_param(n_jobs_within = n_jobs, 
                      N_within = N_within, 
                      param_dict={'beta1' : beta1_diff_3, 'weight' : weights},
                      construct_dict = construct_kwds_3)

umc_3.simulate(dl_beta1_3, n_jobs=n_jobs, 
               directory='beta1_diff_3_two_weights_new3',
               overwrite=True)

# # %%

#*  APPENDIX

# %%
# # #! Change STN so higher regime gets noisier

# y_sd_higher = np.append(
#     np.linspace(3,6,10)[:,np.newaxis],
#     np.ones((10,1)), 
#     axis=1
#     )

# dl_second_noisy = umc.change_param(n_jobs_within = n_jobs, 
#                       N_within = N_within, 
#                       param_dict={'y_sd' : y_sd_higher},
#                       construct_dict = construct_kwds)

# umc.simulate(dl_second_noisy, n_jobs=10, save=True)

# # %%
# # #! Change STN so lower regime gets noisier

# y_sd_lower = np.append(
#     np.ones((10,1)), 
#     np.linspace(1,3,10)[:,np.newaxis],
#     axis=1
#     )

# dl_first_noisy = umc.change_param(n_jobs_within = n_jobs, 
#                       N_within = N_within, 
#                       param_dict={'y_sd' : y_sd_lower},
#                       construct_dict = construct_kwds)

# umc.simulate(dl_first_noisy, n_jobs=10, save=True)