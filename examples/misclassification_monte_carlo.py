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


misclass_levels = np.linspace(0,1, 10)

N_within = 200
n_jobs = 7

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

#! 2 Regimes

umc = UberMonteCarlo()

dl_2 = umc.change_param(n_jobs_within = n_jobs, 
                      N_within = N_within, 
                      param_dict={'weight' : misclass_levels},
                      construct_dict = construct_kwds)

umc.simulate(dl_2, n_jobs=n_jobs, directory = "misclass_2_test", overwrite=True)

#! 3 Regimes 


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

dl_3 = umc_3.change_param(n_jobs_within = n_jobs, 
                      N_within = N_within, 
                      param_dict={'weight' : misclass_levels},
                      construct_dict = construct_kwds_3)

umc_3.simulate(dl_3, 
               n_jobs=n_jobs, 
               directory = "misclass_3_test", overwrite=True)
