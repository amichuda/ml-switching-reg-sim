# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

# %%


# %%
import numpy as np
from UgandaUber.Estimation.modules.monte_carlo import SimulationVisualizer, DirectorySimulationVisualizer
from UgandaUber.Estimation.modules.utils import create_list_covariance_matrices
from functools import partial
from sklearn.model_selection import ParameterGrid


# %%
# Default data construction parameters
construct_kwds = {
    "y_sd": [1,1],
    "drought_mean": [0, 0],
    "drought_cov": [1,1],
    "beta0": [1,2],
    "beta1": [-1,-2],
    "y_name": "y",
    "weight": 0.1,
    "reg_ready": False,
    "output_true_beta": True,
}

def STN(sigma, beta0, beta1, return_single = False, return_other = False):
    
    A = np.array(beta0)/np.sqrt(np.array(beta1)**2 + np.array(sigma)**2)
    
    if return_single:
        if return_other:
            return A[1]
        else:
            return A[0]
    
    return A

STN_partial = partial(STN,
                      beta0 = construct_kwds.get('beta0'),
                      beta1 = construct_kwds.get('beta1'),
                      return_single=True)

# create multivariate signal to noise ratio based on mahalanobis distance
def STN_m(sigma, beta0, beta1):
    """Calculate multivariate STN

    Args:
        mu (np.ndarray): vector of means
        sigma (np.ndarray): variance-covariance matrix
    """
    # First get inverse of var-covar
    if isinstance(sigma, list):
        sigma = np.array(sigma)
    if isinstance(beta0, list):
        beta0 = np.array(beta0)
    if isinstance(beta1, list):
        beta1 = np.array(beta1)
            
    
    sigma_2 = beta1@sigma@beta1 + np.identity(beta1.size)
    
    sigma_inv = numpy.linalg.inv(sigma_2)
    
    return mahalanobis(beta0, 0, sigma_inv)

STN_m_partial = partial(STN_m,
                        beta0 = construct_kwds.get('beta0'),
                        beta1 = construct_kwds.get('beta1'))

STN_partial_other = partial(STN,
                      beta0 = construct_kwds.get('beta0'),
                      beta1 = construct_kwds.get('beta1'),
                      return_single=True,
                      return_other=True)

construct_kwds_3 = {
    "y_sd": [1,1,1],
    "drought_mean": [0, 0,0],
    "drought_cov": [1,1,1],
    "beta0": [1,2,3],
    "beta1": [-1,-2,-3],
    "y_name": "y",
    "weight": 0.1,
    "reg_ready": False,
    "output_true_beta": True,
}

STN_partial_3 = partial(STN,
                      beta0 = construct_kwds_3.get('beta0'),
                      beta1 = construct_kwds_3.get('beta1'),
                      return_single=True)


def get_covar(x):
    
    return np.array(x)[0,1]


# %%
v = DirectorySimulationVisualizer()


# %%
y_sd_sym = v.create_data(
    directory = 'y_sd_sym_two_weights',
    param_name='y_sd',
    other_var='weight',
    f = STN_partial,
    num_replace=1,
    regex=False,
    )

y_sd_sum_stats = v.calculate_statistics(y_sd_sym, other_var=True)

v.plot(y_sd_sum_stats,to_plot='beta_0', level_subset= ('other_var', 0.1))


# %%

y_sd_sym = v.create_data(
    directory='drought_shocks_two_weights',
    param_name='drought_cov',
    other_var='weight',
    f = get_covar,
    regex=True,
    newline_repl=True)

drought_cov_shocks = v.calculate_statistics(y_sd_sym, other_var=True)
v.plot(drought_cov_shocks,to_plot='beta_0', level_subset= ('other_var', 0.7))




# %%
v3 = DirectorySimulationVisualizer(regimes=3)

y_sd_sym = v3.create_data(
    param_name='y_sd',
    directory='y_sd_sym_3_two_weights',
    other_var='weight',
    f = STN_partial_3,
    regex=True,
    )

y_sd_sym_3 = v3.calculate_statistics(y_sd_sym, other_var=True)

v3.plot(y_sd_sym_3,  to_plot='sigma', level_subset= ('other_var', 0.7))


# %%
beta0_diff_data = v.create_data(
    param_name='beta0',
    directory = 'beta0_diff_two_weights',
    other_var='weight',
    f = lambda x: x[1],
    regex=True,
    )

beta0_diff_stats = v.calculate_statistics(beta0_diff_data,  other_var=True)
v.plot(beta0_diff_stats, to_plot='beta_1', level_subset= ('other_var', 0.7))


# %%
beta1_diff_data = v.create_data(
    param_name='beta1',
    directory = 'beta1_diff_two_weights',
    other_var='weight',
    f = lambda x: x[1],
    comma_repl=True,
    num_replace=2,
    )
beta1_diff_stats = v.calculate_statistics(beta1_diff_data,  other_var=True)
v.plot(beta1_diff_stats, to_plot='beta_1', level_subset= ('other_var', 0.1))


# %%

drought_shock_3_data = v3.create_data(
    param_name='drought_cov',
    directory = 'drought_shocks_3_two_weights',
    other_var='weight',
    f = lambda x: np.array(x)[0,1],
    regex=True,
    matrix_repl=True)

drought_shock_3_stats = v3.calculate_statistics(drought_shock_3_data, other_var=True, only_success=True)
v3.plot(drought_shock_3_stats, to_plot='beta_1', level_subset = ('other_var', 0.1))


# %%
beta_0_diff_3_data = v3.create_data(
    directory='beta0_diff_3_two_weights',
    param_name='beta0',
    other_var='weight',
    regex=True,
    )

beta_0_diff_3_stats = v3.calculate_statistics(beta_0_diff_3_data, other_var=True)
v3.plot(beta_0_diff_3_stats, regimes=3, beta='beta_1', level_subset=('other_var', 0.1))


