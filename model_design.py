'''
MODULE: model_design

Support dictionary of initial, transition and likelihood distributions for a
particle filtering solve in qslamr.py.

DICTIONARIES

-- FUNCTIONS

-- PARAMETERS

    MU_W : 0.0, # Qubit position noise mean (dynamics)
    SIG2_W : 1.0, # Qubit position noise variance (dynamics)
    MU_R : 0.0, # Length scale noise mean (dynamics)
    SIG2_R : 1.0, # Length scale noise variance (dynamics)
    MU_MEASR : 0.0, # Map noise mean (measurement)
    SIG2_MEASR : 1.0, # Map noise variance (measurement)
    MU_F : 0.0, # True sigmoid approximation error mean
    SIG2_F : 1.0, # True sigmoid approximation error variance
    LAMBDA : 0.9, # Forgetting factor for quasi-msmt information
    GAMMA_T : 10**8, # Re-sampling threshold
    P_ALPHA : 10, # Number of alpha particles
    P_BETA : 10, # Number of beta particles for each alpha

'''

import numpy as np

GRIDDICT = { "QUBIT_1" : (1., 3.5),
             "QUBIT_2" : (2., 3.5),
             "QUBIT_3" : (3., 3.5),
             "QUBIT_4" : (4., 3.5),
             "QUBIT_5" : (5., 3.5),
            # "QUBIT_6" : (4., 1.5),
            # "QUBIT_7" : (2., 3.5),
            # "QUBIT_8" : (4., 2.3),
            # "QUBIT_9" : (3.7, 1.5),
            # "QUBIT_10" : (3.2, 0.5),
            # "QUBIT_11" : (3.5, 3.5),
            # "QUBIT_12" : (4., 1.9)
           }

def gaussian_kernel(dist_jq, f_est_j, r_est_j):
    '''docstring'''
    argument = -1.0*dist_jq**2 / (2.0*r_est_j**2)
    kernel_val = f_est_j*np.exp(argument)
    return kernel_val

INITIALDICT = {"MU_W" : 0.0, # Qubit position noise mean (dynamics)
               "SIG2_W" : 1.0, # Qubit position noise variance (dynamics)
               "MU_R" : 0.0, # Length scale noise mean (dynamics)
               "SIG2_R" : 0.1**2, # Length scale noise variance (dynamics)
               "MU_MEASR" : 0.0, # Map noise mean (measurement)
               "SIG2_MEASR" : 0.000000000001, # Map noise variance (measurement)
               "MU_F" : 0.0, # True sigmoid approximation error mean
               "SIG2_F" : 0.1*np.pi**2, # True sigmoid approximation error variance
               "LAMBDA" : 0.99, # Forgetting factor for quasi-msmt information
               "GAMMA_T" : 10**8, # Re-sampling threshold
               "P_ALPHA" : 3, # Number of alpha particles
               "P_BETA" : 10, # Numer of beta particles for each alpha
               "kernel_function" : gaussian_kernel
              }
              