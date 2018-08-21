'''
Created on Thu Apr 20 19:20:43 2017
@author: riddhisw

.. module:: name

    :synopsis: descrip.

    Module Level Functions:
    ----------------------
        name : descrp.
        name : descrp.
        name : descrp.

.. moduleauthor:: Riddhi Gupta <riddhi.sw@gmail.com>
'''

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

GRIDDICT = { "QUBIT_1" : (1., 0.0),
             "QUBIT_2" : (2., 0.0),
             "QUBIT_3" : (3., 0.0),
             "QUBIT_4" : (4., 0.0),
             "QUBIT_5" : (5., 0.0),
           }

def gaussian_kernel(dist_jq, f_est_j, r_est_j):
    '''docstring'''
    argument = -1.0*dist_jq**2 / (2.0*r_est_j**2) # TODO : Revist whether there needs to be a normalisation factor
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
               "LAMBDA" : 0.99, # DONE # Forgetting factor for quasi-msmt information
               "GAMMA_T" : 10**8, # Re-sampling threshold
               "P_ALPHA" : 20, # Number of alpha particles
               "P_BETA" : 40, # Numer of beta particles for each alpha
               "kernel_function" : gaussian_kernel
              }

CONTROLDICT = {"Test_0": [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]],
               "Test_1": [[0, 0]*15, [0, 1]*15, [0, 2]*15, [0, 3]*15, [0, 4]*15],
               "Test_2": [ [0, 4]*5, [0, 4]*5 + [2], [0, 2, 4]*5 ],
               "Test_3": [ [0, 4]*20, [0, 4]*20 + [2], [0, 2, 4]*20 ],
              }

NUM_QUBITS = len(GRIDDICT)

TRUENOISE = {"Uniform": np.pi*0.8*np.ones(NUM_QUBITS), 
             "Linear": np.arange(NUM_QUBITS) * np.pi/NUM_QUBITS
            }