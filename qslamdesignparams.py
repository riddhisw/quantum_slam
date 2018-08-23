'''
Created on Thu Apr 20 19:20:43 2017
@author: riddhisw

.. module:: qslamdesignparams

    :synopsis: Sets model design parameters, initial conditions, prior
        probability distributions, transition probability models, and
        filtering parameters for qslam.

    Module Level Dictionaries:
    -------------------------

    GRIDDICT : Defines spatial arrangement for a set of qubits.
        GRIDDICT["KEY"] = VALUE
        "KEY" : Qubit ID
        VALUE : (x, y) spatial coordinates for the qubit.

    NOISEPARAMS : Defines sub-dictionary for each noise injection
        with mean and second moment for a Gaussian white process stored in
        each sub-dictionary.

        List of Sub-Dictionary Keys
            Accessed as NOISEPARAMS["KEY"]
            Mean: NOISEPARAMS["KEY"]["MU"]
            Covariance:  NOISEPARAMS["KEY"]["SIGMA"]
        ============================================================

        "SIGMOID_APPROX_ERROR" :: Approximation error in modelling a continuously varying
            function with a sigmoid.

        "QUANTISATION_UNCERTY" :: Uncertainity in our knowledge of the Born probability
            prior to take a projective (quantised) boolean measurement.

        "CORRELATION_DYNAMICS" :: Noisy time evolution of true correlation length
            scales in the underlying field.

        "SPATIAL_NOISE_JITTER" :: Spatio-temporal environmental jitter that
            manifests as uncertainty in our knowledge of the qubit positions.

    MODELDESIGN : Defines model parameters for particle filtering SLAM solution.
        LAMBDA_1 (`float` | scalar):
            Spatial quasi-msmt forgetting factor for sample probabilities
        LAMBDA_2 (`float` | scalar):
            Spatial quasi-msmt forgetting factor for smeared phase information
        GAMMA_T (`float` | scalar):
            Threshold for initiating adaptive resampling [NOT USED].
        P_ALPHA (`int` | scalar):
            Number of Alpha particles in particle filtering
        P_BETA (`int` | scalar):
            Number of Beta particles for every Alpha parent.
        MULTIPLER_R_MIN (`float` | scalar):
            Sets R_MIN based on qubit grid.
        MULTIPLER_R_MAX (`float` | scalar):
            Sets R_MAX based on qubit grid.
        MULTIPLER_MEAN_RADIUS (`float` | scalar):
            Sets mean radius according to posterior r_state??
        MAX_WEIGHT_CUTOFF (`float` | scalar):
            In ParticleSet.calc_weights_set() object, resets extremely large weights
        MSMTS_PER_NODE (`int`  | scalar): Number of measurements per qubit per iteration before information is
            exchanged with neighbours.
    
    PRIORDICT : Defines sub-dictionaries for each prior distributions
        (functional form and functional arguments) for position and correlation
        lengthscale state variables.

        List of Sub-Dictionary Keys
            Accessed as PRIORDICT["KEY"]
            Functional Form: PRIORDICT["KEY"]["FUNCTION"]
            Functional Args:  PRIORDICT["KEY"]["ARGS"]
        ============================================================

        "SAMPLE_F" : Return samples from the prior distribution for qubit
            phase estimate at a given position, assumed to be:
                FUNCTION: Uniform distribution.
                ARGS["F_MIN"] : Lower bound for uniform distribution
                ARGS["F_MAX"]: Upper bound for uniform distribution.
                ARGS["SIZE"]: Number of samples to return.

        "SAMPLE_R" : Return samples from the prior distribution for qubit
            correlation length at a given position, assumed to be:
                FUNCTION: Uniform distribution.
                ARGS["R_MIN"] : Lower bound for uniform distribution
                ARGS["R_MAX"]: Upper bound for uniform distribution.
                ARGS["SIZE"]: Number of samples to return.

        "SAMPLE_X" : Return samples from the prior distribution for qubit
            X-position state variable, assumed to be:
                FUNCTION: Gaussian distribution.
                ARGS["MEAN"] : Mean of the Gaussian distribution.
                ARGS["VAR"]: Variance of the Gaussian distribution.
                ARGS["SIZE"]: Number of samples to return.

        "SAMPLE_Y" : Return samples from the prior distribution for qubit
            Y-position state variable, assumed to be:
                FUNCTION: Gaussian distribution.
                ARGS["MEAN"] : Mean of the Gaussian distribution.
                ARGS["VAR"]: Variance of the Gaussian distribution.
                ARGS["SIZE"]: Number of samples to return.



moduleauthor:: Riddhi Gupta <riddhi.sw@gmail.com>
'''

########################################### QUBIT SPATIAL ARRANGEMENT ##########

import numpy as np

GRIDDICT = {"QUBIT_1" : (1., 0.0),
            "QUBIT_2" : (2., 0.0),
            "QUBIT_3" : (3., 0.0),
            "QUBIT_4" : (4., 0.0),
            "QUBIT_5" : (5., 0.0),
           }

####################################### PROCESS AND MEASUREMENT NOISE ##########

NOISEPARAMS = {"SIGMOID_APPROX_ERROR" : {"MU": 0.0, "SIGMA" : (np.pi*0.01)**2},
               "QUANTISATION_UNCERTY" : {"MU": 0.0, "SIGMA" : 0.001},
               "CORRELATION_DYNAMICS" : {"MU": 0.0, "SIGMA" : 0.0},
               "SPATIAL_NOISE_JITTER" : {"MU": 0.0, "SIGMA" : 0.0}
              }

########################################## PARTICLE FILTER PARAMETERS ##########

def gaussian_kernel(dist_jq, f_est_j, r_est_j):
    '''docstring'''
    # TODO : Revist whether there needs to be a normalisation factor
    argument = -1.0*dist_jq**2 / (2.0*r_est_j**2)
    kernel_val = f_est_j*np.exp(argument)
    return kernel_val

FIX_LAMBDAS = 0.99
MODELDESIGN = {"LAMBDA_1" : FIX_LAMBDAS, # Forgetting factor for sample probabilities
               "LAMBDA_2" : FIX_LAMBDAS, # Forgetting factor for smeare phases
               "GAMMA_T" : 100000.0, # Re-sampling threshold -- NOT USED
               "P_ALPHA" : 20, # Number of alpha particles
               "P_BETA" : 40, # Numer of beta particles for each alpha
               "kernel_function" : gaussian_kernel,
               "MULTIPLER_R_MIN" : 1.0,  # Sets R_Min based on qubit grid.
               "MULTIPLER_R_MAX" : 10.0, # Sets R_Max based on qubit grid.
               "MULTIPLER_MEAN_RADIUS" : 1.0, # Sets mean radius according to posterior r_state??
               "MAX_WEIGHT_CUTOFF" : 100000.0,
               "MSMTS_PER_NODE" : 1
              }

################################################# PRIOR DISTRIBUTIONS ##########



def sample_s_prior(**args):
    '''Return samples from the prior distribution for x or y coordinate of a qubit.'''
    MEAN = args["MEAN"]
    VAR = args["VAR"]
    SIZE = args["SIZE"]

    samples = np.random.normal(loc=MEAN,
                               scale=VAR, # TODO: st dev or variance?
                               size=SIZE)
    return samples

S_PRIOR_ARGS = {"MEAN" : None,
                "VAR" : None,
                "SIZE" : None}


def sample_f_prior(**args):
    '''Return samples from the prior distribution for phase estimate at a qubit.'''
    F_MIN = args["F_MIN"]
    F_MAX = args["F_MAX"]
    SIZE = args["SIZE"]

    samples = np.random.uniform(low=F_MIN,
                                high=F_MAX,
                                size=SIZE)
    return samples

F_PRIOR_ARGS = {"F_MIN" : 0.0,
                "F_MAX" : np.pi,
                "SIZE" : None}

def sample_r_prior(**args):
    '''Return samples from the prior distribution for correlation length at a qubit.'''
    R_MIN = args["R_MIN"]
    R_MAX = args["R_MAX"]
    SIZE = args["SIZE"]

    samples = np.random.uniform(low=R_MIN,
                                high=R_MAX,
                                size=SIZE)
    return samples

R_PRIOR_ARGS = {"R_MIN" : None,
                "R_MAX" : None,
                "SIZE" : None
               }


PRIORDICT = {"SAMPLE_X" : {"FUNCTION": sample_s_prior, "ARGS": S_PRIOR_ARGS},
             "SAMPLE_Y" : {"FUNCTION": sample_s_prior, "ARGS": S_PRIOR_ARGS},
             "SAMPLE_F" : {"FUNCTION": sample_f_prior, "ARGS": F_PRIOR_ARGS},
             "SAMPLE_R" : {"FUNCTION": sample_r_prior, "ARGS": R_PRIOR_ARGS}
            }
