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
        MAX_NUM_ITERATIONS (`int`  | scalar): Max number of iterations for a qslam
            algorithm. A single control directive corressponds to one iteration.

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

GRIDDICT = {"QUBIT_01" : (0.0, 0.0),
            "QUBIT_02" : (0.0, 1.0),
            "QUBIT_03" : (0.0, 2.0),
            "QUBIT_04" : (0.0, 3.0),
            "QUBIT_05" : (0.0, 4.0),
            "QUBIT_06" : (1.0, 0.0),
            "QUBIT_07" : (1.0, 1.0),
            "QUBIT_08" : (1.0, 2.0),
            "QUBIT_09" : (1.0, 3.0),
            "QUBIT_10" : (1.0, 4.0),
            "QUBIT_11" : (2.0, 0.0),
            "QUBIT_12" : (2.0, 1.0),
            "QUBIT_13" : (2.0, 2.0),
            "QUBIT_14" : (2.0, 3.0),
            "QUBIT_15" : (2.0, 4.0),
            "QUBIT_16" : (3.0, 0.0),
            "QUBIT_17" : (3.0, 1.0),
            "QUBIT_18" : (3.0, 2.0),
            "QUBIT_19" : (3.0, 3.0),
            "QUBIT_20" : (3.0, 4.0),
            "QUBIT_21" : (4.0, 0.0),
            "QUBIT_22" : (4.0, 1.0),
            "QUBIT_23" : (4.0, 2.0),
            "QUBIT_24" : (4.0, 3.0),
            "QUBIT_25" : (4.0, 4.0)
           }


# Make this into  A FUNCTION
LIST_OF_POSITIONS = [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0), (0.0, 4.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0), (1.0, 3.0), (1.0, 4.0), (2.0, 0.0), (2.0, 1.0), (2.0, 2.0), (2.0, 3.0), (2.0, 4.0), (3.0, 0.0), (3.0, 1.0), (3.0, 2.0), (3.0, 3.0), (3.0, 4.0), (4.0, 0.0), (4.0, 1.0), (4.0, 2.0), (4.0, 3.0), (4.0, 4.0)]


####################################### PROCESS AND MEASUREMENT NOISE ##########

NOISEPARAMS = {"SIGMOID_APPROX_ERROR" : {"MU": 0.0, "SIGMA" : 1.0},
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

MODELDESIGN = {"LAMBDA_1" : 0.99, # Forgetting factor for sample probabilities
               "LAMBDA_2" : 0.977, # Forgetting factor for smeare phases
               "GAMMA_T" : 100000.0, # Re-sampling threshold -- NOT USED
               "P_ALPHA" : 5, # Number of alpha particles
               "P_BETA" : 3, # Numer of beta particles for each alpha
               "kernel_function" : gaussian_kernel,
               "MULTIPLER_R_MIN" : 1.0,  # Sets R_Min based on qubit grid.
               "MULTIPLER_R_MAX" : 4.0, # Sets R_Max based on qubit grid.
               "MULTIPLER_MEAN_RADIUS" : 1.0, # Sets mean radius according to posterior r_state??
               "MAX_WEIGHT_CUTOFF" : 100000.0,
               "MSMTS_PER_NODE" : 5,
               "MAX_NUM_ITERATIONS" : 10,
               "ID": 'calibration_run_C0'
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

S_PRIOR_ARGS = {"MEAN" : 0.0,
                "VAR" : 10.0**(-8),
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

#################################### HYPERPARAMETER PRIOR DISTRIBUTIONS #########

def sample_hyper_dist(space_size=None, **hyper_args):
    L_MIN = hyper_args["MIN"]
    L_MAX = hyper_args["MAX"]
    
    if space_size is None:
        sample = np.random.uniform(low=L_MIN, high = L_MAX, size=1)
    
    elif space_size is not None:
        sample = func_x0(space_size)

    return sample

def func_x0(space_size):
    '''
    Returns random samples from a parameter space defined by space_size
    '''
    maxindex = space_size.shape[0]-1
    ind = int(np.random.uniform(low=0, high=maxindex))
    exponent = space_size[ind]
    return np.random.uniform(0,1)*(10.0**exponent)

hyper_args = {"LAMBDA_1": {"MIN": 0.9, "MAX": 1.0},
              "LAMBDA_2": {"MIN": 0.9, "MAX": 1.0},
              "SIGMOID_VAR": {"MIN": 0., "MAX": 1.0},
              "QUANT_VAR": {"MIN": 0., "MAX": 1.0}
              }
HYPERDICT = {"DIST" : sample_hyper_dist,
             "ARGS" : hyper_args
            }

##################################################### RISK MODEL ###############

RISKPARAMS = {"savetopath": './',
              "max_it_qslam": 1,
              "max_it_BR": 50,
              "num_randparams": 1,
              "space_size": None,
              "loss_truncation":0.1
             }

##################################################### GLOBAL MODEL ###############
GLOBALDICT = {"MODELDESIGN": MODELDESIGN,
              "PRIORDICT" : PRIORDICT,
              "NOISEPARAMS": NOISEPARAMS,
              "GRIDDICT" : GRIDDICT,
              "RISKPARAMS" : RISKPARAMS,
              "HYPERDICT": HYPERDICT
             }
