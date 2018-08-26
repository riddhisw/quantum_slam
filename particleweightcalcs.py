'''
Created on Thu Apr 20 19:20:43 2017
@author: riddhisw

.. module:: particleweightcalcs

    :synopsis: Centralises calculation of all particle weights and likelihood functions
        using dynamically updated dictionaries to pass arguments to weight calculation
        functions.

    Module Level Functions:
    ----------------------
        likelihood_func_alpha : Return the likelihood score for a quantised
            measurement model.
        rho : Return rho_0 - an intermediary variable - used in the calculation of a
            likelihood for a quantised measurement model [Helper Function.]
        update_alpha_dictionary : Update LIKELIHOOD_ALPHA dictionary
            based on new qubit measurement for a single node.
        alpha_weight_calc : Return the un-normalised weight of an alpha particle.

        likelihood_func_beta : Return the likelihood score for a noisy qubit phase estimate. Applies
            to weight calculation of an beta-type particle in qslam only.
        beta_weight_calc : Return the un-normalised weight of an beta particle.

    Module Level Dynamic Dictionaries:
    ---------------------------------
        LIKELIHOOD_ALPHA:
            Enables key-based calling of module function likelihood_func_alpha, with
                a dynamic update of arguments passed to likelihood_func_alpha.

        WEIGHTFUNCDICT_ALPHA :
            Enables key-based calling of module function alpha_weight_calc, with
                a dynamic update of arguments passed to alpha_weight_calc.

        LIKELIHOOD_BETA :
            Enables key-based calling of module function likelihood_func_beta, with
                a dynamic update of arguments passed to likelihood_func_beta.

        WEIGHTFUNCDICT_BETA :
            Enables key-based calling of module function beta_weight_calc, with
                a dynamic update of arguments passed to beta_weight_calc.

.. moduleauthor:: Riddhi Gupta <riddhi.sw@gmail.com>
'''

from qslamdesignparams import NOISEPARAMS
from scipy.special import erf
import numpy as np

###############################################################################
# ALPHA PARTICLE WEIGHT CALCULATIONS
###############################################################################

def rho(b_val, var_r):
    ''' Return rho_0 - an intermediary variable - used in the calculation of a
        likelihood for a quantised measurement model [Helper Function.]

        Parameters:
        ----------
            b_val (`float64`, scalar):
            Truncation point for the distribution of errors for a quantised
                measurement outcome.

            var_r (`float64`, scalar):
            Measurement noise variance strength for quantised measurement
                outcomes.
        Returns:
        -------
            rho_0 (`float64`, scalar) : Scaling factor in the likelihood function
                of a quantised measurement model.
'''
    arg = (2*b_val)/(np.sqrt(2*var_r))
    prefactor = (1.0/(arg * np.sqrt(np.pi)))
    rho_0 = erf(arg) + prefactor*np.exp(-1*arg**2) - prefactor
    return rho_0

def likelihood_func_alpha(**args):
    ''' Return the likelihood score for a quantised measurement model. Applies
    to weight calculation of an alpha-type particle in qslam only, following a measurement
    at `j`-th node in Grid.nodes().

    Parameters:
    ----------
        args["msmt_dj"] (`float64`, scalar):
            Single qubit binary (0 or 1) measurement at `j`-th node.
        args["prob_j"] (`float64`, scalar):
            Sample probability at `j`-th node over physical and quasi measurements.
        args["SIGMA"] (`float64`, scalar):
            Measurement noise variance strength for quantised measurement
                outcomes.
    Returns:
    -------
        alpha_likelihood_score (`float64`, scalar):
            The likelihood score for an Alpha particle.
    '''

    msmt_dj = args["msmt_dj"]
    prob_j = args["prob_j"]
    var_r = args["SIGMA"]
    rho_0 = rho(0.5, var_r)

    alpha_likelihood_score = rho_0 / 2.0
    if msmt_dj == 0:
        alpha_likelihood_score += -1.0*rho_0*(2.0*prob_j - 1)
    elif msmt_dj == 1:
        alpha_likelihood_score += 1.0*rho_0*(2.0*prob_j - 1)
    return alpha_likelihood_score

LIKELIHOOD_ALPHA = {"l_func" : likelihood_func_alpha,
                    "l_args" : {"MU" : NOISEPARAMS["QUANTISATION_UNCERTY"]["MU"],
                                "SIGMA" : NOISEPARAMS["QUANTISATION_UNCERTY"]["SIGMA"],
                                "msmt_dj" : -10.0, # TODO: update via PF
                                "prob_j" : -10.0 # TODO: update via PF
                               }}

def alpha_weight_calc(alpha_particle_object, **args):
    '''Return the un-normalised weight of an alpha particle.

    Parameters:
    ----------
    alpha_particle_object (`Particle` class object):
        A single Alpha particle instance.

    args (Dictionary object) :
        "l_func" : contains the likelihood function used to score Alpha particles.
        "1_args" : contains a dictionary of arguments to be passed to args["l_func"].

    Returns:
    -------
        new_raw_weight : The likelihood score for an Alpha particle. Interpreted as
            the unnormalised posterior weight for the Alpha particle under
            importance sampling.
    '''

    old_weight = alpha_particle_object.weight
    # TODO: there needs to be a better way of taking in msmts.
    likelihood = args["l_func"](**args["l_args"])
    new_raw_weight = old_weight*likelihood
    return new_raw_weight


WEIGHTFUNCDICT_ALPHA = {"function": alpha_weight_calc, "args": LIKELIHOOD_ALPHA}


def update_alpha_dictionary(next_phys_msmt_j, prob_j):
    ''' Update LIKELIHOOD_ALPHA dictionary based on new qubit measurement for a
        single node.

        Parameters:
        ----------
            next_phys_msmt_j (`float64`, scalar):
                Single qubit binary (0 or 1) measurement outcome at a particular
                    location indexed by `j`-th Node in the list Grid.nodes.
            prob_j (`float64`, scalar):
                Sample probability at the `j`-th node, calculated over physical and
                    quasi measurements via Node.sample_prob_from_msmts()
        Returns:
        -------
            Updates dictionary values for the following keys to enable a likelihood
                calculation for an Alpha particle  following a measurement on the `j`-th
                node.

                LIKELIHOOD_ALPHA["l_args"]["msmt_dj"]: Single qubit binary (0 or 1) measurement
                LIKELIHOOD_ALPHA["l_args"]["prob_j"]: Sample probability over physical and
                    quasi measurements.

    '''
    LIKELIHOOD_ALPHA["l_args"]["msmt_dj"] = next_phys_msmt_j
    LIKELIHOOD_ALPHA["l_args"]["prob_j"] = prob_j


###############################################################################
# BETA PARTICLE WEIGHT CALCULATIONS
###############################################################################

def likelihood_func_beta(**args):
    ''' Return the likelihood score for a beta particle that depends on
        lengthscale and phase estimate between a pair of qubits.

    Parameters:
    ----------
        args["MU"] (`float64`, scalar):
            True process noise mean (unknown but discovered via optimisation?)
        args["SIGMA"] (`float64`, scalar):
            True process noise covariance scale (unknown but discovered via optimisation?)
        args["new_phase"] (`float64`, scalar):
            Sigmoid smoothened phase from `j` to its neighbourhood at `q`.
        args["old_phase"] (`float64`, scalar):
            State estimate for the phase at `q` given physical and quasi-measurements
            at `q`.

    Returns:
    -------
        Likelihood score for a beta particle that depends on lengthscale and
            phase estimate between a pair of qubits.
    '''
    mean = args["MU"]
    variance = args["SIGMA"]
    new_phase = args["new_phase"]
    old_phase = args["old_phase"]
    prefactor = 1.0 / np.sqrt(2.0 * np.pi * variance)
    argument = -1.0 * ((new_phase - old_phase)- mean)**2 / (2.0 * variance)
    result = prefactor * np.exp(argument)
    return result

LIKELIHOOD_BETA = {"l_func": likelihood_func_beta,
                   "l_args": {"SIGMA" : NOISEPARAMS["SIGMOID_APPROX_ERROR"]["SIGMA"],
                              "MU" : NOISEPARAMS["SIGMOID_APPROX_ERROR"]["MU"],
                              "new_phase" : 0.0, # TODO: update via PF
                              "old_phase" : 0.0 # TODO: update via PF
                             }
                  }

def beta_weight_calc(BetaParticle, **args):
    '''Return the un-normalised weight of a beta particle.

    Parameters:
    ----------
        args["l_func"] : Likelihood function for scoring beta particles.
        args["new_phase"] : A phase estimate at neighbour qubit q due to a physical
            measurment at qubit j.
        args["old_phase"] : A phase estimate at neighbour qubit q due to all previous
            physical and quasi measurements.
    Returns:
    -------
        net_likelihood : Total likelihood over the neighbouhood of phase estimates
            for a single Beta particle.
    '''

    likelihood_neighbours = []
    # print # TODO : Delete code. Printdebug only.
    # print "In beta_weight_calc" # TODO : Delete code. Printdebug only.
    for idx_q in range(len(BetaParticle.neighbourhood_qj)):
        # INCORRECT? SHOULDN"T THIS BE args["l_args"]["new_phase"]
        # args["new_phase"] = BetaParticle.smeared_phases_qj[idx_q]
        args["l_args"]["new_phase"] = BetaParticle.smeared_phases_qj[idx_q]
        args["l_args"]["old_phase"] = BetaParticle.parent[idx_q]
        # INCORRECT? SHOULDN"T THIS BE args["l_args"]["old_phase"]
        # args["old_phase"] = BetaParticle.parent[idx_q]
        likelihood = args["l_func"](**args["l_args"])
        likelihood_neighbours.append(likelihood) 
        # print "For the next beta likelhood calculation,..." # TODO : Delete code. Printdebug only.
        # print "LIKELHOOD BETA args are: ", args["l_args"] # TODO : Delete code. Printdebug only.
        # print "And the likelihood value is ", likelihood # TODO : Delete code. Printdebug only.
        # print # TODO : Delete code. Printdebug only.
    net_likelihood = np.prod(np.asarray(likelihood_neighbours).flatten())
    return net_likelihood


WEIGHTFUNCDICT_BETA = {"function": beta_weight_calc, "args": LIKELIHOOD_BETA}
