from model_design import INITIALDICT
from scipy.special import erf
import numpy as np
###############################################################################
# ALPHA PARTICLES
###############################################################################

def rho(b_val, var_r):
    '''docstring'''
    arg = (2*b_val)/(np.sqrt(2*var_r))
    prefactor = (1.0/(arg * np.sqrt(np.pi)))
    rho_0 = erf(arg) + prefactor*np.exp(-1*arg**2) - prefactor
    return rho_0

def likelihood_func_alpha(**args):
    '''docstring'''
    msmt_dj = args["msmt_dj"]
    prob_j = args["prob_j"]
    var_r = args["var_r"]
    rho_0 = rho(0.5, var_r)

    alpha_weight = rho_0 / 2.0
    if msmt_dj == 0:
        alpha_weight += -1.0*rho_0*(2.0*prob_j - 1)
    elif msmt_dj == 1:
        alpha_weight += 1.0*rho_0*(2.0*prob_j - 1)

    # print
    # print "in likelihood_func_alpha, msmt_dj", msmt_dj
    # print "in likelihood_func_alpha, prob_j", prob_j
    # print "in likelihood_func_alpha, var_r", var_r
    # print "in likelihood_func_alpha, rho_0", rho_0
    # print
    return alpha_weight

LIKELIHOOD_ALPHA = {"l_func" : likelihood_func_alpha,
                    "l_args" : {"mu_R" : INITIALDICT["MU_R"], # TODO:
                                "var_r" : INITIALDICT["SIG2_R"], # TODO:
                                "msmt_dj" : -10.0, # TODO: update via PF
                                "prob_j" : -10.0 # TODO: update via PF 
                               }}

def alpha_weight_calc(alpha_particle_object, **args):
    '''docstring'''
    old_weight = alpha_particle_object.weight
    likelihood = args["l_func"](**args["l_args"]) # TODO: there needs to be a better way of taking in msmts. 
    new_raw_weight = old_weight*likelihood
    print
    print "in alpha_weight_calc, old_weight", old_weight
    print "in alpha_weight_calc, likelihood", likelihood
    print
    return new_raw_weight


WEIGHTFUNCDICT_ALPHA = {"function": alpha_weight_calc, "args": LIKELIHOOD_ALPHA}

def update_alpha_dictionary(next_phys_msmt_j, prob_j):
    '''docstring'''
    LIKELIHOOD_ALPHA["msmt_dj"] = next_phys_msmt_j
    LIKELIHOOD_ALPHA["prob_j"] = prob_j
###############################################################################
# BETA PARTICLES
###############################################################################

def likelihood_func_beta(**args):
    '''E 29'''
    mean = args["mu_f"]
    variance = args["sigma_f"]
    new_phase = args["new_phase"]
    old_phase = args["old_phase"]
    prefactor = 1.0 / np.sqrt(2.0 * np.pi * variance)
    argument = -1.0 * ((new_phase - old_phase)- mean)**2 / (2.0 * variance)
    result = prefactor * np.exp(argument)
    # print
    # print "in likelihood_func_beta, new_phase", new_phase
    # print "in likelihood_func_beta, old_phase", old_phase
    # print "in likelihood_func_beta, argument", argument
    # print "in likelihood_func_beta, result", result
    # print
    return result

LIKELIHOOD_BETA = {"l_func": likelihood_func_beta,
                   "l_args": {"sigma_f" : INITIALDICT["SIG2_F"],
                              "mu_f" : INITIALDICT["MU_F"],
                              "new_phase": 0.0, # TODO: update via PF
                              "old_phase": 0.0  # TODO: update via PF
                             }
                  }

def beta_weight_calc(BetaParticle, **args):
    '''docstring'''
    # old_weight = BetaParticle.weight
    likelihood_neighbours = []

    for idx_q in range(len(BetaParticle.neighbourhood_qj)):
        args["new_phase"] = BetaParticle.smeared_phases_qj[idx_q]
        args["old_phase"] = BetaParticle.parent[idx_q]
        likelihood = args["l_func"](**args["l_args"]) # TODO: there needs to be a better way of taking in new_phase, old_phase.
        likelihood_neighbours.append(likelihood)

    net_likelihood = np.prod(np.asarray(likelihood_neighbours).flatten())
    print
    print "in likelihood_func_beta, likelihood_neighbours", likelihood_neighbours
    print
    print "in likelihood_func_beta, net_likelihood", net_likelihood
    print
    return net_likelihood


WEIGHTFUNCDICT_BETA = {"function": beta_weight_calc, "args": LIKELIHOOD_BETA}
