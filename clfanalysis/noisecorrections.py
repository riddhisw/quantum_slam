'''
This module has helper functions for estimating noise rates and implementing importance re-weighting.

'''
import numpy as np

def estimate_flip_rates(class_probabilities, labels):
    
    '''
    Return estimate flip rates and noise corrupted distribution parameters for 
    class conditional noise learning.
    
    Refs: Menon et. al. Equation 16 in https://pdfs.semanticscholar.org/bd3f/e8875e80f76ff41168c3bb396d9143ffc02e.pdf
    '''
    nu_max = np.max(class_probabilities)
    nu_min = np.min(class_probabilities)
    
    pi_base =  np.sum(labels) / float(len(labels))
    
    alpha_hat = (nu_min * (nu_max - pi_base)) / (pi_base*(nu_max - nu_min))
    beta_hat = ((1 - nu_max)*(pi_base - nu_min)) / ((1 - pi_base)*(nu_max - nu_min))
    pi = (pi_base - nu_min) / (nu_max - nu_min)
    
    rho_minus = (alpha_hat * pi_base) / (1.0 - pi)
    rho_plus = (beta_hat * (1.0 - pi_base)) / (pi)
    
    return alpha_hat, beta_hat, pi, rho_plus, rho_minus
    
    
def estimate_Beta(S, prob, rho0, rho1):
    '''Return beta estimate for importance re-weighting
    Refs: Tutorial 11 COMP5328'''
    
    beta = np.zeros_like(S)
    n = beta.shape[0]
    
    for i in range(n):
        
        if S[i] == 1:
            beta[i] = (prob[i][1] - rho0) / ((1 - rho0 - rho1)*(prob[i][1]))
        else:
            beta[i] = (prob[i][0] - rho1) / ((1 - rho0 - rho1)*(prob[i][0]))    
    
    return beta



def clean_weights(weights):
    '''Return normalised, non-negative weights'''
    # remove negative weights
    for i in range(len(weights)):
        if weights[i] < 0:
            weights[i] = 0.0
            
    # normalise weights
    norm_weights = weights / np.sum(weights)
    
    return norm_weights
    
    
def get_noise_estimates(binaryconfusionmatrix, samples=2000):
    ''' Return pi, rho_0, and rho_1 for noise flip rates based on a 2x2 sklearn confusion matrix.
    '''
    
    p0given0 = binaryconfusionmatrix[0,0] / samples
    p1given0 = binaryconfusionmatrix[0,1] / samples
    p0given1 = binaryconfusionmatrix[1,0] / samples
    p1given1 = binaryconfusionmatrix[1,1] / samples
    
    p0 = p0given0 + p0given1
    p1 = p1given1 + p1given0
    
    return p1, p1given0, p0given1

    
def analyse_confusion_matrix_list(cfnlist, samples=2000):
    '''Helper function for plotting noise rate estimates via a list of confusion matrices'''
    
    repts = cfnlist.shape[0]
    result = np.zeros((repts, 3))
    
    for idx_rept in range(repts):
        
        result[idx_rept, :] = get_noise_estimates(cfnlist[idx_rept, :, :] ,samples=samples)
        
    return result
    

