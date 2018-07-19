hardware.py:import numpy as np
hardware.py:
hardware.py:PARTICLE_STATE = ["x_state", "y_state", "f_state", "r_state"]
hardware.py:
hardware.py:###############################################################################
hardware.py:# CHIP STRUCTURE 
hardware.py:###############################################################################
hardware.py:class Node(object):
hardware.py:    '''[docstring]'''
hardware.py:
hardware.py:    def __init__(self):
hardware.py:        # self.qubitid = 0
hardware.py:        self.f_state = 0.0
hardware.py:        self.r_state = 0.0
hardware.py:        self.x_state = 0.0
hardware.py:        self.y_state = 0.0
hardware.py:        self.counter_tau = 0
hardware.py:        self.counter_beta = 0
hardware.py:        self.__physcmsmtsum = 0.0
hardware.py:        self.__quasimsmtsum = 0.0
hardware.py:
hardware.py:    @property
hardware.py:    def physcmsmtsum(self):
hardware.py:        '''docstring'''
hardware.py:        return self.__physcmsmtsum
hardware.py:    @physcmsmtsum.setter
hardware.py:    def physcmsmtsum(self, next_phys_msmt):
hardware.py:        self.__physcmsmtsum += next_phys_msmt
hardware.py:        self.counter_tau += 1
hardware.py:
hardware.py:    @property
hardware.py:    def quasimsmtsum(self):
hardware.py:        '''docstring'''
hardware.py:        return self.__quasimsmtsum
hardware.py:    @quasimsmtsum.setter
hardware.py:    def quasimsmtsum(self, next_quasi_msmt):
hardware.py:        self.__quasimsmtsum += next_quasi_msmt
hardware.py:        self.counter_beta += 1
hardware.py:
hardware.py:
hardware.py:class Grid(object):
hardware.py:    '''docstring '''
hardware.py:
hardware.py:    def __init__(self, list_of_nodes_positions=None):
hardware.py:        if list_of_nodes_positions is None:
hardware.py:            raise RuntimeError
hardware.py:
hardware.py:        self.list_of_nodes_positions = list_of_nodes_positions
hardware.py:        self.number_of_nodes = len(self.list_of_nodes_positions)
hardware.py:        self.nodes = [Node() for i in range(self.number_of_nodes)]
hardware.py:        self.__state_vector = 0.0
hardware.py:        self.state_vector = np.zeros(self.number_of_nodes*len(PARTICLE_STATE))
hardware.py:
hardware.py:        for item in xrange(self.number_of_nodes):
hardware.py:            self.nodes[item].x_state, self.nodes[item].y_state = self.list_of_nodes_positions[item]
hardware.py:
hardware.py:    def get_all_nodes(self, attribute_list):
hardware.py:        '''Returns attribute for all nodes in a single vector, stacked by nodes, then attributes '''
hardware.py:        vector = [[getattr(node, attr) for node in self.nodes] for attr in attribute_list]
hardware.py:        return np.asarray(vector).flatten()
hardware.py:
hardware.py:    def set_all_nodes(self, single_attribute, attribute_values):
hardware.py:        '''Sets attribute for all nodes '''
hardware.py:        for item in xrange(len(attribute_values)):
hardware.py:            setattr(self.nodes[item], single_attribute, attribute_values[item])
hardware.py:
hardware.py:    @property
hardware.py:    def state_vector(self):
hardware.py:        '''docstring'''
hardware.py:        self.__state_vector = self.get_all_nodes(PARTICLE_STATE)
hardware.py:        return self.__state_vector
hardware.py:    @state_vector.setter
hardware.py:    def state_vector(self, new_state_vector):
hardware.py:        '''docstring'''
hardware.py:        for state in xrange(len(PARTICLE_STATE)):
hardware.py:            bgn = state * self.number_of_nodes
hardware.py:            end = bgn + self.number_of_nodes
hardware.py:            self.set_all_nodes(PARTICLE_STATE[state], new_state_vector[bgn:end])
hardware.py:
model_design.py:'''
model_design.py:MODULE: model_design
model_design.py:
model_design.py:Support dictionary of initial, transition and likelihood distributions for a
model_design.py:particle filtering solve in qslamr.py.
model_design.py:
model_design.py:DICTIONARIES
model_design.py:
model_design.py:-- FUNCTIONS
model_design.py:
model_design.py:-- PARAMETERS
model_design.py:
model_design.py:    MU_W : 0.0, # Qubit position noise mean (dynamics)
model_design.py:    SIG2_W : 1.0, # Qubit position noise variance (dynamics)
model_design.py:    MU_R : 0.0, # Length scale noise mean (dynamics)
model_design.py:    SIG2_R : 1.0, # Length scale noise variance (dynamics)
model_design.py:    MU_MEASR : 0.0, # Map noise mean (measurement)
model_design.py:    SIG2_MEASR : 1.0, # Map noise variance (measurement)
model_design.py:    MU_F : 0.0, # True sigmoid approximation error mean
model_design.py:    SIG2_F : 1.0, # True sigmoid approximation error variance
model_design.py:    LAMBDA : 0.9, # Forgetting factor for quasi-msmt information
model_design.py:    GAMMA_T : 10**8, # Re-sampling threshold
model_design.py:    P_ALPHA : 10, # Number of alpha particles
model_design.py:    P_BETA : 10, # Number of beta particles for each alpha
model_design.py:
model_design.py:'''
model_design.py:
model_design.py:import numpy as np
model_design.py:
model_design.py:GRIDDICT = {"QUBIT_1" : (4., 3.5),
model_design.py:            "QUBIT_2" : (1., 2.5),
model_design.py:            "QUBIT_3" : (0., 3.5),
model_design.py:            # "QUBIT_4" : (4.1, 0),
model_design.py:            # "QUBIT_5" : (2.0, 2.5),
model_design.py:            # "QUBIT_6" : (4., 1.5),
model_design.py:            # "QUBIT_7" : (2., 3.5),
model_design.py:            # "QUBIT_8" : (4., 2.3),
model_design.py:            # "QUBIT_9" : (3.7, 1.5),
model_design.py:            # "QUBIT_10" : (3.2, 0.5),
model_design.py:            # "QUBIT_11" : (3.5, 3.5),
model_design.py:            # "QUBIT_12" : (4., 1.9)
model_design.py:           }
model_design.py:
model_design.py:def gaussian_kernel(dist_jq, f_est_j, r_est_j):
model_design.py:    '''docstring'''
model_design.py:    argument = -1.0*dist_jq**2 / (2.0*r_est_j**2)
model_design.py:    kernel_val = f_est_j*np.exp(argument)
model_design.py:    return kernel_val
model_design.py:
model_design.py:INITIALDICT = {"MU_W" : 0.0, # Qubit position noise mean (dynamics)
model_design.py:               "SIG2_W" : 1.0, # Qubit position noise variance (dynamics)
model_design.py:               "MU_R" : 0.0, # Length scale noise mean (dynamics)
model_design.py:               "SIG2_R" : 1.0, # Length scale noise variance (dynamics)
model_design.py:               "MU_MEASR" : 0.0, # Map noise mean (measurement)
model_design.py:               "SIG2_MEASR" : 1.0, # Map noise variance (measurement)
model_design.py:               "MU_F" : 0.0, # True sigmoid approximation error mean
model_design.py:               "SIG2_F" : 1.0, # True sigmoid approximation error variance
model_design.py:               "LAMBDA" : 0.99, # Forgetting factor for quasi-msmt information
model_design.py:               "GAMMA_T" : 10**8, # Re-sampling threshold
model_design.py:               "P_ALPHA" : 3, # Number of alpha particles
model_design.py:               "P_BETA" : 3, # Numer of beta particles for each alpha
model_design.py:               "kernel_function" : gaussian_kernel
model_design.py:              }
model_design.py:              
particledict.py:from model_design import INITIALDICT
particledict.py:from scipy.special import erf
particledict.py:import numpy as np
particledict.py:###############################################################################
particledict.py:# ALPHA PARTICLES
particledict.py:###############################################################################
particledict.py:
particledict.py:def rho(b_val, var_r):
particledict.py:    '''docstring'''
particledict.py:    arg = (2*b_val)/(np.sqrt(2*var_r))
particledict.py:    prefactor = (1.0/(arg * np.sqrt(np.pi)))
particledict.py:    rho_0 = erf(arg) + prefactor*np.exp(-1*arg**2) - prefactor
particledict.py:    return rho_0
particledict.py:
particledict.py:def likelihood_func_alpha(**args):
particledict.py:    '''docstring'''
particledict.py:    msmt_dj = args["msmt_dj"]
particledict.py:    prob_j = args["prob_j"]
particledict.py:    var_r = args["var_r"]
particledict.py:    rho_0 = rho(0.5, var_r)
particledict.py:
particledict.py:    alpha_weight = rho_0 / 2.0
particledict.py:    if msmt_dj == 0:
particledict.py:        alpha_weight += -1.0*rho_0*(2.0*prob_j - 1)
particledict.py:    elif msmt_dj == 1:
particledict.py:        alpha_weight += 1.0*rho_0*(2.0*prob_j - 1)
particledict.py:
particledict.py:    return alpha_weight
particledict.py:
particledict.py:LIKELIHOOD_ALPHA = {"l_func" : likelihood_func_alpha,
particledict.py:                    "l_args" : {"mu_R" : INITIALDICT["MU_R"], # TODO:
particledict.py:                                "var_r" : INITIALDICT["SIG2_R"], # TODO:
particledict.py:                                "msmt_dj" : -10.0, # TODO: update via PF
particledict.py:                                "prob_j" : -10.0 # TODO: update via PF 
particledict.py:                               }}
particledict.py:
particledict.py:def alpha_weight_calc(alpha_particle_object, **args):
particledict.py:    '''docstring'''
particledict.py:    old_weight = alpha_particle_object.weight
particledict.py:    likelihood = args["l_func"](**args["l_args"]) # TODO: there needs to be a better way of taking in msmts. 
particledict.py:    new_raw_weight = old_weight*likelihood
particledict.py:    return new_raw_weight
particledict.py:
particledict.py:
particledict.py:WEIGHTFUNCDICT_ALPHA = {"function": alpha_weight_calc, "args": LIKELIHOOD_ALPHA}
particledict.py:
particledict.py:def update_alpha_dictionary(next_phys_msmt_j, prob_j):
particledict.py:    '''docstring'''
particledict.py:    LIKELIHOOD_ALPHA["msmt_dj"] = next_phys_msmt_j
particledict.py:    LIKELIHOOD_ALPHA["prob_j"] = prob_j
particledict.py:###############################################################################
particledict.py:# BETA PARTICLES
particledict.py:###############################################################################
particledict.py:
particledict.py:def likelihood_func_beta(**args):
particledict.py:    '''E 29'''
particledict.py:    mean = args["mu_f"]
particledict.py:    variance = args["sigma_f"]
particledict.py:    new_phase = args["new_phase"]
particledict.py:    old_phase = args["old_phase"]
particledict.py:    prefactor = 1.0 / np.sqrt(2.0 * np.pi * variance)
particledict.py:    argument = -1.0 * ((new_phase - old_phase)- mean)**2 / (2.0 * variance)
particledict.py:    result = prefactor * np.exp(argument)
particledict.py:    return result
particledict.py:
particledict.py:LIKELIHOOD_BETA = {"l_func": likelihood_func_beta,
particledict.py:                   "l_args": {"sigma_f" : INITIALDICT["SIG2_F"],
particledict.py:                              "mu_f" : INITIALDICT["MU_F"],
particledict.py:                              "new_phase": 0.0, # TODO: update via PF
particledict.py:                              "old_phase": 0.0  # TODO: update via PF
particledict.py:                             }
particledict.py:                  }
particledict.py:
particledict.py:def beta_weight_calc(BetaParticle, **args):
particledict.py:    '''docstring'''
particledict.py:    # old_weight = BetaParticle.weight
particledict.py:    likelihood_neighbours = []
particledict.py:
particledict.py:    for idx_q in range(len(BetaParticle.neighbourhood_qj)):
particledict.py:        args["new_phase"] = BetaParticle.smeared_phases_qj[idx_q]
particledict.py:        args["old_phase"] = BetaParticle.parent[idx_q]
particledict.py:        likelihood = args["l_func"](**args["l_args"]) # TODO: there needs to be a better way of taking in new_phase, old_phase.
particledict.py:        likelihood_neighbours.append(likelihood)
particledict.py:
particledict.py:    net_likelihood = np.prod(np.asarray(likelihood_neighbours).flatten())
particledict.py:    return net_likelihood
particledict.py:
particledict.py:
particledict.py:WEIGHTFUNCDICT_BETA = {"function": beta_weight_calc, "args": LIKELIHOOD_BETA}
particlesets.py:import numpy as np
particlesets.py:from particledict import WEIGHTFUNCDICT_BETA
particlesets.py:
particlesets.py:###############################################################################
particlesets.py:# PARTICLE STRUCTURE
particlesets.py:###############################################################################
particlesets.py:
particlesets.py:class Particle(object):
particlesets.py:    '''docstring'''
particlesets.py:    def __init__(self):
particlesets.py:
particlesets.py:        self.__particle = 0.0
particlesets.py:        self.__weight = 0.0
particlesets.py:
particlesets.py:    @property
particlesets.py:    def particle(self):
particlesets.py:        '''docstring'''
particlesets.py:        return self.__particle
particlesets.py:    @particle.setter
particlesets.py:    def particle(self, new_state_vector):
particlesets.py:        '''docstring'''
particlesets.py:        self.__particle = new_state_vector
particlesets.py:
particlesets.py:    @property
particlesets.py:    def weight(self):
particlesets.py:        '''docstring'''
particlesets.py:        return self.__weight
particlesets.py:    @weight.setter
particlesets.py:    def weight(self, new_weight):
particlesets.py:        '''docstring'''
particlesets.py:        self.__weight = new_weight
particlesets.py:
particlesets.py:
particlesets.py:class BetaParticle(Particle):
particlesets.py:    '''docstring'''
particlesets.py:
particlesets.py:    def __init__(self, node_j, parent_state):
particlesets.py:        Particle.__init__(self)
particlesets.py:
particlesets.py:        self.parent = parent_state
particlesets.py:        self.particle = np.asarray(parent_state).flatten() # intiialised identically to parent
particlesets.py:        self.total_nodes = int(float(len(parent_state)) / 4.0)
particlesets.py:
particlesets.py:        self.node_j = node_j
particlesets.py:        self.neighbourhood_qj = []
particlesets.py:        self.neighbour_dist_qj = []
particlesets.py:        self.smeared_phases_qj = []
particlesets.py:        self.x_j, self.y_j, self.f_j, self.r_j = self.parent[self.node_j::self.total_nodes]
particlesets.py:
particlesets.py:        self.mean_radius = self.r_j*3.0 #mean_radius_j # TODO: Change to self.r_j*3.0
particlesets.py:
particlesets.py:    def get_neighbourhood_qj(self):
particlesets.py:        '''doc string'''
particlesets.py:
particlesets.py:        self.neighbourhood_qj = []
particlesets.py:        self.neighbour_dist_qj = []
particlesets.py:
particlesets.py:        for idx in range(self.total_nodes):
particlesets.py:
particlesets.py:            xq_ = self.parent[idx]
particlesets.py:            yq_ = self.parent[idx + self.total_nodes]
particlesets.py:            dist = np.sqrt((xq_ - self.x_j)**2 + (yq_ - self.y_j)**2)
particlesets.py:
particlesets.py:            if dist <= self.mean_radius:
particlesets.py:                self.neighbourhood_qj.append(idx)
particlesets.py:                self.neighbour_dist_qj.append(dist)
particlesets.py:
particlesets.py:
particlesets.py:    def smear_fj_on_neighbours(self, **args):
particlesets.py:        '''docstring'''
particlesets.py:
particlesets.py:        self.get_neighbourhood_qj()
particlesets.py:
particlesets.py:        prev_posterior_f_state = args["prev_posterior_f_state"]
particlesets.py:        prev_counter_tau_state = args["prev_counter_tau_state"]
particlesets.py:        lambda_ = args["lambda_"]
particlesets.py:
particlesets.py:        self.smeared_phases_qj = []
particlesets.py:        for idx_q in range(len(self.neighbourhood_qj)):
particlesets.py:
particlesets.py:            node_q = self.neighbourhood_qj[idx_q]
particlesets.py:            dist_jq = self.neighbour_dist_qj[idx_q]
particlesets.py:            tau_q = prev_counter_tau_state[node_q]
particlesets.py:            f_state_q = prev_posterior_f_state[node_q]
particlesets.py:
particlesets.py:            lambda_q = lambda_** tau_q
particlesets.py:            kernel_val = args["kernel_function"](dist_jq, self.f_j, self.r_j)
particlesets.py:
particlesets.py:            smear_phase = (1.0 - lambda_q)*f_state_q + lambda_q*kernel_val
particlesets.py:            self.smeared_phases_qj.append(smear_phase)
particlesets.py:
particlesets.py:
particlesets.py:class AlphaParticle(Particle):
particlesets.py:    '''docstring'''
particlesets.py:    def __init__(self):
particlesets.py:        Particle.__init__(self)
particlesets.py:        self.pset_beta = 0
particlesets.py:        self.node_j = 0.0
particlesets.py:        # self.mean_radius_j = 0.0 # TODO FIX THIS. CANT BE ONE NUMBER FORALL j
particlesets.py:        self.BetaAlphaSet_j = None
particlesets.py:
particlesets.py:    def generate_beta_pset(self, parents): #, number_of_beta_particles):
particlesets.py:        '''docstring'''
particlesets.py:        beta_s = [BetaParticle(self.node_j, state) for state in parents]
particlesets.py:        self.BetaAlphaSet_j = ParticleSet(beta_s, **WEIGHTFUNCDICT_BETA)
particlesets.py:        # return BetaAlphaSet
particlesets.py:
particlesets.py:class ParticleSet(object):
particlesets.py:    '''docstring'''
particlesets.py:    def __init__(self, list_of_particle_objects, **WEIGHTFUNCDICT):
particlesets.py:
particlesets.py:        self.__weights_set = 0.0
particlesets.py:        self.__posterior_state = 0.0
particlesets.py:        self.p_set = len(list_of_particle_objects)
particlesets.py:        self.particles = list_of_particle_objects
particlesets.py:        self.w_dict = WEIGHTFUNCDICT
particlesets.py:        self.weights_set = (1.0 / self.p_set)*np.ones(self.p_set)
particlesets.py:
particlesets.py:    def calc_weights_set(self):
particlesets.py:        '''docstring'''
particlesets.py:        new_weight_set = []
particlesets.py:        for particle in self.particles:
particlesets.py:            new_weight = self.w_dict["function"](particle, **self.w_dict["args"])
particlesets.py:            new_weight_set.append(new_weight)
particlesets.py:            
particlesets.py:        raw_weights = np.asarray(new_weight_set).flatten()
particlesets.py:        normalisation = 1.0/np.sum(raw_weights)
particlesets.py:        return normalisation*raw_weights
particlesets.py:
particlesets.py:    @property
particlesets.py:    def weights_set(self):
particlesets.py:        '''docstring'''
particlesets.py:        self.__weights_set = np.asarray([particle.weight for particle in self.particles]).flatten()
particlesets.py:        return self.__weights_set
particlesets.py:    @weights_set.setter
particlesets.py:    def weights_set(self, new_weights):
particlesets.py:        '''docstring'''
particlesets.py:        for idxp in range(self.p_set):
particlesets.py:            self.particles[idxp].weight = new_weights[idxp]
particlesets.py:
particlesets.py:    @property
particlesets.py:    def posterior_state(self):
particlesets.py:        '''docstring'''
particlesets.py:
particlesets.py:        posterior_state=0.0
particlesets.py:        weight_sum = np.sum(self.weights_set)
particlesets.py:
particlesets.py:        if weight_sum != 1:
particlesets.py:
particlesets.py:        for idxp in range(self.p_set):
particlesets.py:            posterior_state += self.particles[idxp].weight*self.particles[idxp].particle*(1.0/weight_sum)
particlesets.py:
particlesets.py:        return posterior_state
qslamr.py:'''
qslamr.py:MODULE: qslamr
qslamr.py:
qslamr.py:Returns discrete posterior for the SLAM problem for qubit control as a set of
qslamr.py:particles (true states) and their weights (posterior distribution).
qslamr.py:
qslamr.py:CLASS: Node, Grid
qslamr.py:
qslamr.py:METHODS
qslamr.py:
qslamr.py:PARAMETERS
qslamr.py:
qslamr.py:STATIC FUNCTIONS
qslamr.py:
qslamr.py:
qslamr.py:'''
qslamr.py:import numpy as np
qslamr.py:import particledict as pd
qslamr.py:from model_design import GRIDDICT, INITIALDICT
qslamr.py:from hardware import Grid
qslamr.py:from itertools import combinations
qslamr.py:from particlesets import AlphaParticle, ParticleSet, BetaParticle
qslamr.py:from particledict import WEIGHTFUNCDICT_ALPHA, WEIGHTFUNCDICT_BETA, update_alpha_dictionary
qslamr.py:
qslamr.py:class ParticleFilter(Grid):
qslamr.py:    '''doctring
qslamr.py:    
qslamr.py:    MU_W : 0.0, # Qubit position noise mean (dynamics)
qslamr.py:    SIG2_W : 1.0, # Qubit position noise variance (dynamics)
qslamr.py:    MU_R : 0.0, # Length scale noise mean (dynamics)
qslamr.py:    SIG2_R : 1.0, # Length scale noise variance (dynamics)
qslamr.py:    MU_MEASR : 0.0, # Map noise mean (measurement)
qslamr.py:    SIG2_MEASR : 1.0, # Map noise variance (measurement)
qslamr.py:    MU_F : 0.0, # True sigmoid approximation error mean
qslamr.py:    SIG2_F : 1.0, # True sigmoid approximation error variance
qslamr.py:    LAMBDA : 0.9, # Forgetting factor for quasi-msmt information
qslamr.py:    GAMMA_T : 10**8, # Re-sampling threshold
qslamr.py:    P_ALPHA : 10, # Number of alpha particles
qslamr.py:    P_BETA : 10, # Numer of beta particles for each alpha
qslamr.py:
qslamr.py:    '''
qslamr.py:
qslamr.py:    def __init__(self, list_of_nodes_positions, **INITIALDICT):
qslamr.py:        # Load model design and chip config
qslamr.py:        self.QubitGrid = Grid(list_of_nodes_positions=list_of_nodes_positions)
qslamr.py:        self.dgrid, _= self.find_max_distance(self.QubitGrid.list_of_nodes_positions)
qslamr.py:        self.INITIALDICT = INITIALDICT
qslamr.py:
qslamr.py:        # Set up alpha particles
qslamr.py:        self.pset_alpha = self.INITIALDICT["P_ALPHA"]
qslamr.py:        self.pset_beta = self.INITIALDICT["P_BETA"]
qslamr.py:        empty_alpha_particles = [AlphaParticle() for idx in range(self.pset_alpha)]
qslamr.py:        self.AlphaSet = ParticleSet(empty_alpha_particles, **WEIGHTFUNCDICT_ALPHA)
qslamr.py:
qslamr.py:        # resampling threshold
qslamr.py:        self.resample_thresh = INITIALDICT["GAMMA_T"]
qslamr.py:        self.L_factor = self.dgrid * 1.0
qslamr.py:
qslamr.py:
qslamr.py:    def qslamr(self, measurements_controls):
qslamr.py:
qslamr.py:        self.InitializeParticles()
qslamr.py:
qslamr.py:        for t_item in measurements_controls:
qslamr.py:
qslamr.py:            next_phys_msmt_j = t_item[0]
qslamr.py:            control_j = t_item[1]
qslamr.py:            self.ReceiveMsmt(control_j, next_phys_msmt_j)
qslamr.py: 
qslamr.py:            self.PropagateState(control_j)
qslamr.py: 
qslamr.py:            posterior_weights = self.ComputeWeights(control_j)
qslamr.py:
qslamr.py: 
qslamr.py:            self.ResampleParticles(posterior_weights)
qslamr.py:            posterior_state = self.AlphaSet.posterior_state
qslamr.py:            self.QubitGrid.state_vector = posterior_state 
qslamr.py: 
qslamr.py:            self.update_qubitgrid_via_quasimsmts(control_j, posterior_state)
qslamr.py:
qslamr.py:#       ------------------------------------------------------------------------
qslamr.py:#       DATA ASSOCIATION FUNCTIONS
qslamr.py:#       ------------------------------------------------------------------------
qslamr.py:
qslamr.py:    def update_alpha_map_via_born_rule(self, alpha_particle): # TODO: Data association
qslamr.py:        '''doc string
qslamr.py:        this function must always come after we havw computed alpha weights
qslamr.py:        '''
qslamr.py:
qslamr.py:        for alpha_particle in self.AlphaSet.particles:
qslamr.py:            born_prob = self.sample_prob_from_msmts(alpha_particle.node_j)
qslamr.py:            map_val = ParticleFilter.inverse_born(born_prob)
qslamr.py:            map_idx = self.QubitGrid.number_of_nodes * 2 + alpha_particle.node_j
qslamr.py:            parent_particle = alpha_particle.particle # get property
qslamr.py:            parent_particle[map_idx] = map_val
qslamr.py:            alpha_particle.particle = parent_particle # assign updated property
qslamr.py:
qslamr.py:
qslamr.py:    def sample_prob_from_msmts(self, control_j): # TODO Data Association
qslamr.py:        '''docstring'''
qslamr.py:        prob_p = self.QubitGrid.nodes[control_j].physcmsmtsum / self.QubitGrid.nodes[control_j].counter_tau*1.0
qslamr.py:        forgetting_factor = self.INITIALDICT["LAMBDA"]**self.QubitGrid.nodes[control_j].counter_tau
qslamr.py:
qslamr.py:        prob_q = 0.0
qslamr.py:        if self.QubitGrid.nodes[control_j].counter_beta != 0:
qslamr.py:            prob_q = self.QubitGrid.nodes[control_j].quasimsmtsum / self.QubitGrid.nodes[control_j].counter_beta*1.0
qslamr.py:        prob_j = prob_p + forgetting_factor*prob_q
qslamr.py:        return prob_j
qslamr.py:
qslamr.py:
qslamr.py:    @staticmethod
qslamr.py:    def born_rule(map_val):
qslamr.py:        '''docstring'''
qslamr.py:        born_prob = np.cos(map_val / 2.0)**2
qslamr.py:        return born_prob
qslamr.py:
qslamr.py:
qslamr.py:    @staticmethod
qslamr.py:    def inverse_born(born_prob):
qslamr.py:        map_val = np.arccos(2.0*born_prob  - 1.0)
qslamr.py:        return map_val
qslamr.py:
qslamr.py:#       ------------------------------------------------------------------------
qslamr.py:#       SMEARING ACTION VIA QUASI MEASUREMENTS
qslamr.py:#       ------------------------------------------------------------------------
qslamr.py:    
qslamr.py:    def update_qubitgrid_via_quasimsmts(self, control_j, posterior_state):
qslamr.py:        '''docstring
qslamr.py:        this funciton should only be applied after lengthscales have been discovered
qslamr.py:        (alpha, beta_alpha) particles carried over or are resampled with 
qslamr.py:        sufficient alpha diversity; weights are set to uniform. 
qslamr.py:        '''
qslamr.py:        
qslamr.py:        SmearParticle = BetaParticle(control_j, posterior_state)
qslamr.py:        self.generate_beta_neighbourhood(SmearParticle)
qslamr.py:
qslamr.py:        for idx in range(len(SmearParticle.neighbourhood_qj)):
qslamr.py:
qslamr.py:            neighbour_q = SmearParticle.neighbourhood_qj[idx]
qslamr.py:            quasi_phase_q = SmearParticle.smeared_phases_qj[idx]
qslamr.py:            born_prob_q = ParticleFilter.born_rule(quasi_phase_q)
qslamr.py:            quasi_msmt = np.random.binomial(1, born_prob_q)
qslamr.py:            self.QubitGrid.nodes[neighbour_q].quasimsmtsum = quasi_msmt
qslamr.py:
qslamr.py:#       ------------------------------------------------------------------------
qslamr.py:#       SUPPORT FUNCTION 1: INITIALISE AND SAMPLE AT t = 0
qslamr.py:#       ------------------------------------------------------------------------
qslamr.py:
qslamr.py:    def InitializeParticles(self):
qslamr.py:        '''docstring t=1'''
qslamr.py:
qslamr.py:        for alphaparticle in self.AlphaSet.particles:
qslamr.py:            self.set_init_alphaparticle(alphaparticle)
qslamr.py:
qslamr.py:        self.QubitGrid.state_vector = self.AlphaSet.posterior_state
qslamr.py:
qslamr.py:    def set_init_alphaparticle(self, alphaparticle):
qslamr.py:        '''docstring'''
qslamr.py:        sample_s = np.random.normal(loc=self.INITIALDICT["MU_W"],
qslamr.py:                                    scale=self.INITIALDICT["SIG2_W"],
qslamr.py:                                    size=self.QubitGrid.number_of_nodes*2)
qslamr.py:
qslamr.py:        sample_f = np.random.uniform(low=0.,
qslamr.py:                                     high=np.pi, # INITIAL COND
qslamr.py:                                     size=self.QubitGrid.number_of_nodes)
qslamr.py:
qslamr.py:        sample_r = np.random.uniform(low=0.,
qslamr.py:                                     high=self.dgrid*3.0, # INITIAL COND
qslamr.py:                                     size=self.QubitGrid.number_of_nodes)
qslamr.py:
qslamr.py:        alphaparticle.particle = np.concatenate((sample_s, sample_f, sample_r),
qslamr.py:                                                 axis=0)
qslamr.py:        
qslamr.py:        alphaparticle.pset_beta = self.INITIALDICT["P_BETA"]
qslamr.py:        alphaparticle.mean_radius_j = self.dgrid*1.5
qslamr.py:
qslamr.py:#       ------------------------------------------------------------------------
qslamr.py:#       SUPPORT FUNCTION 2: RECEIVE MSMT
qslamr.py:#       ------------------------------------------------------------------------
qslamr.py:
qslamr.py:    def ReceiveMsmt(self, control_j, next_phys_msmt_j):
qslamr.py:        '''docstring'''
qslamr.py:        self.QubitGrid.nodes[control_j].physcmsmtsum = next_phys_msmt_j
qslamr.py:        prob_j = self.sample_prob_from_msmts(control_j)
qslamr.py:
qslamr.py:        
qslamr.py:        for alpha_particle in self.AlphaSet.particles:
qslamr.py:            alpha_particle.pset_beta = self.INITIALDICT["P_BETA"]
qslamr.py:            alpha_particle.node_j = control_j
qslamr.py:            map_index = self.QubitGrid.number_of_nodes*3 + alpha_particle.node_j 
qslamr.py:            # alpha_particle.mean_radius_j = self.AlphaSet.posterior_state[map_index] 
qslamr.py:            # TODO: neighbourbood mean radius update
qslamr.py:
qslamr.py:
qslamr.py:        update_alpha_dictionary(next_phys_msmt_j, prob_j)
qslamr.py:
qslamr.py:#       ------------------------------------------------------------------------
qslamr.py:#       SUPPORT FUNCTION 3: PROPAGATE (ALPHA) STATES
qslamr.py:#       ------------------------------------------------------------------------
qslamr.py:
qslamr.py:    def PropagateState(self, control_j):
qslamr.py:        '''docstring '''
qslamr.py:        for alpha_particle in self.AlphaSet.particles:
qslamr.py:            self.sample_from_transition_dist(alpha_particle, control_j)
qslamr.py:
qslamr.py:    def sample_from_transition_dist(self, alpha_particle, control_j):
qslamr.py:        '''docstring t > 1'''
qslamr.py:
qslamr.py:        sample_s = np.random.normal(loc=INITIALDICT["MU_W"],
qslamr.py:                                    scale=INITIALDICT["SIG2_W"],
qslamr.py:                                    size=self.QubitGrid.number_of_nodes*2)
qslamr.py:
qslamr.py:        sample_f = np.ones(self.QubitGrid.number_of_nodes)
qslamr.py:
qslamr.py:        sample_r = np.random.normal(loc=INITIALDICT["MU_R"],
qslamr.py:                                    scale=INITIALDICT["SIG2_R"],
qslamr.py:                                    size=self.QubitGrid.number_of_nodes)
qslamr.py:
qslamr.py:        alpha_particle.particle = np.concatenate((sample_s, sample_f, sample_r),
qslamr.py:                                                 axis=0)
qslamr.py:        
qslamr.py:#       ------------------------------------------------------------------------
qslamr.py:#       SUPPORT FUNCTION 4: COMPUTE ALPHA WEIGHTS; GENERATE BETA WEIGHTS
qslamr.py:#       ------------------------------------------------------------------------
qslamr.py:
qslamr.py:    def ComputeWeights(self, control_j):
qslamr.py:        '''docstring'''
qslamr.py:
qslamr.py:        new_alpha_weights = self.AlphaSet.calc_weights_set() # Normlaised
qslamr.py:        self.AlphaSet.weights_set = new_alpha_weights
qslamr.py:
qslamr.py:        posterior_weights = []
qslamr.py:
qslamr.py:        for alpha_particle in self.AlphaSet.particles:
qslamr.py:
qslamr.py:            self.update_alpha_map_via_born_rule(control_j)
qslamr.py:            beta_alpha_j_weights = self.generate_beta_layer(alpha_particle)
qslamr.py:            posterior_weights.append(alpha_particle.weight*beta_alpha_j_weights)
qslamr.py:        
qslamr.py:        posterior_weights = np.asarray(posterior_weights).flatten()
qslamr.py:        normalisation = np.sum(posterior_weights)
qslamr.py:
qslamr.py:        normalised_posterior_weights = posterior_weights*(1.0/normalisation)
qslamr.py:        return  normalised_posterior_weights # savedas posterior_weights[alphaindex][betaalphaindex]
qslamr.py:
qslamr.py:
qslamr.py:    def sample_radii(self, previous_length_scale, L_factor=None, Band_factor=None):
qslamr.py:        '''docstring'''
qslamr.py:        if L_factor is None:
qslamr.py:            L_factor = self.L_factor
qslamr.py:
qslamr.py:        if Band_factor is None:
qslamr.py:            Band_factor = 0.0
qslamr.py:
qslamr.py:        sample = np.random.uniform(low=Band_factor, high=L_factor)
qslamr.py:        return sample
qslamr.py:
qslamr.py:
qslamr.py:    def generate_beta_layer(self, alpha_particle):
qslamr.py:        '''docstring'''
qslamr.py:
qslamr.py:        len_idx = self.QubitGrid.number_of_nodes * 3 + alpha_particle.node_j
qslamr.py:        parent_alpha = alpha_particle.particle # get property
qslamr.py:
qslamr.py:        list_of_parent_states = []
qslamr.py:        for idx_beta in range(alpha_particle.pset_beta):
qslamr.py:
qslamr.py:            parent_alpha[len_idx] = self.sample_radii(parent_alpha[len_idx])
qslamr.py:            list_of_parent_states.append(parent_alpha)
qslamr.py:        
qslamr.py:        alpha_particle.generate_beta_pset(list_of_parent_states) #generates beta layer for each alpha
qslamr.py:
qslamr.py:        for beta_particle_object in alpha_particle.BetaAlphaSet_j.particles:
qslamr.py:            self.generate_beta_neighbourhood(beta_particle_object) # compute smeared phases for each beta particle
qslamr.py:
qslamr.py:        beta_alpha_j_weights = alpha_particle.BetaAlphaSet_j.calc_weights_set()
qslamr.py:        
qslamr.py:        return beta_alpha_j_weights # these weights are normalised
qslamr.py:
qslamr.py:
qslamr.py:    def generate_beta_neighbourhood(self, BetaParticle):
qslamr.py:        '''docstring'''
qslamr.py:
qslamr.py:        # BetaParticle.mean_radius = new_neighbourhood_L
qslamr.py:
qslamr.py:        NEIGHBOURDICT = {"prev_posterior_f_state" : self.QubitGrid.get_all_nodes(["f_state"]),
qslamr.py:                         "prev_counter_tau_state" : self.QubitGrid.get_all_nodes(["counter_tau"]),
qslamr.py:                         "lambda_" : self.INITIALDICT["LAMBDA"],
qslamr.py:                         "kernel_function": self.INITIALDICT["kernel_function"]}
qslamr.py:
qslamr.py:        BetaParticle.smear_fj_on_neighbours(**NEIGHBOURDICT)
qslamr.py:
qslamr.py:#       ------------------------------------------------------------------------
qslamr.py:#       SUPPORT FUNCTION 5: RESAMPLE AND BETA COLLAPSE
qslamr.py:#       ------------------------------------------------------------------------
qslamr.py:
qslamr.py:    def ResampleParticles(self, posterior_weights):
qslamr.py:        '''docstring'''
qslamr.py:        resampled_idx = np.arange(self.pset_alpha*self.pset_beta)
qslamr.py:
qslamr.py:        if self.resample_thresh > self.effective_particle_size(posterior_weights):
qslamr.py:            resampled_idx = ParticleFilter.resample_constant_pset_alpha(posterior_weights, self.pset_alpha, self.pset_beta)
qslamr.py:
qslamr.py:        new_alpha_subtrees = self.get_subtrees(resampled_idx, self.pset_beta)
qslamr.py:        new_alpha_list = self.collapse_beta(new_alpha_subtrees, resampled_idx)
qslamr.py:
qslamr.py:        self.pset_alpha = len(new_alpha_list)
qslamr.py:        self.AlphaSet.particles = new_alpha_list # garanteed to be pset_alpha with no second layer
qslamr.py:        self.AlphaSet.weights_set = (1.0/self.pset_alpha)*np.ones(self.pset_alpha)
qslamr.py:
qslamr.py:    def collapse_beta(self, subtree_list, resampled_indices):
qslamr.py:        '''docstring'''
qslamr.py:
qslamr.py:        state_update = 0.
qslamr.py:        new_alpha_particle_list = []
qslamr.py:        for subtree in subtree_list:
qslamr.py:
qslamr.py:            leaves_of_subtree = resampled_indices[subtree[0]:subtree[1]]
qslamr.py:            leaf_count = float(len(leaves_of_subtree))
qslamr.py:
qslamr.py:            if leaf_count != 0:
qslamr.py:
qslamr.py:                normaliser = (1./leaf_count)
qslamr.py:                alpha_node = ParticleFilter.get_alpha_node_from_treeleaf(leaves_of_subtree[0], self.pset_beta)
qslamr.py:                               # resampled_indices[subtree[0]], self.pset_beta)
qslamr.py:                beta_alpha_nodes = [ParticleFilter.get_beta_node_from_treeleaf(leafy, self.pset_beta) for leafy in leaves_of_subtree]
qslamr.py:                               # resampled_indices[subtree[0]:subtree[1]]]
qslamr.py:                r_est_subtree = 0.0
qslamr.py:                for node in beta_alpha_nodes:
qslamr.py:                    beta_state = self.AlphaSet.particles[alpha_node].BetaAlphaSet_j.particles[node].particle
qslamr.py:                    node_j = self.AlphaSet.particles[alpha_node].node_j
qslamr.py:                    beta_lengthscale = beta_state[int(node_j)]
qslamr.py:                    r_est_subtree += normaliser*beta_lengthscale
qslamr.py:
qslamr.py:                parent = self.AlphaSet.particles[alpha_node].particle
qslamr.py:                parent[self.AlphaSet.particles[alpha_node].node_j] = r_est_subtree
qslamr.py:
qslamr.py:                # Beta Layer Collapsed
qslamr.py:                self.AlphaSet.particles[alpha_node].particle = parent
qslamr.py:                self.AlphaSet.particles[alpha_node].BetaAlphaSet_j = None
qslamr.py:
qslamr.py:                # New Alphas Stored
qslamr.py:                new_alpha_particle_list.append(self.AlphaSet.particles[alpha_node])
qslamr.py:
qslamr.py:        return new_alpha_particle_list
qslamr.py:
qslamr.py:
qslamr.py:    def effective_particle_size(self, posterior_weights):
qslamr.py:        '''docstring'''
qslamr.py:        p_size = 1.0/ np.sum(posterior_weights**2)
qslamr.py:        self.L_factor = p_size*self.dgrid
qslamr.py:        return p_size
qslamr.py:
qslamr.py:
qslamr.py:#       ------------------------------------------------------------------------
qslamr.py:#       SUPPORT FUNCTION 6: STATIC METHODS
qslamr.py:#       ------------------------------------------------------------------------
qslamr.py:
qslamr.py:    @staticmethod
qslamr.py:    def compute_dist(one_pair):
qslamr.py:        '''doctring'''
qslamr.py:        xval, yval = one_pair
qslamr.py:        return np.sqrt((xval[0] - yval[0])**2 + (xval[1] - yval[1])**2)
qslamr.py:
qslamr.py:    @staticmethod
qslamr.py:    def find_max_distance(list_of_positions):
qslamr.py:        '''docstring'''
qslamr.py:        distance_pairs = [a_pair for a_pair in combinations(list_of_positions, 2)]
qslamr.py:        distances = [ParticleFilter.compute_dist(one_pair)for one_pair in distance_pairs]
qslamr.py:        return max(distances), distance_pairs[np.argmax(distances)]
qslamr.py:
qslamr.py:    @staticmethod
qslamr.py:    def get_alpha_node_from_treeleaf(leaf_index, pset_beta):
qslamr.py:        '''docstring'''
qslamr.py:        alpha_node = int(leaf_index//float(pset_beta))
qslamr.py:        return alpha_node
qslamr.py:
qslamr.py:    @staticmethod
qslamr.py:    def get_beta_node_from_treeleaf(leaf_index, pset_beta):
qslamr.py:        '''docstring'''
qslamr.py:        beta_node = int(leaf_index - int(leaf_index//float(pset_beta))*pset_beta)
qslamr.py:        return beta_node
qslamr.py:
qslamr.py:    @staticmethod
qslamr.py:    def resample_from_weights(posterior_weights, number_of_samples):
qslamr.py:        '''docstring'''
qslamr.py:        total_particles = len(posterior_weights)
qslamr.py:        cdf_weights = np.asarray([0] + [np.sum(posterior_weights[:idx+1]) for idx in range(total_particles)])
qslamr.py:        pdf_uniform = np.random.uniform(low=0, high=1.0, size=number_of_samples)
qslamr.py:
qslamr.py:        resampled_idx = []
qslamr.py:
qslamr.py:        for u_0 in pdf_uniform:
qslamr.py:            j = 0
qslamr.py:            while u_0 > cdf_weights[j]:
qslamr.py:                j += 1
qslamr.py:                if j > total_particles:
qslamr.py:                    j = total_particles
qslamr.py:                    break   # clip at max particle index, plus zero
qslamr.py:            resampled_idx.append(j-1) # sgift down to match python indices
qslamr.py:
qslamr.py:        return resampled_idx
qslamr.py:
qslamr.py:    @staticmethod
qslamr.py:    def resample_constant_pset_alpha(posterior_weights, pset_alpha, pset_beta):
qslamr.py:        '''Returns indicies for particles picked after sampling from posterior'''
qslamr.py:        # DO WEIGHTS NEED TO BE SORTED? (No, the CDF will be
qslamr.py:        # monotnically increasing; and the shape of the CDF will determine
qslamr.py:        # the frequency at which (x,y) are sampled if y is uniform )
qslamr.py:
qslamr.py:        sufficient_sample = False
qslamr.py:        num_of_samples = pset_alpha
qslamr.py:        total_particles = len(posterior_weights)
qslamr.py:
qslamr.py:
qslamr.py:        if total_particles != int(INITIALDICT["P_ALPHA"]*INITIALDICT["P_BETA"]):
qslamr.py:            raise RuntimeError
qslamr.py:
qslamr.py:        while sufficient_sample is False:
qslamr.py:
qslamr.py:            num_of_samples += 5
qslamr.py:            resampled_indices = ParticleFilter.resample_from_weights(posterior_weights, num_of_samples)
qslamr.py:            resampled_alphas = [ParticleFilter.get_alpha_node_from_treeleaf(leafy, pset_beta) for leafy in resampled_indices]
qslamr.py:            unique_alphas = set(list(resampled_alphas))
qslamr.py:            if len(unique_alphas) == pset_alpha:
qslamr.py:                sufficient_sample = True
qslamr.py:
qslamr.py:        return resampled_indices
qslamr.py:
qslamr.py:    @staticmethod
qslamr.py:    def get_subtrees(resampled_indices, pset_beta):
qslamr.py:        '''docstring'''
qslamr.py:
qslamr.py:        new_sub_trees = []
qslamr.py:
qslamr.py:        resampled_indices.sort()
qslamr.py:        alpha_index_0 = None
qslamr.py:        strt_counter = 0
qslamr.py:        end_counter = 0
qslamr.py:
qslamr.py:        for idx in resampled_indices:
qslamr.py:
qslamr.py:            alpha_index = ParticleFilter.get_alpha_node_from_treeleaf(idx, pset_beta)
qslamr.py:            beta_alpha_idx = ParticleFilter.get_beta_node_from_treeleaf(idx, pset_beta)
qslamr.py:
qslamr.py:            if alpha_index_0 == alpha_index:
qslamr.py:                end_counter += 1
qslamr.py:
qslamr.py:            elif alpha_index_0 != alpha_index:
qslamr.py:
qslamr.py:                new_sub_trees.append([strt_counter, end_counter])
qslamr.py:
qslamr.py:                alpha_index_0 = alpha_index
qslamr.py:                strt_counter = end_counter
qslamr.py:                end_counter += 1
qslamr.py:
qslamr.py:        if end_counter == len(resampled_indices):
qslamr.py:            # end_counter += 1
qslamr.py:            new_sub_trees.append([strt_counter, end_counter])
qslamr.py:
qslamr.py:        return new_sub_trees
qslamr.py:    # def collapse_beta(self, new_sub_trees, resampled_indices):
qslamr.py:    #     '''docstring'''
qslamr.py:
qslamr.py:    #     state_update = 0.
qslamr.py:    #     pset_beta = self.pset_beta
qslamr.py:
qslamr.py:    #     new_alpha_particle_list = []
qslamr.py:    #     for pairs in new_sub_trees:
qslamr.py:
qslamr.py:    #         subtree = resampled_indices[pairs[0]:pairs[1]]
qslamr.py:    #         leaf_count = float(len(subtree))
qslamr.py:
qslamr.py:
qslamr.py:
qslamr.py:    #         if leaf_count != 0:
qslamr.py:
qslamr.py:    #             normaliser = (1./leaf_count)
qslamr.py:    #             alpha_node = ParticleFilter.get_alpha_node_from_treeleaf(pairs[0], pset_beta)
qslamr.py:    #             beta_alpha_nodes = [ParticleFilter.get_beta_node_from_treeleaf(leafy, pset_beta) for leafy in subtree]
