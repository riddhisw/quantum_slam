import numpy as np
from particledict import WEIGHTFUNCDICT_BETA

###############################################################################
# PARTICLE STRUCTURE
###############################################################################

class Particle(object):
    '''docstring'''
    def __init__(self):

        self.__particle = 0.0
        self.__weight = 0.0

    @property
    def particle(self):
        '''docstring'''
        return self.__particle
    @particle.setter
    def particle(self, new_state_vector):
        '''docstring'''
        self.__particle = new_state_vector

    @property
    def weight(self):
        '''docstring'''
        return self.__weight
    @weight.setter
    def weight(self, new_weight):
        '''docstring'''
        self.__weight = new_weight


class BetaParticle(Particle):
    '''docstring'''

    def __init__(self, node_j, parent_state, radius):
        Particle.__init__(self)

        self.parent = parent_state
        self.particle = np.asarray(parent_state).flatten() # COMMENT: Intiialised identically to parent
        self.total_nodes = int(float(len(parent_state)) / 4.0)
        self.node_j = node_j
        self.neighbourhood_qj = []
        self.neighbour_dist_qj = []
        self.smeared_phases_qj = []

        self.x_j, self.y_j, self.f_j, self.r_j = self.particle[self.node_j::self.total_nodes]

        self.mean_radius = radius # TODO: * 3.0 

    def get_neighbourhood_qj(self):
        '''doc string'''

        self.neighbourhood_qj = []
        self.neighbour_dist_qj = []

        for idx in range(self.total_nodes):

            xq_ = self.particle[idx]
            yq_ = self.particle[idx + self.total_nodes]
            dist = np.sqrt((xq_ - self.x_j)**2 + (yq_ - self.y_j)**2)

            if dist <= self.mean_radius:

                if dist > 0.0:
                    self.neighbourhood_qj.append(idx)
                    self.neighbour_dist_qj.append(dist)


    def smear_fj_on_neighbours(self, **args):
        '''docstring'''

        self.get_neighbourhood_qj()

        prev_posterior_f_state = args["prev_posterior_f_state"]
        prev_counter_tau_state = args["prev_counter_tau_state"]
        lambda_ = args["lambda_"]

        self.smeared_phases_qj = []
        for idx_q in range(len(self.neighbourhood_qj)):

            node_q = self.neighbourhood_qj[idx_q]
            dist_jq = self.neighbour_dist_qj[idx_q]
            tau_q = prev_counter_tau_state[node_q]
            f_state_q = prev_posterior_f_state[node_q]

            lambda_q = lambda_** tau_q
            kernel_val = args["kernel_function"](dist_jq, self.f_j, self.r_j)

            smear_phase = (1.0 - lambda_q)*f_state_q + lambda_q*kernel_val
            self.smeared_phases_qj.append(smear_phase)


class AlphaParticle(Particle):
    '''docstring'''
    def __init__(self):
        Particle.__init__(self)
        self.pset_beta = 0
        self.node_j = 0.0
        self.SIG2_MEASR = 0.0

        self.BetaAlphaSet_j = None

    def generate_beta_pset(self, parents, radii):
        '''docstring'''
        beta_s = []
        for idx in range(len(parents)): # TODO: Use enumerate 
            state = parents[idx]
            radius = radii[idx]
            beta_s.append(BetaParticle(self.node_j, state, radius))
        self.BetaAlphaSet_j = ParticleSet(beta_s, **WEIGHTFUNCDICT_BETA)


class ParticleSet(object):
    '''docstring'''
    def __init__(self, list_of_particle_objects, **WEIGHTFUNCDICT):

        self.__weights_set = 0.0
        self.__posterior_state = 0.0
        self.p_set = len(list_of_particle_objects)
        self.particles = list_of_particle_objects
        self.w_dict = WEIGHTFUNCDICT
        self.weights_set = (1.0 / self.p_set)*np.ones(self.p_set)

    def calc_weights_set(self):
        '''docstring'''
        new_weight_set = []
        for particle in self.particles:
            new_weight = self.w_dict["function"](particle, **self.w_dict["args"])
            new_weight_set.append(new_weight)

        raw_weights = np.asarray(new_weight_set).flatten()
        normalisation = 1.0/np.sum(raw_weights)
        return normalisation*raw_weights

    @property
    def weights_set(self):
        '''docstring'''
        self.__weights_set = np.asarray([particle.weight for particle in self.particles]).flatten()
        return self.__weights_set
    @weights_set.setter
    def weights_set(self, new_weights):
        '''docstring'''
        for idxp in range(self.p_set):
            self.particles[idxp].weight = new_weights[idxp]

    @property
    def posterior_state(self):
        '''docstring'''

        posterior_state = 0.0
        weight_sum = np.sum(self.weights_set)
        for idxp in range(self.p_set):
            posterior_state += self.particles[idxp].weight*self.particles[idxp].particle*(1.0/weight_sum)
        return posterior_state
