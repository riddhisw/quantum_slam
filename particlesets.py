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

    def __init__(self, node_j, parent_state):
        Particle.__init__(self)

        self.parent = parent_state
        self.particle = np.asarray(parent_state).flatten() # intiialised identically to parent
        self.total_nodes = int(float(len(parent_state)) / 4.0)

        self.node_j = node_j
        self.neighbourhood_qj = []
        self.neighbour_dist_qj = []
        self.smeared_phases_qj = []
        print
        print "In class BetaParticle, made a  beta particle!"
        print "parent shape", self.parent.shape
        print "total nodes", self.total_nodes
        print "node_j", self.node_j
        print "parent", self.parent
        print "j state",  self.parent[self.node_j::self.total_nodes]
        print
        self.x_j, self.y_j, self.f_j, self.r_j = self.parent[self.node_j::self.total_nodes]

        self.mean_radius = self.r_j*3.0 #mean_radius_j # TODO: Change to self.r_j*3.0

    def get_neighbourhood_qj(self):
        '''doc string'''

        self.neighbourhood_qj = []
        self.neighbour_dist_qj = []

        for idx in range(self.total_nodes):

            xq_ = self.parent[idx]
            yq_ = self.parent[idx + self.total_nodes]
            dist = np.sqrt((xq_ - self.x_j)**2 + (yq_ - self.y_j)**2)

            if dist <= self.mean_radius:
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
        # self.mean_radius_j = 0.0 # TODO FIX THIS. CANT BE ONE NUMBER FORALL j
        self.BetaAlphaSet_j = None

    def generate_beta_pset(self, parents): #, number_of_beta_particles):
        '''docstring'''
        beta_s = [BetaParticle(self.node_j, state) for state in parents]
        self.BetaAlphaSet_j = ParticleSet(beta_s, **WEIGHTFUNCDICT_BETA)
        # return BetaAlphaSet

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
        print "raw_weights in calc_weights_set", raw_weights
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

        posterior_state=0.0
        weight_sum = np.sum(self.weights_set)

        if weight_sum != 1:
            print "Sum of all alpha weight set: "
            print "Normalising weights in posterior state calculation..."

        for idxp in range(self.p_set):
            posterior_state += self.particles[idxp].weight*self.particles[idxp].particle*(1.0/weight_sum)

        return posterior_state
