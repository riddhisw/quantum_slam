import numpy as np
from model_design import INITIALDICT

LAMBDA = INITIALDICT["LAMBDA"]
PARTICLE_STATE = ["x_state", "y_state", "f_state", "r_state"]

###############################################################################
# CHIP STRUCTURE 
###############################################################################
class Node(object):
    '''[docstring]'''

    def __init__(self):

        self._f_state = np.random.uniform(low=0.0, high=np.pi) # cant set _f_state
        self.r_state = 0.0
        self.__r_state_skew = 0.0 
        self.x_state = 0.0
        self.y_state = 0.0
        self.counter_tau = 0
        self.counter_beta = 0
        self.__physcmsmtsum = 0.0
        self.__quasimsmtsum = 0.0
        self.lambda_factor = LAMBDA

    @property
    def physcmsmtsum(self):
        '''docstring'''
        return self.__physcmsmtsum
    @physcmsmtsum.setter
    def physcmsmtsum(self, next_phys_msmt):
        self.__physcmsmtsum += next_phys_msmt
        self.counter_tau += 1

    @property
    def quasimsmtsum(self):
        '''docstring'''
        return self.__quasimsmtsum
    @quasimsmtsum.setter
    def quasimsmtsum(self, next_quasi_msmt):
        self.__quasimsmtsum += next_quasi_msmt
        self.counter_beta += 1


    ############################################################################
    # PLACEHOLDER kew based control for SLAM
    ############################################################################
    @property
    def r_state_skew(self):
        '''docstring'''
        return self.__r_state_skew
    @r_state_skew.setter
    def r_state_skew(self, skew_metric):
        self.__physcmsmtsum = skew_metric
    
    @staticmethod
    





    ############################################################################
    @property
    def f_state(self): # no .setter function
        '''docstring'''
        prob_sample = self.sample_prob_from_msmts()
        if prob_sample is None:
            # self._f_state = np.random.uniform(low=0.0, high=np.pi)
            # print "Map value returned is default; prob_sample NONE in f_state", self._f_state
            return self._f_state 
        if prob_sample >= 0.0 and prob_sample <= 1.0:
            self._f_state = Node.inverse_born(prob_sample)
            return self._f_state
        # print "INVALID prob_sample value encountered in calling f_state", prob_sample
        raise RuntimeError

    def sample_prob_from_msmts(self): # TODO Data Association
        '''docstring'''

        forgetting_factor = self.lambda_factor**self.counter_tau

        w_q = None
        prob_q = 0.0
        if self.counter_beta != 0:
            prob_q = self.quasimsmtsum / self.counter_beta*1.0
            w_q = 0.5*forgetting_factor

        w_p = None
        prob_p =0.0
        if self.counter_tau !=0:
            prob_p = self.physcmsmtsum / self.counter_tau*1.0
            w_p = 0.5 + 0.5*(1 - forgetting_factor)

        if w_p is None and w_q is None:
            # print "NONE returned in sample_prob_from_msmts"
            return  None
        elif w_p is not None  and w_q is None:
            w_p = 1.0
            w_q = 0.0
        elif w_p is None and w_q is not None:
            w_q = 1.0
            w_p = 0.0
        elif w_p is not None and w_q is not None:
            pass
        prob_j = w_p*prob_p + w_q*prob_q

        if prob_j > 1 or prob_j < 0:
            raise RuntimeError
        return prob_j


    @staticmethod
    def born_rule(map_val):
        '''docstring'''
        born_prob = np.cos(map_val / 2.0)**2
        return born_prob


    @staticmethod
    def inverse_born(born_prob):
        map_val = np.arccos(2.0*born_prob  - 1.0)
        return map_val

class Grid(object):
    '''docstring '''

    def __init__(self, list_of_nodes_positions=None):
        if list_of_nodes_positions is None:
            print "No node positions specified"
            raise RuntimeError

        self.list_of_nodes_positions = list_of_nodes_positions
        self.number_of_nodes = len(self.list_of_nodes_positions)
        self.nodes = [Node() for i in range(self.number_of_nodes)]
        self.__state_vector = 0.0
        self.state_vector = np.zeros(self.number_of_nodes*len(PARTICLE_STATE))

        for item in xrange(self.number_of_nodes):
            self.nodes[item].x_state, self.nodes[item].y_state = self.list_of_nodes_positions[item]

    def get_all_nodes(self, attribute_list):
        '''Returns attribute for all nodes in a single vector, stacked by nodes, then attributes '''
        vector = [[getattr(node, attr) for node in self.nodes] for attr in attribute_list]
        return np.asarray(vector).flatten()

    def set_all_nodes(self, single_attribute, attribute_values):
        '''Sets attribute for all nodes '''
        for item in xrange(len(attribute_values)):
            setattr(self.nodes[item], single_attribute, attribute_values[item])

    @property
    def state_vector(self):
        '''docstring'''
        self.__state_vector = self.get_all_nodes(PARTICLE_STATE)
        return self.__state_vector
    @state_vector.setter
    def state_vector(self, new_state_vector):
        '''docstring'''
        for state in xrange(len(PARTICLE_STATE)):
            if state == 2: # f_state cannot be set, only calculated
                pass
            elif state != 2:
                bgn = state * self.number_of_nodes
                end = bgn + self.number_of_nodes
                self.set_all_nodes(PARTICLE_STATE[state], new_state_vector[bgn:end])

