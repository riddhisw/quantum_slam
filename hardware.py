import numpy as np

PARTICLE_STATE = ["x_state", "y_state", "f_state", "r_state"]

###############################################################################
# CHIP STRUCTURE 
###############################################################################
class Node(object):
    '''[docstring]'''

    def __init__(self):
        # self.qubitid = 0
        self.f_state = 0.0
        self.r_state = 0.0
        self.x_state = 0.0
        self.y_state = 0.0
        self.counter_tau = 0
        self.counter_beta = 0
        self.__physcmsmtsum = 0.0
        self.__quasimsmtsum = 0.0

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
            bgn = state * self.number_of_nodes
            end = bgn + self.number_of_nodes
            self.set_all_nodes(PARTICLE_STATE[state], new_state_vector[bgn:end])

