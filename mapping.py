''' Module: mapping.py

The purpose of this module is to generate a continous space time dephasing noise
defined over a grid structure given by nrows x ncols, where a qubit resides at
every node (including edges).

The module contains the following classes:
Map(object): Generates nrow x ncol grid with values defined for each node
TrueMap(Map): Specifies a true dephasing noise field over a Map object

'''

from itertools import product
import numpy as np

class Map(object):
    '''Generates a noise field defined over a grid structure (nrows x ncols),
    where a qubit resides at every node (including edges).
    '''

    def __init__(self, nrows=1, ncols=1, m_type=0, m_vals=None):

        self.m_vals = np.zeros([nrows, ncols]) if m_vals is None else m_vals

        nrows, ncols = np.shape(self.m_vals)[0], np.shape(self.m_vals)[1]
        self.m_nodes_coords = list(product(range(nrows), range(ncols)))

        self.m_nodes = len(self.m_nodes_coords)
        self.m_type = 'Pauli_z_noise'
        self.m_knn = []


    def m_vectorise_map(self):
        '''Returns a vector formed by stacking rows of self.m_vals.
        '''
        vectormap = self.m_vals.reshape(self.m_nodes)
        return vectormap

    def m_knn_list(self, pos, corr_r):
        '''Updates neighest neighbours list given a correlation radius
        that sets the maximally far away neighbour.
        '''
        self.m_knn = []
        for node in self.m_nodes_coords:
            mag = np.linalg.norm(np.subtract(pos, node))
            if mag >= corr_r:
                continue
            else:
                self.m_knn.append(node)
        return self.m_knn

class TrueMap(Map):
    '''Inherits from Map. TrueMap creates spatio-temporal noise fields.'''

    def __init__(self, phi_s=None, phi_t=None, Sigma_s=None, Sigma_t=None,
                 nrows=1, ncols=1, m_type=0, m_vals=None):

        Map.__init__(self,
                     nrows=nrows,
                     ncols=ncols,
                     m_type=m_type,
                     m_vals=m_vals)

        self.m_space_dyn = np.ones_like(self.m_vals) if phi_s is None else phi_s
        self.m_time_dyn = np.ones_like(self.m_vals) if phi_t is None else phi_t
        self.sigma_s = 0 if Sigma_s is None else Sigma_s
        self.sigma_t = 0 if Sigma_t is None else Sigma_t

    def m_evolve(self):
        '''docstring'''
        self.m_vals = self.m_time_dyn * self.m_vals.copy()

    def m_initialise(self):
        '''docstring'''
        self.m_vals = self.m_space_dyn * self.m_vals.copy()


# ############################################################################
# # Test Code
# ############################################################################


# testmap = np.arange(12).reshape(3,4)

# mapobject1 = Map(nrows=3,ncols=4)
# print(mapobject1.m_vectorise_map())

# mapobject2 = Map(m_vals=testmap)
# print(mapobject2.m_vectorise_map())

# print mapobject1.m_knn_list((0,0), 1)
# print mapobject1.m_knn_list((0,0), 1.1)
# print mapobject1.m_knn_list((0,0), 10)
# print mapobject1.m_knn_list((1,1), 1.1)
# print mapobject1.m_knn_list((1,1), 2)