'''
TEST SCRIPT

Type: Module testing
Original Script:  mapping.py
Details: Test functionality of class objects, properties and methods 
Outcome: WIP
'''
import sys
sys.path.append('../')
from qslam.mapping import Map, TrueMap
import numpy as np

###############################################################################
# Class: Map
###############################################################################

'''
Test: default values
'''
newmap = Map()


print(newmap.m_vals == [[0.]])
print(newmap.m_nodes_coords == [(0,0)])
print(newmap.m_nodes ==1)
print(newmap.m_type=='Pauli_z_noise') # placeholder for new functionality
print(newmap.m_knn==[])

mvals = np.zeros((1,2))
specificmap = Map(nrows=10., ncols=10, m_type=1, m_vals=mvals)


'''
Test: non-default values
'''
print(specificmap.m_vals)
print(specificmap.m_nodes_coords)
print(specificmap.m_nodes)
print(specificmap.m_type) # placeholder for new functionality
print(specificmap.m_knn)

'''
Test: Return a vectorise map
'''

mvals2 = np.arange(6).reshape(2,3)
rectmap = Map(m_vals=mvals2)

print(rectmap.m_vals)
print(rectmap.m_nodes_coords)
print(rectmap.m_nodes)
print(rectmap.m_type) # placeholder for new functionality
print(rectmap.m_knn)

print(rectmap.m_vectorise_map())

# 1.1 nearrest neighbour from the first (last) edge excludes the last (first) edge
# middle col with return the whole grid as neighbours

corr_r = 1.42
print("For corr. length ", corr_r, " first diagonal neighbours are included. ")
for pos in rectmap.m_nodes_coords:
    print('At position: ', pos )
    print('... knn are:', rectmap.m_knn_list(pos, corr_r))
    print('')
    print('')


