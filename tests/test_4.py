'''
TEST SCRIPT

Type: Module testing
Original Script:  particle.py
Class: Particle
Methods: update_map_state, predict_scan
Details: Test functionality of place_bot_on_grid, propagate_bot, update_state
Outcome: WIP

1. Particle maps aren't updating
2. the guestbook for each particle is updating twice on each real msmt 
'''

import sys
sys.path.append('../')
from qslam.slampf import ParticleFilter
from qslam.mapping import TrueMap
import numpy as np


###############################################################################
# TESTS 
###############################################################################

slamize = ParticleFilter(num_p=5, localgridcoords_=[5,5])
map_ = np.zeros((5,5))
map_[0,0] = np.pi*0.25
print map_
dunk = TrueMap(m_vals=map_)
controls = [(0, 0, 1)]#, (0, 1, 1)]# , (2, 2, 1), (1, 3, 1), (3, 1, 1)]*100
slamize.qslam_run(dunk, controls)

import matplotlib.pyplot as plt 

plt.figure()
plt.plot(slamize.weights)
plt.show()

