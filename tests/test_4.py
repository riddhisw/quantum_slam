'''
TEST SCRIPT

Type: Module testing
Original Script:  particle.py
Class: Particle
Methods: update_map_state, predict_scan
Details: Test functionality of place_bot_on_grid, propagate_bot, update_state
Outcome: SUCCESSFUL

1. Particle maps aren't updating  - FIXED
2. The guestbook for each particle is updating twice on each real msmt - FIXED
3. Add extended print statements to probe single particle, single control SLAM - DONE
'''

import sys, os
import numpy as np
import matplotlib.pyplot as plt 
sys.path.append('../')
from qslam.slampf import ParticleFilter
from qslam.mapping import TrueMap
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

###############################################################################
# TESTS 
###############################################################################

slamize = ParticleFilter(num_p=1, localgridcoords_=[5,5])
map_ = np.zeros((5,5))
map_[0,0] = np.pi*0.25
print map_
dunk = TrueMap(m_vals=map_)
controls = [(0, 0, 1)]*5 #, (0, 1, 1)]# , (2, 2, 1), (1, 3, 1), (3, 1, 1)]*100
# with suppress_stdout():
slamize.qslam_run(dunk, controls)

