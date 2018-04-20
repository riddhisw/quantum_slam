'''
TEST SCRIPT

Type: Module testing
Original Script:  particle.py
Class: Particle
Methods: update_map_state, predict_scan
Details: Test functionality of place_bot_on_grid, propagate_bot, update_state
Outcome: SUCCESSFUL
'''

import sys
sys.path.append('../')
from qslam.particle import Particle
import numpy as np


###############################################################################
# TESTS 
###############################################################################

''' Update map based on  msmts at (4,2) (edge value)
'''
wheezy = Particle(x=0,y=0, m_vals=np.zeros((5,5)))

X_pos = (4,2)
ucontrol = np.asarray([4,2,1])

print("State: ", wheezy.state)
print("Robot: ", wheezy.r_pose)
print("Map: ", wheezy.m_vals)
print("Guestbook Counter: ", wheezy.r_guestbk_counter)
print("Guestbook: ", wheezy.r_questbk)
print("Feed 4 qubit zero measurements at location X")
print("...")
wheezy.update_map_state([0]*10, [X_pos]*10)
print("State: ", wheezy.state)
print("Robot: ", wheezy.r_pose)
print("Map: ", wheezy.m_vals)
print("Guestbook Counter: ", wheezy.r_guestbk_counter)
print("Guestbook: ", wheezy.r_questbk)
print("...")
print("...")
print("Move to location X")
print("...")
wheezy.r_move(ucontrol)
wheezy.update_state()
print("State: ", wheezy.state)
print("Correlation: ", wheezy.r_pose[-1])
print("...")
print("...")
print("Get neighours at X ")
print("...")
grunters = wheezy.m_knn_list(wheezy.r_xy(), wheezy.r_pose[-1])
print(grunters)
print("...")
print("Manually create 1 as quasi msmts for all neighbours ")
msmts = np.zeros(len(grunters))
print(msmts)
print("Update map")
wheezy.update_map_state(msmts, grunters)
print("Map: ", wheezy.m_vals)
print("Guestbook Counter: ", wheezy.r_guestbk_counter)
print("Guestbook: ", wheezy.r_questbk)
print("Spurt out predictions from map")
spurt_preds = wheezy.predict_scan(grunters)
print spurt_preds
print("State: ", wheezy.state)

