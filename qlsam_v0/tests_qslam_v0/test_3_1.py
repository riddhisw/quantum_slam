'''
TEST SCRIPT

Type: Module testing
Original Script:  particle.py
Class: Particle
Methods: place_bot_on_grid, propagate_bot, update_state
Details: Test functionality of place_bot_on_grid, propagate_bot, update_state
Outcome: SUCCESSFULL
'''

import sys
sys.path.append('../')
from qslam.particle import Particle
import numpy as np

###############################################################################
# TESTS 
###############################################################################

''' Control is ignored
'''
wheezy = Particle(x=0,y=0, m_vals=np.zeros((1,1)))

print( "State :", wheezy.state )
print( "Pose :", wheezy.r_pose )
print( "Map:", wheezy.m_vals  )

ucontrol = np.asarray([1,1,0])
print( "Apply control", ucontrol)
wheezy.propagate_bot(ucontrol)

print( "State after:", wheezy.state )
print( "Pose after:", wheezy.r_pose )
print( "Map after:", wheezy.m_vals )


'''
Initially controls are implemented on a grid
Once robot moves off the map, controls are ignored.
'''

wheezy = Particle(x=0, y=0, m_vals=np.zeros((10, 10)))

print("State :", wheezy.state)
print("Pose :", wheezy.r_pose)
print("Map:", wheezy.m_vals)


for i in range(15):

    ucontrol = np.asarray([i, i, 0])
    print("Apply next control", ucontrol)
    wheezy.propagate_bot(ucontrol)
    print("Pose after:", wheezy.r_pose)

print("State final:", wheezy.state)
print("Pose final:", wheezy.r_pose)
print("Map final:", wheezy.m_vals)


''' Breaks at initialisation
'''

wheezy = Particle(x=1,y=0, m_vals=np.zeros((1,1)))

print( "State :", wheezy.state )
print( "Pose :", wheezy.r_pose )
print( "Map:", wheezy.m_vals  )

ucontrol = np.asarray([1,1,0])
print( "Apply control", ucontrol)
wheezy.propagate_bot(ucontrol)

print( "State after:", wheezy.state )
print( "Pose after:", wheezy.r_pose )
print( "Map after:", wheezy.m_vals )
