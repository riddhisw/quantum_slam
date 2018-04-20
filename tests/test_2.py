'''
TEST SCRIPT

Type: Module testing
Original Script:  sensing.py
Details: Test functionality of class objects, properties and methods 
Outcome: SUCCESSFUL
'''

import sys
sys.path.append('../')
from qslam.sensing import Robot, Scanner
import numpy as np

###############################################################################
# Class: Robot
###############################################################################

'''
Test: run class, check attributes and core functions
'''

walle = Robot()
print(walle.r_pose)
print(walle.r_msmtnoise)
print(walle.r_motionnoise)
print(walle.r_dims)
print(walle.r_phi)
print(walle.r_questbk)
print(walle.r_guestbk_counter)

print("xy", walle.r_xy())
walle.r_move(np.asarray([1,0,2]))
print("move left", walle.r_xy())

###############################################################################
# Class: Scanner
###############################################################################

'''
Test: run class, check attributes and core functions; check inheritance
'''

eyesopen = Scanner([3,3])

print(eyesopen.r_pose)
print(eyesopen.r_msmtnoise)
print(eyesopen.r_motionnoise)
print(eyesopen.r_dims)
print(eyesopen.r_phi)
print(eyesopen.r_questbk)
print(eyesopen.r_guestbk_counter)

print("xy", eyesopen.r_xy())
print type(eyesopen.r_pose)
print type(eyesopen.r_xy())
pose, dynamics, noise = eyesopen.r_move(np.asarray([1,0,2]))
print type(pose)
print pose
print type(dynamics)
print dynamics
print type(noise)
print noise
print("move left", eyesopen.r_xy())

print("check correlation functions")
print(eyesopen.r_corr_length(0), eyesopen.r_corr_length(3), eyesopen.r_corr_length(3.) )

print("check msmts scan")
mval = np.pi*0.2
knn_list = [(0,0),(1,0), (0,1), (2,2)]

print(eyesopen.r_scan_local(mval,knn_list))