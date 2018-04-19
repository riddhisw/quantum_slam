''' Module: particle.py

The purpose of this module is to define a particle using classes Robot and Map

The module contains the following classes:
Particle(Robot, Map)
'''

from qslam.mapping import Map
from qslam.sensing import Robot
import numpy as np

class Particle(Robot, Map):
    ''' doctring
    '''

    def __init__(self, x=0, y=0, corr_r=0, sigma=0, R=0, phi=None,
                 m_type=0, m_vals=None, localgrid=[0, 1]):
        ''' doctring
        '''

        Robot.__init__(self, 
                       localgrid=localgrid,
                       x=x,
                       y=y,
                       corr_r=corr_r,
                       sigma=sigma,
                       R=R,
                       phi=phi)

        Map.__init__(self, nrows=localgrid[0],
                     ncols=localgrid[1],
                     m_type=m_type,
                     m_vals=m_vals)

        self.state = 0
        self.weight = 0

    def update_state(self):
        ''' Updates the state for the particle (both manual and/or in-code)
        '''
        self.state = np.concatenate(self.r_pose, self.m_vectorise_map())

    def propagate_bot(self, u_ctrl):
        ''' Updates the x,y coords of robot with control input and dynamical
        evolution according to phi.
        '''
        self.r_move(u_ctrl)
        self.update_state()

    def update_map_state(self, msmts_from_scan, positions_scanned):
        ''' Updates a particle map based on real and quasi msmts in a scan.
        '''
        pos = iter(positions_scanned)

        idx_updt = 0
        for u_x, u_y in pos: # update robot guestbook
            self.r_addguest(u_x, u_y, msmts_from_scan[idx_updt])
            idx_updt += 1

        for u_x, u_y in pos: # separate loop in case a node is measured twice
            self.m_vals[u_x, u_y] = np.arccos(2.0*self.r_guestbk[u_x, u_y] - 1.)
        self.update_state()

    def predict_scan(self, positions_scanned):
        ''' Returns predicted msmts based on particle map values
        '''
        pos = iter(positions_scanned)
        predictions = []
        for pred_x, pred_y in pos:
            predictions.append(self.r_measure(self.m_vals[pred_x, pred_y]))
        return predictions

self.update_state() # initialises the state vector
