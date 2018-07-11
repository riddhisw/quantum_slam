''' Module: sensing.py

The purpose of this module is to ...

The module contains the following classes:
Robot(object):
Scanner(Map): Inherits from Robot.
'''

import numpy as np
import sys
sys.path.append('../../GIT/')
from qif.common import projected_msmt, calc_z_proj

class Robot(object):
    '''docstring'''
    def __init__(self, x=0, y=0, corr_r=0, sigma=0, R=0, phi=None, 
                 localgridcoords=[1,1]):

        self.r_pose = np.asarray([x, y, corr_r])
        self.r_msmtnoise = R
        self.r_motionnoise = sigma

        self.r_dims = len(self.r_pose)
        self.r_phi = np.eye(self.r_dims) if phi is None else phi

        self.r_questbk = np.nan*np.zeros(localgridcoords) # empty
        self.r_guestbk_counter = np.zeros(localgridcoords)

    def r_xy(self):
        '''docstring'''
        return (self.r_pose[0], self.r_pose[1])

    def r_calc_motion(self, u_cntrl, phi_off=0.0):
        '''docstring'''
        dynamics = np.dot(self.r_pose, self.r_phi)*phi_off + u_cntrl
        return dynamics

    def r_move(self, u_cntrl, noisy=False, phi_off=0.0):

        dynamics = self.r_calc_motion(u_cntrl, phi_off=phi_off)
        noise = np.zeros(self.r_dims)
        if noisy:
            stdev = np.sqrt(self.r_motionnoise)
            noise = np.eye(self.r_dims)*np.random.normal(loc=0,
                                                         scale=stdev)
            noise[self.r_dims-1] = 0.0 # corr_r unaffected
        self.r_pose = dynamics + noise

    def r_measure(self, mapval):
        '''docstring'''
        z_proj = calc_z_proj([mapval]) # potentially non linear msmt h(x)
        msmt = projected_msmt([z_proj]) # quantisation
        print("Inside r_measure, i got mval", mapval, "and I'll return ", msmt) # imported from qif. mapval can be a list
        return msmt

    def r_addguest(self, pos_x, pos_y, msmt):
        '''Updates the guestbook of physical measurements at each nodes'''
        n_x, n_y = self.return_position(pos_x), self.return_position(pos_y)

        old_prob = np.nan_to_num(self.r_questbk[n_x, n_y])*1.0 # convert to 0. if empty
        old_counter = self.r_guestbk_counter[n_x, n_y]*1.0
        self.r_questbk[n_x, n_y] = (old_prob*old_counter + msmt)/(old_counter+1)
        self.r_guestbk_counter[n_x, n_y] += 1
        print(">>> Guestbook counter was updated to:")
        print self.r_guestbk_counter
        print(">>> Guestbook was updated to:")
        print self.r_questbk

    @staticmethod # static methods can be accessed /inhereited using self.
    def get_phase_method(prob):
        'Returns phase between 0 and 1 qubit states given a Born probability'
        return np.arccos(2.0*prob - 1.)

    @staticmethod
    def return_position(pos):
        ''' Converts float to int'''
        return int(pos)


class Scanner(Robot):
    '''docstring
    '''

    def __init__(self, localgridcoords, x=0, y=0, corr_r=1.42, sigma=0, R=0,
                 phi=None):
        '''
        Defines a 'measurement scan' that a robot makes. This consists of a
        physical measurement taken at the robot position (x,y), and quasi
        msmts taken on k nearest neighbours to (x,y). The quasi-msmt procedure
        and the size of the neighbourhood depends on an effective correlation
        length, corr_r.

        localgrid: assumed known and unchanging (array of qubits on hardware)
        corr_r = 1.42 : For k=1, nearest diagnoal neighbours are included
        '''
        Robot.__init__(self,
                       localgridcoords=localgridcoords,
                       x=x,
                       y=y,
                       corr_r=corr_r,
                       sigma=sigma,
                       R=R,
                       phi=phi)

    def r_corr_length(self, v_sep, type='Gaussian'):
        '''Returns the correlation strenght between physical and quasi
        measurements based on a separation distance, v_sep.BaseException

        This needs more careful analysis. 
        '''
        return np.exp(-(v_sep)**2 / 2.0*(1./(self.r_pose[2] + 1e-14))**2) 

    def r_get_quasi_msmts(self, knn_list, born_est):
        '''Returns quasi measurements on the neighbours of the robot,
        using the past measurement record at the robot pose
        '''
        quasi_msmts = []

        print('')
        print("Now I'm in r_get_quasi_msmts and I've got a physical born est: ", born_est)
        
        for neighbour in knn_list:

            vsep = np.linalg.norm(np.subtract(neighbour, self.r_xy()))
            quasi_born = self.r_corr_length(vsep)*born_est
            quasi_msmts.append(self.r_measure(self.get_phase_method(quasi_born)))

            print('For neightbour ', neighbour, 'the separation distance is: ', vsep)
            print('... yielding a quasi_born est of', quasi_born, 'and phase', self.get_phase_method(quasi_born))

        print('Now, the quasi measurements calculations have finished.')
        print('')
        return quasi_msmts

    def r_scan_local(self, mapval, knn_list):
        '''This function takes Born probability estimate based on physical
        msmts in the guestbook, and then blurs each msmt on its nearest
        neighbours to approximate a scan
        '''
        print('')
        print("Now I'm in r_scan_local ...")

        pose_x, pose_y = self.r_xy()
        # print("PRINT POSITION", pose_x, pose_y)

        msmt = self.r_measure(mapval)
        self.r_addguest(pose_x, pose_y, msmt)
        born_est = self.r_questbk[pose_x, pose_y]
        
        print("In r_scan_local, the scan has m_val of ", mapval, "yielding msmt", msmt)
        print("In r_scan_local, the guestbook gives a born estimate has a val of ", born_est)

        scan_msmts = [msmt]  +  self.r_get_quasi_msmts(knn_list, born_est)
        scan_posxy = [self.r_xy()] + knn_list

        print('r_scan_local has finished.')
        print('')
        return zip(scan_posxy, scan_msmts)