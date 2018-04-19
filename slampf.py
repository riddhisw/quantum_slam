''' Module: qslam.py

The purpose of this module is to implement SLAM via particle filtering
The module contains the following classes:
QSlam(Robot, Map)
'''

from qslam.particle import Particle
from qslam.sensing import Scanner
from qslam.mapping import TrueMap

import numpy as np

class ParticleFilter(object):
    '''doctring'''

    def __init__(self,
                 num_p, weights=None,
                 **kwargs):

        self.num_p = num_p
        self.particles = [Particle.__init__(**kwargs) for idx_p in range(self.num_p)]
        weights = np.ones(self.num_p) if weights is None else weights
        self.weights = weights


    @classmethod
    def normalise(cls, x_vec):
        '''Normalises weight vector x'''

        if x_vec.any() < 0:
            print(x_vec, "invalid weight values")
            raise RuntimeError
        return (1./np.sum(x_vec)) * x_vec

    @classmethod
    def sample_from_posterior(cls, weights):
        '''Returns indicies for particles picked after sampling from posterior'''
        # DO WEIGHTS NEED TO BE SORTED? (No, the CDF will be
        # monotnically increasing; and the shape of the CDF will determine
        # the frequency at which (x,y) are sampled if y is uniform )

        num = len(weights)
        cdf_weights = np.asarray([0] + [np.sum(weights[:idx+1]) for idx in range(num)])
        pdf_uniform = np.random(size=num)

        resampled_idx = []

        for u_0 in pdf_uniform:
            j = 0
            while u_0 > cdf_weights[j]:
                j += 1
            resampled_idx.append(j)

        return resampled_idx

    @classmethod
    def likelihood(cls, actuals, predicted, R):
        ''' Likelihood defined in quantised sensor information (trunc Gaussian)
            Dimenionality of likleihood depends on scanned info, but a weight is
            scalar so we have to take the norm'''

        resid_sqr = np.linalg.norm(actuals - predicted)**2
        liklihood_val = (1./np.sqrt(2*np.pi*R))*np.exp(-resid_sqr/(2.0*R))
        return liklihood_val

    @property
    def weights(self):
        '''Particle weights''' #weights only stored as Particle.weight
        self.__weights = np.asarray(list( self.particles[idx_p].weight for idx_p in range(self.num_p)))
        return self.__weights
    @weights.setter
    def weights(self, new_weights):
        new_weights = cls.normalise(cls, new_weights)
        for idx_p in range(self.num_p):
            self.particles[idx_p].weight = new_weights[idx_p]

    def resample(self):
        '''Resamples particles and resets uniform distribution of weights'''
        resampled_idx = cls.sample_from_posterior(self.__weights)
        self.particles = [self.particles[idx_r] for idx_r in resampled_idx]

        for prtcl in iter(self.particles):
            prtcl.weight(self.num_p)

    def filter_update(self, msmt_vector, msmt_locations, control_pose):
        '''Executes particle filtering update step for one measurement scan'''

        particles = iter(self.particles)
        proposed_weights = []

        for single_p in particles:

            single_p.propagate_bot(control_pose)
            predictions = single_p.predict_scan(msmt_locations)

            # predict first (not sure if this is the right R)
            proposed_weights.append([cls.likelihood(msmt_vector, predictions, self.r_msmtnoise)])

            # then update map
            single_p.update_map_state(msmt_vector, msmt_locations)

        self.weights(proposed_weights) # update weights
        self.resample()

    def qslam_run(self, TrueMapobj, control_path, T_terminate,  thres=0):
        ''' Runs SLAM for particle filtering'''
        t = 0
        knn_list = TrueMapobj.m_knn_list()
        while t < T_terminate: # or some other threshold

            u_x, u_y, u_corr_r = control_path[t] # can't depend on state vairables

            msmt_device = Scanner(self.localgrid, # THIS DEPENDS ON **KWARGS 
                                  x=u_x, y=u_y, corr_r=u_corr_r,
                                  sigma=0, R=0, phi=None)
            scandata = msmt_device.r_scan_local(TrueMapobj.m_val[x, y], knn_list)
            scan_posxy, scan_msmts = zip(*scandata)

            self.filter_update(scan_posxy, scan_msmts, control_path[t])
            t += 1
