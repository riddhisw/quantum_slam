''' Module: qslam.py

The purpose of this module is to implement SLAM via particle filtering
The module contains the following classes:
QSlam(Robot, Map)
'''

from qslam.particle import Particle
from qslam.sensing import Scanner

import numpy as np

class ParticleFilter(object):
    '''doctring'''

    def __init__(self,localgridcoords_=[1, 1],
                 num_p=1, particles=None):

        self.num_p = num_p
        self.particles = [Particle(localgridcoords=localgridcoords_) for i in range(self.num_p)] if particles is None else particles
        self.__weights = np.ones(self.num_p)

    @property
    def weights(self):
        '''Particle weights''' #weights only stored as Particle.weight
        self.__weights = np.asarray(list( self.particles[idx_p].weight for idx_p in range(self.num_p)))
        return self.__weights
    @weights.setter
    def weights(self, new_weights):
        new_weights_ = self.normalise(new_weights)
        for idx_p in range(self.num_p):
            self.particles[idx_p].weight = new_weights_[idx_p]
        self.__weights = new_weights_
        print("@wieghts.setter was called:", new_weights_)
    
    @classmethod
    def normalise(self, x_vec_):
        '''Normalises weight vector x'''
        x_vec = np.asarray(x_vec_).flatten()
        if not np.all(x_vec >= 0.0):
            print(x_vec, "invalid weight values")
            raise RuntimeError
        return (1./np.sum(x_vec)) * x_vec

    @classmethod
    def likelihood(self, actuals, predicted, R):
        ''' Likelihood defined in quantised sensor information (trunc Gaussian)
            Dimenionality of likleihood depends on scanned info, but a weight is
            scalar so we have to take the norm'''
        actuals_ = np.asarray(actuals).flatten()
        predicted_ = np.asarray(predicted).flatten()
        resid_sqr = np.linalg.norm(actuals_ - predicted_)**2
        liklihood_val = (1./np.sqrt(2*np.pi*R))*np.exp(-resid_sqr/(2.0*R))
        return liklihood_val


    def resample(self):
        '''Resamples particles and resets uniform distribution of weights'''
        
        print("First, I'll sample particles from the posterior:")
        print('')
        resampled_idx = self.sample_from_posterior()
        print("Then, I'll reset the particle weights to uniform")
        print('')
        self.particles = [self.particles[idx_r] for idx_r in resampled_idx]
        self.weights = np.ones(self.num_p)

        # for prtcl in iter(self.particles):
        #     prtcl.weight(self.num_p)


    def filter_update(self, msmt_vector, msmt_locations, control_pose, R):
        '''Executes particle filtering update step for one measurement scan'''
        
        particles = iter(self.particles)
        proposed_weights = []

        for single_p in particles:

            single_p.propagate_bot(control_pose)
            predictions = single_p.predict_scan(msmt_locations)
            
            print("I'm in filter update. The one step predictions are:")
            print(predictions)
            print("... one step predictions were at the scanned locations:")
            print(msmt_locations)

            # predict first (not sure if this is the right R)
            print('Now i get my likelihood value /weights for each particle ')
            
            particle_likelihood =[self.likelihood(msmt_vector, predictions, R)]
            proposed_weights.append(particle_likelihood)
            
            print("type: ", type(particle_likelihood), "len: ", len(particle_likelihood))
            print(particle_likelihood)

            # then update map
            print('... using msmts, i update the map for each particle...')
            single_p.update_map_state(msmt_vector, msmt_locations)
            print(single_p.m_vals)

            print('... btw, this is what the guestbk for each particle is...')
            print(single_p.r_questbk)

            print('...  while the guestbk counter for each particle is...')
            print(single_p.r_guestbk_counter)

        self.weights = proposed_weights# update weights
        print('After updating weights for each particle, the unnormalised proposed weights are')
        print(proposed_weights)
        print('While the self.weights returns:')
        print(self.weights)
        print("...Now I am going to resample...")
        self.resample()

    def qslam_run(self, TrueMapobj, control_path, R=1, thres=0):
        ''' Runs SLAM for particle filtering'''
        
        t = 0
        T_terminate = len(control_path)
        mapdims = np.shape(TrueMapobj.m_vals)
        global_bot = Scanner(localgridcoords=mapdims, corr_r=0.)

        # self.particles(mapdims) # initiate @particles.setter

        while t < T_terminate: # or some other threshold
            
            print('I moved the global bot to the desired position + correlation:')
            global_bot.r_move(control_path[t])
            print(global_bot.r_pose)

            u_x, u_y, u_corr_r = control_path[t] # can't depend on state vairables
            knn_list = TrueMapobj.m_knn_list( [u_x, u_y],  u_corr_r)
            
            scandata = global_bot.r_scan_local(TrueMapobj.m_vals[u_x, u_y], knn_list)
            scan_posxy, scan_msmts = zip(*scandata)
            print("I took some msmts in a scan. These are:")
            print(scan_msmts)
            print('taken at the following places:')
            print(scan_posxy)
            print('This means that the true questbook counter for the scanner is')
            print(global_bot.r_guestbk_counter)
            print('and the true questbook for the scanner is')
            print(global_bot.r_guestbk_counter)
            print("...and now I'm going to do the filtering...")
            print('')
            self.filter_update(scan_msmts, scan_posxy, control_path[t], R)
            t += 1

    def sample_from_posterior(self):
        '''Returns indicies for particles picked after sampling from posterior'''
        # DO WEIGHTS NEED TO BE SORTED? (No, the CDF will be
        # monotnically increasing; and the shape of the CDF will determine
        # the frequency at which (x,y) are sampled if y is uniform )

        if self.weights is not None:

            print('I am just about to resample, these are the weights:')
            print(self.weights)
            
            cdf_weights = np.asarray([0] + [np.sum(self.weights[:idx+1]) for idx in range(self.num_p)])
            pdf_uniform = np.random.random(size=self.num_p)
            
            resampled_idx = []

            print('Im resampling, and the resampled indices are:')
            
            for u_0 in pdf_uniform:
                j = 0
                while u_0 > cdf_weights[j]:
                    j += 1
                    if j >= self.num_p:
                        j = self.num_p -1
                        print('Break - max particle index reached during sampling')
                        break   # clip at max particle index, plus zero
                resampled_idx.append(j)
            
            print(resampled_idx)

            return resampled_idx