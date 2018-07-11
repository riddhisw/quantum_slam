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
        self.global_bot = None


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
        print("@weights.setter was called:", new_weights_)


    def resample(self):
        '''Resamples particles and resets uniform distribution of weights'''
        print('')
        print("I'm in resample()...")
        print("First, I'll sample particles from the posterior:")
        resampled_idx = self.sample_from_posterior()
        print('indicies are:', resampled_idx)
        self.particles = [self.particles[idx_r] for idx_r in resampled_idx]
        self.weights = np.ones(self.num_p)
        print("Then, I'll reset the particle weights to uniform.")
        print("For consistence, call the wieght attribute of the first particle:")
        print(self.particles[0].weight)
        print("resample complete.")
        print('')

        # for prtcl in iter(self.particles):
        #     prtcl.weight(self.num_p)


    def filter_update(self, msmt_vector, msmt_locations, control_pose, R):
        '''Executes particle filtering update step for one measurement scan'''
        print('')
        print("I'm in filter_update; with some msmts, msmt locations, controls")
        particles = iter(self.particles)
        proposed_weights = []
        print("I'm going to iterate through each particle....")
        for single_p in particles:
            print("... first, I apply the control")
            single_p.propagate_bot(control_pose)
            predictions = single_p.predict_scan(msmt_locations)
            
            print(" ...second, I obtain one step ahead predictions")
            print(predictions)
            print("... one step predictions were at the scanned locations:")
            print(msmt_locations)

            # predict first (not sure if this is the right R)
            print('... third, get likelihoods /weights for each particle ')
            
            particle_likelihood =[self.likelihood(msmt_vector, predictions, R)]
            proposed_weights.append(particle_likelihood)
            
            print("type: ", type(particle_likelihood), "len: ", len(particle_likelihood))
            print(particle_likelihood)

            # then update map
            print('... fourth, using msmts, update the map for each particle...')
            print("Msmt vector", type(msmt_vector), msmt_vector)
            print("Msmt locations", type(msmt_locations), msmt_locations)
            
            single_p.update_map_state(msmt_vector, msmt_locations)
            print("... at the fourth step, the updated particle map is:")
            print(single_p.m_vals)
            print('...and the guestbk for each particle is...')
            print(single_p.r_questbk)
            print('...  while the guestbk counter for each particle is...')
            print(single_p.r_guestbk_counter)

        self.weights = proposed_weights# update weights
        print('All particles have been updated. ')
        print('Now, the unnormalised proposed weights are')
        print(proposed_weights)
        print('While the self.weights returns:')
        print(self.weights)
        print("Next, I am going to resample...")
        self.resample()
        print("filter_update is complete")
        print('')

    def qslam_run(self, TrueMapobj, control_path, R=1, thres=0):
        ''' Runs SLAM for particle filtering'''
        print('')
        print("I'm begining qslam_run")
        print('')
        print("I've got a true environment that looks like this:")
        print TrueMapobj.m_vals
        print("... and a control path (pre determined)")
        t = 0
        T_terminate = len(control_path)
        print('The control path means the robot will make the following no. of msmts:')
        print T_terminate
        print('We set up  a global scnaner (indepednet of particle scanners):')
        
        mapdims = np.shape(TrueMapobj.m_vals)
        if self.global_bot is None:
            self.global_bot = Scanner(localgridcoords=mapdims, 
                                corr_r=0.)
        print('This gives us a global questbook of physical msmts only. Initially, this guestbook is empty:')
        print self.global_bot.r_questbk
        print('')
        print('Now I will walk along the control path and take msmts...')
        
        while t < T_terminate: # or some other threshold
            print('')
            self.global_bot.r_move(control_path[t])
            
            print('... I moved the global bot to the desired pose')
            print(self.global_bot.r_pose)
            
            u_x, u_y, u_corr_r = control_path[t] # can't depend on state vairables
            knn_list = TrueMapobj.m_knn_list( [u_x, u_y],  u_corr_r)
            env_map_val = TrueMapobj.m_vals[u_x, u_y] # noise lesss ?
            
            print("... at this desired pose, the we scan the env", env_map_val)
            scandata = self.global_bot.r_scan_local(env_map_val, knn_list)
            print("... the env scan gives the following scan_data: ")
            print(scandata)
            scan_posxy, scan_msmts = self.unzip_scan_data(scandata)
            
            print("... where the msmts:")
            print(scan_msmts)
            print('...where taken at the following places:')
            print(scan_posxy)
            print('')

            print('...This means that the global bot counts in physical msmts on the grid:')
            print(self.global_bot.r_guestbk_counter)
            print('...and the global bot  physical born probabilities are:')
            print(self.global_bot.r_questbk)
            print("...so this completes the physical env scan and storage of data in global bot.")
            print('')
            print("... this physical scan data is  passed to a slam particle filter...")
            print('')

            self.filter_update(scan_msmts, scan_posxy, control_path[t], R)
            t += 1
            print('')
            print('I completed one step in walking along the control path')
        
        print('')
        print("I've walked along the entire control path")

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
    

    @staticmethod
    def normalise(x_vec_):
        '''Normalises weight vector x'''
        x_vec = np.asarray(x_vec_).flatten()
        if not np.all(x_vec >= 0.0):
            print(x_vec, "invalid weight values")
            raise RuntimeError
        return (1./np.sum(x_vec)) * x_vec
    

    @staticmethod
    def likelihood(actuals, predicted, R):
        ''' Likelihood defined in quantised sensor information (trunc Gaussian)
            Dimenionality of likleihood depends on scanned info, but a weight is
            scalar so we have to take the norm'''
        actuals_ = np.asarray(actuals).flatten()
        predicted_ = np.asarray(predicted).flatten()
        resid_sqr = np.linalg.norm(actuals_ - predicted_)**2
        liklihood_val = (1./np.sqrt(2*np.pi*R))*np.exp(-resid_sqr/(2.0*R))
        return liklihood_val
    
    @staticmethod
    def unzip_scan_data(scan_data):
        '''Helper function for flattening measurements'''

        scan_posxy, scan_msmts = zip(*scan_data)
        scan_msmts_ = np.asarray(scan_msmts).flatten()

        return scan_posxy, scan_msmts_ 
