'''
MODULE: qslamr

Returns discrete posterior for the SLAM problem for qubit control as a set of
particles (true states) and their weights (posterior distribution).

CLASS: Node, Grid

METHODS

PARAMETERS

STATIC FUNCTIONS


'''
import numpy as np
import particledict as pd
from model_design import GRIDDICT, INITIALDICT
from hardware import Grid
from itertools import combinations
from particlesets import AlphaParticle, ParticleSet, BetaParticle
from particledict import WEIGHTFUNCDICT_ALPHA, WEIGHTFUNCDICT_BETA, update_alpha_dictionary

class ParticleFilter(Grid):
    '''doctring
    
    MU_W : 0.0, # Qubit position noise mean (dynamics)
    SIG2_W : 1.0, # Qubit position noise variance (dynamics)
    MU_R : 0.0, # Length scale noise mean (dynamics)
    SIG2_R : 1.0, # Length scale noise variance (dynamics)
    MU_MEASR : 0.0, # Map noise mean (measurement)
    SIG2_MEASR : 1.0, # Map noise variance (measurement)
    MU_F : 0.0, # True sigmoid approximation error mean
    SIG2_F : 1.0, # True sigmoid approximation error variance
    LAMBDA : 0.9, # Forgetting factor for quasi-msmt information
    GAMMA_T : 10**8, # Re-sampling threshold
    P_ALPHA : 10, # Number of alpha particles
    P_BETA : 10, # Numer of beta particles for each alpha

    '''

    def __init__(self, list_of_nodes_positions, **INITIALDICT):
        # Load model design and chip config
        self.QubitGrid = Grid(list_of_nodes_positions=list_of_nodes_positions)
        self.dgrid, _= self.find_max_distance(self.QubitGrid.list_of_nodes_positions)
        self.INITIALDICT = INITIALDICT

        # Set up alpha particles
        self.pset_alpha = self.INITIALDICT["P_ALPHA"]
        self.pset_beta = self.INITIALDICT["P_BETA"]
        empty_alpha_particles = [AlphaParticle() for idx in range(self.pset_alpha)]
        self.AlphaSet = ParticleSet(empty_alpha_particles, **WEIGHTFUNCDICT_ALPHA)

        # resampling threshold
        self.resample_thresh = INITIALDICT["GAMMA_T"]
        self.L_factor = self.dgrid * 1.0


    def qslamr(self, measurements_controls):

        self.InitializeParticles()

        for t_item in measurements_controls:

            print
            print
            print " NEW MSMT"
            print
            next_phys_msmt_j = t_item[0]
            control_j = t_item[1]
            self.ReceiveMsmt(control_j, next_phys_msmt_j)
            print
            print " PROPAGATE"
            print
 
            self.PropagateState(control_j)
            print
            print " COMPUTER WEIGHTS "
            print
 
            posterior_weights = self.ComputeWeights(control_j)

            print
            print " RESAMPLE"
            print
 
            self.ResampleParticles(posterior_weights)
            posterior_state = self.AlphaSet.posterior_state
            self.QubitGrid.state_vector = posterior_state 
            print
            print " QUASI MSMT"
            print
 
            self.update_qubitgrid_via_quasimsmts(control_j, posterior_state)

#       ------------------------------------------------------------------------
#       DATA ASSOCIATION FUNCTIONS
#       ------------------------------------------------------------------------

    def update_alpha_map_via_born_rule(self, alpha_particle): # TODO: Data association
        '''doc string
        this function must always come after we havw computed alpha weights
        '''

        for alpha_particle in self.AlphaSet.particles:
            born_prob = self.sample_prob_from_msmts(alpha_particle.node_j)
            map_val = ParticleFilter.inverse_born(born_prob)
            map_idx = self.QubitGrid.number_of_nodes * 2 + alpha_particle.node_j
            parent_particle = alpha_particle.particle # get property
            parent_particle[map_idx] = map_val
            alpha_particle.particle = parent_particle # assign updated property


    def sample_prob_from_msmts(self, control_j): # TODO Data Association
        '''docstring'''
        prob_p = self.QubitGrid.nodes[control_j].physcmsmtsum / self.QubitGrid.nodes[control_j].counter_tau*1.0
        forgetting_factor = self.INITIALDICT["LAMBDA"]**self.QubitGrid.nodes[control_j].counter_tau

        prob_q = 0.0
        if self.QubitGrid.nodes[control_j].counter_beta != 0:
            prob_q = self.QubitGrid.nodes[control_j].quasimsmtsum / self.QubitGrid.nodes[control_j].counter_beta*1.0
        prob_j = prob_p + forgetting_factor*prob_q
        return prob_j


    @staticmethod
    def born_rule(map_val):
        '''docstring'''
        born_prob = np.cos(map_val / 2.0)**2
        return born_prob


    @staticmethod
    def inverse_born(born_prob):
        map_val = np.arccos(2.0*born_prob  - 1.0)
        return map_val

#       ------------------------------------------------------------------------
#       SMEARING ACTION VIA QUASI MEASUREMENTS
#       ------------------------------------------------------------------------
    
    def update_qubitgrid_via_quasimsmts(self, control_j, posterior_state):
        '''docstring
        this funciton should only be applied after lengthscales have been discovered
        (alpha, beta_alpha) particles carried over or are resampled with 
        sufficient alpha diversity; weights are set to uniform. 
        '''
        
        SmearParticle = BetaParticle(control_j, posterior_state)
        self.generate_beta_neighbourhood(SmearParticle)

        for idx in range(len(SmearParticle.neighbourhood_qj)):

            neighbour_q = SmearParticle.neighbourhood_qj[idx]
            quasi_phase_q = SmearParticle.smeared_phases_qj[idx]
            born_prob_q = ParticleFilter.born_rule(quasi_phase_q)
            quasi_msmt = np.random.binomial(1, born_prob_q)
            self.QubitGrid.nodes[neighbour_q].quasimsmtsum = quasi_msmt

#       ------------------------------------------------------------------------
#       SUPPORT FUNCTION 1: INITIALISE AND SAMPLE AT t = 0
#       ------------------------------------------------------------------------

    def InitializeParticles(self):
        '''docstring t=1'''

        for alphaparticle in self.AlphaSet.particles:
            self.set_init_alphaparticle(alphaparticle)

        self.QubitGrid.state_vector = self.AlphaSet.posterior_state
        print "Initial alpha weights (below) are uniformly distributed..."
        print self.AlphaSet.weights_set
        print "and alpha_weights sum to 1:"
        print np.sum(self.AlphaSet.weights_set)

    def set_init_alphaparticle(self, alphaparticle):
        '''docstring'''
        sample_s = np.random.normal(loc=self.INITIALDICT["MU_W"],
                                    scale=self.INITIALDICT["SIG2_W"],
                                    size=self.QubitGrid.number_of_nodes*2)

        sample_f = np.random.uniform(low=0.,
                                     high=np.pi, # INITIAL COND
                                     size=self.QubitGrid.number_of_nodes)

        sample_r = np.random.uniform(low=0.,
                                     high=self.dgrid*3.0, # INITIAL COND
                                     size=self.QubitGrid.number_of_nodes)

        alphaparticle.particle = np.concatenate((sample_s, sample_f, sample_r),
                                                 axis=0)
        
        alphaparticle.pset_beta = self.INITIALDICT["P_BETA"]
        alphaparticle.mean_radius_j = self.dgrid*1.5

#       ------------------------------------------------------------------------
#       SUPPORT FUNCTION 2: RECEIVE MSMT
#       ------------------------------------------------------------------------

    def ReceiveMsmt(self, control_j, next_phys_msmt_j):
        '''docstring'''
        self.QubitGrid.nodes[control_j].physcmsmtsum = next_phys_msmt_j
        prob_j = self.sample_prob_from_msmts(control_j)

        
        for alpha_particle in self.AlphaSet.particles:
            alpha_particle.pset_beta = self.INITIALDICT["P_BETA"]
            alpha_particle.node_j = control_j
            map_index = self.QubitGrid.number_of_nodes*3 + alpha_particle.node_j 
            # alpha_particle.mean_radius_j = self.AlphaSet.posterior_state[map_index] 
            # TODO: neighbourbood mean radius update


        update_alpha_dictionary(next_phys_msmt_j, prob_j)

#       ------------------------------------------------------------------------
#       SUPPORT FUNCTION 3: PROPAGATE (ALPHA) STATES
#       ------------------------------------------------------------------------

    def PropagateState(self, control_j):
        '''docstring '''
        for alpha_particle in self.AlphaSet.particles:
            self.sample_from_transition_dist(alpha_particle, control_j)

    def sample_from_transition_dist(self, alpha_particle, control_j):
        '''docstring t > 1'''

        sample_s = np.random.normal(loc=INITIALDICT["MU_W"],
                                    scale=INITIALDICT["SIG2_W"],
                                    size=self.QubitGrid.number_of_nodes*2)

        sample_f = np.ones(self.QubitGrid.number_of_nodes)

        sample_r = np.random.normal(loc=INITIALDICT["MU_R"],
                                    scale=INITIALDICT["SIG2_R"],
                                    size=self.QubitGrid.number_of_nodes)

        alpha_particle.particle = np.concatenate((sample_s, sample_f, sample_r),
                                                 axis=0)
        
#       ------------------------------------------------------------------------
#       SUPPORT FUNCTION 4: COMPUTE ALPHA WEIGHTS; GENERATE BETA WEIGHTS
#       ------------------------------------------------------------------------

    def ComputeWeights(self, control_j):
        '''docstring'''

        new_alpha_weights = self.AlphaSet.calc_weights_set() # Normlaised
        self.AlphaSet.weights_set = new_alpha_weights

        posterior_weights = []

        for alpha_particle in self.AlphaSet.particles:

            self.update_alpha_map_via_born_rule(control_j)
            beta_alpha_j_weights = self.generate_beta_layer(alpha_particle)
            print "beta_alpha_j_weights in Compute Weihgts", beta_alpha_j_weights
            posterior_weights.append(alpha_particle.weight*beta_alpha_j_weights)
        
        posterior_weights = np.asarray(posterior_weights).flatten()
        normalisation = np.sum(posterior_weights)
        print "Posterior weights sum to=", normalisation

        normalised_posterior_weights = posterior_weights*(1.0/normalisation)
        print
        print "In ComputeWeights, normalised_posterior_weights", normalised_posterior_weights
        print "type(normalised_posterior_weights)", type(normalised_posterior_weights)
        print 
        return  normalised_posterior_weights # savedas posterior_weights[alphaindex][betaalphaindex]


    def sample_radii(self, previous_length_scale, L_factor=None, Band_factor=None):
        '''docstring'''
        if L_factor is None:
            L_factor = self.L_factor

        if Band_factor is None:
            Band_factor = 0.0

        sample = np.random.uniform(low=Band_factor, high=L_factor)
        return sample


    def generate_beta_layer(self, alpha_particle):
        '''docstring'''

        len_idx = self.QubitGrid.number_of_nodes * 3 + alpha_particle.node_j
        parent_alpha = alpha_particle.particle # get property

        list_of_parent_states = []
        for idx_beta in range(alpha_particle.pset_beta):

            parent_alpha[len_idx] = self.sample_radii(parent_alpha[len_idx])
            list_of_parent_states.append(parent_alpha)
        
        print
        print "In generate_beta_layer, list_of_parent_states =", list_of_parent_states
        alpha_particle.generate_beta_pset(list_of_parent_states) #generates beta layer for each alpha
        print "alpha_particle post generate_beta_pset", alpha_particle.BetaAlphaSet_j.particles

        for beta_particle_object in alpha_particle.BetaAlphaSet_j.particles:
            self.generate_beta_neighbourhood(beta_particle_object) # compute smeared phases for each beta particle

        beta_alpha_j_weights = alpha_particle.BetaAlphaSet_j.calc_weights_set()
        
        print "In generate_beta_layer, beta_alpha_j_weights ", beta_alpha_j_weights.shape
        print beta_alpha_j_weights
        print
        return beta_alpha_j_weights # these weights are normalised


    def generate_beta_neighbourhood(self, BetaParticle):
        '''docstring'''

        # BetaParticle.mean_radius = new_neighbourhood_L

        NEIGHBOURDICT = {"prev_posterior_f_state" : self.QubitGrid.get_all_nodes(["f_state"]),
                         "prev_counter_tau_state" : self.QubitGrid.get_all_nodes(["counter_tau"]),
                         "lambda_" : self.INITIALDICT["LAMBDA"],
                         "kernel_function": self.INITIALDICT["kernel_function"]}

        BetaParticle.smear_fj_on_neighbours(**NEIGHBOURDICT)

#       ------------------------------------------------------------------------
#       SUPPORT FUNCTION 5: RESAMPLE AND BETA COLLAPSE
#       ------------------------------------------------------------------------

    def ResampleParticles(self, posterior_weights):
        '''docstring'''
        resampled_idx = np.arange(self.pset_alpha*self.pset_beta)

        if self.resample_thresh > self.effective_particle_size(posterior_weights):
            resampled_idx = ParticleFilter.resample_constant_pset_alpha(posterior_weights, self.pset_alpha, self.pset_beta)

        new_alpha_subtrees = self.get_subtrees(resampled_idx, self.pset_beta)
        new_alpha_list = self.collapse_beta(new_alpha_subtrees, resampled_idx)

        self.pset_alpha = len(new_alpha_list)
        print "Resample Partices, self.pset_alpha ", self.pset_alpha 
        self.AlphaSet.particles = new_alpha_list # garanteed to be pset_alpha with no second layer
        self.AlphaSet.weights_set = (1.0/self.pset_alpha)*np.ones(self.pset_alpha)

    def collapse_beta(self, subtree_list, resampled_indices):
        '''docstring'''

        state_update = 0.
        new_alpha_particle_list = []
        for subtree in subtree_list:

            leaves_of_subtree = resampled_indices[subtree[0]:subtree[1]]
            leaf_count = float(len(leaves_of_subtree))
            print "The subtree is defined by the endpoint index boundaries", subtree
            print "The leaves of the subtree are ", leaves_of_subtree

            if leaf_count != 0:

                normaliser = (1./leaf_count)
                alpha_node = ParticleFilter.get_alpha_node_from_treeleaf(leaves_of_subtree[0], self.pset_beta)
                               # resampled_indices[subtree[0]], self.pset_beta)
                beta_alpha_nodes = [ParticleFilter.get_beta_node_from_treeleaf(leafy, self.pset_beta) for leafy in leaves_of_subtree]
                               # resampled_indices[subtree[0]:subtree[1]]]
                print "The subtree has alpha node of: ", alpha_node
                print "The leaves of the subtree are labeled by beta indices", beta_alpha_nodes
                print
                r_est_subtree = 0.0
                for node in beta_alpha_nodes:
                    beta_state = self.AlphaSet.particles[alpha_node].BetaAlphaSet_j.particles[node].particle
                    node_j = self.AlphaSet.particles[alpha_node].node_j
                    beta_lengthscale = beta_state[int(node_j)]
                    r_est_subtree += normaliser*beta_lengthscale

                parent = self.AlphaSet.particles[alpha_node].particle
                parent[self.AlphaSet.particles[alpha_node].node_j] = r_est_subtree

                # Beta Layer Collapsed
                self.AlphaSet.particles[alpha_node].particle = parent
                self.AlphaSet.particles[alpha_node].BetaAlphaSet_j = None

                # New Alphas Stored
                new_alpha_particle_list.append(self.AlphaSet.particles[alpha_node])

        return new_alpha_particle_list


    def effective_particle_size(self, posterior_weights):
        '''docstring'''
        print "In effective_particle_size, posterior_weights shape:"
        print posterior_weights
        print "type(posterior_weights)", type(posterior_weights)
        p_size = 1.0/ np.sum(posterior_weights**2)
        self.L_factor = p_size*self.dgrid
        return p_size


#       ------------------------------------------------------------------------
#       SUPPORT FUNCTION 6: STATIC METHODS
#       ------------------------------------------------------------------------

    @staticmethod
    def compute_dist(one_pair):
        '''doctring'''
        xval, yval = one_pair
        return np.sqrt((xval[0] - yval[0])**2 + (xval[1] - yval[1])**2)

    @staticmethod
    def find_max_distance(list_of_positions):
        '''docstring'''
        distance_pairs = [a_pair for a_pair in combinations(list_of_positions, 2)]
        distances = [ParticleFilter.compute_dist(one_pair)for one_pair in distance_pairs]
        return max(distances), distance_pairs[np.argmax(distances)]

    @staticmethod
    def get_alpha_node_from_treeleaf(leaf_index, pset_beta):
        '''docstring'''
        alpha_node = int(leaf_index//float(pset_beta))
        return alpha_node

    @staticmethod
    def get_beta_node_from_treeleaf(leaf_index, pset_beta):
        '''docstring'''
        beta_node = int(leaf_index - int(leaf_index//float(pset_beta))*pset_beta)
        return beta_node

    @staticmethod
    def resample_from_weights(posterior_weights, number_of_samples):
        '''docstring'''
        total_particles = len(posterior_weights)
        cdf_weights = np.asarray([0] + [np.sum(posterior_weights[:idx+1]) for idx in range(total_particles)])
        pdf_uniform = np.random.uniform(low=0, high=1.0, size=number_of_samples)

        resampled_idx = []

        for u_0 in pdf_uniform:
            j = 0
            while u_0 > cdf_weights[j]:
                j += 1
                if j > total_particles:
                    j = total_particles
                    # print('Break - max particle index reached during sampling')
                    break   # clip at max particle index, plus zero
            resampled_idx.append(j-1) # sgift down to match python indices

        return resampled_idx

    @staticmethod
    def resample_constant_pset_alpha(posterior_weights, pset_alpha, pset_beta):
        '''Returns indicies for particles picked after sampling from posterior'''
        # DO WEIGHTS NEED TO BE SORTED? (No, the CDF will be
        # monotnically increasing; and the shape of the CDF will determine
        # the frequency at which (x,y) are sampled if y is uniform )

        sufficient_sample = False
        num_of_samples = pset_alpha
        total_particles = len(posterior_weights)

        print "posterior_weights", posterior_weights

        if total_particles != int(INITIALDICT["P_ALPHA"]*INITIALDICT["P_BETA"]):
            print "Total weights != P_alpha * P_beta"
            raise RuntimeError

        while sufficient_sample is False:

            num_of_samples += 5
            print "Number of sufficient samples are:", num_of_samples
            resampled_indices = ParticleFilter.resample_from_weights(posterior_weights, num_of_samples)
            resampled_alphas = [ParticleFilter.get_alpha_node_from_treeleaf(leafy, pset_beta) for leafy in resampled_indices]
            unique_alphas = set(list(resampled_alphas))
            if len(unique_alphas) == pset_alpha:
                sufficient_sample = True

        return resampled_indices

    @staticmethod
    def get_subtrees(resampled_indices, pset_beta):
        '''docstring'''

        new_sub_trees = []

        resampled_indices.sort()
        alpha_index_0 = None
        strt_counter = 0
        end_counter = 0

        for idx in resampled_indices:

            alpha_index = ParticleFilter.get_alpha_node_from_treeleaf(idx, pset_beta)
            beta_alpha_idx = ParticleFilter.get_beta_node_from_treeleaf(idx, pset_beta)

            if alpha_index_0 == alpha_index:
                end_counter += 1

            elif alpha_index_0 != alpha_index:

                new_sub_trees.append([strt_counter, end_counter])

                alpha_index_0 = alpha_index
                strt_counter = end_counter
                end_counter += 1

        if end_counter == len(resampled_indices):
            # end_counter += 1
            new_sub_trees.append([strt_counter, end_counter])

        print "in get subtrees", new_sub_trees
        return new_sub_trees
    # def collapse_beta(self, new_sub_trees, resampled_indices):
    #     '''docstring'''

    #     state_update = 0.
    #     pset_beta = self.pset_beta

    #     new_alpha_particle_list = []
    #     for pairs in new_sub_trees:

    #         subtree = resampled_indices[pairs[0]:pairs[1]]
    #         leaf_count = float(len(subtree))

    #         print
    #         print "In collapse_beta, and checking variables"
    #         print "subtree", subtree
    #         print "leaf_count", leaf_count


    #         if leaf_count != 0:

    #             normaliser = (1./leaf_count)
    #             alpha_node = ParticleFilter.get_alpha_node_from_treeleaf(pairs[0], pset_beta)
    #             beta_alpha_nodes = [ParticleFilter.get_beta_node_from_treeleaf(leafy, pset_beta) for leafy in subtree]
