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
PARTICLE_STATE = ["x_state", "y_state", "f_state", "r_state"]
from model_design import INITIALDICT

###############################################################################
# ALPHA PARTICLES
###############################################################################

def rho(b_val, var_r):
    '''docstring'''
    rho_0 = 0 # TODO pull from other paper
    return rho_0

def likelihood_func_alpha(**args):
    '''docstring'''
    msmt_dj = args["msmt_dj"]
    prob_j = args["prob_j"]
    var_r = args["var_r"]
    rho_0 = rho(0.5, var_r)

    alpha_weight = rho_0 / 2.0
    if msmt_dj == 0:
        alpha_weight += -1.0*rho_0*(2.0*prob_j - 1)
    elif msmt_dj == 1:
        alpha_weight += 1.0*rho_0*(2.0*prob_j - 1)

    return alpha_weight

LIKELIHOOD_ALPHA = {"l_func" : likelihood_func_alpha,
                    "l_args" : {"mu_R" : INITIALDICT["MU_R"], # TODO:
                                "var_r" : INITIALDICT["SIG2_R"], # TODO:
                                "msmt_dj" : -10.0, # TODO: update via PF
                                "prob_j" : -10.0 # TODO: update via PF 
                               }}

def alpha_weight_calc(alpha_particle_object, **args):
    '''docstring'''
    old_weight = alpha_particle_object.weight
    likelihood = args["l_func"](**args["l_args"]) # TODO: there needs to be a better way of taking in msmts. 
    new_raw_weight = old_weight*likelihood
    return new_raw_weight


WEIGHTFUNCDICT_ALPHA = {"function": alpha_weight_calc, "args": LIKELIHOOD_ALPHA}

###############################################################################
# BETA PARTICLES
###############################################################################

def likelihood_func_beta(**args):
    '''E 29'''
    mean = args["mu_f"]
    variance = args["sigma_f"]
    new_phase = args["new_phase"]
    old_phase = args["old_phase"]
    prefactor = 1.0 / np.sqrt(2.0 * np.pi * variance)
    argument = -1.0 * ((new_phase - old_phase)- mean)**2 / (2.0 * variance)
    result = prefactor * np.exp(argument)
    return result

LIKELIHOOD_BETA = {"l_func": likelihood_func_beta,
                   "l_args": {"sigma_f" : INITIALDICT["SIG2_F"],
                              "mu_f" : INITIALDICT["MU_F"],
                              "new_phase": 0.0, # TODO: update via PF
                              "old_phase": 0.0  # TODO: update via PF
                             }
                  }

def beta_weight_calc(BetaParticle, **args):
    '''docstring'''
    # old_weight = BetaParticle.weight
    likelihood_neighbours = []

    for idx_q in range(len(BetaParticle.neighbourhood_qj)):
        args["new_phase"] = BetaParticle.smeared_phases_qj[idx_q]
        args["old_phase"] = BetaParticle.parent[idx_q]
        likelihood = args["l_func"](**args["l_args"]) # TODO: there needs to be a better way of taking in new_phase, old_phase.
        likelihood_neighbours.append(likelihood)

    net_likelihood = np.prod(np.asarray(likelihood_neighbours).flatten())
    return net_likelihood


WEIGHTFUNCDICT_BETA = {"function": beta_weight_calc, "args": LIKELIHOOD_BETA}

###############################################################################
# CHIP STRUCTURE 
###############################################################################
class Node(object):
    '''[docstring]'''

    def __init__(self):
        # self.qubitid = 0
        self.f_state = 0.0
        self.r_state = 0.0
        self.x_state = 0.0
        self.y_state = 0.0
        self.counter_tau = 0
        self.counter_beta = 0
        self.__physcmsmtsum = 0.0
        self.__quasimsmtsum = 0.0

    @property
    def physcmsmtsum(self):
        '''docstring'''
        return self.__physcmsmtsum
    @physcmsmtsum.setter
    def physcmsmtsum(self, next_phys_msmt):
        self.__physcmsmtsum += next_phys_msmt
        self.counter_tau += 1

    @property
    def quasimsmtsum(self):
        '''docstring'''
        return self.__quasimsmtsum
    @quasimsmtsum.setter
    def quasimsmtsum(self, next_quasi_msmt):
        self.__quasimsmtsum += next_quasi_msmt
        self.counter_beta += 1


class Grid(object):
    '''docstring '''

    def __init__(self, list_of_nodes_positions=None):
        if list_of_nodes_positions is None:
            print "No node positions specified"
            raise RuntimeError

        self.list_of_nodes_positions = list_of_nodes_positions
        self.number_of_nodes = len(self.list_of_nodes_positions)
        self.nodes = [Node() for i in range(self.number_of_nodes)]
        self.__state_vector = 0.0
        self.state_vector = np.zeros(self.number_of_nodes*len(PARTICLE_STATE))

        for item in xrange(self.number_of_nodes):
            self.nodes[item].x_state, self.nodes[item].y_state = self.list_of_nodes_positions[item]

    def get_all_nodes(self, attribute_list):
        '''Returns attribute for all nodes in a single vector, stacked by nodes, then attributes '''
        vector = [[getattr(node, attr) for node in self.nodes] for attr in attribute_list]
        return np.asarray(vector).flatten()

    def set_all_nodes(self, single_attribute, attribute_values):
        '''Sets attribute for all nodes '''
        for item in xrange(len(attribute_values)):
            setattr(self.nodes[item], single_attribute, attribute_values[item])

    @property
    def state_vector(self):
        '''docstring'''
        self.__state_vector = self.get_all_nodes(PARTICLE_STATE)
        return self.__state_vector
    @state_vector.setter
    def state_vector(self, new_state_vector):
        '''docstring'''
        for state in xrange(len(PARTICLE_STATE)):
            bgn = state * self.number_of_nodes
            end = bgn + self.number_of_nodes
            self.set_all_nodes(PARTICLE_STATE[state], new_state_vector[bgn:end])


###############################################################################
# PARTICLE STRUCTURE
###############################################################################

class Particle(object):
    '''docstring'''
    def __init__(self):

        self.__particle = 0.0
        self.__weight = 0.0

    @property
    def particle(self):
        '''docstring'''
        return self.__particle
    @particle.setter
    def particle(self, new_state_vector):
        '''docstring'''
        self.__particle = new_state_vector

    @property
    def weight(self):
        '''docstring'''
        return self.__weight
    @weight.setter
    def weight(self, new_weight):
        '''docstring'''
        self.__weight = new_weight


class BetaParticle(Particle):
    '''docstring'''

    def __init__(self, node_j, mean_radius_j, parent_state):
        Particle.__init__(self)

        self.parent = parent_state
        self.total_nodes = int(float(len(parent_state)) / 4.0)

        self.node_j = node_j
        self.neighbourhood_qj = []
        self.neighbour_dist_qj = []
        self.smeared_phases_qj = []

        self.x_j, self.y_j, self.f_j, self.r_j = self.parent[self.node_j::self.total_nodes]

        self.mean_radius = mean_radius_j # TODO: Change to self.r_j*3.0

    def get_neighbourhood_qj(self):
        '''doc string'''

        self.neighbourhood_qj = []
        self.neighbour_dist_qj = []

        for idx in range(self.total_nodes):

            xq_ = self.parent[idx]
            yq_ = self.parent[idx + self.total_nodes]
            dist = np.sqrt((xq_ - self.x_j)**2 + (yq_ - self.y_j)**2)

            if dist <= self.mean_radius:
                self.neighbourhood_qj.append(idx)
                self.neighbour_dist_qj.append(dist)


    def smear_fj_on_neighbours(self, **args):
        '''docstring'''

        self.get_neighbourhood_qj()

        prev_posterior_f_state = args["prev_posterior_f_state"]
        prev_counter_tau_state = args["prev_counter_tau_state"]
        lambda_ = args["lambda_"]

        self.smeared_phases_qj = []
        for idx_q in range(len(self.neighbourhood_qj)):

            node_q = self.neighbourhood_qj[idx_q]
            dist_jq = self.neighbour_dist_qj[idx_q]
            tau_q = prev_counter_tau_state[node_q]
            f_state_q = prev_posterior_f_state[node_q]

            lambda_q = lambda_** tau_q
            kernel_val = args["kernel_function"](dist_jq, self.f_j, self.r_j)

            smear_phase = (1.0 - lambda_q)*f_state_q + lambda_q*kernel_val
            self.smeared_phases_qj.append(smear_phase)


class AlphaParticle(Particle):
    '''docstring'''
    def __init__(self):
        Particle.__init__(self)
        self.pset_beta = 0
        self.node_j = 0.0
        self.mean_radius_j = 0.0 # TODO FIX THIS. CANT BE ONE NUMBER FORALL j
        self.BetaAlphaSet_j = None

    def generate_beta_pset(self, parents): #, number_of_beta_particles):
        '''docstring'''
        beta_s = [BetaParticle(self.node_j, self.mean_radius_j, state) for state in parents]
        self.BetaAlphaSet_j = ParticleSet(beta_s, **WEIGHTFUNCDICT_BETA)
        # return BetaAlphaSet

class ParticleSet(object):
    '''docstring'''
    def __init__(self, list_of_particle_objects, **WEIGHTFUNCDICT):

        self.__weights_set = 0.0
        self.__posterior_state = 0.0
        self.p_set = len(list_of_particle_objects)
        self.particles = list_of_particle_objects
        self.w_dict = WEIGHTFUNCDICT
        self.weights_set = (1.0 / self.p_set)*np.ones(self.p_set)

    def calc_weights_set(self):
        '''docstring'''
        new_weight_set = []
        for particle in self.particles:
            new_weight = self.w_dict["function"](particle, **self.w_dict["args"])
            new_weight_set.append(new_weight)
        raw_weights = np.asarray(new_weight_set).flatten()
        normalisation = 1.0/np.sum(raw_weights)
        return normalisation*raw_weights

    @property
    def weights_set(self):
        '''docstring'''
        self.__weights_set = np.asarray([particle.weight for particle in self.particles]).flatten()
        return self.__weights_set
    @weights_set.setter
    def weights_set(self, new_weights):
        '''docstring'''
        for idxp in range(self.p_set):
            self.particles[idxp].weight = new_weights[idxp]

    @property
    def posterior_state(self):
        '''docstring'''

        posterior_state=0.0
        weight_sum = np.sum(self.weights_set)

        if weight_sum != 1:
            print "Sum of all alpha weight set: "
            print "Normalising weights in posterior state calculation..."

        for idxp in range(self.p_set):
            posterior_state += self.particles[idxp].weight*self.particles[idxp].particle*(1.0/weight_sum)

        return posterior_state


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
        self.QubitGrid = Grid.__init__(list_of_nodes_positions=list_of_nodes_positions)
        self.dgrid = find_max_distance(self.list_of_nodes_positions)
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

            control_j = t_item[0]
            next_phys_msmt_j = t_item[1]
            self.ReceiveMsmt(control_j, next_phys_msmt_j)
            self.PropagateState(control_j)
            self.ComputeWeights(control_j)
            self.ResampleParticles(posterior_weights)

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
                                    size=self.number_of_nodes)

        sample_f = np.random.uniform(low=0.,
                                     high=np.pi, # INITIAL COND
                                     Size=self.number_of_nodes)

        sample_r = np.random.uniform(low=0.,
                                     high=self.dgrid*3.0, # INITIAL COND
                                     Size=self.number_of_nodes)

        sample_state = [sample_s]*2 + [sample_f, sample_r]
        alphaparticle.particle = np.asarray(sample_state).flatten()
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
            alpha_particle.mean_radius_j = self.AlphaSet.posterior_state[control_j] 
            # TODO: neighbourbood mean radius update

        self.update_alpha_dictionary(control_j, next_phys_msmt_j, prob_j)



    def sample_prob_from_msmts(self, control_j):
        '''docstring'''
        prob_p = self.QubitGrid.nodes[control_j].physcmsmtsum / self.QubitGrid.nodes[control_j].counter_tau*1.0
        forgetting_factor = self.INITIALDICT["LAMBDA"]**self.QubitGrid.nodes[control_j].counter_tau
        prob_q = self.QubitGrid.nodes[control_j].quasimsmtsum / self.QubitGrid.nodes[control_j].counter_beta*1.0
        prob_j = prob_p + forgetting_factor*prob_q
        return prob_j


    def update_alpha_dictionary(self, control_j, next_phys_msmt_j, prob_j):
        '''docstring'''
        LIKELIHOOD_ALPHA["msmt_dj"] = next_phys_msmt_j
        LIKELIHOOD_ALPHA["prob_j"] = prob_j

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
                                    size=self.number_of_nodes*2)

        sample_f = np.zeros(self.number_of_nodes)

        sample_r = np.random.normal(loc=INITIALDICT["MU_R"],
                                    scale=INITIALDICT["SIG2_R"],
                                    size=self.number_of_nodes*2)

        new_state_vector = [sample_s] + [sample_f] + [sample_r]

        alpha_particle.particle = np.asarray(new_state_vector).flatten()

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
            posterior_weights.append([alpha_particle.weight*beta_alpha_j_weights])

        normalisation = np.sum(np.asarray(posterior_weights).flatten())
        print "Posterior weights sum to=", normalisation

        normalised_posterior_weights = posterior_weights*(1.0/normalisation)
        return  normalised_posterior_weights # savedas posterior_weights[alphaindex][betaalphaindex]


    def update_alpha_map_via_born_rule(self, alpha_particle):
        '''doc string
        this function must always come after we havw computed alpha weights
        '''

        for alpha_particle in self.AlphaSet.particles:
            map_idx = self.number_of_nodes * 2 + alpha_particle.node_j
            parent_particle = alpha_particle.particle # get property
            parent_particle[map_idx] = self.sample_prob_from_msmts(alpha_particle.node_j)
            alpha_particle.particle = parent_particle # assign updated property


    def sample_radii(self, previous_length_scale, L_factor=None, Band_factor=None):
        '''docstring'''
        if L_factor is None:
            L_factor = self.L_factor 

        if Band_factor is None:
            Band_factor = 0.0

        sample = np.random.uniform(low=Band_factor, high=L_factor)
        return sample_radii


    def generate_beta_layer(self, alpha_particle):
        '''docstring'''

        len_idx = self.number_of_nodes * 3 + alpha_particle.node_j
        parent_alpha = alpha_particle.particle # get property

        list_of_parent_states = []
        for idx_beta in range(alpha_particle.pset_beta):

            parent_alpha[map_idx] = self.sample_radii(parent_alpha[map_idx])
            list_of_parent_states.append(parent_alpha)

        alpha_particle.generate_beta_pset(list_of_parent_states) #generates beta layer for each alpha

        for beta_particle_object in alpha_particle.BetaAlphaSet_j:
            self.generate_beta_neighbourhood(beta_particle_object) # compute smeared phases for each beta particle

        beta_alpha_j_weights = alpha_particle.BetaAlphaSet_j.calc_weights_set

        return beta_alpha_j_weights # these weights are normalised


    def generate_beta_neighbourhood(self, BetaParticle):
        '''docstring'''

        # BetaParticle.mean_radius = new_neighbourhood_L

        NEIGHBOURDICT = {"prev_posterior_f_state" : self.QubitGrid.get_all_nodes(["f_state"]), 
                        "prev_counter_tau_state" : self.QubitGrid.get_all_nodes(["counter_tau"]),
                        "lambda" : self.INITIALDICT["LAMBDA"]}

        BetaParticle.smear_fj_on_neighbours(**NEIGHBOURDICT)


#       ------------------------------------------------------------------------
#       SUPPORT FUNCTION 5: RESAMPLE AND BETA COLLAPSE
#       ------------------------------------------------------------------------

    def ResampleParticles(self, posterior_weights):
        '''docstring'''
        resampled_idx = np.arange(self.pset_alpha*self.pset_beta)

        if self.resample_thresh > self.effective_particle_size(posterior_weights):
            resampled_idx = self.resample_constant_pset_alpha(posterior_weights)

        new_alpha_subtrees = self.get_subtrees(resampled_idx)
        new_alpha_list = self.collapse_beta(new_alpha_subtrees, resampled_idx)

        self.AlphaSet.particles = new_alpha_list # garanteed to be pset_alpha with no second layer
        self.AlphaSet.weights_set = (1.0/self.pset_alpha)*np.ones(self.pset_alpha)

    def resample_constant_pset_alpha(self, posterior_weights):
        '''Returns indicies for particles picked after sampling from posterior'''
        # DO WEIGHTS NEED TO BE SORTED? (No, the CDF will be
        # monotnically increasing; and the shape of the CDF will determine
        # the frequency at which (x,y) are sampled if y is uniform )

        sufficient_sample = False
        num_of_samples = self.pset_alpha
        total_particles = len(posterior_weights)

        if total_particles != int(INITIALDICT["P_ALPHA"]*INITIALDICT["P_BETA"]):
            print "Total weights != P_alpha * P_beta"
            raise RuntimeError

        while sufficient_sample is False:

            num_of_samples *= 2 # TODO: exponentially growing search
            resampled_indices = self.resample_from_weights(posterior_weights, num_of_samples)
            unique_alphas = list(set([get_alpha_node_from_treeleaf(leafy, pset_beta) for leafy in resampled_indices]))

            if len(unique_alphas) == self.pset_alpha:
                sufficient_sample = True

        return resampled_indices

    def resample_from_weights(self, posterior_weights, number_of_samples):
        '''docstring'''
        total_particles = len(posterior_weights)
        cdf_weights = np.asarray([0] + [np.sum(posterior_weights[:idx+1]) for idx in range(total_particles)])
        pdf_uniform = np.random.random(size=number_of_samples)

        resampled_idx = []

        for u_0 in pdf_uniform:
            j = 0
            while u_0 > cdf_weights[j]:
                j += 1
                if j >= total_particles:
                    j = total_particles -1
                    # print('Break - max particle index reached during sampling')
                    break   # clip at max particle index, plus zero
            resampled_idx.append(j)

        return resampled_idx

    def get_subtrees(self, resampled_indices):
        '''docstring'''

        new_sub_trees = []

        resampled_indices.sort()
        alpha_index_0 = None
        strt_counter = 0
        end_counter = 0

        for idx in resampled_indices:

            alpha_index = get_alpha_node_from_treeleaf(idx, self.pset_beta)
            beta_alpha_idx = get_beta_node_from_treeleaf(idx, self.pset_beta)

            if alpha_index_0 == alpha_index:
                end_counter += 1

            elif alpha_index_0 != alpha_index:

                new_sub_trees.append([strt_counter, end_counter])

                alpha_index_0 = alpha_index
                strt_counter = end_counter
                end_counter += 1

        if end_counter == len(resampled_indices):
            end_counter += 1
            new_sub_trees.append([strt_counter, end_counter])

        return new_sub_trees


    def collapse_beta(self, new_sub_trees, resampled_indices):
        '''docstring'''

        state_update = 0.
        pset_beta = self.pset_beta

        new_alpha_particle_list = []
        for pairs in new_sub_trees:

            subtree = resampled_indices[pairs[0]:pairs[1]]
            leaf_count = float(len(subtree))
            normaliser = (1./leaf_count)

            alpha_node = get_alpha_node_from_treeleaf(pairs[0], pset_beta)
            beta_alpha_nodes = [get_beta_node_from_treeleaf(leafy, pset_beta) for leafy in subtree]

            for node in beta_alpha_nodes:
                beta_state = self.AlphaSet.particles[alpha_node].Beta_Alpha_j[node].particle
                beta_lengthscale = beta_state[self.AlphaSet.particles[alpha_node].node_j]
                len_scale_update += normaliser*beta_lengthscale

            parent = self.AlphaSet.particles[alpha_node].particle
            parent[self.AlphaSet.particles[alpha_node].node_j] = len_scale_update

            # Beta Layer Collapsed
            self.AlphaSet.particles[alpha_node].particle = parent
            self.AlphaSet.particles[alpha_node].BetaAlphaSet_j = None

            # New Alphas Stored
            new_alpha_particle.append(self.AlphaSet.particles[alpha_node])

        return new_alpha_particle_list

    def effective_particle_size(self, posterior_weights):
        '''docstring'''
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
        distances = [compute_dist(one_pair)for one_pair in distance_pairs]
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
