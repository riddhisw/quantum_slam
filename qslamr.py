'''
Created on Thu Apr 20 19:20:43 2017
@author: riddhisw

.. module:: name

    :synopsis: Core particle filter to implement SLAM framework using local
        measurements on a 2D arrangement of qubits to reconstruct spatio-temporal
        dephasing noise.

    Module Level Classes:
    ----------------------
        ParticleFilter : Conducts particle filtering under the qslamr framework.

.. moduleauthor:: Riddhi Gupta <riddhi.sw@gmail.com>
'''

import numpy as np
from itertools import combinations
from scipy.stats import mode

from hardware import Grid, PARTICLE_STATE
from hardware import Node # not initiated, just access static method
from control_action import controller

from particlesets import AlphaParticle, ParticleSet, BetaParticle # has no dictionary imports. 
from particleweightcalcs import ParticleLikelihoods as pl  # update_alpha_dictionary independent function

class ParticleFilter(Grid):
    ''' Conducts particle filtering under the qslamr framework. Inherits from
        class hardware.Grid.

    Attributes:
    ----------
        QubitGrid (`Grid` class object):
            `Grid` class object representing a 2D spatial arrangement of qubits,
            tracking  posterior state estimates, physical and quasi measurements.
        dgrid (`float` | scalar):
            Max spatial pairwise separation for any two qubits on the Grid.
        d_iterq (`float` | scalar):
            Min spatial pairwise separation for any two qubits on the Grid.
        R_MAX (`float` | scalar):
            Upper bound on maximal correlation lengths physically expected in the system.
        R_MIN (`float` | scalar):
            Lower bound on maximal correlation lengths physically expected in the system.
        MODELDESIGN (Dictionary object):
            Stores a set of free parameters and initial conditions.
        pset_alpha (`int` | scalar):
            Total number of of Alpha particles (conserved during re-sampling).
        pset_beta (`int` | scalar):
            Total number of of Beta particles for each Alpha parent.
        resample_thresh (`float` | scalar):
            Effective number of particles given variance of posterior weights.
            Can be used to introduce an adaptive re-sampling scheme.
            [NOT USED : Currently resampling at each iteration].
        measurements_controls (`float` | dims: 2):
             External input for the most recent control directive and measurement
             outcome for the qubit, denoted by locaton index, node_j.
        empty_alpha_particles = [AlphaParticle() for idx in range(self.MODELDESIGN["P_ALPHA"])]
        self.AlphaSet = ParticleSet(empty_alpha_particles, **WEIGHTFUNCDICT_ALPHA)

    Class Methods:
    -------------
        qslamr : Execute core numerical SLAM solver for qslamr module.
        measurement_operation : Return measurement outcome and control directive
            i.e. the location of  the qubit node that is measured.
        particlefilter : Return next iteration of particle filtering algorithm after processing
            an input measurement outcome and control directive.
        update_qubitgrid_via_quasimsmts : Return the posterior neighbourhood associated
            with a qubit measurement at node_j and update all neighbours of node_j
            with quasi-measurement information from the posterior state estimates at node_j.
        InitializeParticles : Intialise properties of QubitGrid and AlphaSet for t = 0 time step
            set_init_alphaparticle : Set initial properties of an Alpha particle.
            Helper function to InitializeParticles().
        ReceiveMsmt : Updates global qubit hardware variables and properties of Alpha
            particles for a new measurement outcome and control directive.
        PropagateState: Propagates all Alpha particle states according to (apriori) transition
            propability distributions for propagating states between measurements.
        ComputeAlphaWeights : Return weights for the distribution defined over Alpha particles.
        ComputePosteriorWeights : Return posterior weights for the joint distribution
            defined over both Alpha and Beta particles.
        generate_beta_layer : Return a set of Beta particle weights for a given Alpha parent.
            Helper function to ComputePosteriorWeights()
        sample_radii : Return a r_state sample from the prior distribution of r_states
            to generate a new Beta particle for a given Alpha parent.
            Helper function to generate_beta_layer().
        generate_beta_neighbourhood : Generate phase estimates over a neighbourhood
            for a candidate Beta particle and its Alpha parent. Helper function
            to generate_beta_layer().
        ResampleParticles : Return a new set of Alpha particles with uniform weights resampled
            according to the joint probability of both Alpha and Beta particles, subject to
            conserving Alpha particle number.
        collapse_beta: Return resampled Alpha parents with collapsed Beta layers. For each Alpha
            parent, store the mode and the spread of r_states in Beta-Particle layer as
            the posterior r_state at node_j. Helper function for ResampleParticles().
        effective_particle_size : Return effective particle number based on posterior weight
            variances for adapative resampling. [Currently not used].

    Static Methods:
    --------------
        resample_constant_pset_alpha (static method) : Return indicies for Alpha
            particles resampled from posterior Alpha-Beta weights, while conserving
            Alpha particle number. Helper function for ResampleParticles().
        calc_posterior_lengthscale (static method) : Return descriptive moments
            for the distribution of r_states in a Beta particle set for an Alpha parent,
            post resampling. Helper function for collapse_beta().
        compute_dist : Return Euclidean distance between two points.
            Helper function for get_distances().
        get_distances :  Return all pairwise distances between an arrangement of qubits.
            Helper function for ParticleFilter.find_max_distance() and
            ParticleFilter.find_min_distance.
        find_max_distance : Return the pair of positions with maximum pair-wise
            separation distance.
        find_min_distance : Return  the pair of positions with minimum pair-wise s
            eparation distance.
        get_alpha_node_from_treeleaf  (static method) : Return Alpha Particle index
            based on global index for flattened layers of Alpha-Beta particles.
            Helper function for collapse_beta().
        get_beta_node_from_treeleaf (static method) : Return Beta Particle index based
            on global index for flattened layers of Alpha-Beta particles.
            Helper function for collapse_beta().
        resample_from_weights (static method) : Return samples from the posterior
            distribution of states approximated by a discrete set of posterior normalised
            weights. Helper function for ParticleFilter.resample_constant_pset_alpha().
        set_uniform_prior_for_correlation (static method) : Return max qubit pairwise separation
            on hardware, and lower and upper bounds on uniform distribution of r_states as a prior.
        get_subtrees (static method) : Return a list of  pairs of (start, end) for a sub-tree
            where each sub-tree represents an Alpha particle, and its element
            leaves represent suriviving resampled Beta particles.
            Helper function for ResampleParticles.

    '''

    def __init__(self, **GLOBALDICT):

        self.GLOBALDICT = GLOBALDICT
        self.MODELDESIGN = self.GLOBALDICT["MODELDESIGN"]
        NOISEPARAMS = self.GLOBALDICT["NOISEPARAMS"]

        self.LikelihoodObj = pl(**NOISEPARAMS)
        self.measurements_controls = None

        self.PRIORDICT = self.GLOBALDICT["PRIORDICT"]

        poskeys = sorted(self.GLOBALDICT["GRIDDICT"].keys())
        posvals = [self.GLOBALDICT["GRIDDICT"][idx_key] for idx_key in poskeys]
        LAMBDA_1 = self.MODELDESIGN["LAMBDA_1"]
        SAMPLE_F = self.PRIORDICT["SAMPLE_F"]

        self.QubitGrid = Grid(LAMBDA_1=LAMBDA_1,
                              list_of_nodes_positions=posvals,
                              **SAMPLE_F)

        self.dgrid, self.diterq, self.R_MIN, self.R_MAX = self.set_uniform_prior_for_correlation()

        empty_alpha_particles = [AlphaParticle() for idx in range(self.MODELDESIGN["P_ALPHA"])]
        self.AlphaSet = ParticleSet(empty_alpha_particles,
                                    **self.LikelihoodObj.WEIGHTFUNCDICT_ALPHA)

    def set_uniform_prior_for_correlation(self):
        '''Return max qubit pairwise separation  on hardware, and lower and upper
        bounds on uniform distribution of r_states as a prior.

        Returns:
        -------
            dgrid (`float` | scalar):
                Max spatial pairwise separation for any two qubits on the Grid.
            d_iterq (`float` | scalar):
                Min spatial pairwise separation for any two qubits on the Grid.
            R_MAX (`float` | scalar):
                Upper bound on maximal correlation lengths physically expected
                in the system.
            R_MIN (`float` | scalar):
                Lower bound on maximal correlation lengths physically expected
                in the system.
        '''
        list_of_positions = self.QubitGrid.list_of_nodes_positions

        d_grid, _ = ParticleFilter.find_max_distance(list_of_positions)
        diterq, _ = ParticleFilter.find_min_distance(list_of_positions)

        R_MAX = d_grid * self.MODELDESIGN["MULTIPLER_R_MAX"]
        R_MIN = diterq * self.MODELDESIGN["MULTIPLER_R_MIN"]

        return d_grid, diterq, R_MIN, R_MAX

    def qslamr(self, measurements_controls=None, autocontrol="OFF",
               max_num_iterations=None,
               var_thres=1.0):
        ''' Execute core numerical SLAM solver for qslamr module.

        Parameters:
        ----------
            measurements_controls : A list containing a measurement set and the control
                directive of the location of the measured qubit. Each control directive
                is a single iteration of the algorithm.
            autocontrol : "OFF" / "ON" flag. specifies whether next qubit measured
                is specified as a user input or chosen by the algorithm
            max_num_iterations : Maximum number of iterations at which the algorithm terminates.
            var_thres :[NOT USED] Error variance threshold where if the variance
                of length scales is less than interqubit separation, then algorithm
                terminates.
        Returns:
        -------
        '''
        if max_num_iterations is None:
            max_num_iterations = self.MODELDESIGN["MAX_NUM_ITERATIONS"]

        self.InitializeParticles()

        run_variance = self.QubitGrid.get_all_nodes(["r_state_variance"])

        if autocontrol == "OFF" and self.measurements_controls is None:
            print "Auto-controller is off and no measurement control protocol is specified "
            raise RuntimeError

        if measurements_controls is not None:
            self.measurements_controls = measurements_controls
            PROTOCOL_ON = True

        if autocontrol == "ON":
            self.measurements_controls = [(0.0, 0.0)]
            PROTOCOL_ON = True

        # COMMENT: maximum no of msmts
        max_iter_condition = max(len(self.measurements_controls), max_num_iterations)

        # COMMENT: threshold error variance is within the min inter-qubit separation
        # stop_protocol = self.R_MIN**2 * var_thres * self.QubitGrid.number_of_nodes

        next_control_neighbourhood = range(0, self.QubitGrid.number_of_nodes)

        protocol_counter = 0
        while PROTOCOL_ON == True:

            msmt_control_pair = self.measurement_operation(autocontrol,
                                                           run_variance,
                                                           protocol_counter,
                                                           next_control_neighbourhood=next_control_neighbourhood)

            next_control_neighbourhood = self.particlefilter(msmt_control_pair)

            if protocol_counter == max_iter_condition - 1:
                print "PROTOCOL - SAFE END - Max number of measurements taken"
                PROTOCOL_ON = False

            run_variance = self.QubitGrid.get_all_nodes(["r_state_variance"])

            protocol_counter += 1

    def measurement_operation(self, autocontrol, listofcontrolparameters, protocol_counter,
                              next_control_neighbourhood=None):
        ''' Return measurement outcome and control directive i.e. the location of
            the qubit node that is measured.

        Parameters:
        ----------

        Returns:
        -------

        '''

        if autocontrol == "OFF":
            return self.measurements_controls[protocol_counter]

        elif autocontrol == "ON":

            node_j = controller(listofcontrolparameters,
                                next_control_neighbourhood,
                                number_of_nodes=1)[0]
            # TODO: adapt for arbiraty number of simultaneous msmts , number_of_nodes > 1
            msmtset_j = [self.QubitGrid.measure_node(node_j) for idx in range(self.MODELDESIGN["MSMTS_PER_NODE"])]

            return [msmtset_j, node_j]


    def particlefilter(self, msmt_control_pair):
        ''' Return next iteration of particle filtering algorithm after processing
            an input measurement outcome and control directive.

        Parameters:
        ----------

        Returns:
        -------
        '''

        msmtset_j = msmt_control_pair[0]
        control_j = msmt_control_pair[1]

        for alpha_particle in self.AlphaSet.particles:
            alpha_particle.pset_beta = self.MODELDESIGN["P_BETA"]
            alpha_particle.node_j = control_j

        ###### BEGIN MSMT LOOP / ESTIMATE LOCALLY
        for next_phys_msmt_j in msmtset_j:

            self.ReceiveMsmt(control_j, next_phys_msmt_j)
            self.PropagateState(control_j)
            self.ComputeAlphaWeights()

            # self.QubitGrid.state_vector = self.AlphaSet.posterior_state*1.0 # ?????
            # Dont update the posterior state as the full posterior
            # has not been computed.
        ###### END  MSMT LOOP / ESTIMATE LOCALLY

        ###### SHARE WITH NEIGHBOURHOOD / SMOOTHEN GLOBALLY
        posterior_weights = self.ComputePosteriorWeights(control_j,
                                                         **self.LikelihoodObj.WEIGHTFUNCDICT_BETA)
        self.ResampleParticles(posterior_weights) # no QubitGrid update

        # COMMENT: Update node j neighbourhood and map estimate
        posterior_state = self.AlphaSet.posterior_state
        self.QubitGrid.state_vector = posterior_state*1.0

        # COMMENT: Sprinkle quasi msmts / share info with neighbours.
        next_control_neighbourhood = self.update_qubitgrid_via_quasimsmts(control_j,
                                                                          posterior_state)
        return next_control_neighbourhood # neighbourhood of next control action.


#       ------------------------------------------------------------------------
#       SUPPORT FUNCTION 1: INITIALISE AND SAMPLE AT t = 0
#       ------------------------------------------------------------------------

    def InitializeParticles(self):
        '''Intialise properties of QubitGrid and AlphaSet for t = 0 time step.
        Note f_state cannot be set, merely read from QubitGrid.
        '''
        for alphaparticle in self.AlphaSet.particles:
            self.set_init_alphaparticle(alphaparticle)

        self.QubitGrid.state_vector = self.AlphaSet.posterior_state


    def set_init_alphaparticle(self, alphaparticle):
        ''' Set initial properties of an Alpha particle.
            Helper function to InitializeParticles().

        Parameters:
        ----------

        Returns:
        -------
        '''

        self.PRIORDICT["SAMPLE_R"]["ARGS"]["R_MIN"] = self.R_MIN
        self.PRIORDICT["SAMPLE_R"]["ARGS"]["R_MAX"] = self.R_MAX
        size = self.QubitGrid.number_of_nodes

        substate_list = []
        for idx_key in ["SAMPLE_X", "SAMPLE_Y", "SAMPLE_F", "SAMPLE_R"]: # f_state cannot be set, merely read from QubitGrid

            if idx_key == "SAMPLE_F":
                samples = self.QubitGrid.get_all_nodes(["f_state"]) # Currently  all alpha particles have the same prior ie. same f_state values. 

            if idx_key != "SAMPLE_F":

                self.PRIORDICT[idx_key]["ARGS"]["SIZE"] = size
                samples = self.PRIORDICT[idx_key]["FUNCTION"](**self.PRIORDICT[idx_key]["ARGS"])

                if idx_key == "SAMPLE_X":
                    samples += self.QubitGrid.get_all_nodes(["x_state"])

                if idx_key == "SAMPLE_Y":
                    samples += self.QubitGrid.get_all_nodes(["y_state"])

            substate_list.append(samples)

        alphaparticle.particle = np.asarray(substate_list).flatten()
        alphaparticle.pset_beta = self.MODELDESIGN["P_BETA"]

#       ------------------------------------------------------------------------
#       SUPPORT FUNCTION 2: RECEIVE MSMT
#       ------------------------------------------------------------------------

    def ReceiveMsmt(self, control_j, next_phys_msmt_j):
        ''' Updates global qubit hardware variables and properties of Alpha
            particles for a new measurement outcome and control directive.

        Parameters:
        ----------

        Returns:
        -------
        '''

        self.QubitGrid.nodes[control_j].physcmsmtsum = next_phys_msmt_j
        prob_j = self.QubitGrid.nodes[control_j].sample_prob_from_msmts()
        pl.update_alpha_dictionary(next_phys_msmt_j,
                                   prob_j,
                                   **self.LikelihoodObj.LIKELIHOOD_ALPHA)

#       ------------------------------------------------------------------------
#       SUPPORT FUNCTION 3: PROPAGATE (ALPHA) STATES
#       ------------------------------------------------------------------------

    def PropagateState(self, control_j):
        '''Propagates all Alpha particle states according to (apriori) transition
        propability distributions for propagating states between measurements.

        Parameters:
        ----------

        Returns:
        -------

        '''
        for alpha_particle in self.AlphaSet.particles:
            self.sample_from_transition_dist(alpha_particle, control_j)


    def sample_from_transition_dist(self, alpha_particle, control_j):
        ''' Helper function for PropagateState(). Under identity dynamics (time invariance)
        this function does nothing in qslamr.

        Parameters:
        ----------

        Returns:
        -------
        '''
        # Implements time invariant states
        # Currently implements identity (do nothing).
        # TODO: Placehodler for time-invariance


#       ------------------------------------------------------------------------
#       SUPPORT FUNCTION 4: COMPUTE ALPHA WEIGHTS; GENERATE BETA WEIGHTS
#       ------------------------------------------------------------------------
    def ComputeAlphaWeights(self):
        ''' Update weights for the distribution defined over Alpha particles.

        Parameters:
        ----------

        Returns:
        -------
        '''
        new_alpha_weights = self.AlphaSet.calc_weights_set() # Normlaised
        self.AlphaSet.weights_set = new_alpha_weights

    def ComputePosteriorWeights(self, control_j, **BETADICT):
        ''' Return posterior weights for the joint distribution defined over both
        Alpha and Beta particles.

        Parameters:
        ----------

        Returns:
        -------
        '''
        f_state_index = 2*self.QubitGrid.number_of_nodes + control_j

        posterior_weights = []

        for idx_alpha in range(self.MODELDESIGN["P_ALPHA"]):

            alpha_particle = self.AlphaSet.particles[idx_alpha]
            alpha_particle.particle[f_state_index] = self.QubitGrid.nodes[control_j].f_state


            beta_alpha_j_weights = self.generate_beta_layer(alpha_particle, **BETADICT)

            posterior_weights.append(alpha_particle.weight*beta_alpha_j_weights)

        posterior_weights = np.asarray(posterior_weights).flatten()
        normalisation = np.sum(posterior_weights)

        if normalisation == 0.0:
            print "Zero value normalisation in ComputePosteriorWeights()"
            print posterior_weights
            normalisation = 1.0 # All weights are zero.
            # TODO Add a way to restart the syetem.

        normalised_posterior_weights = posterior_weights*(1.0/normalisation)

        if np.any(np.isnan(normalised_posterior_weights)):
            print "Invalid Nan values encountered in ComputePosteriorWeights()"
            print normalised_posterior_weights
            raise RuntimeError

        return  normalised_posterior_weights


    def generate_beta_layer(self, alpha_particle, **BETADICT):
        ''' Return a set of Beta particle weights for a given Alpha parent. Helper
            function to ComputePosteriorWeights().
        Parameters:
        ----------

        Returns:
        -------
        '''
        len_idx = self.QubitGrid.number_of_nodes*3 + alpha_particle.node_j
        parent_alpha = alpha_particle.particle.copy()
        new_beta_state = parent_alpha * 1.0 # get property

        list_of_parent_states = []
        list_of_length_samples = []

        for idx_beta in range(alpha_particle.pset_beta):
            new_length_sample = self.sample_radii()
            new_beta_state[len_idx] = new_length_sample*1.0
            list_of_parent_states.append(new_beta_state.copy())
            list_of_length_samples.append(new_length_sample)

        alpha_particle.generate_beta_pset(list_of_parent_states,
                                          list_of_length_samples,
                                          **BETADICT)
        for beta_particle_object in alpha_particle.BetaAlphaSet_j.particles:

            self.generate_beta_neighbourhood(beta_particle_object)

        beta_alpha_j_weights = alpha_particle.BetaAlphaSet_j.calc_weights_set()

        return beta_alpha_j_weights


    def sample_radii(self, previous_length_scale=0.0):
        ''' Return a r_state sample from the prior distribution of r_states to
        generate a new Beta particle for a given Alpha parent.
        Helper function to generate_beta_layer().

        Parameters:
        ----------

        Returns:
        -------
        '''

        if previous_length_scale < 0:
            print "Previous length scale is less than zero:", previous_length_scale
            raise RuntimeError
        lower_bound = (previous_length_scale + self.R_MIN)*0.1 + 0.9*self.R_MIN
        sample = np.random.uniform(low=lower_bound, high=self.R_MAX)
        return sample


    def generate_beta_neighbourhood(self, BetaParticle):
        ''' Generate phase estimates over a neighbourhood for input Beta particle
            and its Alpha parent. Helper function to generate_beta_layer().
         '''

        NEIGHBOURDICT = {"prev_posterior_f_state" : self.QubitGrid.get_all_nodes(["f_state"]),
                         "prev_counter_tau_state" : self.QubitGrid.get_all_nodes(["counter_tau"]),
                         "lambda_" : self.MODELDESIGN["LAMBDA_2"],
                         "kernel_function": self.MODELDESIGN["kernel_function"]}

        BetaParticle.smear_fj_on_neighbours(**NEIGHBOURDICT)

#       ------------------------------------------------------------------------
#       SUPPORT FUNCTION 5: RESAMPLE AND BETA COLLAPSE
#       ------------------------------------------------------------------------

    def ResampleParticles(self, posterior_weights):
        '''Return a new set of Alpha particles with uniform weights resampled
        according to the joint probability of both Alpha and Beta particles, subject to
        conserving Alpha particle number.

        Parameters:
        ----------

        Returns:
        -------
        '''

        resampled_idx = ParticleFilter.resample_constant_pset_alpha(posterior_weights,
                                                                    self.MODELDESIGN["P_ALPHA"],
                                                                    self.MODELDESIGN["P_BETA"])

        new_alpha_subtrees = self.get_subtrees(resampled_idx, self.MODELDESIGN["P_BETA"])

        new_alpha_list = self.collapse_beta(new_alpha_subtrees, resampled_idx)

        self.AlphaSet.particles = new_alpha_list # this a list of particle objects

        # COMMENT: Resampled, uniform weights
        uniformprob = (1.0/self.MODELDESIGN["P_ALPHA"])
        self.AlphaSet.weights_set = uniformprob*np.ones(self.MODELDESIGN["P_ALPHA"])

    def collapse_beta(self, subtree_list, resampled_indices):
        ''' Return resampled Alpha parents with collapsed Beta layers. For each Alpha
        parent, store the mode and the spread of r_states in Beta-Particle layer as
        the posterior r_state at node_j.  Helper function for ResampleParticles().

        Parameters:
        ----------

        Returns:
        -------
        '''
        state_update = 0.
        new_alpha_particle_list = []

        # for estimating variange at node accross all Alphas.
        normaliser = (1./float(len(subtree_list)))
        uncertainity_at_j = 0.0

        for subtree in subtree_list:

            leaves_of_subtree = resampled_indices[subtree[0]:subtree[1]]

            leaf_count = float(len(leaves_of_subtree))

            if leaf_count != 0: # excludees the [0,0] subtree

                # COMMENT: Get indicies and reset controlparam
                alpha_node = ParticleFilter.get_alpha_node_from_treeleaf(leaves_of_subtree[0], self.MODELDESIGN["P_BETA"])
                beta_alpha_nodes = [ParticleFilter.get_beta_node_from_treeleaf(leafy, self.MODELDESIGN["P_BETA"]) for leafy in leaves_of_subtree]
                r_est_index = self.QubitGrid.number_of_nodes*3 + self.AlphaSet.particles[alpha_node].node_j
                r_est_subtree_list = []

                for node in beta_alpha_nodes:
                    beta_state = self.AlphaSet.particles[alpha_node].BetaAlphaSet_j.particles[node].particle.copy()
                    beta_lengthscale = beta_state[r_est_index]*1.0
                    if np.isnan(beta_lengthscale):
                        raise RuntimeError
                    r_est_subtree_list.append(beta_lengthscale)

                # COMMENT: new posterior for alpha lenthscales based on mode
                r_est_subtree, r_est_subtree_variance = ParticleFilter.calc_posterior_lengthscale(np.asarray(r_est_subtree_list))
                parent = self.AlphaSet.particles[alpha_node].particle.copy()*1.0
                parent[r_est_index] = r_est_subtree

                if np.any(np.isnan(parent)):
                    print "A resampled parent particle has an invalid value."
                    raise RuntimeError

                # Collapse Beta Layer
                self.AlphaSet.particles[alpha_node].particle = parent*1.0

                # COMMENT: The line of code below resets beta layers for an alpha
                # parent. This code only works if the alpha_nodes are unique. For 
                # alpha parents that are not unique, it will wipe out the BetaLayer
                # and the loop over sub-trees will fail if the parent is repeated.
                # For re-sampled parents, we will use null_beta_layer() function
                # to kill the beta layer for an Alpha parent.
                # ERROR GENERATING CODE:
                # self.AlphaSet.particles[alpha_node].BetaAlphaSet_j = None

                # New Alphas Stored, control parameters updated
                uncertainity_at_j += r_est_subtree_variance * normaliser
                new_alpha_particle_list.append(self.AlphaSet.particles[alpha_node])

        self.QubitGrid.nodes[self.AlphaSet.particles[alpha_node].node_j].r_state_variance = uncertainity_at_j

        number_of_new_alphas = len(new_alpha_particle_list)

        if  number_of_new_alphas < self.MODELDESIGN["P_ALPHA"]:

            alpha_sample_size = self.MODELDESIGN["P_ALPHA"] - number_of_new_alphas
            alpha_sampling_vector = np.random.randint(low=0, high=number_of_new_alphas, size=alpha_sample_size)
            extended_alpha_list  = new_alpha_particle_list + [new_alpha_particle_list[idx_] for idx_ in alpha_sampling_vector] 
            self.null_beta_layer(extended_alpha_list)

            return extended_alpha_list

        self.null_beta_layer(new_alpha_particle_list)

        return new_alpha_particle_list

    def null_beta_layer(self, list_of_alpha_particles):
        ''' Strips input list of Alpha particles of its individual Beta layers. Helpfer function to self.collapse_beta()'''

        for alpha_idx in list_of_alpha_particles:
            alpha_idx.BetaAlphaSet_j = None

    def effective_particle_size(self, posterior_weights):
        '''Return effective particle number based on posterior weight variances
        for adapative resampling. [Currently not used].

        Parameters:
        ----------

        Returns:
        -------
        '''

        p_size = 1.0/ np.sum(posterior_weights**2)
        return p_size

#       ------------------------------------------------------------------------
#       SUPPORT FUNCTIONS 6: SMEARING ACTION VIA QUASI MEASUREMENTS
#       ------------------------------------------------------------------------

    def update_qubitgrid_via_quasimsmts(self, control_j, posterior_state):
        '''Return the posterior neighbourhood associated with a qubit measurement
            at node_j and update all neighbours of node_j with quasi-measurement
            information from the posterior state estimates at node_j.

        Parameters:
        ----------

        Returns:
        -------

        Warning -- this function should only be applied after lengthscales have
            been discovered (alpha, beta_alpha) particles carried over or are
            resampled with sufficient alpha diversity; weights are set to uniform.
        '''

        posterior_radius = posterior_state[self.QubitGrid.number_of_nodes*3 + control_j]
        posterior_beta_particle = BetaParticle(control_j, posterior_state, posterior_radius)
        self.generate_beta_neighbourhood(posterior_beta_particle)

        for idx in range(len(posterior_beta_particle.neighbourhood_qj)):

            neighbour_q = posterior_beta_particle.neighbourhood_qj[idx]
            quasi_phase_q = posterior_beta_particle.smeared_phases_qj[idx]

            if quasi_phase_q >= 0.0 and quasi_phase_q <= np.pi:
                born_prob_q = Node.born_rule(quasi_phase_q)
                quasi_msmt = np.random.binomial(1, born_prob_q)
                self.QubitGrid.nodes[neighbour_q].quasimsmtsum = quasi_msmt

            elif quasi_phase_q < 0.0 or quasi_phase_q > np.pi:
                print "quasi-phase posterior at q=", neighbour_q
                print "...was invalid, q_phase ", quasi_phase_q
                print "... no quasi_msmts were added."

        return posterior_beta_particle.neighbourhood_qj

#       ------------------------------------------------------------------------
#       SUPPORT FUNCTIONS 7: STATIC METHODS
#       ------------------------------------------------------------------------

    @staticmethod
    def calc_posterior_lengthscale(r_lengthscales_array):
        '''Return descriptive moments for the distribution of r_states in a
        Beta particle set for an Alpha parent, post resampling.
        Helper function for collapse_beta().

        Parameters:
        ----------

        Returns:
        -------
        '''
        r_posterior_post_beta_collapse, r_posterior_variance = ParticleFilter.calc_skew(r_lengthscales_array)
        return r_posterior_post_beta_collapse, r_posterior_variance

    @staticmethod
    def calc_skew(r_lengthscales_array):
        '''Helper function for  calc_posterior_lengthscale().

        Parameters:
        ----------

        Returns:
        -------
        '''
        totalcounts = len(r_lengthscales_array)
        mean_ = np.mean(r_lengthscales_array)
        mode_, counts = mode(r_lengthscales_array)
        median_ = np.sort(r_lengthscales_array)[int(totalcounts/2) - 1]

        variance = np.var(r_lengthscales_array)

        if mean_ < mode_ and counts > 1:
            return mode_, variance
        if mean_ > mode_ and counts > 1:
            return mode_, variance
        return mean_, variance


    @staticmethod
    def compute_dist(one_pair):
        '''Return Euclidean distance between two points.
        Helper function for ParticleFilter.get_distances().'''
        xval, yval = one_pair
        return np.sqrt((xval[0] - yval[0])**2 + (xval[1] - yval[1])**2)

    @staticmethod
    def get_distances(list_of_positions):
        '''Return all pairwise distances between an arrangement of qubits.
        Helper function for ParticleFilter.find_max_distance() and
        ParticleFilter.find_min_distance.
        '''
        distance_pairs = [a_pair for a_pair in combinations(list_of_positions, 2)]
        distances = [ParticleFilter.compute_dist(one_pair)for one_pair in distance_pairs]
        return distances, distance_pairs

    @staticmethod
    def find_max_distance(list_of_positions):
        '''Return the pair of positions with maximum pair-wise separation distance. '''
        distances, distance_pairs = ParticleFilter.get_distances(list_of_positions)
        return max(distances), distance_pairs[np.argmax(distances)]

    @staticmethod
    def find_min_distance(list_of_positions):
        '''Return  the pair of positions with minimum pair-wise separation distance. '''
        distances, distance_pairs = ParticleFilter.get_distances(list_of_positions)
        return min(distances), distance_pairs[np.argmin(distances)]

    @staticmethod
    def get_alpha_node_from_treeleaf(leaf_index, pset_beta):
        '''Return Alpha Particle index based on global index for flattened layers
        of Alpha-Beta particles. Helper function for collapse_beta().
        Parameters:
        ----------

        Returns:
        -------
        '''
        alpha_node = int(leaf_index//float(pset_beta))
        return alpha_node

    @staticmethod
    def get_beta_node_from_treeleaf(leaf_index, pset_beta):
        '''Return Beta Particle index based on global index for flattened layers
        of Alpha-Beta particles. Helper function for collapse_beta().
        Parameters:
        ----------

        Returns:
        -------
        '''
        beta_node = int(leaf_index - int(leaf_index//float(pset_beta))*pset_beta)
        return beta_node

    @staticmethod
    def resample_from_weights(posterior_weights, number_of_samples):
        ''' Return samples from the posterior distribution of states approximated
        by a discrete set of posterior normalised weights.
        Helper function for static method ParticleFilter.resample_constant_pset_alpha().
        Parameters:
        ----------

        Returns:
        -------
        '''
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
                    break
            resampled_idx.append(j-1)
        return resampled_idx

    @staticmethod
    def resample_constant_pset_alpha(posterior_weights, pset_alpha, pset_beta):
        ''' Return indicies for Alpha particles resampled from posterior Alpha-Beta weights, while
        conserving Alpha particle number. Helper function for ResampleParticles().
        Parameters:
        ----------

        Returns:
        -------
        '''
        sufficient_sample = False
        num_of_samples = pset_alpha
        total_particles = len(posterior_weights)

        resampled_indices = ParticleFilter.resample_from_weights(posterior_weights,
                                                                 num_of_samples)
        return resampled_indices

    @staticmethod
    def get_subtrees(resampled_indices, pset_beta):
        '''
        Return a list of  pairs of (start, end) for a sub-tree
        where each sub-tree represents an Alpha particle, and its element
        leaves represent suriviving resampled Beta particles.
        Helper function for ResampleParticles.
        Parameters:
        ----------

        Returns:
        -------
        '''
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
            new_sub_trees.append([strt_counter, end_counter])

        return new_sub_trees
