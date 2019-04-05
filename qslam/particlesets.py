'''
Created on Thu Apr 20 19:20:43 2017
@author: riddhisw

.. module:: name

    :synopsis: Defines particle objects for particle filtering. Distinguishes
        between Alpha and Beta particles used in qslam.

    Module Level Classes:
    --------------------
        Particle : Defines properties of any Particle instance, for Alpha and
            Beta particles.
        ParticleSet : Defines features to specify and control properties for a
            fixed set of Particle objects.
        Alpha : Initiates a single Alpha particle. Inherits from Particle.
        Beta : Initiates a single Beta particle. Inherits from Particle.

.. moduleauthor:: Riddhi Gupta <riddhi.sw@gmail.com>
'''

import numpy as np

###############################################################################
# PARTICLE STRUCTURE
###############################################################################

class Particle(object):
    '''Defines properties of any Particle instance, for Alpha and
            Beta particles.

    Properties:
    ----------
        particle (`float64` | dims: 4 * Grid.number_of_nodes):
            Extended, vectorised state variable consisting of stacked substates:
                x_state : Spatial x estimate of all qubits on grid in vectorised form.
                y_state : Spatial y estimate of all qubits on grid in vectorised form.
                f_state : Dephasing noise field estimate at each qubit on grid in vectorised form.
                r_state : Correlation length estimate at each qubit on grid in vectorised form.
            Initial Value: 0.0
            Setter Function: particle(new_state_vector).

        weight (`float64` | scalar ):
            Weight of the particle (normalised or un-normalised).
            Initial Value: 0.0
            Setter Function: weight(new_weight).

    '''
    def __init__(self):

        self.__particle = 0.0
        self.__weight = 0.0

    @property
    def particle(self):
        ''' Getter function. Extended, vectorised state variable consisting of stacked substates:
                x_state : Spatial x estimate of all qubits on grid in vectorised form.
                y_state : Spatial y estimate of all qubits on grid in vectorised form.
                f_state : Dephasing noise field estimate at each qubit on grid in vectorised form.
                r_state : Correlation length estimate at each qubit on grid in vectorised form.
        Dims: 4 * Grid.number_of_nodes (or number of qubits on a Grid).
        '''
        return self.__particle
    @particle.setter
    def particle(self, new_state_vector):
        ''' Setter function. Sets the value of extended, vectorised state variable consisting of
            stacked substates:
                x_state : Spatial x estimate of all qubits on grid in vectorised form.
                y_state : Spatial y estimate of all qubits on grid in vectorised form.
                f_state : Dephasing noise field estimate at each qubit on grid in vectorised form.
                r_state : Correlation length estimate at each qubit on grid in vectorised form.
        Dims: 4 * Grid.number_of_nodes (or number of qubits on a Grid).
        '''
        self.__particle = new_state_vector

    @property
    def weight(self):
        '''Getter function. Scalar weight of the particle (normalised or un-normalised).'''
        return self.__weight
    @weight.setter
    def weight(self, new_weight):
        '''Setter function. Scalar weight of the particle (normalised or un-normalised).'''
        self.__weight = new_weight

class ParticleSet(object):
    ''' Defines features to specify and control properties for a
            fixed set of Particle objects.

    Attributes:
    ----------
        p_set (`int` | scalar):
            Number of particle objects in attribute particles.
        particles | list_of_particle_objects (List of `Particle` class objects):
             List of particle objects which define a set of Particles to model
             a particular posterior distribution; such that the weights of the particles
             sum to 1.0.
        w_dict | WEIGHTFUNCDICT (Dictionary object):
             A dictionary object that defines the weight calculation for all particles
             in ParticleSet:
                w_dict["function"]: Calls function to return the weight of the particle.
                w_dict["args"]: Dictionary object of arguments for w_dict["function"].

    Properties:
    ----------
        weights_set (`float64` | dims: p_set ):
            An array of weights correponding to `Particle.weight` attribute for each
            `Particle` for p_set number of total particles. weights_set maybe
            normalised or un-normalised.
        Initial Value: 0.0
        Setter Function: weights_set(new_weights).

        posterior (`float64` | dims: Particle.particle.shape[0] ):
            Return the posterior mean state vector over a distribution of particles;
            computing the weighted average sum over Particle.particle states with
            weights given by the set of Particle.weight.
        Initial Value: 0.0.
        Setter Function: None.

    Methods:
    -------
        calc_weights_set : Return an array of normalised weights for a list of
            Particle objects.
    '''

    def __init__(self, list_of_particle_objects,
                 MAX_WEIGHT_CUTOFF=100000.0,
                 **WEIGHTFUNCDICT):
        ''' Initiates a ParticleSet instance.'''

        self.__weights_set = 0.0
        self.__posterior_state = 0.0
        self.p_set = len(list_of_particle_objects)
        self.particles = list_of_particle_objects
        self.w_dict = WEIGHTFUNCDICT
        self.weights_set = (1.0 / self.p_set)*np.ones(self.p_set)
        self.MAX_WEIGHT_CUTOFF = MAX_WEIGHT_CUTOFF


    def calc_weights_set(self):
        '''Return an array of normalised weights for a list of Particle objects.'''
        new_weight_set = []


        for particle in self.particles:
            new_weight = self.w_dict["function"](particle, **self.w_dict["args"])

            if new_weight > self.MAX_WEIGHT_CUTOFF: # avoid inf
                # print "Large weight reset"
                new_weight = float(self.MAX_WEIGHT_CUTOFF)

            new_weight_set.append(new_weight)

        raw_weights = np.asarray(new_weight_set).flatten()

        unnormalised_total = np.sum(raw_weights)

        if unnormalised_total == 0.0: # avoid zeros
            # print "Weights all zeros =", raw_weights
            normalisation = 1.0
            return normalisation*raw_weights

        normalisation = 1.0/unnormalised_total
        return normalisation*raw_weights

    @property
    def weights_set(self):
        '''Getter function. Returns Particle.weight attribute for all Particles in
            the set.'''
        self.__weights_set = np.asarray([particle.weight for particle in self.particles]).flatten()
        return self.__weights_set
    @weights_set.setter
    def weights_set(self, new_weights):
        '''Setter function. Sets new_weights as Particle.weight attribute for all Particles in
            the set.'''
        for idxp in range(self.p_set):
            self.particles[idxp].weight = new_weights[idxp]

    @property
    def posterior_state(self):
        ''' Getter function. Return the posterior mean state vector over a ParticleSet
            distribution of particles.
        '''
        posterior_state = 0.0
        weight_sum = np.sum(self.weights_set)
        for idxp in range(self.p_set):
            posterior_state += self.particles[idxp].weight*self.particles[idxp].particle*(1.0/weight_sum)
        return posterior_state

class AlphaParticle(Particle):
    '''Initiates a single Alpha particle. Inherits from Particle.

    Attributes:
    ----------
        pset_beta (`int` | scalar):
             Number of Beta particles (children) to initiate for Alpha particle (parent).
        node_j (`int` | scalar):
             Location of the node at which a physical measurement has taken place,
                expressed as an index of vectorised set of ordered qubit coordinates in Grid.nodes.
        SIG2_MEASR (`float` | scalar):
             Measurement noise covariance strength. Equivalent to uncertainty in the result of a
                non-linear measurement model prior to quantisation of measurement outcomes. Used in
                computing the likelihood function for Beta particles (children).
        BetaAlphaSet_j (`ParticleSet` class object initiated with a list of `BetaParticle` objects):
             A set of Beta Particle,  associated with the Alpha Particle as a common parent.
                Initiated as a `ParticleSet` class object using a list of `BetaParticle` instances.
                BetaAlphaSet_j is reset after resampling procedure in qslamr.

    Methods:
    -------
        generate_beta_pset :
            Set BetaAlphaSet_j with a Beta particle distribution for a common AlphaParticle
                parent, and for a physical measurement at node_j.
    '''
    def __init__(self):
        Particle.__init__(self)
        self.pset_beta = 0
        self.node_j = 0.0
        self.SIG2_MEASR = 0.0
        self.BetaAlphaSet_j = None

    def generate_beta_pset(self, parents, radii, **BETADICT):
        '''docstring'''


        beta_s = []
        for idx in range(len(parents)): # TODO: Use enumerate
            state = parents[idx]
            radius = radii[idx]
            beta_s.append(BetaParticle(self.node_j, state, radius))
        self.BetaAlphaSet_j = ParticleSet(beta_s, **BETADICT)


class BetaParticle(Particle):
    '''Initiates a single Beta particle. Inherits from Particle.

    Attributes:
    ----------
        parent (`float` | dims: 4*total_nodes):
             Copy of the original Alpha Particle parent state.
        node_j (`int` | scalar):
             Location of the node at which a physical measurement has taken place,
                expressed as an index of vectorised set of ordered qubit
                coordinates in parent Alpha state.
        particle (`float` | dims: 4*total_nodes):
             Beta Particle child state representing the distributions of
                correlation lengthscales at measurement location, node_j,
                for a state vector given by parent.
        mean_radius (`float` | scalar):
            Length defining the radius within which another qubit is considered
                to be in the neighbourhood of a qubit at node_j.
        total_nodes (`int` | scalar):
             Total number of qubit nodes represented by parent Alpha state.
        neighbourhood_qj (List of `int` variables | len: <= total_nodes):
             List of qubits that are neighbours of qubit at node_j, expressed
                as a list of indices for each qubit location as vectorised in the
                parent state.
        neighbour_dist_qj (List of `float` scalars | len == len(neighbourhood_qj)):
             List of pairwise distances between qubits in neighbourhood_qj and
                qubit at node_j.
        smeared_phases_qj (List of `float` scalars | len == len(neighbourhood_qj)):
            List of phase estimates for each node q in the neighbourhood of node_j,
                based on correlation length scale, r_state, and phase estimate, f_state, at
                node_j. Information at node_j is blurred by  `MODELDESIGN["kernel_function"]`
                accessed in smear_fj_on_neighbours().
        x_j (`float` | scalar):
            Spatial x coordinate estimate at node_j.
        y_j (`float` | scalar):
            Spatial y coordinate estimate at node_j.
        f_j (`float` | scalar):
            Dephasing noise field estimate at node_j, updated for a measurement
                outcome recieved at node_j.
        r_j (`float` | scalar):
            Correlation length estimate at node_j, prior to receiving a measurement
                outcome at node_j.
    Methods:
    -------
        get_neighbourhood_qj : Builds neighbour_dist_qj, the list of neighbouring qubits
            around node_j where the pairwise distance between node_j and a neighbour
            is less than mean_radius.

        smear_fj_on_neighbours : Builds smeared_phases_qj, the list of phase estimates for all
            neighbouring qubits around node_j, after a local measurement at node_j.
    '''

    def __init__(self, node_j, parent_state, radius):
        ''' Initiates a single Beta particle. Inherits from Particle. '''

        Particle.__init__(self)

        self.parent = parent_state
        # COMMENT: Initialised identically to parent
        self.particle = np.asarray(parent_state).flatten()
        self.total_nodes = int(float(len(parent_state)) / 4.0)
        self.node_j = node_j
        self.neighbourhood_qj = []
        self.neighbour_dist_qj = []
        self.smeared_phases_qj = []
        self.x_j, self.y_j, self.f_j, self.r_j = self.particle[self.node_j::self.total_nodes]
        self.mean_radius = radius # TODO: * 3.0

    def get_neighbourhood_qj(self):
        ''' Builds neighbour_dist_qj, the list of neighbouring qubits around node_j
            where the pairwise distance between node_j and a neighbour
            is less than mean_radius.
        '''

        self.neighbourhood_qj = []
        self.neighbour_dist_qj = []

        for idx in range(self.total_nodes):

            xq_ = self.particle[idx]
            yq_ = self.particle[idx + self.total_nodes]
            dist = np.sqrt((xq_ - self.x_j)**2 + (yq_ - self.y_j)**2)

            if dist <= self.mean_radius:

                if dist > 0.0:
                    self.neighbourhood_qj.append(idx)
                    self.neighbour_dist_qj.append(dist)


    def smear_fj_on_neighbours(self, **args):
        ''' Builds smeared_phases_qj, the list of phase estimates for all
            neighbouring qubits around node_j, after a local measurement at node_j.
        '''
        
        self.get_neighbourhood_qj()

        prev_posterior_f_state = args["prev_posterior_f_state"]
        prev_counter_tau_state = args["prev_counter_tau_state"]
        lambda_ = args["lambda_"]

        self.smeared_phases_qj = []
        for idx_q in range(len(self.neighbourhood_qj)):

            node_q = self.neighbourhood_qj[idx_q]
            dist_jq = self.neighbour_dist_qj[idx_q]
            tau_q = prev_counter_tau_state[node_q]

            if tau_q == 0:
                # TODO: Add functionality here for dealing with unmeasured qubits.
                pass
            f_state_q = prev_posterior_f_state[node_q]

            lambda_q = lambda_** tau_q
            kernel_val = args["kernel_function"](dist_jq, self.f_j, self.r_j)

            smear_phase = (1.0 - lambda_q)*f_state_q + lambda_q*kernel_val


            self.smeared_phases_qj.append(smear_phase)
