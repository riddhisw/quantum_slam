'''
Created on Thu Apr 20 19:20:43 2017
@author: riddhisw

.. module:: name

    :synopsis: Computes expected value of Bayes Risk for qslam.

    Module Level Classes:
    ----------------------
        CreateQslamExpt : Optimises qslam filter parameters and uses 
            Bayes Risk metric for predictive power analysis.

.. moduleauthor:: Riddhi Gupta <riddhi.sw@gmail.com>
'''
import copy 
import qslamr as qs
import numpy as np
import os

H_PARAM = ['LAMBDA_1', 'LAMBDA_2', 'SIGMOID_VAR', 'QUANT_VAR']

from hardware import Node

class NaiveEstimator(object):

    def __init__(self, dims=25, typeofmap='Uniform', msmt_per_node=1, numofnodes=None):

        self.truth_generator = EngineeredTruth(dims=dims, typeofmap=typeofmap)
        self.msmt_per_node = msmt_per_node
        self.numofnodes = numofnodes

        if self.numofnodes is None:
            self.numofnodes = dims

        self.__total_msmt_budget = self.numofnodes * self.msmt_per_node
        self.empirical_estimate = None

    @property
    def total_msmt_budget(self):
        return self.numofnodes * self.msmt_per_node

    def get_empirical_est(self):

        phase_map = self.truth_generator.get_map()

        if len(phase_map) == self.numofnodes:
            mask = np.ones(self.numofnodes, dtype=bool)

        if len(phase_map) > self.numofnodes:

            randomly_choose = np.random.randint(low=0,
                                                high=len(phase_map),
                                                size=self.numofnodes)
            mask = np.zeros(len(phase_map), dtype=bool) # Mask for hiding all values.
            mask[randomly_choose] = True

        node_labels = np.arange(len(phase_map))
        self.empirical_estimate = np.zeros(len(phase_map))

        for idx_node in node_labels[mask]:
            single_shots = [ Node.quantiser(Node.born_rule(phase_map[idx_node])) for idx_shot in range(self.msmt_per_node)]
            self.empirical_estimate[idx_node] = Node.inverse_born(np.mean(np.asarray(single_shots, dtype=float)))

        maperrors =  self.empirical_estimate - phase_map
        
        return  maperrors

class EngineeredTruth(object):
    ''' Generates true maps for Bayes Risk analysis'''
    def __init__(self, dims=25, typeofmap='Uniform'):

        self.type = typeofmap
        self.dims = dims

    def get_map(self):

        if self.type == 'Uniform':

            truemap = np.ones(self.dims)*0.456*np.pi

        if self.type == 'Gaussian':

            mu_x = 2.0
            mu_y = 2.0
            scl = 0.8

            truemap = []
            sqrdims = int(np.sqrt(self.dims))
            for xidx in range(sqrdims):
                for yidx in range(sqrdims):
                    phase = 2.5*np.pi*(1.0 / (np.sqrt(2.0*np.pi*scl)))*np.exp(-((float(xidx) - mu_x)**2 + (float(yidx) - mu_y)**2)/ 2*scl)

                    if phase > np.pi:
                        phase = np.pi

                    if phase < 0.0:
                        phase = 0.0

                    truemap.append(phase)

            truemap = np.asarray(truemap)

        return truemap


class Bayes_Risk(object):
    ''' Stores Bayes Risk map for a scenario specified by (testcase, variation)

    Attributes:
    ----------
        bayes_params (`dtype`) : Parameters to intiate Bayes Risk class:
            max_it_BR (`int`) : Number of repetitions for a Bayes Risk calculation.
            num_randparams (`int`) : Number of random (sigma, R) sample pairs.
            space_size (`int`) : Exponent parameter to set orders of magnitude
                spanned by unknown noise variance parameters.
            loss_truncation (`int`) : Pre-determined threshold for number of
                lowest input values to return in modcule function,
                get_tuned_params(), in module common.
        doparallel (`Boolean`) : Enable parallelisation of Bayes Risk calculations [DEPRECIATED].
        lowest_pred_BR_pair (`float64`) : (sigma, R) pair with min Bayes Risk in state estimation.
        lowest_fore_BR_pair (`float64`) : (sigma, R) pair with min Bayes Risk in prediction.
        means_list (`float64`) : Helper calculation for Bayes Risk.
        skip_msmts (`int`) : Number of time-steps to skip between measurements.
            To receive measurement at every time-step, set skip_msmts=1.
        did_BR_Map (`Boolean`) : When True, indicates that a BR Map has been created.
        macro_truth (`float64`) : Matrix data container for set of true noise realisations,
            generated for the purposes of calculating the Bayes Risk metric for all
            (sigma, R) random samples.
        macro_prediction_errors (`float64`) : Matrix data container for set of state estimates.
        macro_forecastng_errors (`float64`) : Matrix data containter for set of forecasts.
        macro_hyperparams (`float64`) : Matrix data containter for random
            samples of (sigma, R).
    '''

    def __init__(self, truthtype='Uniform', **RISKPARAMS):
        '''Initiates a Bayes_Risk class instance. '''

        self.savetopath = RISKPARAMS["savetopath"]
        self.max_it_BR = RISKPARAMS["max_it_BR"]
        self.max_it_qslam = RISKPARAMS["max_it_qslam"]
        self.num_randparams = RISKPARAMS["num_randparams"]
        self.space_size  = RISKPARAMS["space_size"]
        self.loss_truncation = RISKPARAMS["loss_truncation"]

        self.truemap_generator = EngineeredTruth(typeofmap=truthtype)

        self.filename_br = None
        self.macro_true_fstate = None
        self.macro_predictions = None
        self.macro_residuals = None
        self.macro_hyperparams = None
        self.lowest_pred_BR_pair = None
        self.did_BR_Map = True
        self.means_list = None

        pass

class CreateQslamExpt(Bayes_Risk):
    '''docstring'''

    def __init__(self, truthtype='Uniform', **GLOBALDICT):

        self.GLOBALDICT = GLOBALDICT
        RISKPARAMS = self.GLOBALDICT["RISKPARAMS"]
        Bayes_Risk.__init__(self, truthtype=truthtype, **RISKPARAMS)
        self.qslamobj = None
        self.filename_br = self.GLOBALDICT["MODELDESIGN"]["ID"] + '_BR_Map'


    def loss(self, posterior_state, true_state_):
        '''Return squared error cost (objective function) between
        engineered truth and algorithm posterior state.

        Parameters:
        ----------
            posterior_state (`float64` | Numpy array | dims: N) :
                Posterior state estimates in vector form.
            true_state_ (`float64` | Numpy array | dims: N) :
                One realisation of (engineered) true state in vector form.
        Returns:
        -------
            residuals_errors : errors between algorithm
                output and engineered truths.
        '''
        residuals_errors = posterior_state - true_state_
        return residuals_errors

    def map_loss_trial(self, true_map_,
                       measurements_controls_=None,
                       autocontrol_="ON",
                       var_thres_=1.0, **SAMPLE_GLOBAL_MODEL):

        '''Return an error vector for map reconstruction from one trial of an algorithm.'''

        self.qslamobj = qs.ParticleFilter(**SAMPLE_GLOBAL_MODEL)
        self.qslamobj.QubitGrid.engineeredtruemap = true_map_

        self.qslamobj.qslamr(measurements_controls=measurements_controls_,
                             autocontrol=autocontrol_,
                             max_num_iterations=SAMPLE_GLOBAL_MODEL["MODELDESIGN"]["MAX_NUM_ITERATIONS"],
                             var_thres=var_thres_)

        posterior_map = self.qslamobj.QubitGrid.get_all_nodes(["f_state"])

        map_residuals = self.loss(posterior_map, true_map_)

        return posterior_map, map_residuals

    def rand_param(self, **SAMPLE_GLOBAL_MODEL):
        ''' Return a randomly sampled hyper-parameter vector. '''

        HYPERDICT = SAMPLE_GLOBAL_MODEL["HYPERDICT"]
        samples = [HYPERDICT["DIST"](space_size=self.space_size, **HYPERDICT["ARGS"][param]) for param in H_PARAM]
        return samples

    def modify_global_model(self, samples, **SAMPLE_GLOBAL_MODEL):

        SAMPLE_GLOBAL_MODEL["MODELDESIGN"]["LAMBDA_1"] = samples[0]
        SAMPLE_GLOBAL_MODEL["MODELDESIGN"]["LAMBDA_2"] = samples[1]
        SAMPLE_GLOBAL_MODEL["NOISEPARAMS"]["SIGMOID_APPROX_ERROR"]["SIGMA"] = samples[2]
        SAMPLE_GLOBAL_MODEL["NOISEPARAMS"]["QUANTISATION_UNCERTY"]["SIGMA"] = samples[3]

        return SAMPLE_GLOBAL_MODEL

    def one_bayes_trial(self, samples=None):
        ''' Return true realisations, state etimation errors and prediction errors
        over max_it_BR repetitions for one (sigma, R) pair. '''

        SAMPLE_GLOBAL_MODEL = copy.deepcopy(self.GLOBALDICT)

        if samples is None:
            samples = self.rand_param(**SAMPLE_GLOBAL_MODEL)

        SAMPLE_GLOBAL_MODEL = self.modify_global_model(samples, **SAMPLE_GLOBAL_MODEL)

        predictions = []
        map_errors = []
        true_maps = []

        for ind in xrange(self.max_it_BR):

            # true_map_ = 0.75 * np.pi * np.ones(len(SAMPLE_GLOBAL_MODEL["GRIDDICT"]))
            true_map_ = self.truemap_generator.get_map()
            posterior, errors = self.map_loss_trial(true_map_, **SAMPLE_GLOBAL_MODEL)

            true_maps.append(true_map_)
            predictions.append(posterior)
            map_errors.append(errors)
        
        return true_maps, predictions, map_errors, samples

    def naive_implementation(self, randomise='OFF'):
        ''' Return Bayes Risk analysis as a saved .npz file over max_it_BR
        repetitions of true dephasing noise and simulated datasets; for
        num_randparams number of random hyperparameters.

        Returns:
        -------
            Output .npz file containing all Bayes Risk data for analysis.
        '''
        fix_hyperparams = None
        self.macro_hyperparams = []
        self.macro_predictions = []
        self.macro_residuals = []
        self.macro_true_fstate = []

        # start_outer_multp = t.time()

        for ind in xrange(self.num_randparams):

            if randomise == 'OFF':
                fix_hyperparams = np.ones(4)
                fix_hyperparams[0] = self.GLOBALDICT["MODELDESIGN"]["LAMBDA_1"]
                fix_hyperparams[1] = self.GLOBALDICT["MODELDESIGN"]["LAMBDA_2"]
                fix_hyperparams[2] = self.GLOBALDICT["NOISEPARAMS"]["SIGMOID_APPROX_ERROR"]["SIGMA"]
                fix_hyperparams[3] = self.GLOBALDICT["NOISEPARAMS"]["QUANTISATION_UNCERTY"]["SIGMA"]

            full_bayes_map = self.one_bayes_trial(samples=fix_hyperparams)

            self.macro_true_fstate.append(full_bayes_map[0])
            self.macro_predictions.append(full_bayes_map[1])
            self.macro_residuals.append(full_bayes_map[2])
            self.macro_hyperparams.append(full_bayes_map[3])

            np.savez(os.path.join(self.savetopath, self.filename_br),
                     macro_true_fstate=self.macro_true_fstate,
                     macro_predictions=self.macro_predictions,
                     macro_residuals=self.macro_residuals,
                     macro_hyperparams=self.macro_hyperparams,
                     max_it_BR=self.max_it_BR,
                     num_randparams=self.num_randparams,
                     savetopath=self.savetopath)

        self.did_BR_Map = True



    # def get_tuned_params(self, max_forecast_loss):
    #     '''[Helper function for Bayes Risk mapping]'''
    #     self.means_lists_, self.lowest_pred_BR_pair, self.lowest_fore_BR_pair = get_tuned_params_(max_forecast_loss,
    #                                                                                               np.array(self.num_randparams),
    #                                                                                               np.array(self.macro_prediction_errors)
    # def set_tuned_params(self):
    #     '''[Helper function for Bayes Risk mapping]'''
    #     self.optimal_sigma = self.lowest_pred_BR_pair[0]
    #     self.optimal_R = self.lowest_pred_BR_pair[1]


