import numpy as np
from visualiserisk import Metric
import qslamr as qs
import copy 
from experimentaldata import RealData
from hardware import Node


class EmpiricalRisk(object):

    def __init__(self, GLOBALDICT):

        self.GLOBALDICT = GLOBALDICT

    def qslam_trial(self, measurements_controls_=None,
                         autocontrol_="ON",
                         var_thres_=1.0 ):
        
        qslamobj = qs.ParticleFilter(copy.deepcopy(self.GLOBALDICT), real_data=True)
        qslamobj.qslamr(max_num_iterations=self.GLOBALDICT["MODELDESIGN"]["MAX_NUM_ITERATIONS"],
                        measurements_controls=measurements_controls_,
                        autocontrol=autocontrol_,
                        var_thres=var_thres_)

        posterior_map = qslamobj.QubitGrid.get_all_nodes(["f_state"])
        posterior_corrs = qslamobj.QubitGrid.get_all_nodes(["r_state"])
        controlpath = qslamobj.QubitGrid.control_sequence

        return posterior_map, posterior_corrs, controlpath

    def naive_trial(self):

        RealDataObject = RealData()

        naiveobj = NaiveEstimatorExpt(RealDataObject,
                                        msmt_per_node=self.GLOBALDICT["MODELDESIGN"]["MSMTS_PER_NODE"],
                                        numofnodes=len(self.GLOBALDICT["GRIDDICT"]),
                                        max_num_iterations=self.GLOBALDICT["MODELDESIGN"]["MAX_NUM_ITERATIONS"])

        posterior_map, true_map_ = naiveobj.get_empirical_est()
        return posterior_map, true_map_

    def calculate_risk(self, number_of_trials=50):

        ssim_array = np.zeros((number_of_trials, 2))
        empr_array = np.zeros((number_of_trials, 2))

        for idx_run in range(number_of_trials):

            posterior_qslam_map = self.qslam_trial()[0]
            posterior_naive_map, true_map_ = self.naive_trial()
            posterior_map_list = [posterior_qslam_map, posterior_naive_map]

            for idx in range(2):

                residuals = posterior_map_list[idx] - true_map_
                ssim_array[idx_run, idx] = Metric.score_ssim(posterior_map_list[idx],
                                                        true_map_,
                                                        Cone=0.01, Ctwo=0.01)

                empr_array[idx_run, idx] = Metric.singlemap_rmse(residuals, axis=0)

        return np.mean(empr_array, axis=0), np.mean(ssim_array, axis=0)


class NaiveEstimatorExpt(object):
    
    def __init__(self,
                RealDataObject,
                msmt_per_node=1,
                numofnodes=25,
                max_num_iterations=None):

        self.msmt_per_node = msmt_per_node
        self.numofnodes = numofnodes
        self.max_num_iterations = max_num_iterations
        self.expt_data_generator = RealDataObject
        self.empirical_estimate = None

        self.__total_msmt_budget = self.msmt_per_node * self.max_num_iterations


    @property
    def total_msmt_budget(self):
        return self.msmt_per_node * self.max_num_iterations

    def get_empirical_truth(self):
        ''' Return phase map implied by the empirical mean of msmt data'''
        empirical_mean = self.expt_data_generator.get_empirical_mean()
        phasemap = Node.inverse_born(empirical_mean)
        return phasemap

    def get_empirical_est(self):

        phase_map = self.get_empirical_truth()

        if self.numofnodes <= self.max_num_iterations:

            if self.max_num_iterations / self.numofnodes == self.msmt_per_node:
                mask = np.ones(self.numofnodes, dtype=bool)

            if self.max_num_iterations / self.numofnodes != self.msmt_per_node:
                self.msmt_per_node = int(self.total_msmt_budget / self.numofnodes)
                mask = np.ones(self.numofnodes, dtype=bool)

        if self.numofnodes > self.max_num_iterations:

            randomly_choose = np.random.choice(self.numofnodes, self.max_num_iterations, replace=False)
            mask = np.zeros(self.numofnodes, dtype=bool) # Mask for hiding all values.
            mask[randomly_choose] = True


        node_labels = np.arange(self.numofnodes)
        
        
        ### INITIALISATION IMPORTANT!!!
        self.empirical_estimate = np.ones(self.numofnodes) * np.random.randint(low=0.0,
                                                                               high=np.pi,
                                                                               size=1)
        for idx_node in node_labels[mask]:

            noisy_single_shots = [self.expt_data_generator.get_real_data(idx_node) for idx_shot in range(self.msmt_per_node)]
            self.empirical_estimate[idx_node] = Node.inverse_born(np.mean(np.asarray(noisy_single_shots, dtype=float)))

        return self.empirical_estimate, phase_map

