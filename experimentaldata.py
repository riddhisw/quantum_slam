import numpy as np
from  scipy.stats import binom as bnm



class RealData(object):

    def __init__(self, ion_bright_path2file, bayeseval_path2file, ion_bright_key):
        ''' Accesses output classifier data of prob of seeing a bright ion by Hempel et al. '''

        # HARDCODED
        self.high_grad_seq = set([4, 6, 11, 13, 15, 16, 18, 24, 25, 28, 29, 32, 35, 37, 39, 46, 49, 51, 57])

        # Data
        exp_data = np.loadtxt(bayeseval_path2file).transpose()
        self.bayesprimitives = exp_data[:, ::2]
        ions_bright = np.load(ion_bright_path2file)[ion_bright_key]
        self.primatives = ions_bright[:, ::2, :, 1]
        # ::2 - skips bb1 data
        # 0 - probability bright?
        # 1 - probability dark? (reverse definition)

        # Set truth
        self.pick_seq_qslam = list(self.high_grad_seq)[np.random.randint(low=0, high=len(self.high_grad_seq))]
        print "Set sequence:", self.pick_seq_qslam

        # Other parameters
        self.total_ions = self.primatives.shape[0]
        self.total_seq = self.primatives.shape[1]
        self.total_repetitions = self.primatives.shape[2]
        self.recommended_amplification = np.round(1.0 / np.max(self.primatives.flatten()))

        print "Recommended amplification for real data:",  self.recommended_amplification

    def get_real_data(self, node_j, amplification=1):
        '''Return a msmt from analysis of an experimental dataset
        node_j: postion index for ion
        p_bright: probability that the ion is bright according to classifier
        amplification: amplifies p_bright if data is close to extreme values'''

        pick_repetition = np.random.randint(low=0, high=self.total_repetitions)
        p_bright = self.primatives[node_j, self.pick_seq_qslam, pick_repetition]

        msmt = bnm.rvs(1, p_bright*amplification)

        return msmt

    def get_bayes_pbright(self, pick_seq=None):
        '''Return the experimentalist analysis for p-bright'''

        if pick_seq is None:
            pick_seq = self.pick_seq_qslam

        empirical_p_bright = self.bayesprimitives[:, pick_seq]
        empirical_p_var = np.var(self.primatives[:, pick_seq, :], axis=1)
        empirical_p_mean = np.mean(self.primatives[:, pick_seq, :], axis=1)
        return empirical_p_bright, empirical_p_var, empirical_p_mean




# Data feed links [HARDCODED]
PATH1 = '/home/riddhisw/Documents/SLAM_project/qslam/expt_data/'
PATH2 = '/home/riddhisw/Documents/SLAM_project/qslam/expt_data/'
BAYESEVAL = PATH1 + '20181003-153821BayesEval_probabilities.txt'
IONBRIGHT = PATH2 + 'ion_bright.npz'
IONBRIGHT_KEY = "ion_bright"

# Data generator
RealDataGenerator = RealData(IONBRIGHT, BAYESEVAL, IONBRIGHT_KEY)
