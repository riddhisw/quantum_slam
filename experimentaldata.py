import numpy as np
from  scipy.stats import binom as bnm



class RealData(object):

    def __init__(self, ion_bright_path2file, bayeseval_path2file, ion_bright_key, pA=0):
        ''' Accesses output classifier data of prob of seeing a bright ion by Hempel et al. '''

        pick_classifer_prob = 1
        # 0 - probability bright?
        # 1 - probability dark? (reverse definition)

        # RB DATA: HARDCODED
#        self.high_grad_seq = set([4, 6, 11, 13, 15, 16, 18, 24, 25, 28, 29, 32, 35, 37, 39, 46, 49, 51, 57])
#        exp_data = np.loadtxt(bayeseval_path2file).transpose()
#        ions_bright = np.load(ion_bright_path2file)[ion_bright_key]
#        self.bayesprimitives = exp_data[:, ::2]
#        self.primatives = ions_bright[:, ::2, :, pick_classifer_prob] # ::2 - skips bb1 data
#        self.pick_seq_qslam = list(self.high_grad_seq)[np.random.randint(low=0, high=len(self.high_grad_seq))]
#       print "Set sequence:", self.pick_seq_qslam


        # MODIFIED RAMSEY DATA HARDCODED
        # self.high_grad_seq = None
        # self.pick_seq_qslam = pA# # pA or pB experiment, pA data looks better
        # self.primatives = np.load(ion_bright_path2file)[ion_bright_key][:, :, :, pick_classifer_prob]
        # self.bayesprimitives = None

        # # SIMPLE RAMSEY DATA HARDCODED
        self.high_grad_seq = None # static or scanned experiment
        self.pick_seq_qslam = 7 # set a ramsey time 
        # Set ramsey for static expt as == 40ms; scanned expt s.t. < pi / 2 )
        self.primatives = np.load(ion_bright_path2file)[ion_bright_key][:, :, :, pick_classifer_prob]
        self.bayesprimitives = None

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
        '''Return the experimentalist analysis for p-bright for experiment using 
        Bayes and empirical statistcs'''

        if pick_seq is None:
            pick_seq = self.pick_seq_qslam
        bayes_p_bright = None
        if self.bayesprimitives is not None:
            bayes_p_bright = self.bayesprimitives[:, pick_seq]
        empirical_p_var = np.var(self.primatives[:, pick_seq, :], axis=1)
        empirical_p_mean = np.mean(self.primatives[:, pick_seq, :], axis=1)

        return bayes_p_bright, empirical_p_var, empirical_p_mean




# # Data feed links [HARDCODED]
PATH2 = '/home/riddhisw/Documents/SLAM_project/qslam/expt_data/'
IONBRIGHT_KEY = "ion_bright"

# ------------------------------------------------------------- RB Seqs expts---
# # RB:
# BAYESEVAL = PATH2 + '20181003-153821BayesEval_probabilities.txt'
# IONBRIGHT = PATH2 + 'ion_bright.npz'

# ------------------------------------------------------------- pA & pB expts---
# # MODIFIED RAMSEY:
# IONBRIGHT = PATH2 + 'Riddhi_SLAM/20181008-210159ion_bright_matrix.npz'
# BAYESEVAL = ''

# Data generator for standard / pA experiments
# pA = 0
# RealDataGenerator = RealData(IONBRIGHT, BAYESEVAL, IONBRIGHT_KEY)

# Data generator for standard for pB experiments
# pA = 1
# RealDataGenerator = RealData(IONBRIGHT, BAYESEVAL, IONBRIGHT_KEY, pA=pA)

# ------------------------------------------------------------- Simple ramsey ---
# # SIMPLE RAMSEY:

# Expt 1: SCANNED
# IONBRIGHT = PATH2 + 'Claire_ramseydata/20181010-114733ion_bright_matrix.npz'

# Expt 2: STATIC
IONBRIGHT = PATH2 + 'Claire_ramseydata/20181010-115434ion_bright_matrix.npz'

BAYESEVAL = ''
pA = 0
RealDataGenerator = RealData(IONBRIGHT, BAYESEVAL, IONBRIGHT_KEY)

print "Dataset chosen:"
print "IONBRIGHT:", IONBRIGHT
print "BAYESEVAL:", BAYESEVAL
print "pA:", pA