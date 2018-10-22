import numpy as np
from  scipy.stats import binom as bnm



# # DATA TYPE [HARDCODED]
PATH2 = '/home/riddhisw/Documents/SLAM_project/qslam/expt_data/'
KEY_1 = "ion_bright" # take a biased coin flip using class probabilities
KEY_2 = "labels_bright" # take classification labels as black box output

# DATAPATHS BY EXPERIMENT

# ------------------------------------------------------------- RB Seqs expts---
# # RB:
# BAYESEVAL = PATH2 + '20181003-153821BayesEval_probabilities.txt'
# IONBRIGHT = PATH2 + 'ion_bright.npz'

# ------------------------------------------------------------- pA & pB expts---
# # MODIFIED RAMSEY:
# IONBRIGHT = PATH2 + 'Riddhi_SLAM/'+'20181008-'+'concatenated_labels.npz'
IONBRIGHT = PATH2 + 'Riddhi_SLAM/20181008-210159ion_bright_matrix.npz'
# BAYESEVAL = ''

# Data generator for standard / pA experiments
# pA = 0
# RealDataGenerator = RealData(IONBRIGHT, BAYESEVAL, IONBRIGHT_KEY)
# max_msmts = 500

# Data generator for standard for pB experiments
# pA = 1
# RealDataGenerator = RealData(IONBRIGHT, BAYESEVAL, IONBRIGHT_KEY, pA=pA)
# max_msmts = 500

# ------------------------------------------------------------- Simple ramsey ---
# # SIMPLE RAMSEY:

# Expt 1: SCANNED
# IONBRIGHT = PATH2 + 'Claire_ramseydata/20181010-114733ion_bright_matrix.npz'

# Expt 2: STATIC
# IONBRIGHT = PATH2 + 'Claire_ramseydata/20181010-115434ion_bright_matrix.npz'
# IONBRIGHT = PATH2 + 'Claire_ramseydata/'+'20181010-114733_concatenated_labels.npz'
# max_msmts = 51*500

BAYESEVAL = ''
pA = 0

print IONBRIGHT

# print "Expt chosen:"
# print "IONBRIGHT:", IONBRIGHT
# print "BAYESEVAL:", BAYESEVAL
# print "pA:", pA
class RealData(object):

    def __init__(self, ion_bright_path2file=IONBRIGHT, 
                bayeseval_path2file=BAYESEVAL, data_key=KEY_2, pA=pA):
        ''' Accesses output classifier data of prob of seeing a bright ion by Hempel et al. '''

        self.data_key = data_key
        if self.data_key == "labels_bright":
            pick_classifer_prob = 0
        if self.data_key == KEY_1:
            pick_classifer_prob = 1

        # 0 - probability bright?
        # 1 - probability dark? (reverse definition)

        # RB DATA: HARDCODED
#        self.high_grad_seq = set([4, 6, 11, 13, 15, 16, 18, 24, 25, 28, 29, 32, 35, 37, 39, 46, 49, 51, 57])
#        exp_data = np.loadtxt(bayeseval_path2file).transpose()
#        ions_bright = np.load(ion_bright_path2file)[self.data_key]
#        self.bayesprimitives = exp_data[:, ::2]
#        self.primatives = ions_bright[:, ::2, :, pick_classifer_prob] # ::2 - skips bb1 data
#        self.pick_seq_qslam = list(self.high_grad_seq)[np.random.randint(low=0, high=len(self.high_grad_seq))]
#       print "Set sequence:", self.pick_seq_qslam


        # MODIFIED RAMSEY DATA HARDCODED
        self.high_grad_seq = None
        self.pick_seq_qslam = pA# # pA or pB experiment, pA data looks better
        self.primatives = np.load(ion_bright_path2file)[self.data_key][:, :, :, pick_classifer_prob]
        self.bayesprimitives = None

        # # SIMPLE STATIC RAMSEY DATA HARDCODED
        # self.high_grad_seq = None # static or scanned experiment
        # self.pick_seq_qslam = 0 #7 # set a ramsey time 
        # # Set ramsey for static expt as == 40ms; scanned expt s.t. < pi / 2 )
        # self.primatives = np.load(ion_bright_path2file)[self.data_key][:, :, :, pick_classifer_prob]
        # self.bayesprimitives = None

        # # SCANNED  RAMSEY DATA HARDCODED
        # self.high_grad_seq = None # static or scanned experiment
        # self.pick_seq_qslam = 7 # set a ramsey time 
        # # Set ramsey for static expt as == 40ms; scanned expt s.t. < pi / 2 )
        # self.primatives = np.load(ion_bright_path2file)[self.data_key][:, :, :, pick_classifer_prob]
        # self.bayesprimitives = None

        # OTHER

        self.total_ions = self.primatives.shape[0]
        self.total_seq = self.primatives.shape[1]
        self.total_repetitions = self.primatives.shape[2]
        self.recommended_amplification = np.round(1.0 / np.max(self.primatives.flatten()))
        self.ion_bright_path2file = IONBRIGHT
        self.sample_repts = list(range(self.total_repetitions))


    def get_real_data(self, node_j, amplification=1):
        '''Return a msmt from analysis of an experimental dataset
        node_j: postion index for ion
        p_bright: probability that the ion is bright according to classifier
        amplification: amplifies p_bright if data is close to extreme values'''

        pick_repetition = self.sample_repetitions_without_replacement()
        image_data_point = self.primatives[node_j, self.pick_seq_qslam, pick_repetition]

        if self.data_key == KEY_1:
            msmt = bnm.rvs(1, image_data_point*amplification)
            return msmt

        if self.data_key == KEY_2:
            return image_data_point

    def sample_repetitions_without_replacement(self):

        total_samples_left = len(self.sample_repts)

        if total_samples_left > 0:
            pick_sample = self.sample_repts[np.random.randint(low=0, high=total_samples_left)]
            self.sample_repts.remove(pick_sample)

            return pick_sample

        elif total_samples_left == 0:
            print "No more expt measurements avail"
            raise RuntimeError


    def get_empirical_mean(self):
        ''' Return the empirical mean of msmt data'''
        empirical_p_mean = np.mean(self.primatives[:, self.pick_seq_qslam, :], axis=1)
        return empirical_p_mean

    def get_bayes_pbright(self, pick_seq=None):
        '''Return the experimentalist analysis for p-bright for experiment using
        Bayes and empirical statistcs'''

        if pick_seq is None:
            pick_seq = self.pick_seq_qslam
        bayes_p_bright = None
        if self.bayesprimitives is not None:
            bayes_p_bright = self.bayesprimitives[:, pick_seq]
        empirical_p_var = np.var(self.primatives[:, pick_seq, :], axis=1)
        empirical_p_mean = self.get_empirical_mean()

        return bayes_p_bright, empirical_p_var, empirical_p_mean
