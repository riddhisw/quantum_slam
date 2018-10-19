import sys
import os
import copy
import traceback

########################
# Find qslam modules
########################
sys.path.append('../')

from qslamdesignparams import GLOBALDICT
from riskanalysis import CreateQslamExpt as riskqslam
from riskanalysis import CreateNaiveExpt as risknaive
from visualiserisk import *

########################
# Taking in bash parameters
########################
idx_prefix = int(sys.argv[1]) # 
idx_noise_var = int(sys.argv[2])
idx_prevar = int(sys.argv[3])


########################
# Save to path 
########################

savetopath = '/scratch/QCL_KF/qslamdata/'
# savetopath = './data/'
prefix_list = ['NSL_tfloor', 'NSL_theight']

prefix = prefix_list[idx_prefix]

########################
# Truth Parameters
########################

TRUTHKWARGS = {}
TRUTHKWARGS["truthtype"] = 'OneStepd' 
TRUTHKWARGS["OneStepdheight"] = {"low": 0.25*np.pi,
                                 "high": 0.75*np.pi}

TRUTHKWARGS["OneStepdfloorarea"] = 0.25
change_gridconfig = True


########################
# Set Hardware 
########################

if change_gridconfig is True:
    
    GLOBALDICT["GRIDDICT"] = {}
    
    for idx_posy in range(25):
        
        if idx_posy < 10 :
            
            GLOBALDICT["GRIDDICT"]["QUBIT_0" + str(idx_posy)] = (0.0, float(idx_posy))
            
        if idx_posy >= 10 :
            
            GLOBALDICT["GRIDDICT"]["QUBIT_" + str(idx_posy)] = (0.0, float(idx_posy))


########################
# Set Defaults
########################

change_MAX_NUM_ITERATIONS = 100
change_MSMTS_PER_NODE = 1
change_SIGMOID_APPROX_ERROR = 10.0**(-6)
change_QUANTISATION_UNCERTY = 10.0**(-4)
change_P_ALPHA = 15 
change_P_BETA = 10 
change_LAMBDA_1 = 0.820
change_LAMBDA_2 = 0.968

GLOBALDICT["MODELDESIGN"]["MAX_NUM_ITERATIONS"] = change_MAX_NUM_ITERATIONS
GLOBALDICT["MODELDESIGN"]["MSMTS_PER_NODE"] = change_MSMTS_PER_NODE
GLOBALDICT["NOISEPARAMS"]["SIGMOID_APPROX_ERROR"]["SIGMA"] = change_SIGMOID_APPROX_ERROR
GLOBALDICT["NOISEPARAMS"]["QUANTISATION_UNCERTY"]["SIGMA"] = change_QUANTISATION_UNCERTY
GLOBALDICT["MODELDESIGN"]["P_ALPHA"] = change_P_ALPHA
GLOBALDICT["MODELDESIGN"]["P_BETA"] = change_P_BETA
GLOBALDICT["MODELDESIGN"]["LAMBDA_1"] = change_LAMBDA_1
GLOBALDICT["MODELDESIGN"]["LAMBDA_2"] = change_LAMBDA_2



########################
# Set All Loop Parameters
########################

meta_truth_floor_scan = [0.2, 0.4, 0.6, 0.8, 1.0]

lowscan = np.asarray([0.2]*5)*np.pi
highscan = np.asarray([0.2, 0.4, 0.6, 0.8, 1.0])*np.pi
truth_step_scan = zip(lowscan, highscan)


meta_max_iter_scan = [ 5, 10, 15, 20, 25, 50, 75, 100, 125, 250]


noiseclasses = ['noiseless'] + ['alwaysdark', 'spnoise']*4
noisestrengths = [0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.5, 0.5]*2
meta_noisevar_scan = zip(noiseclasses, noisestrengths)


lambda1 = [0.99, 0.956, 0.922, 0.888, 0.854, 0.820, 0.786, 0.752, 0.718, 0.684, 0.65]
lambda2 = [0.977, 0.9752, 0.9734, 0.9716, 0.9698, 0.968, 0.9662, 0.9644, 0.9626, 0.9608, 0.959]
lambda_scan = zip(lambda1, lambda2)


msmt_per_qubit_scan = [1, 2, 4, 5, 6, 8, 10, 15, 20, 25, 50]


LOOPS_DICT = {"meta_truth_floor_scan": meta_truth_floor_scan,
              "meta_max_iter_scan":meta_max_iter_scan, 
              "meta_noisevar_scan": meta_noisevar_scan,
              "truth_step_scan": truth_step_scan,
              "lambda_scan":lambda_scan,
              "msmt_per_qubit_scan": msmt_per_qubit_scan}

########################
# Run Script
########################

ParamUpdater = DataCube(LOOPS_DICT)

for idx_msmt_var in range(len(ParamUpdater.meta_max_iter_scan)):

    for idx_var in range(max(len(ParamUpdater.lambda_scan), len(ParamUpdater.msmt_per_qubit_scan))):

        vardict = copy.deepcopy(GLOBALDICT)
        vardict_truth = copy.deepcopy(TRUTHKWARGS)
        vardict, vardict_truth = ParamUpdater.meta_loop_update(vardict, vardict_truth,
                                                               idx_prevar, idx_msmt_var, idx_noise_var,
                                                               flag='floor')
        
        vardict_0 = ParamUpdater.inner_loop_update(copy.deepcopy(vardict), idx_var, flag='weights')
        vardict_1 = ParamUpdater.inner_loop_update(copy.deepcopy(vardict), idx_var, flag='msmtinfo')
        
        for idx_var_dict in range(2):
            
            regime_ID = prefix + '_n_' + str(idx_noise_var) +'_vset_' + str(idx_var_dict)
            testcase_ID = regime_ID + '_t_' + str(idx_prevar) + '_m_' + str(idx_msmt_var)
            
            unique_id = savetopath + testcase_ID + '_v_' + str(idx_var)
            
            vars()['vardict_'+str(idx_var_dict)]["MODELDESIGN"]["ID"] = unique_id
              
            qslam_br = 0.
            naive_br = 0.
            qslamdata = 0.
            naivedata = 0.
        
            try:
                qslam_br = riskqslam(vardict_truth, vars()['vardict_'+str(idx_var_dict)])
                naive_br = risknaive(vardict_truth, vars()['vardict_'+str(idx_var_dict)])
                qslam_br.naive_implementation(randomise='OFF')
                naive_br.naive_implementation()
                print "Variation | Set: %s | Index: %s successful..." %(idx_var_dict, idx_var)

            except:
                print "Variation | Set: %s | Index: %s was not completed..." %(idx_var_dict, idx_var)
                print "Error information:"
                print "Type", sys.exc_info()[0]
                print "Value", sys.exc_info()[1]
                print "Traceback", traceback.format_exc()


