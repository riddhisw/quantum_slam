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
idx_prefix = int(sys.argv[1]) # truth flag type (three options)
idx_var = int(sys.argv[2]) - 1 # lambda scan (100 options). Array index starts at 1 so subtract 1


########################
# Truth Parameters
########################

# Choose defaults to match floor case (heights didn't work)
TRUTHKWARGS = {}

BARRIER_FLOOR = 0.25*np.pi
BARRIER_HEIGHT = 0.75*np.pi
FLOOR_RATIO = 0.4 # matches floor case

TRUTHKWARGS["OneStepdheight"] = {"low": BARRIER_FLOOR, 
                                 "high": BARRIER_HEIGHT} 
TRUTHKWARGS["OneStepdfloorarea"] = FLOOR_RATIO 



########################
# Save to path 
########################

savetopath = '/scratch/QCL_KF/qslamdata/' # on Artemis
# savetopath = './data/' # local

########################
# Set true field
########################

prefix_list = ['2019_Feb_1D', '2019_Feb_2D', '2019_Feb_2D_Gssn']
prefix = prefix_list[idx_prefix]

if idx_prefix == 0:
    change_gridconfig = True # 1D
    TRUTHFLAG = None # use TRUTHKWARGS
    TRUTHKWARGS["truthtype"] = 'OneStepd' 

if idx_prefix == 1:
    change_gridconfig = False # 2D
    TRUTHFLAG = None # use TRUTHKWARGS
    TRUTHKWARGS["truthtype"] = 'OneStepq' 

if idx_prefix == 2:
    change_gridconfig = False # 2D
    TRUTHFLAG = None # use TRUTHKWARGS
    TRUTHKWARGS["truthtype"] = 'Gaussian' 

########################
# Set 1D Hardware if req
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


# ------------------------------------------------------------------------------
# TURN OFF PARAMETER SCANS
# ------------------------------------------------------------------------------

idx_prevar = 0 
# Fix truth configurations
meta_truth_floor_scan = [FLOOR_RATIO] # [0.2, 0.4, 0.6, 0.8, 1.0]
lowscan = np.asarray(BARRIER_FLOOR) # np.asarray([0.2]*5)*np.pi
highscan = np.asarray(BARRIER_HEIGHT) # np.asarray([0.2, 0.4, 0.6, 0.8, 1.0])*np.pi
truth_step_scan = zip(lowscan, highscan)


idx_noise_var = 0 
# Fix to noiseless case
noiseclasses = ['noiseless'] 
noisestrengths = [0.0]
meta_noisevar_scan = zip(noiseclasses, noisestrengths)


# Fix msmt scan - and turn it off!
msmt_per_qubit_scan = [1] # [1, 2, 4, 5, 6, 8, 10, 15, 20, 25, 50]

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# NEW PARAMETER SCANS
# ------------------------------------------------------------------------------

meta_max_iter_scan = [ 5, 10, 15, 20, 25, 50, 75, 100, 125, 250]

lambda_databse = np.load('./lambda_pairs.npz')
lambda1 = list(lambda_databse['lambda_1']) # [0.99, 0.956, 0.922, 0.888, 0.854, 0.820, 0.786, 0.752, 0.718, 0.684, 0.65]
lambda2 = list(lambda_databse['lambda_2']) # [0.977, 0.9752, 0.9734, 0.9716, 0.9698, 0.968, 0.9662, 0.9644, 0.9626, 0.9608, 0.959]
lambda_scan = zip(lambda1, lambda2)

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

    vardict = copy.deepcopy(GLOBALDICT)
    vardict_truth = copy.deepcopy(TRUTHKWARGS)
    vardict, vardict_truth = ParamUpdater.meta_loop_update(vardict, vardict_truth,
                                                           idx_prevar, idx_msmt_var, idx_noise_var,
                                                           flag=TRUTHFLAG)
    
    vardict_0 = ParamUpdater.inner_loop_update(copy.deepcopy(vardict), idx_var, flag='weights')
    # vardict_1 = ParamUpdater.inner_loop_update(copy.deepcopy(vardict), idx_var, flag='msmtinfo') # Turn off msmt per iteration
    
    for idx_var_dict in range(1): # range(2): # Turn off msmt per iteration
        
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


