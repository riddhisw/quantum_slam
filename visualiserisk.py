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
import numpy as np
import matplotlib
import sys
import os

fsize=12#8
Fsize=12#8
# Set global parameters
matplotlib.rcParams['font.size'] = fsize # global
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['axes.linewidth'] = 0.5
matplotlib.rcParams['mathtext.default'] ='regular' # makes mathtext mode Arial. note mathtext is used as ticklabel font in log plots

# Set global tick mark parameters
matplotlib.rcParams['xtick.major.width'] = 0.5
matplotlib.rcParams['ytick.major.width'] = 0.5
matplotlib.rcParams['xtick.labelsize']= fsize
matplotlib.rcParams['ytick.labelsize'] = fsize
matplotlib.rcParams['xtick.minor.visible'] = False
matplotlib.rcParams['ytick.minor.visible'] = False
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'

from matplotlib.gridspec import GridSpec as gs
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
Path = mpath.Path


# DATA HANDLING

DATAVIEWKEYS = {"truth" : "macro_true_fstate",
                "pred_f" : "macro_predictions",
                "pred_r" : "macro_correlations",
                "path" : "macro_paths",
                "errs" : "macro_residuals"
               }

FLAG = {"q" : "qslamr",
        "n" : "naive"
       }

NPZFLAG = {FLAG["q"]: '_BR_Map.npz',
           FLAG["n"]: '_NE_Map.npz'
          }

PATHDICT = {"pdir": './data',
            "fle": 'empty'}

def path_to_file(PATHDICT, flag="q"):
    'Return path to file'
    pathtofile = os.path.join(PATHDICT["pdir"],
                              PATHDICT["fle"] + NPZFLAG[FLAG[flag]])
    return pathtofile

def check_savefig(current_indices, controls=None):
    'Return True if parameter regimes for plotting match controls'
    if controls is None:
        return False
    for item in controls:
        result = np.all(np.asarray(current_indices) == np.asarray(item))
        if result:
            return result

def cm2inch(cm):
    return cm / 2.54
    
    
    
# Metrics

class DataCube(object):

    def __init__(self, loops_dictionary):

        for key in loops_dictionary.keys():
            setattr(self, key, loops_dictionary[key])

    def meta_loop_update(self, vardict, vardict_truth, idx_prevar, idx_msmt_var, idx_noise_var,
                  flag='floor'):

        if flag =='floor':
            vardict_truth["OneStepdfloorarea"]  = self.meta_truth_floor_scan[idx_prevar]

        if flag =='height':
            vardict_truth["OneStepdheight"]["low"] = self.truth_step_scan [idx_prevar][0]
            vardict_truth["OneStepdheight"]["high"] = self.truth_step_scan[idx_prevar][1]

        vardict["MODELDESIGN"]["MAX_NUM_ITERATIONS"] = self.meta_max_iter_scan[idx_msmt_var]

        vardict["ADDNOISE"]["args"]["noise_type"] = self.meta_noisevar_scan[idx_noise_var][0]
        vardict["ADDNOISE"]["args"]["prob_hit"] = self.meta_noisevar_scan[idx_noise_var][1]

        return vardict, vardict_truth

    def inner_loop_update(self, vardict, idx_var, flag='weights'):

        if flag == 'weights':
            vardict["MODELDESIGN"]["LAMBDA_1"] = self.lambda_scan[idx_var][0]
            vardict["MODELDESIGN"]["LAMBDA_2"] = self.lambda_scan[idx_var][1]

        if flag == 'msmtinfo':
            vardict["MODELDESIGN"]["MSMTS_PER_NODE"] = self.msmt_per_qubit_scan[idx_var]

        return vardict


class Metric(object):

    def __init__(self):
        pass


    @staticmethod
    def original_err_metric(macro_residuals):
        result = (1.0/ np.sqrt(macro_residuals.shape[2]))*np.mean(np.linalg.norm(macro_residuals[0,:,:], axis=1), axis=0)
        return result / np.pi

    @staticmethod
    def err(macro_residuals):
        error = np.mean(Metric.singlemap_err(macro_residuals[0,:,:], axis=1), axis=0)
        return error 

    @staticmethod
    def singlemap_err(residualsdata, axis):
        normaliser = 1.0 / np.sqrt(residualsdata.shape[axis])
        result = normaliser * np.linalg.norm(residualsdata, axis=axis) / np.pi
        return result

    @staticmethod
    def rms(macro_residuals):
        rmse_vector = Metric.singlemap_rmse(macro_residuals[0, :, :], axis=0)
        normaliser = 1.0 / np.sqrt(rmse_vector.shape[0])
        error = normaliser*np.linalg.norm(rmse_vector)
        return error

    @staticmethod
    def singlemap_rmse(residualsdata, axis):
        result = np.sqrt(np.mean(residualsdata**2, axis=axis)) / np.pi
        return result

    @staticmethod
    def ssim(dataobj, Cone=0.01, Ctwo=0.01):

        scores = np.zeros(50)
        
        for idx_run in range(50):
            
            pred = dataobj["macro_predictions"][0, idx_run, :]
            truth =  dataobj["macro_true_fstate"][0, idx_run, :]
            scores[idx_run] = Metric.score_ssim(pred, truth, Cone=Cone, Ctwo=Ctwo)
        
    #     avgssim = np.mean(scores) # returing ssim (raw)
    #     result = abs(1 - avgssim) # == np.mean(scores)

    #     avgssim = np.mean(abs(1.0 - scores)) # returing ssim (raw)
    #     result = avgssim*1.0

        result = np.mean(scores) # returing deviations abs(1 - ssim)
        
        return result, scores

    @staticmethod    
    def score_ssim(sigx, sigy, Cone=0.01, Ctwo=0.01):
        '''Equation 13 in Image quality assessment'''
        mu_x = np.mean(sigx)
        mu_y = np.mean(sigy)
        stdx = np.std(sigx, ddof=1)
        stdy = np.std(sigx, ddof=1)

        Nminus1 = sigx.shape[0] - 1
        covarxy = (1.0 / Nminus1)*np.sum((sigx - mu_x)*(sigy - mu_y))
        
        ssim = (2.0*mu_x*mu_y + Cone) * (2.0*covarxy + Ctwo)
        ssim /= (mu_x**2 + mu_y**2 + Cone) * (stdx**2 + stdy**2 + Ctwo)
        
        deviation = abs(1.0 - ssim)  # returing deviation abs(1 - ssim)
        return deviation # ssim 


# CONTROL PATH PLOTTER

PKWG = {"pc": 'w', # path base color
        "palpha": 0.25, # path color alpha value
        "pss": 10, # path start / stop size 
        "ac": 'w', # arrow base color
        "aalpha": 0.3, # arrow color alpha value
        "adxdy": 0.1, # arrow dx and dy as ratio of path segment
        "alw": 1.0, # arrow line width
        "ahw": 0.05, # arrow head width
        }


HEATMAP = {"vmin" : 0.0,
            "vmax" : np.pi,
            "cmap" : 'viridis',
            "origin": 'lower'
            }

CORRMAP = {"vmin" : 0.0,
            "vmax" : 30.,
            "cmap" : 'Purples',
            "origin": 'lower'
            }

class qPlotter(object):

    def __init__(self, 
                 userPKWG=None,
                 userHEATMAP=None,
                 userCORRMAP=None):

        self.PKWG = userPKWG
        self.HEATMAP = userHEATMAP
        self.CORRMAP = userCORRMAP        

        if self.PKWG is None:
            self.PKWG = PKWG

        if self.HEATMAP is None:
            self.HEATMAP = HEATMAP

        if self.CORRMAP is None:
            self.CORRMAP = CORRMAP



    def get_single_run(self, dataobj, viewtype, pickone):
        
        totalruns = dataobj[DATAVIEWKEYS[viewtype]].shape[1]
        
        if pickone is None:
            pickone = np.random.randint(low=0, high=totalruns)
        
        data = dataobj[DATAVIEWKEYS[viewtype]][0, pickone, :]
        return data
        

    def show_map(self, ax, dataobj, viewtype, pickone=None, linear=False):
        
        # if viewtype == 'path':
        #     print "Skip control path"
        #     return ax
        
        statedata = self.get_single_run(dataobj, viewtype, pickone)
        
        # print statedata.shape
        
        if linear is False:
        
            if statedata.shape[0] < 4:
                cax = ax.imshow(statedata[np.newaxis, :], **self.HEATMAP)
            
            if statedata.shape[0] >= 4:
                mapdims = int(np.sqrt(statedata.shape))
                mapdata = statedata.reshape(mapdims, mapdims)
                cax = ax.imshow(mapdata, **self.HEATMAP)
        
        if linear is True:
                mapdims = statedata.shape[0]
                mapdata = np.zeros((3, mapdims))
                for idx in range(3): # make imshow fatter
                    mapdata[idx, : ] = statedata
                cax = ax.imshow(mapdata, **self.HEATMAP)
        
        return ax, cax


    def show_control_path(self, ax, dataobj, GRIDDICT, viewtype="path",  pickone=None):
        
        controlpath = self.get_single_run(dataobj, viewtype, pickone)
        
        points = self.get_control_path(controlpath, GRIDDICT) 
        
        codes = [Path.LINETO] * len(points)
        codes[0] = Path.MOVETO
        
        path = mpath.Path(points, codes)
        patch = mpatches.PathPatch(path, 
                                facecolor='None', 
                                edgecolor=self.PKWG["pc"], 
                                alpha=self.PKWG["palpha"])
        ax.add_patch(patch)
        
        ax.plot(points[0][0], points[0][1], 'o',
                color=self.PKWG["pc"], 
                # edgecolor=self.PKWG["pc"], 
                ms=self.PKWG["pss"])
        ax.plot(points[-1][0], points[-1][1], 's',
                color=self.PKWG["pc"], 
                # edgecolor=self.PKWG["pc"], 
                ms=self.PKWG["pss"])
        
        for idp in range(len(points)-1):
            
            ax.arrow(points[idp][0], points[idp][1], 
                    self.PKWG["adxdy"]*(points[idp+1][0] - points[idp][0]), 
                    self.PKWG["adxdy"]*(points[idp+1][1] - points[idp][1]), 
                    shape='full',  
                    lw=self.PKWG["alw"], 
                    facecolor=self.PKWG["ac"], 
                    edgecolor=self.PKWG["ac"], 
                    alpha=self.PKWG["aalpha"],
                    length_includes_head=True, 
                    width=self.PKWG["ahw"])

        return ax


    def get_control_path(self, path, griddict):
        '''Returns qubit positions (x,y) visited based on a control path and input dictionary of qubits'''
        
        result = []
        
        for qubit in path:
            
            if qubit + 1 < 10:
                result.append(griddict["QUBIT_0" + str(qubit +1 )])
                
            if qubit + 1 >= 10:
                result.append(griddict["QUBIT_" + str(qubit + 1 )])
                
        if len(path) == len(result):
            return result
        
        raise RuntimeError

