#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 3 11:41:00 2018

@author: chempel
"""

import numpy as np
import time
from pathlib import Path
import pandas as pd
# import zipfile
# RG 
# from Bayes2class import maximum_likelihood_state_estimation

def runAnalysisN (N, prefix, year, date, datafile, reps, startrep, img_shape, div, clf, points=None):
      
    #zf = zipfile.ZipFile(Path(prefix + year+ '/' + date + '/' + date +'-'+datafile[:-2] +'/cimg' + date + '-' + datafile[:-2] + '.zip')) 
    #file = pd.read_csv(zf.open(zipfile.ZipFile.namelist(zf)[0]), sep="\t", header=None)
    zf = Path(prefix + year+ '/' + date + '/' + date +'-'+datafile +'/cimg' + date + '-' + datafile + '.dat')
    file = pd.read_csv(str(zf), sep="\t", header=None)
    
    dpts = file.values.shape[0]
    repetitions = int(file.values.shape[1] / img_shape[0] / img_shape[1])
    print ('read ' + str(dpts) + ' data points with ' + str(repetitions) + ' repetitions (' + str(points) + ' points, ' + str(reps) +  ' repetitions requested)')
    
    if points is None:
        points = np.arange(dpts)

    
    #%% Classify data
    ion_bright = []
    tic = time.clock()
    for ion in range(0,N):  
        if (ion == 0):
            img_width = int(div[0])+1
            print('image width '+str(img_width)+' pixels from 0 to ' + str(int(div[0])))
            data = file.values.reshape(dpts,reps,img_shape[0],img_shape[1])[:,:,:,0:int(div[0])+1].reshape(dpts*reps,img_shape[0]*img_width)
        elif (ion == N-1):
            img_width = img_shape[1]-(int(div[N-2])+1)
            print('image width '+str(img_width)+' pixels from ' + str(int(div[N-2])+1) +' to ' + str(img_shape[1]-1))
            data = file.values.reshape(dpts,reps,img_shape[0],img_shape[1])[:,:,:,int(div[N-2])+1:img_shape[1]].reshape(dpts*reps,img_shape[0]*img_width)
        else:
            img_width = int(div[ion])-int(div[ion-1])
            print('image width '+str(img_width)+' pixels from ' + str(int(div[ion-1])+1) +' to ' + str(int(div[ion])))
            data = file.values.reshape(dpts,reps,img_shape[0],img_shape[1])[:,:,:,int(div[ion-1])+1:int(div[ion])+1].reshape(dpts*reps,img_shape[0]*img_width)     
    
        ion_bright.append(clf[ion].predict_proba(data)[:,:].reshape(dpts,repetitions, 2))
   
    # RG
    filename = date + '-' + datafile
    np.savez(prefix + '/' + filename + 'ion_bright_matrix', ion_bright=ion_bright)

# # Mute Bayes2class.maximum_likelihood_state_estimation() as theory unclear.
#    
#     n_points = len(points)
#     pb = np.zeros((N, n_points)) # probabilities bright per ion
#     pb_err = np.zeros((N, 2, n_points))# probabilities bright per ion
#     for ion in range(0,N):
#         # Classify dark/bright
#         tic = time.clock()
#         print('states classified: ' + str(time.clock()-tic) + ' seconds')
#         print('starting Bayesian analysis')
#         (pbright, pbright_err)  = maximum_likelihood_state_estimation(ion_bright[ion][:, startrep:])
#         pb[ion] = pbright
#         pb_err[ion] = pbright_err
#     return [pb, pb_err]
