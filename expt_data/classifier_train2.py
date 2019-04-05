#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 10:25:10 2018

@author: chempel
"""

from pathlib import Path
import pandas as pd
import zipfile

import numpy as np
from lmfit.models import GaussianModel
from lmfit import Parameters
import peakutils

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

def classifier_train2(N,reps,img_shape,prefix,year,date,brightID,darkID):

    #%% LOAD CALIBRATION DATA
    brightFN = (prefix+year+'/'+date+'/ROI_ref_bright_'+str(img_shape[0])+'x'+str(img_shape[1])+'_'+str(reps)+'_cyc_'+brightID+'.txt')
    darkFN = (prefix+year+'/'+date+'/ROI_ref_bgnd_'+str(img_shape[0])+'x'+str(img_shape[1])+'_'+str(reps)+'_cyc_'+darkID+'.txt')
    
    ##
    zf = zipfile.ZipFile(Path(brightFN+'.zip')) 
    file = pd.read_csv(zf.open(zipfile.ZipFile.namelist(zf)[0]), sep="\t", header=None)
    train_bright = file.values.reshape((reps, img_shape[0] * img_shape[1]))
    
    zf = zipfile.ZipFile(Path(darkFN+'.zip')) 
    file = pd.read_csv(zf.open(zipfile.ZipFile.namelist(zf)[0]), sep="\t", header=None)
    train_dark = file.values.reshape((reps, img_shape[0] * img_shape[1]))
    
    #%% CREATE 1D projections for region finding and find peaks (starting vals)
    bright_avg_bgs = np.subtract(np.sum(train_bright, axis=0),np.sum(train_dark, axis=0))
    bright_col_sum_bgs = np.sum(bright_avg_bgs.reshape(img_shape),axis=0)
    
    x = np.arange(0,img_shape[1])
    y = bright_col_sum_bgs
    
    indexes = peakutils.indexes(y, thres=0.5, min_dist=2)    
    
    #%% initialize model function for Gaussian fits and run fit
    pars = Parameters()
    for num in range(1,N+1):
       #print('ion ' + str(num))
       model = GaussianModel(prefix='ion'+ str(num) + '_')
       pars.update(model.make_params())
       pars['ion'+ str(num) + '_center'].set(indexes[N-num])
       pars['ion'+ str(num) + '_sigma'].set(0.5)
       pars['ion'+ str(num) + '_amplitude'].set(np.max(y))
       if (num == 1):
           mod = model
       else:
           mod = mod + model
    
    out = mod.fit(y, pars, x=x)   
    
    div=np.zeros(N-1)
    for num in range(1,N):
        center1 = out.params['ion'+ str(num) + '_center'].value
        center2 = out.params['ion'+ str(num+1) + '_center'].value
        div[num-1] = ((center1+center2)/2)
    div=np.flip(div,0)
    
    #%% Plot result and divisions
    fig = plt.figure
    plt.imshow(np.sum(train_bright, axis=0).reshape(img_shape))
    plt.show()
    
    fig, ax = plt.subplots(figsize=(16, 4))
    #ax.plot(x,y,'bo')
    out.plot_fit(numpoints = 300, xlabel='pixel', ylabel='integrated counts')
    for d in div:
        ax.plot([(d), (d)],[0, max(bright_col_sum_bgs)],'--')
        ax.plot([int(d), int(d)],[0, max(bright_col_sum_bgs)])
    plt.title('ion locations and regions')
    plt.legend(('data','best-fit','ideal division','max pixel div'))
    plt.show()
    
    #%% Classifier training
    clf=[]
    scores=[]
    for ion in range(0,N):
        print('Ion ' + str(ion))
        if (ion == 0):
            img_width = int(div[0])+1
            print('image width '+str(img_width)+' pixels from 0 to ' + str(int(div[0])))
            data_ion_bright = train_bright.reshape(reps,img_shape[0],img_shape[1])[:,:,0:int(div[0])+1].reshape(reps,img_shape[0]*img_width)
            data_ion_dark = train_dark.reshape(reps,img_shape[0],img_shape[1])[:,:,0:int(div[0])+1].reshape(reps,img_shape[0]*img_width)
        elif (ion == N-1):
            img_width = img_shape[1]-(int(div[N-2])+1)
            print('image width '+str(img_width)+' pixels from ' + str(int(div[N-2])+1) +' to ' + str(img_shape[1]-1))
            data_ion_bright = train_bright.reshape(reps,img_shape[0],img_shape[1])[:,:,int(div[N-2])+1:img_shape[1]].reshape(reps,img_shape[0]*img_width)
            data_ion_dark = train_dark.reshape(reps,img_shape[0],img_shape[1])[:,:,int(div[N-2])+1:img_shape[1]].reshape(reps,img_shape[0]*img_width)
        else:
            img_width = int(div[ion])-int(div[ion-1])
            print('image width '+str(img_width)+' pixels from ' + str(int(div[ion-1])+1) +' to ' + str(int(div[ion])))
            data_ion_bright = train_bright.reshape(reps,img_shape[0],img_shape[1])[:,:,int(div[ion-1])+1:int(div[ion])+1].reshape(reps,img_shape[0]*img_width)     
            data_ion_dark = train_dark.reshape(reps,img_shape[0],img_shape[1])[:,:,int(div[ion-1])+1:int(div[ion])+1].reshape(reps,img_shape[0]*img_width)   
                
        data = np.concatenate((data_ion_dark, data_ion_bright))
        labels = np.concatenate(([0] * reps, [1] * reps))
    
        clfier = RandomForestClassifier(n_estimators=100, n_jobs=-1, bootstrap=False)
        clfier = clfier.fit(data, labels)

        clf.append(clfier)
        scores.append(cross_val_score(clfier, data, labels, cv=10))
        print('Cross-validation accuracy: {:.3f} +/- {:.3f}'.format(np.mean(scores), np.std(scores)))
    return [clf,div,scores]