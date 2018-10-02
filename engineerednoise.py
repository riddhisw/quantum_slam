'''
Created on Thu Sep 28 17:40:43 2017
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

class EngineeredNoise(object):
    
    def __init__(self):
        '''Creates a noise class object'''
        self.NOISE = {"alwaysdark": EngineeredNoise.alwaysdark, 
                      "spnoise": EngineeredNoise.spnoise, 
                      "noiseless":EngineeredNoise.noiseless}

    @staticmethod
    def alwaysdark(msmt):
        return np.zeros_like(msmt)

    @staticmethod
    def spnoise(msmt, lightdarksplit = 0.5):
        
        rand = np.random.uniform(0, 1, size=1)
        
        if rand <= lightdarksplit:
            return EngineeredNoise.alwaysdark(msmt)
        
        return np.ones_like(msmt)

    @staticmethod   
    def noiseless(msmt):
        return msmt


    def add_noise(self, msmts, 
                  prob_hit=0.1, 
                  noise_type='noiseless'):
        
        msmts = np.asarray(msmts, dtype=float)
        original_shape = msmts.shape
        msmts = msmts.flatten()
        
        totalmsmts = msmts.shape[0]
        rand = np.random.uniform(0, 1, size=totalmsmts)
        
        for idx_msmt in range(msmts.shape[0]):
            
            if rand[idx_msmt] <= prob_hit:
                
                msmts[idx_msmt] = self.NOISE[noise_type](msmts[idx_msmt])
        
        return msmts.reshape(original_shape)
