'''

This module contains support functions to pre-process camera data.
'''
import numpy as np
from pathlib import Path
import pandas as pd
import zipfile

import numpy as np
from lmfit.models import GaussianModel
from lmfit import Parameters
import peakutils

import matplotlib.pyplot as plt


def make_training_set(N, prefix, year, date, brightID, darkID, img_shape, cycles, ionpos):
    
    train_bright, train_dark, div = divide_img(N, prefix, year, date, brightID, darkID, img_shape, cycles)
    
    data = [train_dark, train_bright]
    training_images =[]
    training_labels =[]
    
    for idx_data in range(2):
        
        imageset = get_ion_imgs(ionpos, data[idx_data], img_shape, div, 1, cycles)
        # imageset = imageset*1.0 / np.max(imageset) # rescale
        training_images.append(imageset)
        training_labels.append(np.ones(imageset.shape[0])*idx_data*1.0)

    Xtr = np.vstack((training_images[0], training_images[1]))
    Str = np.concatenate((training_labels[0], training_labels[1]))
    
    return Xtr, Str, div


def make_test_set(N, prefix, year, date, datafile_p, img_shape, ionpos, div):

    testmatrix, dpts, reps = load_data(N, prefix, year, date, datafile_p, img_shape, 'predict', cycles=None)
    test_data = testmatrix.values.reshape(dpts * reps, img_shape[0]*img_shape[1])
    Xts = get_ion_imgs(ionpos, test_data, img_shape, div, dpts, reps)
    # Xts = Xts*1.0 / np.max(Xts) # rescale
    return Xts
    
    
def divide_img(N, prefix, year, date, brightID, darkID, img_shape, cycles):
    '''Return divisions of a camera picture for N ions.
    Reference: Extracted classifier_train2() function written by from C. Hempel et.al. '''
    
    brightf, dpts_b, reps_b = load_data(N, prefix, year, date, brightID, img_shape, 'train_bright', cycles=cycles)
    darkf, dpts_d, reps_d = load_data(N, prefix, year, date, darkID, img_shape, 'train_dark', cycles=cycles)

    train_bright = brightf.values.reshape((reps_b, img_shape[0] * img_shape[1]))
    train_dark = darkf.values.reshape((reps_d, img_shape[0] * img_shape[1]))
    
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
    
#     fig, ax = plt.subplots(figsize=(16, 4))
#     #ax.plot(x,y,'bo')
#     out.plot_fit(numpoints = 300, xlabel='pixel', ylabel='integrated counts')
    
#     for d in div:
#         ax.plot([(d), (d)],[0, max(bright_col_sum_bgs)],'--')
#         ax.plot([int(d), int(d)],[0, max(bright_col_sum_bgs)])
#     plt.title('ion locations and regions')
#     plt.legend(('data','best-fit','ideal division','max pixel div'))
#     plt.show()
    
    return train_bright, train_dark, div


def load_data(N, prefix, year, date, datafile, img_shape, flag, cycles=None):
    '''
    Return set of images from experimental data files.
    
    image_set - matrix of data with dims= [dpts, reps, img_shape[0], img_shape[1] ]
    flag - 'predict', 'train_bright', 'train_dark'
    cycles - must be specified if flag == 'train_bright' or 'train_dark'
    
    '''
    
    file=None
    
    if type(datafile) == list:
        # concatenate images for prediction only
        print('Request that following prediction datasets are concatenated:')
        print(datafile)
        data=[]
        for igor_ID in datafile:
            print('Adding file with IGOR ID: ', igor_ID)
            datafile_ext = igor_ID +'/cimg' + date + '-' + igor_ID + '.dat'
            zf = Path(prefix + year+ '/' + date + '/' + date +'-'+ datafile_ext)
            datacube = pd.read_csv(str(zf), sep="\t", header=None)
            data.append(datacube)
            
        file = pd.concat(data, axis=1)
        
    if flag=='predict':
        
        if file is None:
            datafile_ext = datafile +'/cimg' + date + '-' + datafile + '.dat'
            zf = Path(prefix + year+ '/' + date + '/' + date +'-'+ datafile_ext)
            file = pd.read_csv(str(zf), sep="\t", header=None)
        
        dpts = file.values.shape[0]
        reps = int(file.values.shape[1] / img_shape[0] / img_shape[1])
        
        return file, dpts, reps
        
    
    if flag[0:5]=='train':
        # Assumes a single calibration datset, one each for bright or dark ions
        
        if flag[6:] == 'bright':
            calibration_type = '/ROI_ref_bright_'
        elif flag[6:] =='dark':
            calibration_type = '/ROI_ref_bgnd_'
        
        filename =( prefix+year+'/'+date+calibration_type+str(img_shape[0])+'x'+str(img_shape[1])+'_'+str(cycles)+'_cyc_'+datafile+'.txt')

        zf = zipfile.ZipFile(Path(filename+'.zip')) 
        file = pd.read_csv(zf.open(zipfile.ZipFile.namelist(zf)[0]), sep="\t", header=None)
        dpts = file.values.shape[0]
        reps = cycles
    
        return file, dpts, reps
    


def get_ion_imgs(ionpos, data, img_shape, div, dpts, reps):
    ''' Return imgs of an ion at position ionpos from camera pictures.
    Reference: Extracted classifier_train2() function written by from C. Hempel et.al.
    
    ionpos - position index of ion taking values in [0, ..., N-1]
    data - set of camera images for N ions with dims [dpts*reps, img_shape[0] * img_shape[1]]
    img_shape - tuple of image dims ( img_shape[0] , img_shape[1])
    div - N-1 division locations of full camera image for N ions
    
    '''
    total_imgs = reps * dpts
    N = len(div) + 1
    if (ionpos == 0):
        img_width = int(div[0])+1
        ion_imgs = data.reshape(total_imgs, img_shape[0], img_shape[1])[:,:,0:int(div[0])+1].reshape(total_imgs, img_shape[0]*img_width)
    elif (ionpos == N-1):
        img_width = img_shape[1]-(int(div[N-2])+1)
        ion_imgs = data.reshape(total_imgs, img_shape[0],img_shape[1])[:,:,int(div[N-2])+1:img_shape[1]].reshape(total_imgs, img_shape[0]*img_width)
    else:
        img_width = int(div[ionpos])-int(div[ionpos - 1])
        ion_imgs = data.reshape(total_imgs, img_shape[0],img_shape[1])[:,:,int(div[ionpos -1])+1:int(div[ionpos])+1].reshape(total_imgs, img_shape[0]*img_width)
    
    return ion_imgs


def get_widths(N, div, img_shape):
    
    widths = []
    for ionpos in range(N) :
        if (ionpos == 0):
            img_width = int(div[0])+1
    #         dims = 0:int(div[0])+1
        elif (ionpos == N-1):
            img_width = img_shape[1]-(int(div[N-2])+1)
    #         dims = int(div[N-2])+1:img_shape[1]
        else:
            img_width = int(div[ionpos])-int(div[ionpos - 1])
    #         dims = int(div[ionpos -1])+1:int(div[ionpos])+1
        widths.append(img_width)
    return widths
# ---------------------------------------------- IGNORE -----------------

def rescale(data, maxval=255):
    '''Return the scaled image dataset where each value is between 0 and 1.
    Maxval is set to 255 for image data.'''
    return data*(1.0 / float(maxval))


def greyscale(colorimg):
    ''' Return a grey scale image, where RGB three channel images have
    dims: width x height x 3'''
    
    greyimg = np.mean(colorimg, axis=2)
    return greyimg
