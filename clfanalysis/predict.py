'''
This module contains a script to predict and classify ion data, and store results of training. 
Noise flip rates are also estimated from data.
Paratermised by ion position and classifier type.
'''

import numpy as np
from clfanalysis.preprocess import make_test_set, make_training_set
from clfanalysis.noisecorrections import estimate_Beta, estimate_flip_rates, clean_weights
from sklearn.metrics import confusion_matrix
from clfanalysis.globalparams import *
from clfanalysis.trainclf import train_classifier


def run_classifier_predictions(ionpos, clf_flag, datafile_p, N, prefix, year, date, brightID, darkID, img_shape, cycles, save):

    Xtr, Str, div = make_training_set(N, prefix, year, date, brightID, darkID, img_shape, cycles, ionpos)
    Xts = make_test_set(N, prefix, year, date, datafile_p, img_shape, ionpos, div)

    results, class_prob_list, best_model_list = train_classifier(clfdict[clf_flag]['obj'], 
                                                                 clfdict[clf_flag]['prm'], 
                                                                 Xtr, 
                                                                 Str, 
                                                                 N_ITER, 
                                                                 n_repeats, 
                                                                 cv=cv, scoring='accuracy')

    estimatortype = [ 'estf']

    if clf_flag == 'mlp':
        estimatortype = [None]

    rho_plus_list = []
    rho_minus_list = []
    params_list = []
    pi_list = []

    vars()['bestscore_'+'bare'] =0.
    vars()['bestscore_'+'estf'] =0.
    vars()['scoretracker_'+'bare']=[]
    vars()['scoretracker_'+'estf']=[]
    vars()['predY_'+'bare'] =-1.
    vars()['predY_'+'estf'] =-1.
    
    for pickrept in range(n_repeats):

        # Pick best model from each RandomisedCV trial with 5 Fold CV.
        class_prob = class_prob_list[pickrept]
        bestmodel = best_model_list[pickrept]
        params = results[pickrept]["params"][bestmodel]

        # Estimate Noise Rates from class probabilities of the model acting on training data
        alpha, beta, pi, rho_plus, rho_minus = estimate_flip_rates(class_prob[:, 1], Str.flatten())

        # Constrain physically
        if rho_minus >= 0. and rho_plus>=0. and pi >= 0. and rho_minus + rho_plus < 0.5 and pi <= 1.0:

            rho_plus_list.append(rho_plus)
            rho_minus_list.append(rho_minus)
            pi_list.append(pi)
            params_list.append(params)

            # Calculate weights using true and estimated noise rates
            for idx_type in estimatortype:

                if idx_type is not None:

                    if idx_type == 'estf':
                        vars()['weights_'+idx_type] = estimate_Beta(Str, class_prob, rho_minus, rho_plus)

                    vars()['weights_'+idx_type] =  clean_weights(vars()['weights_'+idx_type])

            pick_data = np.random.randint(low=0, high=900, size=800)
            score = np.random.randint(low=800, high=1000, size=100)

            # Run new classifiers with best model parameters and importance re-weighting
            for idx_type in estimatortype + ['bare']:

                if idx_type is not None:

                    vars()['clf_'+idx_type] = clfdict[clf_flag]['func'](**params)

                    X = Xtr[pick_data]
                    S = Str.flatten()[pick_data]
                    W = None

                    if idx_type != 'bare':
                        W = vars()['weights_'+idx_type].flatten()[pick_data]

                    # Fit models

                    if clf_flag != 'mlp':
                        vars()['clf_'+idx_type].fit(X, S, sample_weight=W)
                    elif clf_flag == 'mlp':
                        vars()['clf_'+idx_type].fit(X, S)

                    #Calculate score:
                    current_score = vars()['clf_'+idx_type].score(Xtr[score], Str.flatten()[score])

                    if current_score > vars()['bestscore_'+idx_type]: # retain perfect models which occur first 

                        # Predict on Test Data
                        print('Making a prediction', pickrept, idx_type, current_score, vars()['bestscore_'+idx_type])
                        vars()['predY_'+idx_type] = vars()['clf_'+idx_type].predict(Xts)
                        vars()['bestscore_'+idx_type] = current_score *1.0
                        vars()['scoretracker_'+idx_type].append(pickrept)


    mean_fit_time = np.zeros(n_repeats)
    mean_score_time = np.zeros(n_repeats)
    mean_train_score = np.zeros(n_repeats)
    mean_test_score = np.zeros(n_repeats)
    cv_test_score = np.zeros((n_repeats, cv))
    cv_train_score = np.zeros((n_repeats, cv))

    for idx_rept in range(n_repeats):

        best_model = best_model_list[idx_rept]

        mean_fit_time[idx_rept] = results[idx_rept]["mean_fit_time"][best_model]
        mean_score_time[idx_rept] = results[idx_rept]["mean_score_time"][best_model]
        mean_train_score[idx_rept] = results[idx_rept]["mean_train_score"][best_model]
        mean_test_score[idx_rept] = results[idx_rept]["mean_test_score"][best_model]

        for idx_cv in range(cv):

            cv_test_score[idx_rept, idx_cv] = results[idx_rept]['split'+str(idx_cv)+'_test_score'][best_model]
            cv_train_score[idx_rept, idx_cv] = results[idx_rept]['split'+str(idx_cv)+'_train_score'][best_model]


    if save is not None:
        np.savez(clf_flag+'_ionpos'+str(ionpos) + save, 
                mean_fit_time=mean_fit_time,
                mean_score_time=mean_score_time,
                mean_train_score=mean_train_score,
                mean_test_score=mean_test_score,
                cv_test_score=cv_test_score,
                cv_train_score=cv_train_score,
                best_model_list=best_model_list,
                class_prob_list=class_prob_list,
                rho_minus_list=rho_minus_list,
                rho_plus_list=rho_plus_list,
                pi_list=pi_list,
                params_list=params_list,
                predY_bare=vars()['predY_'+'bare'],
                predY_estf=vars()['predY_'+'estf'], 
                scoretracker_bare = vars()['scoretracker_'+'bare'],
                scoretracker_estf = vars()['scoretracker_'+'estf']
                )

    return vars()['predY_'+'bare'], vars()['predY_'+'estf'], rho_minus_list, rho_plus_list, pi_list