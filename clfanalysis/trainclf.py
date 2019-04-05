'''
This module contains helper functions for training classifer objects,
where classifer objects are RandomForest, MPLClassifier, and LogisticRegression
from sklearn. 
'''

# http://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score
# http://scikit-learn.org/stable/modules/model_evaluation.html
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import copy

def train_classifier(clf, clfparams, Xtr, Str, N_ITER, n_repeats, cv=5, scoring='accuracy'):
    '''Return trained classifer models:
    results: list of results (cv_results_ dictionary objects) for each repetition [len: n_repeats]
    class_prob_list: list of class probabilities from predict_proba() method for the best ranked model for each repetition. [len: n_repeats]
    best_model_list: list of n_iter index of best ranked moderl for each repetition [len: n_repeats]
    '''
    results = []
    class_prob_list = []
    best_model_list = []

    for idx_ in range(n_repeats):

        # print('Trial:', idx_)
        randomsearch_clf = 0
        randomsearch_clf = RandomizedSearchCV(estimator=clf,
                                             param_distributions=clfparams,
                                             n_iter=N_ITER,
                                             scoring=scoring,
                                             cv=cv,
                                             return_train_score=True)

        randomsearch_clf.fit(Xtr, Str.flatten())   
        results.append(copy.deepcopy(randomsearch_clf.cv_results_))
        class_prob_list.append(randomsearch_clf.predict_proba(Xtr))
        best_model_list.append(randomsearch_clf.best_index_)
        
    return results, class_prob_list, best_model_list

# def computeAccuracy(Y, pred_Y):
#    ''' Return accuracy score. sklearn metric (accuracy_score()) and definition
#    in COMP5328 is virtually the same except sklearn accounts for a single 
#    statistical degree of freedom for the true accuracy being estimated. For large 
#    datasets, both metrics are asympotically the same. '''
#    
#     acc = 0.0
#     for i in range(len(Y)):
#         if Y[i] == pred_Y[i]:
#             acc += 1.0
#     result = acc/len(Y)
    
    result = accuracy_score(Y, pred_Y)
    return result

# scoring = ['accuracy']#, 'balanced_accuracy'] #brier_score_loss, f1, precision, balanced_accuracy, average_precision
