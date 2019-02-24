'''
This module stores parameters that affect all simulations
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


# Global Training Parameters
n_repeats = 10
N_ITER = 7
cv = 5 # cross validation = 5 means that we train on 80% of the data, as required   

# Classifier definitions and parameter grids

rf = RandomForestClassifier()

param_dict_rf = {'n_estimators': [50, 100, 150, 200, 250],
                 'max_depth': [None, 5, 10, 15],
                 'bootstrap':[False], #, True], overfitting
                 'criterion' : ['gini', 'entropy']}

lr = LogisticRegression()

param_dict_lr = {'solver': ['saga', 'liblinear', 'lbfgs'],
                 'max_iter': [100, 150, 200, 250],
                 'fit_intercept': [True, False],
                 'tol': [10**(-2), 10**(-1), 0.5, 1],
                 #'penalty': ['l1', 'l2']
                 }

mlp = MLPClassifier()

param_dict_mlp = {'hidden_layer_sizes': [ (100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100)],
                 'activation': ['relu', 'logistic', 'tanh'],
                 'learning_rate': ['constant', 'invscaling'],
                 'learning_rate_init': [0.001, 0.01, 0.1],
                 'solver': ['adam', 'lbfgs'],
                 'max_iter': [150, 200, 250, 300],
                 'tol': [10**(-5), 10**(-4), 10**(-3)], #, 10**(-1), 0.5],
                 }

clfdict = {'mlp': {'obj': mlp, 'prm': param_dict_mlp, 'func': MLPClassifier},
           'lr': {'obj': lr, 'prm': param_dict_lr, 'func': LogisticRegression},
           'rf': {'obj': rf, 'prm': param_dict_rf, 'func': RandomForestClassifier},
    
}
