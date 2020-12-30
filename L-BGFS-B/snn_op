from models import SOD, POISSON, NBGLM, my_score_fun
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

def my_mse_fun(clf, X, y):
    x0 = clf.op.x
    y_pred = clf.predict(X)
    mse = mean_squared_error(y, y_pred)
    return mse

if __name__ == '__main__':
    X_file = '/Users/panpan/Documents/EB_spikes/snn_data/X.npy'
    y_file = '/Users/panpan/Documents/EB_spikes/snn_data/y.npy'
    X = np.load(X_file)
    y_all = np.load(y_file)
    LL_poisson_list = []
    mse_poisson_list = []
    LL_sod_list =[]
    mse_sod_list=[]
    scores_fun = {'ll_f':my_score_fun,
                  'mse_f':my_mse_fun}

    for i in range(y_all.shape[1]):
        y=y_all[:,i]

        lam = [0.1, 1, 10]
        param_grid = dict(lam=lam)
        # estimator of poisson
        #grid search
        '''
        grid_poisson = GridSearchCV(estimator=POISSON(), param_grid=param_grid, #n_jobs=15,
                                    cv=5, verbose=10, scoring=scores_fun)
        grid_poisson_result = grid_poisson.fit(X, y)
        print("Best: {0}, using {1}".format(grid_poisson_result.best_score_, grid_poisson_result.best_params_))
        # cv
        '''
        # grid_poisson_cv = cross_validate(POISSON(lam=grid_poisson_result.best_params_), X, y, #n_jobs=15,
        grid_poisson_cv = cross_validate(POISSON(lam=0.1), X, y,  # n_jobs=15,
                                         scoring=scores_fun, cv=5)
        LL_poisson_list.extend(grid_poisson_cv['test_ll_f'])
        mse_poisson_list.extend(grid_poisson_cv['test_mse_f'])


        # estimator of sod
        #grid search
        grid_sod = GridSearchCV(estimator=SOD(), param_grid=param_grid, #n_jobs=15,
                                    cv=5, verbose=10, scoring=my_score_fun)
        grid_sod_result = grid_sod.fit(X, y)
        print("Best: {0}, using {1}".format(grid_sod_result.best_score_, grid_sod_result.best_params_))
        # cv
        grid_sod_cv = cross_validate(SOD(lam=grid_sod_result.best_params_), X, y, #n_jobs=15,
                                         scoring=my_score_fun, cv=5)
        LL_sod_list.extend(grid_sod_cv['test_ll_f'])
        mse_sod_list.extend(grid_sod_cv['test_mse_f'])

    df = dict(LL_sod = LL_sod_list, mse_sod=mse_sod_list, LL_poisson=LL_poisson_list, mse_poisson = mse_poisson_list)
    df = pd.DataFrame(df)
    df.to_csv('snn_50sim_500bin_10neuron.csv')
