import scipy
import scipy.io
import matplotlib.pyplot as plt
import os
from models import SOD, POISSON, NBGLM
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.pipeline import Pipeline
import pandas as pd
import h5py


def my_score_fun(clf, X, y):
    x0 = clf['clf'].op.x
    X_transform = clf['scale'].transform(X)
    possitive_LLsum = -1*clf['clf'].objective(x0, X_transform, y)
    return possitive_LLsum


def my_ll_fun(clf, X, y):
    x0 = clf['clf'].op.x
    X_transform = clf['scale'].transform(X)
    possitive_LLsum = clf['clf'].LL(x0, X_transform, y)
    return possitive_LLsum


def my_mse_fun(clf, X, y):
    X_transform = clf['scale'].transform(X)
    y_pred = clf['clf'].predict(X_transform, real=False)
    mse = mean_squared_error(y, y_pred)
    return mse

def my_r2_fun(clf, X, y):
    X_transform = clf['scale'].transform(X)
    y_pred = clf['clf'].predict(X_transform, real=False)
    r2 = r2_score(y, y_pred)
    return r2

if __name__ == '__main__':
    result = dict()
    scores_fun = {'ll_f':my_ll_fun,
                  'mse_f':my_mse_fun,
                  'r2_f':my_r2_fun}

    wkdir = '/home/msc/PycharmProjects/EB_spike/experimental_data/'
    files = os.listdir(wkdir)
    for file in files:
  
        LL_list = []
        mse_list = []
        r2_list = []
        lam_list = []
        

        spike_count = h5py.File(wkdir+file,'r')['spike_count'][()]
        spike_count = spike_count.T

        for i in range(spike_count.shape[1]):
            # preparing training and test data for each neuron
            y = spike_count[:, i]
            X = np.delete(spike_count, i, axis=1)

            lam = [0.1, 1, 10]
            param_grid = dict(clf__lam=lam)

            pipe = Pipeline([
                ('scale', StandardScaler()),
                ('clf', SOD())])

            grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring=my_score_fun)
            grid.fit(X, y)
            lam_list.append(grid.best_params_['clf__lam'])

            pipe_op = Pipeline([
                ('scale', StandardScaler()),
                ('clf', SOD(lam=grid.best_params_['clf__lam']))])
            # cv
            grid_cv = cross_validate(pipe_op, X, y, scoring=scores_fun, cv=5, n_jobs=-1)
            LL_list.extend(grid_cv['test_ll_f'])
            mse_list.extend(grid_cv['test_mse_f'])
            r2_list.extend(grid_cv['test_r2_f'])

        df = dict(LL=LL_list,mse=mse_list,r2=r2_list)
        df = pd.DataFrame(df)
        df.to_csv('file_{}_neuron_sod_true.csv'.format(file[:-4]))

        df2 = dict(lam=lam_list)
        df2 = pd.DataFrame(df2)
        df2.to_csv('file_{}_neuron_sod_lam.csv'.format(file[:-4]))
