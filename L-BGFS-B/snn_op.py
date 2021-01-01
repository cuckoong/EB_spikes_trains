from models import SOD, POISSON, NBGLM
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def my_score_fun(clf, X, y):
    x0 = clf['clf'].op.x
    possitive_LLsum = -1 * clf['clf'].objective(x0, X, y)
    return possitive_LLsum


def my_mse_fun(clf, X, y):
    x0 = clf.name_steps.clf.op.x
    y_pred = clf.predict(X)
    mse = mean_squared_error(y, y_pred)
    return mse

if __name__ == '__main__':
    X_file = 'X.npy'
    y_file = 'y.npy'
    X = np.load(X_file)
    y_all = np.load(y_file)
    LL_list = []
    # mse_list = []
    # scores_fun = {'ll_f':my_score_fun,
    #               'mse_f':my_mse_fun}

    for i in range(y_all.shape[1]):
        y=y_all[:,i]

        lam = [0.1, 1, 10]
        param_grid = dict(clf__lam=lam)

        pipe = Pipeline([
            ('scale', StandardScaler()),
            ('clf', SOD())])

        grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring=my_score_fun)
        grid.fit(X, y)

        pipe_op = Pipeline([
            ('scale', StandardScaler()),
            ('clf', SOD(lam=grid.best_params_['clf__lam']))])
        #cv
        grid_cv = cross_validate(pipe_op, X, y, scoring=my_score_fun, cv=5, n_jobs=-1)
        LL_list.extend(grid_cv['test_score'])
        # mse_list.extend(grid_cv['test_scoring'])

    df = dict(LL = LL_list)
    df = pd.DataFrame(df)
    df.to_csv('snn_sod_50sim_500bin_10neuron.csv')
