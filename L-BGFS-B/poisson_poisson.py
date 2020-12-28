import pandas as pd
from models import SOD, POISSON, NBGLM, my_score_fun
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer
import numpy as np


if __name__ == '__main__':
    # simulation sod + solution sod
    seed_betas = [12,22,32,42,52,62,72,82,92,102]
    n_sim = 50
    n_bin = 500
    n_neuron = 100
    density = 0.2
    lam = [0.1, 1, 10]

    r2_list = []
    mse_list = []

    for seed_beta in seed_betas:
        # training
        X_train, y_train, y_train_true = POISSON().simulate(n_sim=n_sim, n_bin=n_bin,
                                                            n_neuron=n_neuron, density=density,
                                                            seed_beta=seed_beta, seed_X=101)

        param_grid = dict(lam=lam)
        # change estimator
        grid = GridSearchCV(estimator=POISSON(), param_grid=param_grid,#n_jobs=-1,
                            cv=5, verbose=10, scoring=my_score_fun)
        grid_result = grid.fit(X_train, y_train)
        print("Best: {0}, using {1}".format(grid_result.best_score_, grid_result.best_params_))

        # change estimator
        model_tune = POISSON(lam = grid_result.best_params_['lam'])
        model_tune.fit(X_train, y_train)

        # change estimator name
        filename1 = 'poisson_poisson_sim_{}_bin_{}_neuron_{}_beta_{}_X_{}_density{}.pickle'.\
            format(n_sim, n_bin, n_neuron, seed_beta, 101, density)
        model_tune.save(filename=filename1)
