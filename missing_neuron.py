import pandas as pd
from models import SOD, POISSON, NBGLM, my_score_fun
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer
import numpy as np


if __name__ == '__main__':
    # simulation sod + solution sod
    seed_betas = [102]
    n_sim = 50
    n_bin = 1000
    n_neuron = 100
    s = 50
    r = 5
    gam = 7
    density = 0.2
    lam = [0.1, 1, 10]
    partial=0.3
    seed_partials = [29,23,19,17,13,11]

    r2_list = []
    mse_list = []

    for seed_beta in seed_betas:
        # training
        X_train_all, y_train, y_train_true = SOD().simulate(n_sim=n_sim, n_bin=n_bin, n_neuron=n_neuron, density=density,
                                                        seed_beta=seed_beta, seed_X=101,
                                                        s = s, r = r, gam = gam)

        for seed_partial in seed_partials:
            # add missing neurons
            rng_select = np.random.default_rng(seed_partial)
            select = np.arange(n_neuron)
            rng_select.shuffle(select)
            X_train = X_train_all[:, select[:int(partial*n_neuron)]]

            param_grid = dict(lam=lam)
            # change estimator
            grid = GridSearchCV(estimator=POISSON(), param_grid=param_grid, n_jobs=-1,
                                cv=5, verbose=10, scoring=my_score_fun)
            grid_result = grid.fit(X_train, y_train)
            print("Best: {0}, using {1}".format(grid_result.best_score_, grid_result.best_params_))

            # change estimator
            model_tune = POISSON(lam = grid_result.best_params_['lam'])
            model_tune.fit(X_train, y_train)

            # change estimator name
            filename1 = 'sod_poisson_r_{}_s_{}_gam_{}_sim_{}_bin_{}_neuron_{}_beta_{}_X_{}_density{}_partial_{}_seed_{}.pickle'.\
                format(r, s, gam, n_sim, n_bin, n_neuron, seed_beta, 101, density, partial, seed_partial)
            model_tune.save(filename=filename1)
