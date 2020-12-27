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
    s = 50
    r = 5
    gam = 7
    density = 1
    lam = [0.1, 1, 10]
    partials = [0.8, 0.5, 0.3]
    seed_partials = [11, 13, 17, 19, 23]

    r2_list = []
    mse_list = []

    for seed_beta in seed_betas:
        # training
        X_train_all, y_train, y_train_true = SOD().simulate(n_sim=n_sim, n_bin=n_bin, n_neuron=n_neuron, density=density,
                                                        seed_beta=seed_beta, seed_X=101,
                                                        s = s, r = r, gam = gam)

        for seed_partial in seed_partials:
            # add missing neurons (remain neurons = 0.8*n_neuron)
            rng_select = np.random.default_rng(seed_partial)
            select = np.arange(n_neuron)
            rng_select.shuffle(select)
            # , seed = seed_partial)[:select_num]
            X_train = X_train_all[:, select[0:80]]

            param_grid = dict(lam=lam)
            # change estimator
            grid = GridSearchCV(estimator=SOD(), param_grid=param_grid,#n_jobs=-1,
                                cv=5, verbose=10, scoring=my_score_fun)
            grid_result = grid.fit(X_train, y_train)
            print("Best: {0}, using {1}".format(grid_result.best_score_, grid_result.best_params_))

            # change estimator
            model_tune = NBGLM(lam = grid_result.best_params_['lam'])
            model_tune.fit(X_train, y_train)

            # change estimator name
            filename1 = 'sod_sod_r_{}_s_{}_gam_{}_sim_{}_bin_{}_neuron_{}_beta_{}_X_{}_density{}_partial_{}.pickle'.\
                format(r, s, gam, n_sim, n_bin, n_neuron, seed_beta, 101, density, seed_partial)
            model_tune.save(filename=filename1)
