import pandas as pd
from models import SOD, POISSON, NBGLM, my_score_fun
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer


if __name__ == '__main__':
    # simulation sod + solution sod
    seed_betas = [12,22,32,42,52,62,72,82,92,102]
    n_sim = 50
    n_bin = 500
    n_neuron = 100
    s = 50
    r = 5
    gam = 7

    sod_r2_list = []
    sod_mse_list = []

    for seed_beta in seed_betas:
        # training
        sod = SOD()
        X_train, y_train, y_train_true = sod.simulate(n_sim=n_sim, n_bin=n_bin, n_neuron=n_neuron,
                                                      seed_beta=seed_beta, seed_X=101, s = s, r = r, gam = gam)
        # sod solution
        lam = [0.1, 1, 10]
        param_grid = dict(lam=lam)
        grid = GridSearchCV(estimator=SOD(), param_grid=param_grid,
                            cv=5, verbose=10, scoring=my_score_fun)
        grid_result = grid.fit(X_train, y_train)
        print("Best: {0}, using {1}".format(grid_result.best_score_, grid_result.best_params_))

        sod_tune = SOD(lam = grid_result.best_params_)
        sod_tune.fit(X_train, y_train)
        filename_sod = 'sod_sod_r_{}_s_{}_gam_{}_sim_{}_bin_{}_neuron_{}_beta_{}_X_{}.pickle'.\
            format(r, s, gam, n_sim, n_bin, n_neuron, seed_beta, 42)
        sod_tune.save(filename=filename_sod)

        # evaluation
        X_test, y_test, y_test_true = sod.simulate(n_sim=n_sim, n_bin=n_bin, n_neuron=n_neuron,
                                                   seed_beta=seed_beta, seed_X=42, s = s, r = r, gam = gam)

        sod_r2_list.append(sod_tune.evaluate(X_test, y_test_true, 'r2'))
        sod_mse_list.append(sod_tune.evaluate(X_test, y_test_true, 'mse'))

    # log results
    filename2 = 'sod_data_r_{}_s_{}_gam_{}_sim_{}_bin_{}_neuron_{}_X_{}_results.csv'.\
        format(r, s, gam, n_sim, n_bin, n_neuron, 101)
    result = dict(sod_r2_list = sod_r2_list, sod_mse_list=sod_mse_list)

    df = pd.DataFrame(result)
    df.to_csv(filename2)

