import pandas as pd
from models import SOD, POISSON, NBGLM


if __name__ == '__main__':
    # simulation sod + solution sod
    seed_betas = [12,22,32,42,52,62,72,82,92,102]
    n_sims = [50]
    n_bins = [50, 100, 500, 1000]
    n_neuron = 100
    s = 50
    r = 5
    gam = 7
    lam = 1

    for n_sim in n_sims:
        for n_bin in n_bins:
            sod_r2_list = []
            sod_mse_list = []
            nbglm_r2_list = []
            nbglm_mse_list = []
            poisson_r2_list = []
            poisson_mse_list = []
            for seed_beta in seed_betas:
                # training
                sod = SOD()
                X_train, y_train, y_train_true = sod.simulate(n_sim=n_sim, n_bin=n_bin, n_neuron=n_neuron,
                                                              seed_beta=seed_beta, seed_X=101, s = s, r = r, gam = gam)
                # sod solution
                sod.fit(X_train, y_train, lam=lam)
                filename_sod = 'sod_sod_r_{}_s_{}_gam_{}_sim_{}_bin_{}_neuron_{}_beta_{}_X_{}.pickle'.\
                    format(r, s, gam,n_sim, n_bin, n_neuron, seed_beta, 101)
                sod.save(filename_sod)

                # nbglm solution
                nbglm = NBGLM()
                nbglm.fit(X_train, y_train, lam=lam)
                filename_nb = 'sod_nbglm_r_{}_s_{}_gam_{}_sim_{}_bin_{}_neuron_{}_beta_{}_X_{}.pickle'. \
                    format(r, s, gam, n_sim, n_bin, n_neuron, seed_beta, 101)
                nbglm.save(filename_nb)

                # poisson solution
                poisson = POISSON()
                poisson.fit(X_train, y_train, lam=lam)
                filename_poi = 'sod_poisson_r_{}_s_{}_gam_{}_sim_{}_bin_{}_neuron_{}_beta_{}_X_{}.pickle'. \
                    format(r, s, gam, n_sim, n_bin, n_neuron, seed_beta, 101)
                poisson.save(filename_poi)

                # evaluation
                X_test, y_test, y_test_true = sod.simulate(n_sim=n_sim, n_bin=n_bin, n_neuron=n_neuron,
                                                           seed_beta=seed_beta, seed_X=42, s = s, r = r, gam = gam)

                sod_r2_list.append(sod.evaluate(X_test, y_test_true, 'r2'))
                sod_mse_list.append(sod.evaluate(X_test, y_test_true, 'mse'))

                nbglm_r2_list.append(nbglm.evaluate(X_test, y_test_true, 'r2'))
                nbglm_mse_list.append(nbglm.evaluate(X_test, y_test_true, 'mse'))

                poisson_r2_list.append(poisson.evaluate(X_test, y_test_true, 'r2'))
                poisson_mse_list.append(poisson.evaluate(X_test, y_test_true, 'mse'))

            # log results
            filename2 = 'sod_data_r_{}_s_{}_gam_{}_sim_{}_bin_{}_neuron_{}_X_{}_results.csv'.\
                format(r, s, gam, n_sim, n_bin, n_neuron, 42)
            result = dict(sod_r2_list = sod_r2_list, sod_mse_list=sod_mse_list,
                          nbglm_r2_list = nbglm_r2_list, nbglm_mse_list = nbglm_mse_list,
                          poisson_r2_list = poisson_r2_list, poisson_mse_list = poisson_mse_list)

            df = pd.DataFrame(result)
            df.to_csv(filename2)

