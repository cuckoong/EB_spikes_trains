import numpy as np
import scipy.special as sc
from scipy import stats
import scipy
from scipy import optimize
from scipy.optimize import basinhopping
import scipy.sparse as sps
from scipy.special import digamma, polygamma
import pickle
import pandas as pd


def optimize_fun(fun, der, x0, X, y, lam, bnds, basin=False):
    if basin:
        op = basinhopping(fun, x0, niter=100, T = 10, disp=True, niter_success = 20,
                          minimizer_kwargs={'method': 'L-BFGS-B', 'bounds':bnds, 'jac': der, 'args':(X, y, lam)},)
    else:
        op = scipy.optimize.minimize(fun, x0, args=(X, y, lam), bounds=bnds, method='L-BFGS-B', jac= der,
                                     options={'disp': True, 'maxiter': 5000})
    return op


class SOD:
    def __init__(self):
        self.type = 'SOD'

    def simulate(self, n_sim, n_bin, n_neuron, seed_beta=42, seed_X=42, s=None, r=None, gam=None, density=0.2,
                 get_y_true = True):
        """
        :param n_sim: simulation trial
        :param n_bin: data length
        :param n_neuron: neuron number
        :param seed_beta: seed for beta
        :param seed_X: seed for X
        :param s: sigma
        :param r: negative binomial shape
        :param gam: link function parameters
        :param density: density for sparse weights
        :return: X, y (simulated data)
        """
        np.random.seed(seed_beta)
        beta0 = np.random.uniform(-1, 1)
        rvs = stats.uniform(loc=-1, scale=2).rvs
        beta = sps.random(1, n_neuron, density=density, data_rvs=rvs).toarray()[0]

        # simulate data X
        np.random.seed(seed_X)
        X = np.random.normal(0.0, 1.0, [n_bin, n_neuron])
        mu = (gam * np.exp(beta0 + np.dot(X, beta)) + 1) ** (-1 / gam)
        y_true = r * (1 / mu - 1)  # true y value

        # simulate y_sim
        rng = np.random.default_rng(42)
        phi = rng.beta(s * mu, s * (1 - mu))
        y = rng.negative_binomial(n=r, p=phi, size=(n_sim, len(phi)))

        X = np.vstack([X for i in range(n_sim)])
        y = np.reshape(y, (n_sim * n_bin,))
        y_true = np.hstack([y_true for i in range(n_sim)])

        if get_y_true:
            return X, y, y_true
        else:
            return X, y

    def objective(self, x0, X, y, lam, alpha=0.5):
        r = x0[0]
        gam = x0[1]
        s = x0[2]
        beta0 = x0[3]
        beta = x0[4:]
        mu = (gam * np.exp(beta0 + np.dot(X, beta)) + 1) ** (-1 / gam)
        LL_sum = -1 * np.sum(sc.gammaln(r + s * mu) +
                             sc.gammaln(y + s - s * mu) +
                             sc.gammaln(r + y) +
                             sc.gammaln(s) -
                             sc.gammaln(r + y + s) -
                             sc.gammaln(r) -
                             sc.gammaln(s * mu) -
                             sc.gammaln(s - s * mu))
        LL_sum += lam * (alpha * np.sum(np.abs(beta)) + 1 / 2 * (1 - alpha) * np.sum(beta ** 2))
        return LL_sum

    def der(self, x0, X, y, lam, alpha=0.5):
        r = x0[0]
        gam = x0[1]
        s = x0[2]
        beta0 = x0[3]
        beta = x0[4:]
        der = np.zeros_like(x0)

        g_est = np.exp(beta0 + np.dot(X, beta))
        mu = (gam * g_est + 1) ** (-1 / gam)
        mu_gam = gam * g_est + 1
        der_mu_over_gam = (np.log(mu_gam) / gam ** 2 - g_est / gam / mu_gam) * mu

        der_mu_over_beta = -1 * X * (g_est * mu / mu_gam).reshape(-1, 1)
        der_mu_over_beta0 = -1 * g_est * (gam * g_est + 1) ** (-1 / gam - 1)

        A = digamma(r + s * mu) - digamma(s * mu)
        B = digamma(y + s - s * mu) - digamma(s - s * mu)
        C = digamma(s) - digamma(r + y + s)

        der[0] = -1 * np.sum(digamma(r + s * mu) + digamma(r + y) - digamma(r + y + s) - digamma(r))
        der[1] = -1 * np.sum(der_mu_over_gam * (A - B))
        der[2] = -1 * np.sum(A * mu + B * (1 - mu) + C)
        der[3] = -1 * s * np.sum(der_mu_over_beta0 * (A - B))
        der[4:] = -1 * s * np.sum(der_mu_over_beta * (A - B).reshape(-1, 1), axis=0) + \
                  lam * (alpha * np.sign(beta) + (1 - alpha) * beta)
        return der

    def fit(self, X, y, lam, partial=1, seed=1):
        n_neuron = X.shape[1]
        # bounds
        r_bound = [1, np.inf]  # need attention
        gam_bound = [1, np.inf]
        s_bound = [1e-9, np.inf]
        beta0_bound = [None, None]
        beta_bound = [-1, 1]
        bnds = [r_bound, gam_bound, s_bound, beta0_bound]
        bnds.extend([beta_bound] * int(n_neuron * partial))

        # initials
        r0 = 3
        gam0 = 3
        s0 = 3
        beta0_initial = np.random.uniform(-1, 1, 1)[0]
        beta_initial = np.random.uniform(-1, 1, n_neuron)[0:int(n_neuron * partial)]
        x0 = [r0, gam0, s0, beta0_initial]
        x0.extend(beta_initial)

        # optimization using scipy
        self.op = optimize_fun(self.objective, self.der,  x0, X, y, lam, bnds, basin=True)

    def predict(self, X):
        op = self.op
        r_op = op['x'][0]
        gam_op = op['x'][1]
        s_op = op['x'][2]
        beta0_op = op['x'][3]
        beta_op = op['x'][4:]
        mu_op = (gam_op * np.exp(beta0_op + np.dot(X, beta_op)) + 1) ** (-1 / gam_op)
        y_op = r_op * (1 / mu_op - 1)  # estimated y value
        return y_op

    def evaluate(self, X, y, metrics):
        from sklearn.metrics import r2_score, mean_squared_error
        y_op = self.predict(X)
        if metrics == 'mse':
            mse = mean_squared_error(y, y_op)
            print(mse)
            return mse
        elif metrics == 'r2':
            r2 = r2_score(y, y_op)
            print(r2)
            return r2

    def save(self,filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

class NBGLM:
    def __init__(self):
        self.type = 'NBGLM'

    def simulate(self, n_sim, n_bin, n_neuron, seed_beta=42, seed_X=42, r=None, gam=None, density=0.2,
                 get_y_true = True):
        """
        :param n_sim: simulation trial
        :param n_bin: data length
        :param n_neuron: neuron number
        :param seed_beta: seed for beta
        :param seed_X: seed for X
        :param s: sigma
        :param r: negative binomial shape
        :param gam: link function parameters
        :param density: density for sparse weights
        :return: X, y (simulated data)
        """
        np.random.seed(seed_beta)
        beta0 = np.random.uniform(-1, 1)
        rvs = stats.uniform(loc=-1, scale=2).rvs
        beta = sps.random(1, n_neuron, density=density, data_rvs=rvs).toarray()[0]

        # simulate data X
        np.random.seed(seed_X)
        X = np.random.normal(0.0, 1.0, [n_bin, n_neuron])
        mu = (gam * np.exp(beta0 + np.dot(X, beta)) + 1) ** (-1 / gam)
        y_true = r * (1 / mu - 1)  # true y value

        # simulate y_sim
        rng = np.random.default_rng(42)
        y = rng.negative_binomial(n=r, p=mu, size=(n_sim, len(mu)))

        X = np.vstack([X for i in range(n_sim)])
        y = np.reshape(y, (n_sim * n_bin,))
        y_true = np.hstack([y_true for i in range(n_sim)])

        if get_y_true:
            return X, y, y_true
        else:
            return X, y

    def objective(self, x0, X, y, lam, alpha=0.5):
        r = x0[0]
        gam = x0[1]
        beta0 = x0[2]
        beta = x0[3:]
        mu = (gam * np.exp(beta0 + np.dot(X, beta)) + 1) ** (-1 / gam)
        LL_sum = -1 * np.sum(sc.gammaln(r + y) - sc.gammaln(r) + r * np.log(mu) + y * np.log(1 - mu))
        LL_sum += lam * (alpha * np.sum(np.abs(beta)) + 1 / 2 * (1 - alpha) * np.sum(beta ** 2))
        return LL_sum

    def der(self, x0, X, y, lam, alpha=0.5):
        r = x0[0]
        gam = x0[1]
        beta0 = x0[2]
        beta = x0[3:]
        der = np.zeros_like(x0)
        g_est = np.exp(beta0 + np.dot(X, beta))
        mu = (gam * g_est + 1) ** (-1 / gam)
        mu_gam = gam * g_est + 1
        der_mu_over_gam = (np.log(mu_gam) / gam ** 2 - g_est / gam / mu_gam) * mu

        der_mu_over_beta = -1 * X * (g_est * mu / mu_gam).reshape(-1, 1)
        der_mu_over_beta0 = -1 * g_est * (gam * g_est + 1) ** (-1 / gam - 1)

        A = r / mu - y / (1-mu)

        der[0] = -1 * np.sum(digamma(r + y) - digamma(r) + np.log(mu))
        der[1] = -1 * np.sum(A * der_mu_over_gam)
        der[2] = -1 * np.sum(A * der_mu_over_beta0)
        der[3:] = -1 * np.sum(der_mu_over_beta * A.reshape(-1,1), axis=0) + \
                  lam * (alpha * np.sign(beta) + (1 - alpha) * beta)
        return der

    def fit(self, X, y, lam):
        n_neuron = X.shape[1]
        # bounds
        r_bound = [1, np.inf]  # need attention
        gam_bound = [1, np.inf]
        beta0_bound = [None, None]
        beta_bound = [-1, 1]
        bnds = [r_bound, gam_bound, beta0_bound]
        bnds.extend([beta_bound] * n_neuron)

        # initial guess
        # initials
        r0 = 3
        gam0 = 3
        beta0_initial = np.random.uniform(-1, 1, 1)[0]
        beta_initial = np.random.uniform(-1, 1, n_neuron)
        x0 = [r0, gam0, beta0_initial]
        x0.extend(beta_initial)

        # optimization using scipy
        self.op = optimize_fun(self.objective, self.der, x0, X, y, lam, bnds, basin=True)

    def predict(self, X):
        op = self.op
        r_op = op['x'][0]
        gam_op = op['x'][1]
        beta0_op = op['x'][2]
        beta_op = op['x'][3:]
        mu_op = (gam_op * np.exp(beta0_op + np.dot(X, beta_op)) + 1) ** (-1 / gam_op)
        y_op = r_op * (1 / mu_op - 1)  # estimated y value
        return y_op

    def evaluate(self, X, y, metrics):
        from sklearn.metrics import r2_score, mean_squared_error
        y_op = self.predict(X)
        if metrics == 'mse':
            mse = mean_squared_error(y, y_op)
            print(mse)
            return mse
        elif metrics == 'r2':
            r2 = r2_score(y, y_op)
            print(r2)
            return r2

    def save(self,filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


class POISSON:
    def __init__(self):
        self.type = 'poisson'

    def simulate(self, n_sim, n_bin, n_neuron, seed_beta=42, seed_X=42, density=0.2, get_y_true = True):
        """
        :param n_sim: simulation trial
        :param n_bin: data length
        :param n_neuron: neuron number
        :param seed_beta: seed for beta
        :param seed_X: seed for X
        :param s: sigma
        :param r: negative binomial shape
        :param gam: link function parameters
        :param density: density for sparse weights
        :return: X, y (simulated data)
        """

        np.random.seed(seed_beta)
        beta0 = np.random.uniform(-1, 1)
        rvs = stats.uniform(loc=-1, scale=2).rvs
        beta = sps.random(1, n_neuron, density=density, data_rvs=rvs).toarray()[0]

        # simulate data X
        np.random.seed(seed_X)
        X = np.random.normal(0.0, 1.0, [n_bin, n_neuron])
        y_true = np.exp(beta0 + np.dot(X, beta))

        # simulate y_sim
        rng = np.random.default_rng(42)
        y = rng.poisson(lam=y_true, size=(n_sim, len(y_true)))

        X = np.vstack([X for _ in range(n_sim)])
        y = np.reshape(y, (n_sim * n_bin,))
        y_true = np.hstack([y_true for _ in range(n_sim)])

        if get_y_true:
            return X, y, y_true
        else:
            return X, y

    def objective(self, x0, X, y, lam, alpha=0.5):
        beta0 = x0[0]
        beta = x0[1:]

        mu = np.exp(beta0 + np.dot(X, beta))
        LL_sum = -1 * np.sum(y * np.log(mu) - mu)
        LL_sum += lam * (alpha * np.sum(np.abs(beta)) + 1 / 2 * (1 - alpha) * np.sum(beta ** 2))
        return LL_sum

    def der(self, x0, X, y, lam, alpha=0.5):
        beta0 = x0[0]
        beta = x0[1:]

        der = np.zeros_like(x0)
        der[0] = -1 * np.sum(y - np.exp(beta0 + np.dot(X, beta)))
        der[1:] = -1 * np.sum(X * (y - np.exp(beta0 + np.dot(X, beta))).reshape(-1,1), axis=0) +\
                  lam * (alpha * np.sign(beta) + (1 - alpha) * beta)

        return der

    def fit(self, X, y, lam):
        n_neuron = X.shape[1]
        # bounds
        beta0_bound = [None, None]
        beta_bound = [-1, 1]
        bnds = [beta0_bound]
        bnds.extend([beta_bound] * n_neuron)

        # initial guess
        # initials
        beta0_initial = np.random.uniform(-1, 1, 1)[0]
        beta_initial = np.random.uniform(-1, 1, n_neuron)
        x0 = [beta0_initial]
        x0.extend(beta_initial)

        # optimization using scipy
        self.op = optimize_fun(self.objective, self.der, x0, X, y, lam, bnds, basin=True)


    def predict(self, X):
        op = self.op
        beta0_op = op['x'][0]
        beta_op = op['x'][1:]
        y_op = np.exp(beta0_op + np.dot(X, beta_op))  # estimated y value
        return y_op

    def evaluate(self, X, y, metrics):
        from sklearn.metrics import r2_score, mean_squared_error
        y_op = self.predict(X)
        if metrics == 'mse':
            mse = mean_squared_error(y, y_op)
            print(mse)
            return mse
        elif metrics == 'r2':
            r2 = r2_score(y, y_op)
            print(r2)
            return r2

    def save(self,filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


if __name__ == '__main__':
    # simulation sod + solution sod
    seed_betas = [12,22,32,42,52,62,72,82,92,102]
    n_sim = 50
    n_bin = 500
    n_neuron = 100
    ss = [5, 10, 50, 100]
    rs = [1,3,5,7,9]
    gams = [3,5,7,9]

    # for s in ss:
    #     r = 5
    #     gam = 7
    #     sod_r2_list = []
    #     sod_mse_list = []
    #     nbglm_r2_list = []
    #     nbglm_mse_list = []
    #     poisson_r2_list = []
    #     poisson_mse_list = []
    #     for seed_beta in seed_betas:
    #         # training
    #         sod = SOD()
    #         X_train, y_train, y_train_true = sod.simulate(n_sim=n_sim, n_bin=n_bin, n_neuron=n_neuron,
    #                                                       seed_beta=seed_beta, seed_X=101, s = s, r = r, gam = gam)
    #         # sod solution
    #         sod.fit(X_train, y_train, lam=10)
    #         filename_sod = 'sod_sod_r_{}_s_{}_gam_{}_sim_{}_bin_{}_neuron_{}_beta_{}_X_{}.pickle'.\
    #             format(r, s, gam,n_sim, n_bin, n_neuron, seed_beta, 101)
    #         sod.save(filename_sod)
    #
    #         # nbglm solution
    #         nbglm = NBGLM()
    #         nbglm.fit(X_train, y_train, lam=10)
    #         filename_nb = 'sod_nbglm_r_{}_s_{}_gam_{}_sim_{}_bin_{}_neuron_{}_beta_{}_X_{}.pickle'. \
    #             format(r, s, gam, n_sim, n_bin, n_neuron, seed_beta, 101)
    #         nbglm.save(filename_nb)
    #
    #         # poisson solution
    #         poisson = POISSON()
    #         poisson.fit(X_train, y_train, lam=10)
    #         filename_poi = 'sod_poisson_r_{}_s_{}_gam_{}_sim_{}_bin_{}_neuron_{}_beta_{}_X_{}.pickle'. \
    #             format(r, s, gam, n_sim, n_bin, n_neuron, seed_beta, 101)
    #         poisson.save(filename_poi)
    #
    #         # evaluation
    #         X_test, y_test, y_test_true = sod.simulate(n_sim=n_sim, n_bin=n_bin, n_neuron=n_neuron,
    #                                                    seed_beta=seed_beta, seed_X=42, s = s, r = r, gam = gam)
    #
    #         sod_r2_list.append(sod.evaluate(X_test, y_test_true, 'r2'))
    #         sod_mse_list.append(sod.evaluate(X_test, y_test_true, 'mse'))
    #
    #         nbglm_r2_list.append(nbglm.evaluate(X_test, y_test_true, 'r2'))
    #         nbglm_mse_list.append(nbglm.evaluate(X_test, y_test_true, 'mse'))
    #
    #         poisson_r2_list.append(poisson.evaluate(X_test, y_test_true, 'r2'))
    #         poisson_mse_list.append(poisson.evaluate(X_test, y_test_true, 'mse'))
    #     # log results
    #     filename2 = 'sod_data_r_{}_s_{}_gam_{}_sim_{}_bin_{}_neuron_{}_X_{}_results.csv'.\
    #         format(r, s, gam, n_sim, n_bin, n_neuron, 42)
    #     result = dict(sod_r2_list = sod_r2_list, sod_mse_list=sod_mse_list,
    #                   nbglm_r2_list = nbglm_r2_list, nbglm_mse_list = nbglm_mse_list,
    #                   poisson_r2_list = poisson_r2_list, poisson_mse_list = poisson_mse_list)
    #
    #     df = pd.DataFrame(result)
    #     df.to_csv(filename2)

    # for r in rs:
    #     s = 50
    #     gam = 7
    #     sod_r2_list = []
    #     sod_mse_list = []
    #     nbglm_r2_list = []
    #     nbglm_mse_list = []
    #     poisson_r2_list = []
    #     poisson_mse_list = []
    #     for seed_beta in seed_betas:
    #         # training
    #         sod = SOD()
    #         X_train, y_train, y_train_true = sod.simulate(n_sim=n_sim, n_bin=n_bin, n_neuron=n_neuron,
    #                                                       seed_beta=seed_beta, seed_X=101, s=s, r=r, gam=gam)
    #         # sod solution
    #         sod.fit(X_train, y_train, lam=10)
    #         filename_sod = 'sod_sod_r_{}_s_{}_gam_{}_sim_{}_bin_{}_neuron_{}_beta_{}_X_{}.pickle'. \
    #             format(r, s, gam, n_sim, n_bin, n_neuron, seed_beta, 101)
    #         sod.save(filename_sod)
    #
    #         # nbglm solution
    #         nbglm = NBGLM()
    #         nbglm.fit(X_train, y_train, lam=10)
    #         filename_nb = 'sod_nbglm_r_{}_s_{}_gam_{}_sim_{}_bin_{}_neuron_{}_beta_{}_X_{}.pickle'. \
    #             format(r, s, gam, n_sim, n_bin, n_neuron, seed_beta, 101)
    #         nbglm.save(filename_nb)
    #
    #         # poisson solution
    #         poisson = POISSON()
    #         poisson.fit(X_train, y_train, lam=10)
    #         filename_poi = 'sod_poisson_r_{}_s_{}_gam_{}_sim_{}_bin_{}_neuron_{}_beta_{}_X_{}.pickle'. \
    #             format(r, s, gam, n_sim, n_bin, n_neuron, seed_beta, 101)
    #         poisson.save(filename_poi)
    #
    #         # evaluation
    #         X_test, y_test, y_test_true = sod.simulate(n_sim=n_sim, n_bin=n_bin, n_neuron=n_neuron,
    #                                                    seed_beta=seed_beta, seed_X=42, s=s, r=r, gam=gam)
    #
    #         sod_r2_list.append(sod.evaluate(X_test, y_test_true, 'r2'))
    #         sod_mse_list.append(sod.evaluate(X_test, y_test_true, 'mse'))
    #
    #         nbglm_r2_list.append(nbglm.evaluate(X_test, y_test_true, 'r2'))
    #         nbglm_mse_list.append(nbglm.evaluate(X_test, y_test_true, 'mse'))
    #
    #         poisson_r2_list.append(poisson.evaluate(X_test, y_test_true, 'r2'))
    #         poisson_mse_list.append(poisson.evaluate(X_test, y_test_true, 'mse'))
    #     # log results
    #     filename2 = 'sod_data_r_{}_s_{}_gam_{}_sim_{}_bin_{}_neuron_{}_X_{}_results.csv'. \
    #         format(r, s, gam, n_sim, n_bin, n_neuron, 42)
    #     result = dict(sod_r2_list=sod_r2_list, sod_mse_list=sod_mse_list,
    #                   nbglm_r2_list=nbglm_r2_list, nbglm_mse_list=nbglm_mse_list,
    #                   poisson_r2_list=poisson_r2_list, poisson_mse_list=poisson_mse_list)
    #
    #     df = pd.DataFrame(result)
    #     df.to_csv(filename2)

    for gam in gams:
        s = 50
        r = 5
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
                                                          seed_beta=seed_beta, seed_X=101, s=s, r=r, gam=gam)
            # sod solution
            sod.fit(X_train, y_train, lam=10)
            filename_sod = 'sod_sod_r_{}_s_{}_gam_{}_sim_{}_bin_{}_neuron_{}_beta_{}_X_{}.pickle'. \
                format(r, s, gam, n_sim, n_bin, n_neuron, seed_beta, 101)
            sod.save(filename_sod)

            # nbglm solution
            nbglm = NBGLM()
            nbglm.fit(X_train, y_train, lam=10)
            filename_nb = 'sod_nbglm_r_{}_s_{}_gam_{}_sim_{}_bin_{}_neuron_{}_beta_{}_X_{}.pickle'. \
                format(r, s, gam, n_sim, n_bin, n_neuron, seed_beta, 101)
            nbglm.save(filename_nb)

            # poisson solution
            poisson = POISSON()
            poisson.fit(X_train, y_train, lam=10)
            filename_poi = 'sod_poisson_r_{}_s_{}_gam_{}_sim_{}_bin_{}_neuron_{}_beta_{}_X_{}.pickle'. \
                format(r, s, gam, n_sim, n_bin, n_neuron, seed_beta, 101)
            poisson.save(filename_poi)

            # evaluation
            X_test, y_test, y_test_true = sod.simulate(n_sim=n_sim, n_bin=n_bin, n_neuron=n_neuron,
                                                       seed_beta=seed_beta, seed_X=42, s=s, r=r, gam=gam)

            sod_r2_list.append(sod.evaluate(X_test, y_test_true, 'r2'))
            sod_mse_list.append(sod.evaluate(X_test, y_test_true, 'mse'))

            nbglm_r2_list.append(nbglm.evaluate(X_test, y_test_true, 'r2'))
            nbglm_mse_list.append(nbglm.evaluate(X_test, y_test_true, 'mse'))

            poisson_r2_list.append(poisson.evaluate(X_test, y_test_true, 'r2'))
            poisson_mse_list.append(poisson.evaluate(X_test, y_test_true, 'mse'))
        # log results
        filename2 = 'sod_data_r_{}_s_{}_gam_{}_sim_{}_bin_{}_neuron_{}_X_{}_results.csv'. \
            format(r, s, gam, n_sim, n_bin, n_neuron, 42)
        result = dict(sod_r2_list=sod_r2_list, sod_mse_list=sod_mse_list,
                      nbglm_r2_list=nbglm_r2_list, nbglm_mse_list=nbglm_mse_list,
                      poisson_r2_list=poisson_r2_list, poisson_mse_list=poisson_mse_list)

        df = pd.DataFrame(result)
        df.to_csv(filename2)



