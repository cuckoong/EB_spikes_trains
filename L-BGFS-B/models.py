import numpy as np
import scipy.special as sc
from scipy import stats
import scipy
from scipy import optimize
from scipy.optimize import basinhopping
import scipy.sparse as sps
from scipy.special import digamma
import pickle
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, mean_squared_error


def my_score_fun(clf, X, y):
    x0 = clf.op.x
    possitive_LLsum = -1 * clf.objective(x0, X, y)
    return possitive_LLsum


def optimize_fun(fun, der, x0, X, y, bnds, basin=True):
    if basin:
        op = basinhopping(fun, x0, niter=50, T=10, niter_success=20, disp=True,
                          minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bnds, 'jac': der,
                                            'args': (X, y), 'options': {'maxiter': 5000}})
    else:
        op = scipy.optimize.minimize(fun, x0, args=(X, y), bounds=bnds, method='L-BFGS-B', jac=der,
                                     options={'disp': True, 'maxiter': 5000})
    return op


class SOD(BaseEstimator):
    def __init__(self, lam=1, alpha=0.5, op=None):
        self.type = 'SOD'
        self.lam = lam
        self.alpha = alpha
        self.op = op

    def simulate(self, n_sim, n_bin, n_neuron, seed_beta=42, seed_X=42, s=None, r=None, gam=None, density=0.2,
                 get_y_true=True):
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

    def get_params(self, deep=True):
        return {'lam': self.lam, 'op': self.op}

    def objective(self, x0, X, y):
        r = x0[0]
        gam = x0[1]
        s = x0[2]
        beta0 = x0[3]
        beta = x0[4:]
        mu = (gam * np.exp(beta0 + np.dot(X, beta)) + 1) ** (-1 / gam)
        LL_sum = -1 * np.sum(sc.gammaln(r + s * mu) +
                             sc.gammaln(y + s - s * mu + np.finfo(np.float32).eps) +
                             sc.gammaln(r + y) +
                             sc.gammaln(s) -
                             sc.gammaln(r + y + s) -
                             sc.gammaln(r) -
                             sc.gammaln(s * mu + np.finfo(np.float32).eps) -
                             sc.gammaln(s - s * mu + np.finfo(np.float32).eps))
        LL_sum += self.lam * (self.alpha * np.sum(np.abs(beta)) + 1 / 2 * (1 - self.alpha) * np.sum(beta ** 2))
        return LL_sum

    def der(self, x0, X, y):
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

        A = digamma(r + s * mu) - digamma(s * mu + np.finfo(np.float32).eps)
        B = digamma(y + s - s * mu + np.finfo(np.float32).eps) - digamma(s - s * mu + np.finfo(np.float32).eps)
        C = digamma(s) - digamma(r + y + s)

        der[0] = -1 * np.sum(digamma(r + s * mu) + digamma(r + y) - digamma(r + y + s) - digamma(r))
        der[1] = -1 * np.sum(der_mu_over_gam * (A - B))
        der[2] = -1 * np.sum(A * mu + B * (1 - mu) + C)
        der[3] = -1 * s * np.sum(der_mu_over_beta0 * (A - B))
        der[4:] = -1 * s * np.sum(der_mu_over_beta * (A - B).reshape(-1, 1), axis=0) + \
                  self.lam * (self.alpha * np.sign(beta) + (1 - self.alpha) * beta)
        return der

    def fit(self, X, y):
        n_neuron = X.shape[1]
        # bounds
        r_bound = [1, np.inf]  # need attention
        gam_bound = [np.finfo(np.float32).eps, np.inf]  # [1, np.inf]
        s_bound = [1, np.inf]
        beta0_bound = [None, None]
        beta_bound = [-1, 1]
        bnds = [r_bound, gam_bound, s_bound, beta0_bound]
        bnds.extend([beta_bound] * X.shape[1])

        # initials
        r0 = np.random.uniform(1, 100, 1)[0]
        gam0 = np.random.uniform(np.finfo(np.float32).eps, 100, 1)[0]
        s0 = np.random.uniform(1, 100, 1)[0]

        beta0_initial = np.random.uniform(-1, 1, 1)[0]
        beta_initial = np.random.uniform(-1, 1, n_neuron)[0:n_neuron]
        x0 = [r0, gam0, s0, beta0_initial]
        x0.extend(beta_initial)

        # con = (
        #     {'type': 'ineq',
        #      'fun':lambda x, X, y, lam: (np.exp(x[1]) * np.exp(x[3] + np.dot(X, x[4:])) + 1) ** (-1 / np.exp(x[1])) - 1e-9,
        #      'args': (X, y, lam)},
        #
        #     {'type':'ineq',
        #      'fun': lambda x, X, y, lam: 1-(np.exp(x[1]) * np.exp(x[3] + np.dot(X, x[4:])) + 1) ** (-1 / np.exp(x[1]))-1e-9,
        #      'args': (X, y, lam)},
        #
        #     {'type': 'ineq',
        #      'fun': lambda x : x[0] - 1e-9 - 1},
        #
        #     {'type': 'ineq',
        #      'fun': lambda x : x[2] - 1e-9 - 1}
        # )

        # optimization using scipy
        self.op = optimize_fun(self.objective, self.der, x0, X, y, bnds)

        return self

    def predict(self, X):
        op = self.op
        r_op = op['x'][0]
        gam_op = op['x'][1]
        # s_op = np.exp(op['x'][2])
        beta0_op = op['x'][3]
        beta_op = op['x'][4:]
        mu_op = (gam_op * np.exp(beta0_op + np.dot(X, beta_op)) + 1) ** (-1 / gam_op)
        y_op = r_op * (1 / mu_op - 1)  # estimated y value
        return y_op

    def evaluate(self, X, y_true, metrics='r2'):
        y_op = self.predict(X)
        if metrics == 'mse':
            mse = mean_squared_error(y_true, y_op)
            # print(mse)
            return mse
        elif metrics == 'r2':
            r2 = r2_score(y_true, y_op)
            # print(r2)
            return r2

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


class NBGLM(BaseEstimator):
    def __init__(self, lam=1, alpha=0.5, op=None):
        self.type = 'NBGLM'
        self.lam = lam
        self.alpha = alpha
        self.op = op

    def simulate(self, n_sim, n_bin, n_neuron, seed_beta=42, seed_X=42, r=None, gam=None, density=0.2,
                 get_y_true=True):
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

    def get_params(self, deep=True):
        return {'lam': self.lam, 'op': self.op}

    def objective(self, x0, X, y):
        r = x0[0]
        gam = x0[1]
        beta0 = x0[2]
        beta = x0[3:]
        mu = (gam * np.exp(beta0 + np.dot(X, beta)) + 1) ** (-1 / gam)
        LL_sum = -1 * np.sum(sc.gammaln(r + y) - sc.gammaln(r) +
                             r * np.log(mu + np.finfo(np.float32).eps) +
                             y * np.log(1 - mu + np.finfo(np.float32).eps))
        LL_sum += self.lam * (self.alpha * np.sum(np.abs(beta)) + 1 / 2 * (1 - self.alpha) * np.sum(beta ** 2))
        return LL_sum

    def der(self, x0, X, y):
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

        A = r / mu - y / (1 - mu)

        der[0] = -1 * np.sum(digamma(r + y) - digamma(r) + np.log(mu + np.finfo(np.float32).eps))
        der[1] = -1 * np.sum(A * der_mu_over_gam)
        der[2] = -1 * np.sum(A * der_mu_over_beta0)
        der[3:] = -1 * np.sum(der_mu_over_beta * A.reshape(-1, 1), axis=0) + \
                  self.lam * (self.alpha * np.sign(beta) + (1 - self.alpha) * beta)
        return der

    def fit(self, X, y):
        n_neuron = X.shape[1]
        # bounds
        r_bound = [1, np.inf]  # need attention
        gam_bound = [np.finfo(np.float32).eps, np.inf]
        beta0_bound = [None, None]
        beta_bound = [-1, 1]
        bnds = [r_bound, gam_bound, beta0_bound]
        bnds.extend([beta_bound] * n_neuron)

        # initial guess
        # initials
        r0 = np.random.uniform(1, 100, 1)[0]
        gam0 = np.random.uniform(np.finfo(np.float32).eps, 100, 1)[0]

        beta0_initial = np.random.uniform(-1, 1, 1)[0]
        beta_initial = np.random.uniform(-1, 1, n_neuron)
        x0 = [r0, gam0, beta0_initial]
        x0.extend(beta_initial)

        # optimization using scipy
        self.op = optimize_fun(self.objective, self.der, x0, X, y, bnds)

        return self

    def predict(self, X):
        op = self.op
        r_op = op['x'][0]
        gam_op = op['x'][1]
        beta0_op = op['x'][2]
        beta_op = op['x'][3:]
        mu_op = (gam_op * np.exp(beta0_op + np.dot(X, beta_op)) + 1) ** (-1 / gam_op)
        y_op = r_op * (1 / mu_op - 1)  # estimated y value
        return y_op

    def evaluate(self, X, y, metrics='r2'):
        y_op = self.predict(X)
        if metrics == 'mse':
            mse = mean_squared_error(y, y_op)
            print(mse)
            return mse
        elif metrics == 'r2':
            r2 = r2_score(y, y_op)
            print(r2)
            return r2

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


class POISSON(BaseEstimator):
    def __init__(self, lam=1, alpha=0.5, op=None):
        self.type = 'poisson'
        self.lam = lam
        self.alpha = alpha
        self.op = op

    def simulate(self, n_sim, n_bin, n_neuron, seed_beta=42, seed_X=42, density=0.2, get_y_true=True):
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
        beta0 = np.random.uniform(-1, 1)[0]
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

    def get_params(self, deep=True):
        return {'lam': self.lam, 'op': self.op}


    def objective(self, x0, X, y):
        beta0 = x0[0]
        beta = x0[1:]

        mu = np.exp(beta0 + np.dot(X, beta))
        LL_sum = -1 * np.sum(y * np.log(mu) - mu)
        LL_sum += self.lam * (self.alpha * np.sum(np.abs(beta)) + 1 / 2 * (1 - self.alpha) * np.sum(beta ** 2))
        return LL_sum

    def der(self, x0, X, y):
        beta0 = x0[0]
        beta = x0[1:]

        der = np.zeros_like(x0)
        der[0] = -1 * np.sum(y - np.exp(beta0 + np.dot(X, beta)))
        der[1:] = -1 * np.sum(X * (y - np.exp(beta0 + np.dot(X, beta))).reshape(-1, 1), axis=0) + \
                  self.lam * (self.alpha * np.sign(beta) + (1 - self.alpha) * beta)

        return der

    def fit(self, X, y):
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
        self.op = optimize_fun(self.objective, self.der, x0, X, y, bnds)

    def predict(self, X):
        op = self.op
        beta0_op = op['x'][0]
        beta_op = op['x'][1:]
        y_op = np.exp(beta0_op + np.dot(X, beta_op))  # estimated y value
        return y_op

    def evaluate(self, X, y, metrics='r2'):
        y_op = self.predict(X)
        if metrics == 'mse':
            mse = mean_squared_error(y, y_op)
            print(mse)
            return mse
        elif metrics == 'r2':
            r2 = r2_score(y, y_op)
            print(r2)
            return r2

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
