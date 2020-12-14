import numpy as np
from scipy import stats
import scipy.sparse as sps

def simulate_data(n_sim, n_bin, n_neuron, case, seed=42, s=None, r=None, gam=None, density=0.2):
    # NB-GLM simulation
    # sample a sparse model
    np.random.seed(42)
    beta0 = np.random.uniform(-1, 1)
    rvs = stats.uniform(loc=-1, scale=2).rvs
    beta = sps.random(1, n_neuron, density=density, data_rvs=rvs).toarray()[0]

    # simulate data X
    np.random.seed(seed)
    X = np.random.normal(0.0, 1.0, [n_bin, n_neuron])
    if case == 'POISSON':
        y_true = np.exp(beta0 + np.dot(X, beta))
        rng = np.random.default_rng(42)
        y_sim = rng.poisson(lam=y_true, size=(2 * n_sim, len(y_true)))
        # data record
        data = dict(X=X, y_sim=y_sim, y_true=y_true, beta0=beta0, beta=beta,
                    n_bin=n_bin, n_neuron=n_neuron,
                    # r=r, gam=gam,
                    n_sim=n_sim, sparse_density=density)
    elif case == '2step':
        np.random.seed(seed)
        X = np.random.normal(0.0, 1.0, [n_bin+1, n_neuron])
        np.random.seed(84)
        beta0 = np.random.uniform(-1, 1)
        rvs = stats.uniform(loc=-1, scale=2).rvs
        beta = sps.random(2, n_neuron, density=density, data_rvs=rvs).toarray()
        y_true = np.exp(beta0 + np.dot(X, beta[0,:])[:-1] + np.dot(X, beta[1,:])[1:])
        rng = np.random.default_rng(42)
        y_sim = rng.poisson(lam=y_true, size=(2 * n_sim, len(y_true)))
        # data record
        data = dict(X=X[1:,:], y_sim=y_sim, y_true=y_true, beta0=beta0, beta=beta,
                    n_bin=n_bin, n_neuron=n_neuron,
                    # r=r, gam=gam,
                    n_sim=n_sim, sparse_density=density)
    elif case in ['SOD', 'NBGLM']:
        mu = (gam * np.exp(beta0 + np.dot(X, beta)) + 1) ** (-1 / gam)
        y_true = r * (1 / mu - 1)  # true y value
        # simulate y_sim
        rng = np.random.default_rng(42)
        if case == 'SOD':
            phi = rng.beta(s * mu, s * (1 - mu))
            y_sim = rng.negative_binomial(n=r, p=phi, size=(2 * n_sim, len(phi)))
        elif case == 'NBGLM':
            y_sim = rng.negative_binomial(n=r, p=mu, size=(2 * n_sim, len(mu)))

        # data record
        data = dict(X=X, y_sim=y_sim, y_true=y_true, beta0=beta0, beta=beta,
                    n_bin=n_bin, n_neuron=n_neuron, r=r, gam=gam,
                    n_sim=n_sim, sparse_density=density)
        if s is not None:
            data['s'] = s
    print('simulation mse')
    print(np.mean((y_sim - y_true) ** 2))
    return data