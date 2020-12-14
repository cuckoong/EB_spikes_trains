import numpy as np
import scipy.special as sc
from scipy import stats
import scipy
from scipy import optimize
import scipy.sparse as sps

def S0D_objective(x0, y, X):
    r = x0[0]
    beta0 = x0[1]
    gam = x0[2]
    s = x0[3]
    beta = x0[4:]
    mu = (gam * np.exp(beta0 + np.dot(X, beta)) + 1) ** (-1 / gam)
    LL_sum = -1 * np.sum(sc.gammaln(r + s * mu) +
                         sc.gammaln(y + s - s * mu) +
                         sc.gammaln(r + y) +
                         sc.gammaln(s) -
                         sc.gammaln(r + y + s) -
                         sc.gammaln(r) -
                         sc.gammaln(s * mu) -
                         sc.gammaln(s - s * mu)) + \
             np.sum(np.abs(beta))
    return LL_sum


def mse_op(op, data, case='sod'):
    # load simulated data
    X = data['X']
    y_sim = data['y_sim']
    y_true = data['y_true']
    n_neuron = data['n_neuron']
    # load estimated parameters
    if case == 'sod' or case == 'nbglm':
        r_op = op['x'][0]
        beta0_op = op['x'][1]
        gam_op = op['x'][2]
        beta_op = op['x'][-n_neuron:]
        mu_op = (gam_op * np.exp(beta0_op + np.dot(X, beta_op)) + 1) ** (-1 / gam_op)
        y_op = r_op * (1 / mu_op - 1)  # estimated y value
    elif case == 'poisson':
        beta0_op = op['x'][0]
        beta_op = op['x'][-n_neuron:]
        y_op = np.exp(beta0_op + np.dot(X, beta_op))

    return np.mean((y_op - y_true) ** 2), y_op, np.mean((y_op - y_sim) ** 2)


def NBGLM_objective(x0, y, X):
    r = x0[0]
    b0 = x0[1]
    gam = x0[2]
    b = x0[3:]
    mu = (gam * np.exp(b0 + np.dot(X, b)) + 1) ** (-1 / gam)
    LL_sum = -1 * np.sum(sc.gammaln(r + y) - sc.gammaln(r) + r * np.log(mu) + y * np.log(1 - mu))
    LL_sum += np.sum(np.abs(b))
    return LL_sum


def Poisson_objecitve(x0, y, X):
    b0 = x0[0]
    b = x0[1:]
    lam = np.exp(b0 + np.dot(X, b))
    LL_sum = -1 * np.sum(y * np.log(lam) - lam - sc.gammaln(1 + y))
    LL_sum += np.sum(np.abs(b))
    return LL_sum


def Poisson_optimize(data):
    n_neuron = data['n_neuron']
    n_sim = data['n_sim']
    beta_bound = [-1, 1]

    # initial guess
    beta0 = np.random.uniform(-1, 1)
    rvs = stats.uniform(loc=-1, scale=2).rvs
    beta_initial = sps.random(1, n_neuron, density=0.2, data_rvs=rvs).toarray()[0]

    beta0_bnds =  [-np.inf, np.inf]
    bnds = [beta0_bnds]
    bnds.extend([beta_bound] * n_neuron)
    # beta_initial = np.random.normal(-1, 1, (1 + n_neuron))  # sps.random(1, n_neuron, density=0.2, data_rvs=rvs).toarray()[0]
    # beta_initial = [0] * (1 + n_neuron)
    x0 = [beta0]
    x0.extend(beta_initial)

    # optimization using scipy
    # t0 = time.time()
    y_sim = data['y_sim']
    y_sim_train = y_sim[:n_sim, ]
    y_sim_test = y_sim[n_sim:, ]

    op = scipy.optimize.minimize(Poisson_objecitve, x0, args=(y_sim_train, data['X'],),
                                 bounds=bnds, method='SLSQP', options=dict(maxiter=5000))  # need attention
    if not op.success:
        print('not converge in poisson model')

    beta0_op = op['x'][0]
    beta_op = op['x'][-n_neuron:]
    y_op = np.exp(beta0_op + np.dot(data['X'], beta_op))
    return op, y_op, np.mean((y_op - y_sim_test) ** 2)


def SOD_optimize(data, partial=1, seed=1):
    # optimization for SODS
    # load some parameters from data
    n_neuron = data['n_neuron']
    n_sim = data['n_sim']
    # bounds and constraints
    r_bound = [1, np.inf]  # need attention
    beta0_bound = [-1, 1]
    gam_bound = [-np.inf, np.inf]
    s_bound = [0, np.inf]
    beta_bound = [-1, 1]
    #
    bnds_sod = [r_bound, beta0_bound, gam_bound, s_bound]
    bnds_sod.extend([beta_bound] * int(n_neuron*partial))
    r0 = np.random.choice(np.arange(1, 1000), 1)  # need attention
    gam0 = np.random.uniform(0, 1000, 1)
    s0 = np.random.choice(np.arange(1, 1000), 1)  # need attention

    beta0_initial = np.random.uniform(-1, 1, 1)
    beta_initial = np.random.uniform(-1, 1, n_neuron)[0:int(n_neuron*partial)]  # sps.random(1, n_neuron, density=0.2, data_rvs=rvs).toarray()[0]
    x0 = [r0, beta0_initial, gam0, s0]
    x0.extend(beta_initial)

    # optimization using scipy
    # t0 = time.time()
    y_sim = data['y_sim']
    y_sim_train = y_sim[:n_sim, ]
    y_sim_test = y_sim[n_sim:, ]
    if partial < 1:
        np.random.seed(seed)
        select = np.random.choice(n_neuron, int(n_neuron*partial), replace= False)
    elif partial == 1:
        select = np.arange(n_neuron)
    op_sod = scipy.optimize.minimize(S0D_objective, x0, args=(y_sim_train, data['X'][:, select],),
                                     bounds=bnds_sod, method='SLSQP', options=dict(maxiter=5000))  # need attention

    if not op_sod.success:
        print('not converge in sod model')

    # calculate mse
    r_op = op_sod['x'][0]
    beta0_op = op_sod['x'][1]
    gam_op = op_sod['x'][2]
    # s_op = op_sod['x'][3]
    beta_op = op_sod['x'][-int(n_neuron*partial):]
    mu_op = (gam_op * np.exp(beta0_op + np.dot(data['X'][:, select], beta_op)) + 1) ** (-1 / gam_op)
    # theta_op = (n_sim*r_op + s_op*mu_op)/(n_sim*r_op+n_sim*np.mean(y_sim_train, axis=0)+s_op)
    y_op = r_op * (1 / mu_op - 1)  # estimated y value
    return op_sod, y_op, np.mean((y_op - y_sim_train) ** 2), select


def NBGLM_optimize(data):
    # optimization for SODS
    # load some parameters from data
    n_neuron = data['n_neuron']
    n_sim = data['n_sim']
    # bounds and constraints
    r_bound = [1, np.inf]  # need attention
    beta0_bound = [-1, 1]
    gam_bound = [-np.inf, np.inf]
    beta_bound = [-1, 1]
    #
    bnds = [r_bound, beta0_bound, gam_bound]
    bnds.extend([beta_bound] * n_neuron)

    # initial guess
    r0 = np.random.choice(np.arange(1, 1000), 1)  # need attention
    gam0 = np.random.uniform(0, 1000, 1)
    beta0_initial = np.random.uniform(-1, 1, 1)
    beta_initial = np.random.uniform(-1, 1, n_neuron)  # sps.random(1, n_neuron, density=0.2, data_rvs=rvs).toarray()[0]
    x0 = [r0, beta0_initial, gam0]
    x0.extend(beta_initial)
    # x0.extend(beta_initial.tolist())

    # optimization using scipy
    # t0 = time.time()
    y_sim = data['y_sim']
    y_sim_train = y_sim[:n_sim, ]
    y_sim_test = y_sim[n_sim:, ]
    op_nbglm = scipy.optimize.minimize(NBGLM_objective, x0, args=(y_sim_train, data['X'],),
                                       bounds=bnds, method='SLSQP', options=dict(maxiter=5000))  # need attention
    if not op_nbglm.success:
        print('not converge in sod model')

    # calculate mse
    r_op = op_nbglm['x'][0]
    beta0_op = op_nbglm['x'][1]
    gam_op = op_nbglm['x'][2]
    beta_op = op_nbglm['x'][-n_neuron:]
    mu_op = (gam_op * np.exp(beta0_op + np.dot(data['X'], beta_op)) + 1) ** (-1 / gam_op)
    y_op = r_op * (1 / mu_op - 1)  # estimated y value
    return op_nbglm, y_op, np.mean((y_op - y_sim_test) ** 2)

