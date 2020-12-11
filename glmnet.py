from itertools import product
import numpy as np
import scipy.sparse as sps
import scipy
from scipy import optimize
from scipy import stats
import scipy.special as sc
import pickle
import pystan
from multiprocessing import Pool, cpu_count


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
    if case == 'linear':
        y_true = np.exp(beta0 + np.dot(X, beta))
        rng = np.random.default_rng(42)
        y_sim = rng.poisson(lam=y_true, size=(2 * n_sim, len(y_true)))
        # data record
        data = dict(X=X, y_sim=y_sim, y_true=y_true, beta0=beta0, beta=beta,
                    n_bin=n_bin, n_neuron=n_neuron,
                    # r=r, gam=gam,
                    n_sim=n_sim, sparse_density=density)
    else:
        mu = (gam * np.exp(beta0 + np.dot(X, beta)) + 1) ** (-1 / gam)
        y_true = r * (1 / mu - 1)  # true y value
        # simulate y_sim
        rng = np.random.default_rng(42)
        if case == 'sod':
            phi = rng.beta(s * mu, s * (1 - mu))
            y_sim = rng.negative_binomial(n=r, p=phi, size=(2 * n_sim, len(phi)))
        elif case == 'nbglm':
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

# initial guess
def initial_guess(n_neuron):
    r0 = np.random.choice(np.arange(1, 100), 1)[0]  # need attention
    gam0 = np.random.uniform(1, 10, 1)[0]
    s0 = np.random.choice(np.arange(1, 1000), 1)[0]  # need attention
    beta0_initial = 0 #np.random.uniform(-1, 1, 1)[0]
    beta_initial = [0]*n_neuron #np.random.uniform(-1, 1, n_neuron) #np.random.uniform(-1, 1, n_neuron).tolist()  #
    return dict(r=r0, b0 = beta0_initial, b =beta_initial, s= s0, gam = gam0)

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


def SOD_optimize(data):
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
    bnds_sod.extend([beta_bound] * n_neuron)

    r0 = np.random.choice(np.arange(1, 1000), 1)  # need attention
    gam0 = np.random.uniform(0, 1000, 1)
    s0 = np.random.choice(np.arange(1, 1000), 1)  # need attention

    beta0_initial = np.random.uniform(-1, 1, 1)
    beta_initial = np.random.uniform(-1, 1, n_neuron)  # sps.random(1, n_neuron, density=0.2, data_rvs=rvs).toarray()[0]
    x0 = [r0, beta0_initial, gam0, s0]
    x0.extend(beta_initial)

    # optimization using scipy
    # t0 = time.time()
    y_sim = data['y_sim']
    y_sim_train = y_sim[:n_sim, ]
    y_sim_test = y_sim[n_sim:, ]
    op_sod = scipy.optimize.minimize(S0D_objective, x0, args=(y_sim_train, data['X'],),
                                     bounds=bnds_sod, method='SLSQP', options=dict(maxiter=5000))  # need attention

    if not op_sod.success:
        print('not converge in sod model')

    # calculate mse
    r_op = op_sod['x'][0]
    beta0_op = op_sod['x'][1]
    gam_op = op_sod['x'][2]
    s_op = op_sod['x'][3]
    beta_op = op_sod['x'][-n_neuron:]
    mu_op = (gam_op * np.exp(beta0_op + np.dot(data['X'], beta_op)) + 1) ** (-1 / gam_op)
    theta_op = (n_sim*r_op + s_op*mu_op)/(n_sim*r_op+n_sim*np.mean(y_sim_train, axis=0)+s_op)
    y_op = r_op * (1 / theta_op - 1)  # estimated y value
    return op_sod, y_op, np.mean((y_op - y_sim_test) ** 2)


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


def testing(r, gam, s, n_sim, n_bin, n_neuron, n_iter=50):
    # simulation parameters
    # simulation

    # '''
    # sods simulation
    # data = simulate_data(n_sim=n_sim, n_bin=n_bin, r = r, gam = gam, s=s, n_neuron=n_neuron, case='sod', seed=42, density=0.2)

    # NB-GLM simulation
    # data = simulate_data(n_sim=n_sim, n_bin=n_bin, r=r, gam=gam, s=s, n_neuron=n_neuron, case='nbglm', seed=42, density=0.2)

    # Poisson simulation
    data = simulate_data(n_sim=n_sim, n_bin=n_bin, n_neuron=n_neuron, case='linear', seed=42, density=0.2)

    # initial x0 for SODS
    # 50 iteration of random initial
    mse_list = []
    beta_list = []
    op_list = []
    y_estimate_list = []
    for _ in range(n_iter):
        op, y_estimate, mse_test = SOD_optimize(data)
        # op, y_estimate, mse_test = NBGLM_optimize(data)
        # op, y_estimate, mse_test = Poisson_optimize(data)
        if not op.success:
            print(op.message)
        # log op
        op_list.append(op)
        # log beta
        beta_list.append(op['x'][-n_neuron:])
        # log mse of test dataset
        mse_list.append(mse_test)
        print(mse_test)
        y_estimate_list.append(y_estimate)

    result = (mse_list, beta_list, op_list, y_estimate_list, data)
    filename = 'Poisson_poisson_' + str(r) + 'gam' + str(gam) + 's' + str(s) + \
               'sim' + str(n_sim) + 'bin' + str(n_bin) + 'neuron' + str(n_neuron) + '.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(result, f)

def fb_testing(r, gam, s, n_sim, n_bin, n_neuron, n_iter):
    # get simulation data
    model = """
    data {
      int<lower=1> N_bin;    // rows of data, bins 
      int<lower=1> N_sim;    // columns of data, trials
      int<lower=1> N_neuron;    // rows of neurons
      real X[N_neuron, N_bin];       // M neurons, N bins
      int<lower=0> y[N_sim,N_bin]; // response
      real y_mean[N_bin];           // mean of y train
    }

    parameters {
      real<lower=1, upper=100> r; // neg. binomial mean parameter. [1, np.inf] 
      real<lower=-1,upper=1>  b0;  // intercept. [-1, 1] 
      real<lower=-1,upper=1>  b[N_neuron];  // slopes. [-1, 1] 
      real<lower=1> s;  //sigma. [1, np.inf]
      real<lower=1, upper=10> gam; //gamma. [-np.inf, np.inf]
    }

    transformed parameters{
      real mu[N_bin]; 
      real phi[N_bin];

      for(i in 1:N_bin){
        mu[i] = (gam*exp(b0 + dot_product(X[,i],b))+1)^(-1/gam);
        phi[i] = (N_sim*r + s*mu[i])/(N_sim*r+N_sim*y_mean[i]+s);
        }
    }

    model {   
      // data model:
      for(i in 1:N_bin){
        for(j in 1:N_sim){
         y[j,i] ~ neg_binomial(r, phi[i]/(1-phi[i]));
        }
      }
    }
    generated quantities {
      vector[N_bin] y_new;
      for (n in 1:N_bin){
        y_new[n] = r*(1/phi[n]-1);
        }
    }
    """

    sm = pystan.StanModel(model_code=model, verbose=0)
    data = simulate_data(n_sim, n_bin, n_neuron, case ='sod', seed=42, s=s, r=r, gam=gam, density=0.2)

    X = data['X'].T
    n_sim = data['n_sim']
    y_sim = data['y_sim']
    y_sim_train = y_sim[:n_sim, ]
    y_sim_test = y_sim[n_sim:, ]
    y_mean = np.mean(y_sim_train, axis=0)
    y_true = data['y_true']

    # MCMC model
    fit = sm.sampling(data=dict(N_bin=n_bin, N_neuron=n_neuron, N_sim=n_sim, y=y_sim_train,
                                X=X, y_mean = y_mean),
                      init= [initial_guess(n_neuron) for i in range(n_iter)],
                      seed=42,
                      n_jobs=4,
                      chains=n_iter,
                      verbose=True)
    y_op = fit['y_new']
    b_op = fit['b']
    b0_op = fit['b0']
    r_op = fit['r']
    gam_op = fit['gam']
    s_op = fit['s']

    fit_op = (y_op, b_op, b0_op, r_op, gam_op, s_op)
    y_est = np.mean(y_op, axis=0)
    beta_list = np.mean(b_op, axis=0)
    mse_list = np.mean((y_sim_test-y_est)**2)

    result = (mse_list, beta_list, fit_op, y_est, data)

    filename = 'SODS_fb_' + str(r) + 'gam' + str(gam) + 's' + str(s) + \
               'sim' + str(n_sim) + 'bin' + str(n_bin) + 'neuron' + str(n_neuron) + '.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    # start "number of cores" processes
    r = [5]
    gam = [7]
    s = [50]
    n_sim = [10, 100]
    n_bin = [100, 500, 1000]
    n_neuron = [100]
    n_iter = [5]
    # testing(r, gam, s, n_sim, n_bin, n_neuron, n_iter)
    # fb_testing(5,7,50,10,500,100,2)
    # iter_list = product(r, gam, s, n_sim, n_bin, n_neuron, n_iter)
    # pool = Pool(processes=5)
    pool.starmap(fb_testing, iter_list)
    # pool.starmap(testing, iter_list)
    # pool.close()
