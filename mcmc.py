import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano.tensor as tt
from pymc3 import *
from models import SOD
import theano

az.rcParams['plot.matplotlib.show'] = True

if __name__ == '__main__':
    #
    seed_beta = 12
    n_sim = 50
    n_bin = 500
    n_neuron = 100
    s = 50
    n_sample = n_sim * n_bin
    r = 5
    gam = 7
    density = 0.2

    filename = 'sod_sod_fb_sim_{}_bin{}_r_{}_s_{}_gam_{}_neuron_{}_density_{}_beta_{}_X_{}_10x_sigma_trace.trace'.format(n_sim,
                                                                                                               n_bin, r,
                                                                                                               s, gam,
                                                                                                               n_neuron,
                                                                                                               density,
                                                                                                               seed_beta,
                                                                                                               101)

    x, y, y_true = SOD().simulate(n_sim=n_sim, n_bin=n_bin, n_neuron=n_neuron, density=density, seed_beta=seed_beta,
                                  seed_X=101, s=s, r=r, gam=gam)
    y_mean = y.reshape(n_sim, n_bin)
    y_mean = np.mean(y_mean, axis=0)
    y_mean = np.tile(y_mean, n_sim)

    # data = dict(x=x, y=y)
    x_shared = theano.shared(x)
    y_shared = theano.shared(y)
    y_mean = theano.shared(y_mean)

    with Model() as model:  # model specifications in PyMC3 are wrapped in a with-statement
        # Define priors
        # BoundNormal0 = Bound(Normal, lower=1)
        r = TruncatedNormal("r", mu=5, sigma=10, lower=0)  # , mu=5, sd=1)
        gam = TruncatedNormal("gam", mu=7, sigma=10, lower=0)  # , mu=5, sd=1)
        s = TruncatedNormal("s", mu=50, sigma=10, lower=0)  # , mu=50, sd=1)
        beta0 = Normal("beta0", mu=0, sigma=1)
        beta = TruncatedNormal("beta", mu=0, sigma=1, lower=-1, upper=1, shape=n_neuron)

        # mu
        mu = (gam * pm.math.exp(beta0 + pm.math.dot(x_shared, beta)) + 1) ** (-1 / gam)
        y_op = Deterministic('y_op', r * (1 / mu - 1))

        # phi
        phi = Deterministic('phi', (s*mu+n_sim*r)/(s+n_sim*r+n_sim*y_mean))#, shape=n_sample)

        # Define likelihood
        likelihood = NegativeBinomial("y", alpha=r, mu=r * (1 / phi - 1), observed=y_shared)  # attention
        #Inference!
        idata = sample(1000, cores=4, progressbar=True, chains=4, tune=2000, return_inferencedata=False)

        az.to_netcdf(idata, filename)
        print(az.summary(idata, var_names=['r', 'gam', 's']))
       
    '''
    with model:
        idata = az.from_netcdf(filename)

    # az.plot_trace(idata, var_names=['r','gam','s','beta0'])
    print(az.summary(idata, var_names=['r', 'gam', 's']))
    # print('')
    '''
