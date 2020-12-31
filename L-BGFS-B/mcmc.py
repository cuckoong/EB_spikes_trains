import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano.tensor as tt
from pymc3 import *
from models import SOD
import theano


if __name__ == '__main__':
    #
    seed_beta = 12
    n_sim = 50
    n_bin = 500
    n_neuron = 100
    s = 50
    n_sample = n_sim*n_bin
    r = 5
    gam = 7
    density = 0.2

    filename = 'sod_sod_fb_sim_{}_bin{}_r_{}_s_{}_gam_{}_neuron_{}_density_{}_beta_{}_X_{}_trace.trace'.format(n_sim, n_bin,r, s, gam, n_neuron, density, seed_beta, 101)

    #filename = 'sod_sod_fb_sim_50_bin500_r_r ~ Bound-Normal_s_s ~ Bound-Normal_gam_gam ~ Bound-Normal_neuron_100_density_0.2_beta_12_X_101_trace.trace'

    x, y, y_true = SOD().simulate(n_sim=n_sim, n_bin=n_bin, n_neuron=n_neuron, density=density,
                                                            seed_beta=seed_beta, seed_X=101,
                                                            s = s, r = r, gam = gam)
    y_mean = y.reshape(n_sim, n_bin)
    y_mean = np.mean(y_mean, axis=0)
    y_mean = np.tile(y_mean, n_sim)

    # data = dict(x=x, y=y)
    x_shared = theano.shared(x)
    y_shared = theano.shared(y)
    y_mean = theano.shared(y_mean)

    with Model() as model:  # model specifications in PyMC3 are wrapped in a with-statement
        # Define priors
        BoundNormal0 = Bound(Normal, lower=1)
        r = BoundNormal0("r", mu=5, sd=5)
        gam = BoundNormal0("gam", mu=5, sd=5)
        s = BoundNormal0("s", mu=50, sd=5)

        beta0 = Normal("beta0", mu=0, sd=1)
        BoundNormal = Bound(Normal, upper=1, lower=-1)
        beta = BoundNormal("beta", mu=0, sd =5, shape=n_neuron)

        # mu
        mu = Deterministic('mu', (gam*pm.math.exp(beta0 + pm.math.dot(x_shared, beta))+1)**(-1/gam))
        # phi
        phi = Beta('phi', alpha=s*mu+n_sim*r, beta=s*(1-mu)+n_sim*y_mean, shape=n_sample)

        # Define likelihood
        likelihood = NegativeBinomial("y", alpha =r, mu=r*(1/phi-1), observed=y_shared) # attention

        # Inference!
        idata = sample(1000, cores=1, progressbar=True, return_inferencedata=True, tune=2000, chains=4)
     
        az.to_netcdf(idata, filename)

    #with model:
    #    trace = az.from_netcdf(filename)
     # az.plot_trace(trace)
    #print(az.summary(trace))
