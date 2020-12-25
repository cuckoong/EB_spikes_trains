import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano.tensor as tt
from pymc3 import *
from models import SOD

if __name__ == '__main__':
    #
    seed_beta = 12
    n_sim = 50
    n_bin = 500
    n_neuron = 100
    s = 50
    r = 5
    gam = 7
    density = 0.2

    x, y, y_true = SOD().simulate(n_sim=n_sim, n_bin=n_bin, n_neuron=n_neuron, density=density,
                                                            seed_beta=seed_beta, seed_X=101,
                                                            s = s, r = r, gam = gam)

    # data = dict(x=x, y=y)

    with Model() as model:  # model specifications in PyMC3 are wrapped in a with-statement
        # Define priors
        r = Uniform("r", lower=1, upper=100, shape=1)
        gam = Uniform("gam", lower=np.finfo(np.float32).eps, upper=100)
        s = Uniform("s", lower=np.finfo(np.float32).eps, upper=100)

        beta0 = Uniform("beta0", lower=-1, upper=1)
        beta = Uniform("beta", lower=-1, upper=1, shape=n_neuron)

        # mu
        mu = pm.Deterministic('mu', (gam*math.exp(beta0 + math.dot(x,beta))+1)**(-1/gam))
        # phi
        phi = Beta('phi', alpha=s*mu, beta=s*(1-mu))

        # Define likelihood
        likelihood = NegativeBinomial("y", n=r, p=phi, observed=y) # attention

        # Inference!
        trace = sample(100, cores=2)  # draw 3000 posterior samples using NUTS sampling
