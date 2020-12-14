from itertools import product
from multiprocessing import Pool, cpu_count
from running_op import testing

if __name__ == '__main__':
    # start "number of cores" processes
    r = 5
    gam = 7
    s = 50
    n_sim = 50
    n_bin = 500
    n_neuron = 100
    case = 'SOD'
    solutions = ['sod']
    n_iter = 20
    seeds = [101] #, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
    partials = [0.8] #, 0.5, 0.3]
    for partial in partials:
        for solution in solutions:
            for seed in seeds:
                testing(r=r, gam=gam, s=s, n_sim=n_sim, n_bin=n_bin, n_neuron=n_neuron, case=case, solution=solution,
                        partial=partial, seed=seed, n_iter=n_iter)
        # fb_testing(5,7,50,10,500,100,2)
        # iter_list = product(r, gam, s, n_sim, n_bin, n_neuron, case, solution, n_iter)
        # pool = Pool(processes=5)
        # pool.starmap(testing, iter_list)
        # pool.close()