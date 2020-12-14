from simulation import simulate_data
from model_op import SOD_optimize, NBGLM_optimize, Poisson_optimize
import pickle

def testing(r, gam, s, n_sim, n_bin, n_neuron, case, solution, partial=1, seed = 42, n_iter=50):
    # simulation parameters
    if case in ['SOD', 'NBGLM']:
        # sods simulation & NB-GLM simulation
        data = simulate_data(n_sim=n_sim, n_bin=n_bin, r=r, gam=gam, s=s, n_neuron=n_neuron, case=case, seed=42, density=0.2)
    elif case in ['POISSON', '2step']:
        # Poisson simulation
        data = simulate_data(n_sim=n_sim, n_bin=n_bin, n_neuron=n_neuron, case=case, seed=42, density=0.2)

    # initial x0 for SODS
    # 50 iteration of random initial
    mse_list = []
    beta_list = []
    op_list = []
    y_estimate_list = []
    if solution == 'sod':
        if partial < 1:
            select_list = []
    for _ in range(n_iter):
        if solution == 'sod':
            op, y_estimate, mse_test, select = SOD_optimize(data, partial, seed = seed)
            if partial < 1:
                select_list.append(select)
        elif solution == 'nbglm':
            op, y_estimate, mse_test = NBGLM_optimize(data)
        elif solution == 'poisson':
            op, y_estimate, mse_test = Poisson_optimize(data)
        if not op.success:
            print(op.message)
        # log op
        op_list.append(op)
        # log beta
        if solution == 'sod':
            beta_list.append(op['x'][-int(n_neuron*partial):])
        else:
            beta_list.append(op['x'][-n_neuron:])
        # log mse of test dataset
        mse_list.append(mse_test)
        print(mse_test)
        y_estimate_list.append(y_estimate)
    if solution == 'sod' and partial < 1:
        result = (mse_list, beta_list, op_list, y_estimate_list, select_list, data)
        filename = case + '_' + solution + '_' + str(r) + 'gam' + str(gam) + 's' + str(s) + 'sim' + str(n_sim) + \
                   'bin' + str(n_bin) + 'neuron' + str(n_neuron) + 'partial' + str(partial) + '.pickle'
    else:
        result = (mse_list, beta_list, op_list, y_estimate_list, data)
        filename = case + '_' + solution + '_' + str(r) + 'gam' + str(gam) + 's' + str(s) + 'sim' + str(n_sim) + \
                   'bin' + str(n_bin) + 'neuron' + str(n_neuron)  + '.pickle'

    with open(filename, 'wb') as f:
        pickle.dump(result, f)