from brian2 import *
from models import SOD, NBGLM, POISSON
import pandas as pd
import brian2genn
set_device('genn')#, use_GPU= False)
prefs.devices.genn.cuda_backend.extra_compile_args_nvcc += ['--verbose']
prefs.devices.genn.path='/home/msc/genn'
prefs.devices.genn.cuda_backend.cuda_path='/usr/local/cuda'

if __name__ == '__main__':
    N = 100
    F = 50 * Hz
    input = PoissonGroup(N, rates=F)

    taum = 20 * ms
    taue = 5 * ms
    taui = 10 * ms
    Vt = -50 * mV
    Vr = -60 * mV
    El = -49 * mV
    sigma = 0.5 * (Vt - Vr)

    eqs = '''
    dv/dt  = (ge+gi-(v-El))/taum+sigma*xi*taum**-0.5 : volt (unless refractory)
    dge/dt = -ge/taue : volt
    dgi/dt = -gi/taui : volt
    '''

    # second part
    P = NeuronGroup(100, eqs, threshold='v>Vt', reset='v = Vr', refractory=5 * ms,
                    # method='exact')
                    method='euler')
    P.v = 'Vr + rand() * (Vt - Vr)'
    P.ge = 0 * mV
    P.gi = 0 * mV

    we = (60 * 0.27 / 10) * mV  # excitatory synaptic weight (voltage)
    wi = (-20 * 4.5 / 10) * mV  # inhibitory synaptic weight
    Ce = Synapses(input, P, model="""p : 1""",
                    on_pre="ge += we*(rand()<p)")

    Ci = Synapses(input, P, model="""p : 1""",
                    on_pre="gi += wi*(rand()<p)")
    Ce.connect('i<80', p=0.2)
    Ci.connect('i>=80', p=0.2)
    Ce.p = 'rand()'
    Ci.p = 'rand()'

    mon = SpikeMonitor(input)
    s_mon = SpikeMonitor(P)

    duration = 2500*1000
    run(duration * ms)

    out, idx = pd.cut([0, duration], bins=int(duration / 50), retbins=True)  # 50ms per bin
    # spike_count_y = pd.cut(s_mon.spike_trains()[0]/second*1000, bins=idx)
    # y = spike_count_y.value_counts().values

    y = []
    for tmp in range(100):
        spike_time = s_mon.spike_trains()[tmp] / second * 1000
        spike_count = pd.cut(spike_time, bins=idx)
        y.append(spike_count.value_counts().values)
    y = np.vstack(y).T

    X = []
    for tmp in range(100):
        spike_time = mon.spike_trains()[tmp] / second * 1000
        spike_count = pd.cut(spike_time, bins=idx)
        X.append(spike_count.value_counts().values)
    X = np.vstack(X).T

    i = 40
    # X_train = X[:int(len(X)/5*4), :]
    # X_test = X[int(len(X)/5*4):-1, :]
    # y_train = y[1:int(len(X)/5*4)+1, i]
    # y_test = y[int(len(X)/5*4)+1:, i]
    #
    X_train = X[:int(len(X) / 5 * 4), :]
    X_test = X[int(len(X) / 5 * 4):, :]

    y_train = y[:int(len(X) / 5 * 4), i]
    y_test = y[int(len(X) / 5 * 4):, i]

    # sod
    sod = SOD()
    sod.fit(X_train, y_train, lam=1)
    # sod.evaluate(X_train, y_train, 'mse')
    mse = sod.evaluate(X_test, y_test, 'mse')
    llsum = -1*sod.objective(sod.op.x, X_test, y_test)
    print(mse)
    print(llsum)


    poisson = POISSON()
    poisson.fit(X_train, y_train, lam=1)
    # poisson.evaluate(X_train, y_train, 'r2')
    mse = poisson.evaluate(X_test, y_test, 'mse')
    llsum = -1*poisson.objective(poisson.op.x, X_test, y_test)
    print(mse)
    print(llsum)

    print('done')