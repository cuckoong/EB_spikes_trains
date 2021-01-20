from brian2 import *
import pandas as pd
import numpy as np
#import brian2genn
#set_device('genn')#, use_GPU= False)
#prefs.devices.genn.cuda_backend.extra_compile_args_nvcc += ['--verbose']
#prefs.devices.genn.path='/home/msc/genn'
#prefs.devices.genn.cuda_backend.cuda_path='/usr/local/cuda-9.0'

if __name__ == '__main__':
    N = 100
    F = 50 * Hz
    input = PoissonGroup(N, F)
    P_neurons = 100

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
    Ce = Synapses(input, P, on_pre="ge += we", delay=10*ms)

    Ci = Synapses(input,  P, on_pre="gi += wi", delay=10*ms)
    Ce.connect('i<80', p=0.2)
    Ci.connect('i>=80', p=0.2)


    mon = SpikeMonitor(input)
    s_mon = SpikeMonitor(P)

    duration = 2500*1000
    run(duration * ms, report='text')

    out, idx = pd.cut([0, duration], bins=int(duration / 100), retbins=True)  # 100ms per bin
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

    np.save('X_100ms_100.npy', X)
    np.save('y_100ms_100.npy', y)
    
    print('done')
