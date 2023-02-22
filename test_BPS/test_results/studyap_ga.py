import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from scipy.signal import find_peaks




# Baseline tor-ord
mod1, proto1, x = myokit.load('./BPS3.mmt')

proto1.schedule(5.3, 0.1, 1, 1000, 0)    
sim1 = myokit.Simulation(mod1, proto1)
#sim1.pre(1000 * 600) 
dat_normal3 = sim1.run(1000)
t1 = dat_normal3['engine.time']
v1 = dat_normal3['membrane.v']

# Best Ind Plot
ind_1 = pd.read_excel('Best_ind_noCa_morph10.xlsx')
ind_1 = ind_1.to_dict('index')
#ind_1 = None
print(ind_1)

def get_ind_data(ind):

    mod, proto, x = myokit.load('./BPS3.mmt')
    if ind is not None:
        for k, v in ind[0].items():
            mod['multipliers'][k].set_rhs(v)

    return mod, proto

def get_normal_sim_dat(ind, IC):
    """
        Runs simulation for a given individual. If the individuals is None,
        then it will run the baseline model

        Returns
        ------
            t, v, cai, i_ion
    """

    mod, proto = get_ind_data(ind)
    proto.schedule(5.3, 0.1, 1, 1000, 0) 
    sim = myokit.Simulation(mod, proto)
    
    if IC == None:
        IC = sim.state()

    sim.set_state(IC)
    dat = sim.run(50000)

    # Get t, v, and cai for second to last AP#######################
    i_stim = dat['stimulus.i_stim']
    peaks = find_peaks(-np.array(i_stim), distance=100)[0]
    start_ap = peaks[-3] #TODO change start_ap to be after stim, not during
    end_ap = peaks[-2]

    t = np.array(dat['engine.time'][start_ap:end_ap])
    t = t - t[0]
    max_idx = np.argmin(np.abs(t-900))
    t = t[0:max_idx]
    end_ap = start_ap + max_idx

    v = np.array(dat['membrane.v'][start_ap:end_ap])
    cai = np.array(dat['intracellular_ions.cai'][start_ap:end_ap])
    i_ion = np.array(dat['membrane.i_ion'][start_ap:end_ap])

    return (t, v, cai, i_ion)

t2, v2, cai, i_ion = get_normal_sim_dat(ind_1, IC=None)

# Create the figure
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#fig.suptitle(f'Prepacing', fontsize=14)
ax.set_xlabel('Time [ms]', fontsize=14)
ax.set_ylabel('Voltage [mV]', fontsize=14)

ax.plot(t1, v1, 'k', label = 'Baseline BPS')
ax.plot(t2, v2, 'm--', label = 'Best IND BPS')



plt.legend(loc='best')
plt.show()