import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd




# Baseline tor-ord
mod1, proto1, x = myokit.load('./BPS3_ko54.mmt')

proto1.schedule(5.3, 0.1, 1, 1000, 0)    
sim1 = myokit.Simulation(mod1, proto1)
sim1.pre(1000 * 600) 
dat_normal3 = sim1.run(1000)
t1 = dat_normal3['engine.time']
v1 = dat_normal3['membrane.v']

# Best Ind Plot

ind_1 = pd.read_excel('Best_ind_ko54_ca.xlsx')
ind_1 = ind_1.to_dict('index')

mod, proto, x = myokit.load('./BPS3_ko54.mmt')
for k, v in ind_1[0].items():
        mod['multipliers'][k].set_rhs(v)



proto.schedule(5.3, 0.1, 1, 1000, 0) 
sim = myokit.Simulation(mod,proto)
sim.pre(1000 * 600) 
dat = sim.run(1000)
t2 = np.array(dat['engine.time'])
v2 = np.array(dat['membrane.v'])

# Create the figure
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.suptitle(f'Prepacing', fontsize=14)
ax.set_xlabel('Time [ms]', fontsize=14)
ax.set_ylabel('Voltage [mV]', fontsize=14)

ax.plot(t1, v1, 'k', label = 'Baseline BPS')
ax.plot(t2, v2, 'm--', label = 'Best IND BPS')



plt.legend(loc='best')
plt.show()