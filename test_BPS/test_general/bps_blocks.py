import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd


# Baseline tor-ord
mod1, proto1, x = myokit.load('./BPS33.mmt')

proto1.schedule(5.3, 0.1, 1, 1000, 0)    
sim1 = myokit.Simulation(mod1, proto1)
sim1.pre(1000*600)
dat_normal1 = sim1.run(1000)
t1 = dat_normal1['engine.time']
v1 = dat_normal1['membrane.v']



# ICaL 
mod2, proto2, x = myokit.load('./BPS33.mmt')
mod2['multipliers']['i_cal_pca_multiplier'].set_rhs(8)
proto2.schedule(5.3, 0.1, 1, 1000, 0)    
sim2 = myokit.Simulation(mod2, proto2)
sim2.pre(1000*600)
dat_normal2 = sim2.run(1000)
t2 = dat_normal2['engine.time']
v2 = dat_normal2['membrane.v']


# IKr
mod3, proto3, x = myokit.load('./BPS33.mmt')
mod3['multipliers']['i_kr_multiplier'].set_rhs(0.1)
mod3['multipliers']['i_kb_multiplier'].set_rhs(0.5)
proto3.schedule(5.3, 0.1, 1, 1000, 0)    
sim3 = myokit.Simulation(mod3, proto3)
sim3.pre(1000*600)
dat_normal3 = sim3.run(1000)
t3 = dat_normal3['engine.time']
v3 = dat_normal3['membrane.v']



# Create the figure
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Time [ms]', fontsize=14)
ax.set_ylabel('Voltage [mV]', fontsize=14)

ax.plot(t1, v1, 'k--', label = 'BPS Model')
ax.plot(t2, v2, 'r', label = 'Enhancement ICaL')
ax.plot(t3, v3, 'c', label = 'Block IKr and IKb')



plt.legend(loc='best')
plt.show()