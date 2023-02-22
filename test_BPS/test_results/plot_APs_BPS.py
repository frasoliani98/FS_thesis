import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd


# Baseline tor-ord
mod1, proto1, x = myokit.load('./BPS3.mmt')

proto1.schedule(5.3, 0.1, 1, 1000, 0)    
sim1 = myokit.Simulation(mod1, proto1)
sim1.pre(1000 * 600) 
dat_normal3 = sim1.run(1000)
tb = dat_normal3['engine.time']
vb = dat_normal3['membrane.v']

# Best Ind Plot 1

ind_1 = pd.read_excel('Best_ind_noCa_morph1.xlsx')
ind_1 = ind_1.to_dict('index')

mod, proto, x = myokit.load('./BPS3.mmt')
for k, v in ind_1[0].items():
        mod['multipliers'][k].set_rhs(v)

proto.schedule(5.3, 0.1, 1, 1000, 0) 
sim = myokit.Simulation(mod,proto)
sim.pre(1000 * 600) 
dat = sim.run(1000)
t1 = np.array(dat['engine.time'])
v1 = np.array(dat['membrane.v'])

# Best Ind Plot 2

ind_2 = pd.read_excel('Best_ind_noCa_morph2.xlsx')
ind_2 = ind_2.to_dict('index')

mod, proto, x = myokit.load('./BPS3.mmt')
for k, v in ind_2[0].items():
        mod['multipliers'][k].set_rhs(v)

proto.schedule(5.3, 0.1, 1, 1000, 0) 
sim = myokit.Simulation(mod,proto)
sim.pre(1000 * 600) 
dat = sim.run(1000)
t2 = np.array(dat['engine.time'])
v2 = np.array(dat['membrane.v'])

# Best Ind Plot 3
ind_3 = pd.read_excel('Best_ind_noCa_morph3.xlsx')
ind_3 = ind_3.to_dict('index')

mod, proto, x = myokit.load('./BPS3.mmt')
for k, v in ind_3[0].items():
        mod['multipliers'][k].set_rhs(v)

proto.schedule(5.3, 0.1, 1, 1000, 0) 
sim = myokit.Simulation(mod,proto)
sim.pre(1000 * 600) 
dat = sim.run(1000)
t3 = np.array(dat['engine.time'])
v3 = np.array(dat['membrane.v'])

# Best Ind Plot 4
ind_4 = pd.read_excel('Best_ind_noCa_morph4.xlsx')
ind_4 = ind_4.to_dict('index')

mod, proto, x = myokit.load('./BPS3.mmt')
for k, v in ind_4[0].items():
        mod['multipliers'][k].set_rhs(v)

proto.schedule(5.3, 0.1, 1, 1000, 0) 
sim = myokit.Simulation(mod,proto)
sim.pre(1000 * 600) 
dat = sim.run(1000)
t4 = np.array(dat['engine.time'])
v4 = np.array(dat['membrane.v'])

# Best Ind Plot 5

ind_5 = pd.read_excel('Best_ind_noCa_morph5.xlsx')
ind_5 = ind_5.to_dict('index')

mod, proto, x = myokit.load('./BPS3.mmt')
for k, v in ind_5[0].items():
        mod['multipliers'][k].set_rhs(v)

proto.schedule(5.3, 0.1, 1, 1000, 0) 
sim = myokit.Simulation(mod,proto)
sim.pre(1000 * 600) 
dat = sim.run(1000)
t5 = np.array(dat['engine.time'])
v5 = np.array(dat['membrane.v'])

# Best Ind Plot 6

ind_6 = pd.read_excel('Best_ind_noCa_morph12.xlsx')
ind_6 = ind_6.to_dict('index')

mod, proto, x = myokit.load('./BPS3.mmt')
for k, v in ind_6[0].items():
        mod['multipliers'][k].set_rhs(v)

proto.schedule(5.3, 0.1, 1, 1000, 0) 
sim = myokit.Simulation(mod,proto)
sim.pre(1000 * 600) 
dat = sim.run(1000)
t6 = np.array(dat['engine.time'])
v6 = np.array(dat['membrane.v'])

# Best Ind Plot 7

ind_7 = pd.read_excel('Best_ind_noCa_morph7.xlsx')
ind_7 = ind_7.to_dict('index')

mod, proto, x = myokit.load('./BPS3.mmt')
for k, v in ind_7[0].items():
        mod['multipliers'][k].set_rhs(v)

proto.schedule(5.3, 0.1, 1, 1000, 0) 
sim = myokit.Simulation(mod,proto)
sim.pre(1000 * 600) 
dat = sim.run(1000)
t7 = np.array(dat['engine.time'])
v7 = np.array(dat['membrane.v'])

# Best Ind Plot 8

ind_8 = pd.read_excel('Best_ind_noCamorph14.xlsx')
ind_8 = ind_8.to_dict('index')

mod, proto, x = myokit.load('./BPS3.mmt')
for k, v in ind_8[0].items():
        mod['multipliers'][k].set_rhs(v)

proto.schedule(5.3, 0.1, 1, 1000, 0) 
sim = myokit.Simulation(mod,proto)
sim.pre(1000 * 600) 
dat = sim.run(1000)
t8 = np.array(dat['engine.time'])
v8 = np.array(dat['membrane.v'])



# Create the figure
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.suptitle(f'Prepacing', fontsize=14)
ax.set_xlabel('Time [ms]', fontsize=14)
ax.set_ylabel('Voltage [mV]', fontsize=14)

ax.plot(tb, vb, 'k--', label = 'Baseline BPS')
ax.plot(t1, v1, 'r', label = 'Best IND 1 BPS')
ax.plot(t2, v2, 'g', label = 'Best IND 2 BPS')
ax.plot(t3, v3, 'b', label = 'Best IND 3 BPS')
ax.plot(t4, v4, 'm', label = 'Best IND 4 BPS')
ax.plot(t5, v5, 'c', label = 'Best IND 5 BPS')
ax.plot(t6, v6, 'y', label = 'Best IND 6 BPS')
ax.plot(t7, v7, color = 'orangered', label = 'Best IND 7 BPS')
ax.plot(t8, v8, color = 'deeppink', label = 'Best IND 8 BPS')



plt.legend(loc='best')
plt.show()