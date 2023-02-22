import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd


# Create the figure
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Time [ms]', fontsize=14)
ax.set_ylabel('Voltage [mV]', fontsize=14)


# Baseline AP BPS
t11 = time.time()
mod1, proto1, x = myokit.load('./BPS.mmt')

proto1.schedule(5.3, 0.1, 1, 1000, 0)    
sim1 = myokit.Simulation(mod1, proto1)
sim1.pre(1000*600)
dat_normal1 = sim1.run(510)
print(f'It took {time.time() - t11} s')
t = dat_normal1['engine.time']
v = dat_normal1['membrane.v']

# Baseline AP BPS eads
t22 = time.time()
mod2, proto2, x = myokit.load('./BPS_f.mmt')

proto2.schedule(5.3, 0.1, 1, 4000, 0)    
sim2 = myokit.Simulation(mod2, proto2)
sim2.pre(1000*600)
dat_normal2 = sim2.run(510)
print(f'It took {time.time() - t22} s')
t2 = dat_normal2['engine.time']
v2 = dat_normal2['membrane.v']

# Baseline tor-ord
t33 = time.time()
mod3, proto3, x = myokit.load('./tor_ord_endo2.mmt')

proto3.schedule(5.3, 0.1, 1, 1000, 0)    
sim3 = myokit.Simulation(mod3, proto3)
sim3.pre(1000*600)
dat_normal3 = sim3.run(510)
print(f'It took {time.time() - t33} s')
t3 = dat_normal3['engine.time']
v3 = dat_normal3['membrane.v']

# Plot boundaries tor ord
torord = pd.read_csv('torord_physiologicData.csv')
torord = pd.DataFrame(torord)
timetor = torord['t'].values.tolist()
v10 = torord['v_10'].values.tolist()
v25 = torord['v_25'].values.tolist()
v75 = torord['v_75'].values.tolist()
v90 = torord['v_90'].values.tolist()

ax.plot(t, v, 'k--', label = 'Baseline BPS')
ax.plot(t2, v2, 'm--', label = 'Baseline BPS EADs')
ax.plot(t3, v3, 'y--', label = 'Baseline TOR ORD')
ax.plot(timetor, v10, 'r', label = 'v 10 tor-ord')
ax.plot(timetor, v25, 'g', label = 'v 25 tor-ord')
ax.plot(timetor, v75, 'b', label = 'v 75 tor-ord')
ax.plot(timetor, v90, 'c', label = 'v 90 tor-ord')


plt.legend(loc='best')
plt.show()
