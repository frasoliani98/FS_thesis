import myokit
import matplotlib.pyplot as plt
import time
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Time [ms]', fontsize=14)
ax.set_ylabel('Voltage [mV]', fontsize=14)
#fig.suptitle(f'9 EADs profiles', fontsize=14)

'''
# 1) TOR ORD
t = time.time()
mod, proto, x = myokit.load('./tor_ord_endo2.mmt')

proto.schedule(5.3, 0.1, 1, 1000, 0)    
sim = myokit.Simulation(mod, proto)
sim.pre(1000*600)
dat_normal = sim.run(1000)
print(f'It took {time.time() - t} s')
t1 = dat_normal['engine.time']
v1 = dat_normal['membrane.v']


# 2) BPS NORMAL
t = time.time()
mod, proto, x = myokit.load('./BPS.mmt')

proto.schedule(5.3, 0.1, 1, 1000, 0)    
sim = myokit.Simulation(mod, proto)
sim.pre(1000*600)
dat_normal = sim.run(1000)
print(f'It took {time.time() - t} s')
t2 = dat_normal['engine.time']
v2 = dat_normal['membrane.v']
'''

# 3) BPS EADS
t = time.time()
mod, proto, x = myokit.load('./BPS_f.mmt')

proto.schedule(5.3, 0.1, 1, 1000, 0)    
sim = myokit.Simulation(mod, proto)
sim.pre(1000*600)
dat_normal = sim.run(1000)
print(f'It took {time.time() - t} s')
t3 = dat_normal['engine.time']
v3 = dat_normal['membrane.v']





#ax.plot(t1, v1, 'k', label = 'TOR ORD')

#ax.plot(t2, v2, 'k--', label = 'BPS NORMAL BASELINE')

ax.plot(t3, v3, 'b', label = 'BPS DOFETILIDE BASELINE')

#plt.legend(loc='best')
plt.show()
