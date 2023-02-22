import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Time [ms]', fontsize=14)
ax.set_ylabel('Voltage [mV]', fontsize=14)
fig.suptitle(f'9 EADs profiles', fontsize=14)


# 2) BPS NORMAL
t = time.time()
mod, proto, x = myokit.load('./BPS.mmt')

proto.schedule(5.3, 0.1, 1, 1000, 0)    
sim = myokit.Simulation(mod, proto)
sim.pre(1000*600)
dat_normal = sim.run(1000)
print(f'It took {time.time() - t} s')
t1 = dat_normal['engine.time']
v1 = dat_normal['membrane.v']
t1 = t1.tolist()
v1 = v1.tolist()


time_base = pd.DataFrame(t1, columns=["t"])
voltage_base = pd.DataFrame(v1, columns=["v"])
df = time_base.join(voltage_base, how="outer")
df.to_csv('baseline_bps_data.csv', index=False)



