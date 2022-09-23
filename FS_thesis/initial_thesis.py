# 1) Use Myokit and python to run an action potential 
# using the ToR-ORd model

import myokit
import matplotlib.pyplot as plt

mod, proto, x = myokit.load('./tor_ord_endo.mmt')
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(5)
#mod['multipliers']['i_ks_multiplier'].set_rhs(0.1)
#mod['IKr']['IKr'].set_rhs(0.15)

#proto.schedule(3, 10, 1, 0)
sim = myokit.Simulation(mod, proto)
sim.pre(100*1000) #to see 100 AP before mine
dat_normal = sim.run(1000)


# Plot
fig, axs = plt.subplots(2, 1)
axs[0].plot(dat_normal['engine.time'], dat_normal['membrane.v'])
axs[1].plot(dat_normal['engine.time'], dat_normal['membrane.i_ion'])

# To remove the up and right part in the graph
for ax in axs:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

# Set Labels
axs[1].set_xlabel('Time (ms)', fontsize=14)
axs[0].set_ylabel('Voltage (mV)', fontsize=14)
axs[1].set_ylabel('Current (pA/pF)', fontsize=14)
axs[1].set_ylim(-1, 4)


plt.show()