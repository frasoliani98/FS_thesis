import myokit
import matplotlib.pyplot as plt
import time
import numpy as np


def plot_bps():
    t = time.time()
    mod, proto, x = myokit.load('./BPS.mmt')
    proto.schedule(5.3, 0.1, 1, 1000, 0)
    sim = myokit.Simulation(mod, proto)
    sim.pre(1000*500)
    dat_normal = sim.run(1000)
    print(f'It took {time.time() - t} s')

    #mod['multipliers']['i_cal_pca_multiplier'].set_rhs(8)
    #sim = myokit.Simulation(mod, proto)
    #dat_ical_scale = sim.run(10000)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    #t = np.concatenate((dat_normal['engine.time'], np.array(dat_ical_scale['engine.time']) + 1000))
    #v = np.concatenate((dat_normal['membrane.v'], dat_ical_scale['membrane.v']))

    ax.plot(dat_normal['engine.time'], dat_normal['membrane.v'], linewidth=0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.set_ylabel('Voltage (mV)', fontsize=14)
    plt.show()


plot_bps()