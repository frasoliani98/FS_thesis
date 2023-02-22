import myokit
import matplotlib.pyplot as plt
import time
import numpy as np


def plot_bps():
    t = time.time()
    mod, proto, x = myokit.load('./BPS33.mmt')

    
    
    #mod['D'].set_rhs(100)
    #mod['IKr']['D'].set_rhs(0)
    #print(mod.get('IKr.D').eq())
    mod['IKr']['Kmax'].set_rhs(1e8)
    mod['IKr']['Ku'].set_rhs(1.79e-5)
    mod['IKr']['Vhalf'].set_rhs(-1.147)
    mod['IKr']['halfmax'].set_rhs(548300000)
    mod['extracellular']['cao'].set_rhs(2)
    mod['extracellular']['ko'].set_rhs(5)
    mod['extracellular']['nao'].set_rhs(137)

    mod['multipliers']['i_na_multiplier'].set_rhs(0.3411)
    mod['multipliers']['i_nal_multiplier'].set_rhs(0.3241)
    mod['multipliers']['i_cal_pca_multiplier'].set_rhs(1.4129)
    mod['multipliers']['i_to_multiplier'].set_rhs(0.4533)
    mod['multipliers']['i_kr_multiplier'].set_rhs(0.8998)
    mod['multipliers']['i_ks_multiplier'].set_rhs(0.4015)
    mod['multipliers']['i_k1_multiplier'].set_rhs(1.9379)
    mod['multipliers']['i_NCX_multiplier'].set_rhs(1.2201)
    mod['multipliers']['i_nak_multiplier'].set_rhs(0.6743)
    mod['multipliers']['jrel_multiplier'].set_rhs(0.8059)
    mod['multipliers']['jup_multiplier'].set_rhs(1.4250)

    

    proto.schedule(5.3, 0.1, 1, 4000, 0)
    sim = myokit.Simulation(mod, proto)
    sim.pre(4000*600)
    dat_normal = sim.run(4000)
    print(f'It took {time.time() - t} s')

  

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))


    ax.plot(dat_normal['engine.time'], dat_normal['membrane.v'], linewidth=0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.set_ylabel('Voltage (mV)', fontsize=14)
    plt.show()


plot_bps()