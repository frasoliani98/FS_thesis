import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd




# Baseline tor-ord
modb, protob, x = myokit.load('./BPS3.mmt')
modb['IKr']['D'].set_state_value(100)
modb['IKr']['Kmax'].set_rhs(1e8)
modb['IKr']['Ku'].set_rhs(1.79e-5)
modb['IKr']['Vhalf'].set_rhs(-1.147)
modb['IKr']['halfmax'].set_rhs(548300000)
modb['extracellular']['cao'].set_rhs(2)
modb['extracellular']['ko'].set_rhs(5)
modb['extracellular']['nao'].set_rhs(137)

modb['multipliers']['i_na_multiplier'].set_rhs(0.3411)
modb['multipliers']['i_nal_multiplier'].set_rhs(0.3241)
modb['multipliers']['i_cal_pca_multiplier'].set_rhs(1.4129)
modb['multipliers']['i_to_multiplier'].set_rhs(0.4533)
modb['multipliers']['i_kr_multiplier'].set_rhs(0.8998)
modb['multipliers']['i_ks_multiplier'].set_rhs(0.4015)
modb['multipliers']['i_k1_multiplier'].set_rhs(1.9379)
modb['multipliers']['i_NCX_multiplier'].set_rhs(1.2201)
modb['multipliers']['i_nak_multiplier'].set_rhs(0.6743)
modb['multipliers']['jrel_multiplier'].set_rhs(0.8059)
modb['multipliers']['jup_multiplier'].set_rhs(1.4250)

protob.schedule(5.3, 0.1, 1, 4000, 0)    
sim1 = myokit.Simulation(modb, protob)
sim1.pre(4000 * 50) 
dat_normal3 = sim1.run(4000)
tb = dat_normal3['engine.time']
vb = dat_normal3['membrane.v']

# Best Ind Plot 1

ind_1 = pd.read_excel('Best_ind_noCa_morph1.xlsx')
ind_1 = ind_1.to_dict('index')


mod, proto, x = myokit.load('./BPS3.mmt')
for k, v in ind_1[0].items():
        mod['multipliers'][k].set_rhs(v)
        
mod['IKr']['D'].set_state_value(100)
mod['IKr']['Kmax'].set_rhs(1e8)
mod['IKr']['Ku'].set_rhs(1.79e-5)
mod['IKr']['Vhalf'].set_rhs(-1.147)
mod['IKr']['halfmax'].set_rhs(548300000)
mod['extracellular']['cao'].set_rhs(2)
mod['extracellular']['ko'].set_rhs(5)
mod['extracellular']['nao'].set_rhs(137)

mod['multipliers']['i_na_multiplier'].set_rhs(0.3411)
mod['multipliers']['i_nal_multiplier'].set_rhs(0.3241)
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(1.4129*ind_1[0]['i_cal_pca_multiplier'])
mod['multipliers']['i_to_multiplier'].set_rhs(0.4533)
mod['multipliers']['i_kr_multiplier'].set_rhs(0.8998*ind_1[0]['i_kr_multiplier'])
mod['multipliers']['i_ks_multiplier'].set_rhs(0.4015)
mod['multipliers']['i_k1_multiplier'].set_rhs(1.9379)
mod['multipliers']['i_NCX_multiplier'].set_rhs(1.2201)
mod['multipliers']['i_nak_multiplier'].set_rhs(0.6743)
mod['multipliers']['jrel_multiplier'].set_rhs(0.8059)
mod['multipliers']['jup_multiplier'].set_rhs(1.4250)

proto.schedule(5.3, 0.1, 1, 4000, 0) 
sim = myokit.Simulation(mod,proto)
sim.pre(4000 * 50) 
dat = sim.run(4000)
t1 = np.array(dat['engine.time'])
v1 = np.array(dat['membrane.v'])

# Best Ind Plot 2

ind_2 = pd.read_excel('Best_ind_noCa_morph2.xlsx')
ind_2 = ind_2.to_dict('index')


mod, proto, x = myokit.load('./BPS3.mmt')
for k, v in ind_2[0].items():
        mod['multipliers'][k].set_rhs(v)
        
mod['IKr']['D'].set_state_value(100)
mod['IKr']['Kmax'].set_rhs(1e8)
mod['IKr']['Ku'].set_rhs(1.79e-5)
mod['IKr']['Vhalf'].set_rhs(-1.147)
mod['IKr']['halfmax'].set_rhs(548300000)
mod['extracellular']['cao'].set_rhs(2)
mod['extracellular']['ko'].set_rhs(5)
mod['extracellular']['nao'].set_rhs(137)

mod['multipliers']['i_na_multiplier'].set_rhs(0.3411)
mod['multipliers']['i_nal_multiplier'].set_rhs(0.3241)
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(1.4129*ind_2[0]['i_cal_pca_multiplier'])
mod['multipliers']['i_to_multiplier'].set_rhs(0.4533)
mod['multipliers']['i_kr_multiplier'].set_rhs(0.8998*ind_2[0]['i_kr_multiplier'])
mod['multipliers']['i_ks_multiplier'].set_rhs(0.4015)
mod['multipliers']['i_k1_multiplier'].set_rhs(1.9379)
mod['multipliers']['i_NCX_multiplier'].set_rhs(1.2201)
mod['multipliers']['i_nak_multiplier'].set_rhs(0.6743)
mod['multipliers']['jrel_multiplier'].set_rhs(0.8059)
mod['multipliers']['jup_multiplier'].set_rhs(1.4250)

proto.schedule(5.3, 0.1, 1, 4000, 0) 
sim = myokit.Simulation(mod,proto)
sim.pre(4000 * 50) 
dat = sim.run(4000)
t2 = np.array(dat['engine.time'])
v2 = np.array(dat['membrane.v'])

# Best Ind Plot 3

ind_3 = pd.read_excel('Best_ind_noCa_morph3.xlsx')
ind_3 = ind_3.to_dict('index')


mod, proto, x = myokit.load('./BPS3.mmt')
for k, v in ind_3[0].items():
        mod['multipliers'][k].set_rhs(v)
        
mod['IKr']['D'].set_state_value(100)
mod['IKr']['Kmax'].set_rhs(1e8)
mod['IKr']['Ku'].set_rhs(1.79e-5)
mod['IKr']['Vhalf'].set_rhs(-1.147)
mod['IKr']['halfmax'].set_rhs(548300000)
mod['extracellular']['cao'].set_rhs(2)
mod['extracellular']['ko'].set_rhs(5)
mod['extracellular']['nao'].set_rhs(137)

mod['multipliers']['i_na_multiplier'].set_rhs(0.3411)
mod['multipliers']['i_nal_multiplier'].set_rhs(0.3241)
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(1.4129*ind_3[0]['i_cal_pca_multiplier'])
mod['multipliers']['i_to_multiplier'].set_rhs(0.4533)
mod['multipliers']['i_kr_multiplier'].set_rhs(0.8998*ind_3[0]['i_kr_multiplier'])
mod['multipliers']['i_ks_multiplier'].set_rhs(0.4015)
mod['multipliers']['i_k1_multiplier'].set_rhs(1.9379)
mod['multipliers']['i_NCX_multiplier'].set_rhs(1.2201)
mod['multipliers']['i_nak_multiplier'].set_rhs(0.6743)
mod['multipliers']['jrel_multiplier'].set_rhs(0.8059)
mod['multipliers']['jup_multiplier'].set_rhs(1.4250)

proto.schedule(5.3, 0.1, 1, 4000, 0) 
sim = myokit.Simulation(mod,proto)
sim.pre(4000 * 50) 
dat = sim.run(4000)
t3 = np.array(dat['engine.time'])
v3 = np.array(dat['membrane.v'])

# Best Ind Plot 4

ind_4 = pd.read_excel('Best_ind_noCa_morph4.xlsx')
ind_4 = ind_4.to_dict('index')


mod, proto, x = myokit.load('./BPS3.mmt')
for k, v in ind_4[0].items():
        mod['multipliers'][k].set_rhs(v)
        
mod['IKr']['D'].set_state_value(100)
mod['IKr']['Kmax'].set_rhs(1e8)
mod['IKr']['Ku'].set_rhs(1.79e-5)
mod['IKr']['Vhalf'].set_rhs(-1.147)
mod['IKr']['halfmax'].set_rhs(548300000)
mod['extracellular']['cao'].set_rhs(2)
mod['extracellular']['ko'].set_rhs(5)
mod['extracellular']['nao'].set_rhs(137)

mod['multipliers']['i_na_multiplier'].set_rhs(0.3411)
mod['multipliers']['i_nal_multiplier'].set_rhs(0.3241)
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(1.4129*ind_4[0]['i_cal_pca_multiplier'])
mod['multipliers']['i_to_multiplier'].set_rhs(0.4533)
mod['multipliers']['i_kr_multiplier'].set_rhs(0.8998*ind_4[0]['i_kr_multiplier'])
mod['multipliers']['i_ks_multiplier'].set_rhs(0.4015)
mod['multipliers']['i_k1_multiplier'].set_rhs(1.9379)
mod['multipliers']['i_NCX_multiplier'].set_rhs(1.2201)
mod['multipliers']['i_nak_multiplier'].set_rhs(0.6743)
mod['multipliers']['jrel_multiplier'].set_rhs(0.8059)
mod['multipliers']['jup_multiplier'].set_rhs(1.4250)

proto.schedule(5.3, 0.1, 1, 4000, 0) 
sim = myokit.Simulation(mod,proto)
sim.pre(4000 * 50) 
dat = sim.run(4000)
t4 = np.array(dat['engine.time'])
v4 = np.array(dat['membrane.v'])

# Best Ind Plot 5

ind_5 = pd.read_excel('Best_ind_noCa_morph5.xlsx')
ind_5 = ind_5.to_dict('index')


mod, proto, x = myokit.load('./BPS3.mmt')
for k, v in ind_5[0].items():
        mod['multipliers'][k].set_rhs(v)
        
mod['IKr']['D'].set_state_value(100)
mod['IKr']['Kmax'].set_rhs(1e8)
mod['IKr']['Ku'].set_rhs(1.79e-5)
mod['IKr']['Vhalf'].set_rhs(-1.147)
mod['IKr']['halfmax'].set_rhs(548300000)
mod['extracellular']['cao'].set_rhs(2)
mod['extracellular']['ko'].set_rhs(5)
mod['extracellular']['nao'].set_rhs(137)

mod['multipliers']['i_na_multiplier'].set_rhs(0.3411)
mod['multipliers']['i_nal_multiplier'].set_rhs(0.3241)
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(1.4129*ind_5[0]['i_cal_pca_multiplier'])
mod['multipliers']['i_to_multiplier'].set_rhs(0.4533)
mod['multipliers']['i_kr_multiplier'].set_rhs(0.8998*ind_5[0]['i_kr_multiplier'])
mod['multipliers']['i_ks_multiplier'].set_rhs(0.4015)
mod['multipliers']['i_k1_multiplier'].set_rhs(1.9379)
mod['multipliers']['i_NCX_multiplier'].set_rhs(1.2201)
mod['multipliers']['i_nak_multiplier'].set_rhs(0.6743)
mod['multipliers']['jrel_multiplier'].set_rhs(0.8059)
mod['multipliers']['jup_multiplier'].set_rhs(1.4250)

proto.schedule(5.3, 0.1, 1, 4000, 0) 
sim = myokit.Simulation(mod,proto)
sim.pre(4000 * 50) 
dat = sim.run(4000)
t5 = np.array(dat['engine.time'])
v5 = np.array(dat['membrane.v'])

# Best Ind Plot 6

ind_6 = pd.read_excel('Best_ind_noCa_morph12.xlsx')
ind_6 = ind_6.to_dict('index')


mod, proto, x = myokit.load('./BPS3.mmt')
for k, v in ind_6[0].items():
        mod['multipliers'][k].set_rhs(v)
        
mod['IKr']['D'].set_state_value(100)
mod['IKr']['Kmax'].set_rhs(1e8)
mod['IKr']['Ku'].set_rhs(1.79e-5)
mod['IKr']['Vhalf'].set_rhs(-1.147)
mod['IKr']['halfmax'].set_rhs(548300000)
mod['extracellular']['cao'].set_rhs(2)
mod['extracellular']['ko'].set_rhs(5)
mod['extracellular']['nao'].set_rhs(137)

mod['multipliers']['i_na_multiplier'].set_rhs(0.3411)
mod['multipliers']['i_nal_multiplier'].set_rhs(0.3241)
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(1.4129*ind_6[0]['i_cal_pca_multiplier'])
mod['multipliers']['i_to_multiplier'].set_rhs(0.4533)
mod['multipliers']['i_kr_multiplier'].set_rhs(0.8998*ind_6[0]['i_kr_multiplier'])
mod['multipliers']['i_ks_multiplier'].set_rhs(0.4015)
mod['multipliers']['i_k1_multiplier'].set_rhs(1.9379)
mod['multipliers']['i_NCX_multiplier'].set_rhs(1.2201)
mod['multipliers']['i_nak_multiplier'].set_rhs(0.6743)
mod['multipliers']['jrel_multiplier'].set_rhs(0.8059)
mod['multipliers']['jup_multiplier'].set_rhs(1.4250)

proto.schedule(5.3, 0.1, 1, 4000, 0) 
sim = myokit.Simulation(mod,proto)
sim.pre(4000 * 50) 
dat = sim.run(4000)
t6 = np.array(dat['engine.time'])
v6 = np.array(dat['membrane.v'])

# Best Ind Plot 7

ind_7 = pd.read_excel('Best_ind_noCa_morph7.xlsx')
ind_7 = ind_7.to_dict('index')


mod, proto, x = myokit.load('./BPS3.mmt')
for k, v in ind_7[0].items():
        mod['multipliers'][k].set_rhs(v)
        
mod['IKr']['D'].set_state_value(100)
mod['IKr']['Kmax'].set_rhs(1e8)
mod['IKr']['Ku'].set_rhs(1.79e-5)
mod['IKr']['Vhalf'].set_rhs(-1.147)
mod['IKr']['halfmax'].set_rhs(548300000)
mod['extracellular']['cao'].set_rhs(2)
mod['extracellular']['ko'].set_rhs(5)
mod['extracellular']['nao'].set_rhs(137)

mod['multipliers']['i_na_multiplier'].set_rhs(0.3411)
mod['multipliers']['i_nal_multiplier'].set_rhs(0.3241)
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(1.4129*ind_7[0]['i_cal_pca_multiplier'])
mod['multipliers']['i_to_multiplier'].set_rhs(0.4533)
mod['multipliers']['i_kr_multiplier'].set_rhs(0.8998*ind_7[0]['i_kr_multiplier'])
mod['multipliers']['i_ks_multiplier'].set_rhs(0.4015)
mod['multipliers']['i_k1_multiplier'].set_rhs(1.9379)
mod['multipliers']['i_NCX_multiplier'].set_rhs(1.2201)
mod['multipliers']['i_nak_multiplier'].set_rhs(0.6743)
mod['multipliers']['jrel_multiplier'].set_rhs(0.8059)
mod['multipliers']['jup_multiplier'].set_rhs(1.4250)

proto.schedule(5.3, 0.1, 1, 4000, 0) 
sim = myokit.Simulation(mod,proto)
sim.pre(4000 * 50) 
dat = sim.run(4000)
t7 = np.array(dat['engine.time'])
v7 = np.array(dat['membrane.v'])

# Best Ind Plot 8

ind_8 = pd.read_excel('Best_ind_noCamorph14.xlsx')
ind_8 = ind_8.to_dict('index')


mod, proto, x = myokit.load('./BPS3.mmt')
for k, v in ind_8[0].items():
        mod['multipliers'][k].set_rhs(v)
        
mod['IKr']['D'].set_state_value(100)
mod['IKr']['Kmax'].set_rhs(1e8)
mod['IKr']['Ku'].set_rhs(1.79e-5)
mod['IKr']['Vhalf'].set_rhs(-1.147)
mod['IKr']['halfmax'].set_rhs(548300000)
mod['extracellular']['cao'].set_rhs(2)
mod['extracellular']['ko'].set_rhs(5)
mod['extracellular']['nao'].set_rhs(137)

mod['multipliers']['i_na_multiplier'].set_rhs(0.3411)
mod['multipliers']['i_nal_multiplier'].set_rhs(0.3241)
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(1.4129*ind_8[0]['i_cal_pca_multiplier'])
mod['multipliers']['i_to_multiplier'].set_rhs(0.4533)
mod['multipliers']['i_kr_multiplier'].set_rhs(0.8998*ind_8[0]['i_kr_multiplier'])
mod['multipliers']['i_ks_multiplier'].set_rhs(0.4015)
mod['multipliers']['i_k1_multiplier'].set_rhs(1.9379)
mod['multipliers']['i_NCX_multiplier'].set_rhs(1.2201)
mod['multipliers']['i_nak_multiplier'].set_rhs(0.6743)
mod['multipliers']['jrel_multiplier'].set_rhs(0.8059)
mod['multipliers']['jup_multiplier'].set_rhs(1.4250)

proto.schedule(5.3, 0.1, 1, 4000, 0) 
sim = myokit.Simulation(mod,proto)
sim.pre(4000 * 50) 
dat = sim.run(4000)
t8 = np.array(dat['engine.time'])
v8 = np.array(dat['membrane.v'])



# Create the figure
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
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