import myokit
import matplotlib.pyplot as plt
import time
import numpy as np

t11 = time.time()
mod1, proto1, x = myokit.load('./BPS_f.mmt')

''''
#4661
mod['multipliers']['i_na_multiplier'].set_rhs(1.0098443639461787)
mod['multipliers']['i_nal_multiplier'].set_rhs(0.6206623655072978)
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(1.0590088264565551)
mod['multipliers']['i_to_multiplier'].set_rhs(0.6488514437487292)
mod['multipliers']['i_kr_multiplier'].set_rhs(1.0934320925025014)
mod['multipliers']['i_ks_multiplier'].set_rhs(0.3788272959938376)
mod['multipliers']['i_k1_multiplier'].set_rhs(0.3385690921528329)
mod['multipliers']['i_NCX_multiplier'].set_rhs(1.0199980271059927)
mod['multipliers']['i_nak_multiplier'].set_rhs(1.2280671971397352)
mod['multipliers']['jrel_multiplier'].set_rhs(1.8108983603248088)
mod['multipliers']['jup_multiplier'].set_rhs(1.3100270267215237)
'''
'''
#451
mod['multipliers']['i_na_multiplier'].set_rhs(0.5644782688023217)
mod['multipliers']['i_nal_multiplier'].set_rhs(1.148986324360065)
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(1.5046827675485108)
mod['multipliers']['i_to_multiplier'].set_rhs(1.7938554768398853)
mod['multipliers']['i_kr_multiplier'].set_rhs(1.3623227045353679)
mod['multipliers']['i_ks_multiplier'].set_rhs(0.382775126800872)
mod['multipliers']['i_k1_multiplier'].set_rhs(0.3226046672184516)
mod['multipliers']['i_NCX_multiplier'].set_rhs(1.9395159324129767)
mod['multipliers']['i_nak_multiplier'].set_rhs(1.8123731254721296)
mod['multipliers']['jrel_multiplier'].set_rhs(1.4615780576626078)
mod['multipliers']['jup_multiplier'].set_rhs(1.4811662264379706)
'''
'''
#1101
mod['multipliers']['i_na_multiplier'].set_rhs(0.8669)
mod['multipliers']['i_nal_multiplier'].set_rhs(0.9470)
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(1.9614)
mod['multipliers']['i_to_multiplier'].set_rhs(1.0514)
mod['multipliers']['i_kr_multiplier'].set_rhs(1.7717)
mod['multipliers']['i_ks_multiplier'].set_rhs(0.5371)
mod['multipliers']['i_k1_multiplier'].set_rhs(1.7493)
mod['multipliers']['i_NCX_multiplier'].set_rhs(1.2452)
mod['multipliers']['i_nak_multiplier'].set_rhs(1.4675)
mod['multipliers']['jrel_multiplier'].set_rhs(1.2809)
mod['multipliers']['jup_multiplier'].set_rhs(0.5615)
'''
'''
mod['multipliers']['i_na_multiplier'].set_rhs(0.34108326348348433)
mod['multipliers']['i_nal_multiplier'].set_rhs(0.3240617429836477)
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(1.4128622523699432)
mod['multipliers']['i_to_multiplier'].set_rhs(0.45326551931043363)
mod['multipliers']['i_kr_multiplier'].set_rhs(0.8998214801859381)
mod['multipliers']['i_ks_multiplier'].set_rhs(0.4014664293882376)
mod['multipliers']['i_k1_multiplier'].set_rhs(1.9379197645151593)
mod['multipliers']['i_NCX_multiplier'].set_rhs(1.2201451074180292)
mod['multipliers']['i_nak_multiplier'].set_rhs(0.6743151653882478)
mod['multipliers']['jrel_multiplier'].set_rhs(0.8058525595905954)
mod['multipliers']['jup_multiplier'].set_rhs(1.4249714441425554)
'''

proto1.schedule(5.3, 0.1, 1, 4000, 0)
sim1 = myokit.Simulation(mod1, proto1)
sim1.pre(4000*1000)
dat_normal1 = sim1.run(1000)
print(f'It took {time.time() - t11} s')
t1 = dat_normal1['engine.time']
v1 = dat_normal1['membrane.v']

# $$
t22 = time.time()
mod2, proto2, x = myokit.load('./BPS.mmt')

proto2.schedule(5.3, 0.1, 1, 1000, 0)
sim2 = myokit.Simulation(mod2, proto2)
sim2.pre(4000*1000)
dat_normal2 = sim2.run(1000)
print(f'It took {time.time() - t22} s')
t2 = dat_normal2['engine.time']
v2 = dat_normal2['membrane.v']

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.plot(t2, v2,  'k--', label = 'BPS NORMAL BASELINE', linewidth=1)
ax.plot(t1, v1, 'B', label = 'BPS PROTOCOL DOFETILIDE BASELINE',linewidth=1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Time (ms)', fontsize=14)
ax.set_ylabel('Voltage (mV)', fontsize=14)

plt.show()