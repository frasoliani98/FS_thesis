import myokit
import matplotlib.pyplot as plt
import time
import numpy as np

# Create the figure
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Time [ms]', fontsize=14)
ax.set_ylabel('Voltage [mV]', fontsize=14)
fig.suptitle(f'9 EADs profiles', fontsize=14)


# %%
t44 = time.time()
mod4, proto4, x = myokit.load('./BPS_f.mmt')


# 4th profile: 1461
mod4['multipliers']['i_na_multiplier'].set_rhs(0.34108326348348433)
mod4['multipliers']['i_nal_multiplier'].set_rhs(0.3240617429836477)
mod4['multipliers']['i_cal_pca_multiplier'].set_rhs(1.4128622523699432)
mod4['multipliers']['i_to_multiplier'].set_rhs(0.45326551931043363)
mod4['multipliers']['i_kr_multiplier'].set_rhs(0.8998214801859381)
mod4['multipliers']['i_ks_multiplier'].set_rhs(0.4014664293882376)
mod4['multipliers']['i_k1_multiplier'].set_rhs(1.9379197645151593)
mod4['multipliers']['i_NCX_multiplier'].set_rhs(1.2201451074180292)
mod4['multipliers']['i_nak_multiplier'].set_rhs(0.6743151653882478)
mod4['multipliers']['jrel_multiplier'].set_rhs(0.8058525595905954)
mod4['multipliers']['jup_multiplier'].set_rhs(1.4249714441425554)

proto4.schedule(5.3, 0.1, 1, 4000, 0)    
sim4 = myokit.Simulation(mod4, proto4)
sim4.pre(1000*600)
dat_normal4 = sim4.run(4000)
print(f'It took {time.time() - t44} s')
t4 = dat_normal4['engine.time']
v4 = dat_normal4['membrane.v']

# 1st profile: 451
t11 = time.time()
mod1, proto1, x = myokit.load('./BPS_f.mmt')

mod1['multipliers']['i_na_multiplier'].set_rhs(0.5644782688023217)
mod1['multipliers']['i_nal_multiplier'].set_rhs(1.148986324360065)
mod1['multipliers']['i_cal_pca_multiplier'].set_rhs(1.5046827675485108)
mod1['multipliers']['i_to_multiplier'].set_rhs(1.7938554768398853)
mod1['multipliers']['i_kr_multiplier'].set_rhs(1.3623227045353679)
mod1['multipliers']['i_ks_multiplier'].set_rhs(0.382775126800872)
mod1['multipliers']['i_k1_multiplier'].set_rhs(0.3226046672184516)
mod1['multipliers']['i_NCX_multiplier'].set_rhs(1.9395159324129767)
mod1['multipliers']['i_nak_multiplier'].set_rhs(1.8123731254721296)
mod1['multipliers']['jrel_multiplier'].set_rhs(1.4615780576626078)
mod1['multipliers']['jup_multiplier'].set_rhs(1.4811662264379706)

proto1.schedule(5.3, 0.1, 1, 4000, 0)    
sim1 = myokit.Simulation(mod1, proto1)
sim1.pre(1000*4000)
dat_normal1 = sim1.run(4000)
print(f'It took {time.time() - t11} s')
t1 = dat_normal1['engine.time']
v1 = dat_normal1['membrane.v']

# 2 profile: 900
t = time.time()
mod, proto, x = myokit.load('./BPS_f.mmt')

mod['multipliers']['i_na_multiplier'].set_rhs(1.1873467070305843)
mod['multipliers']['i_nal_multiplier'].set_rhs(1.38233541563373)
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(1.8403752529856816)
mod['multipliers']['i_to_multiplier'].set_rhs(1.7167955493335276)
mod['multipliers']['i_kr_multiplier'].set_rhs(1.62896643979671)
mod['multipliers']['i_ks_multiplier'].set_rhs(0.3272179689765393)
mod['multipliers']['i_k1_multiplier'].set_rhs(1.439108031881329)
mod['multipliers']['i_NCX_multiplier'].set_rhs(1.4481351626265873)
mod['multipliers']['i_nak_multiplier'].set_rhs(0.6428416063240505)
mod['multipliers']['jrel_multiplier'].set_rhs(0.7234731786835848)
mod['multipliers']['jup_multiplier'].set_rhs(1.07873445458111)

proto.schedule(5.3, 0.1, 1, 4000, 0)    
sim = myokit.Simulation(mod, proto)
sim.pre(1000*600)
dat_normal = sim.run(4000)
print(f'It took {time.time() - t} s')
t2 = dat_normal['engine.time']
v2 = dat_normal['membrane.v']

# 3 profile: 1101
t = time.time()
mod, proto, x = myokit.load('./BPS_f.mmt')

mod['multipliers']['i_na_multiplier'].set_rhs(0.8669048556312773)
mod['multipliers']['i_nal_multiplier'].set_rhs(0.9470163134744838)
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(1.96142889200044)
mod['multipliers']['i_to_multiplier'].set_rhs(1.0514344728736338)
mod['multipliers']['i_kr_multiplier'].set_rhs(1.7717118510060725)
mod['multipliers']['i_ks_multiplier'].set_rhs(0.5370509997391757)
mod['multipliers']['i_k1_multiplier'].set_rhs(1.7493012799990504)
mod['multipliers']['i_NCX_multiplier'].set_rhs(1.2452423032767352)
mod['multipliers']['i_nak_multiplier'].set_rhs(1.4674903262460632)
mod['multipliers']['jrel_multiplier'].set_rhs(1.280947686335993)
mod['multipliers']['jup_multiplier'].set_rhs(0.5614876272819089)

proto.schedule(5.3, 0.1, 1, 4000, 0)    
sim = myokit.Simulation(mod, proto)
sim.pre(600*1000)
dat_normal = sim.run(4000)
print(f'It took {time.time() - t} s')
t3 = dat_normal['engine.time']
v3 = dat_normal['membrane.v']

# 5 profile: 1740
t = time.time()
mod, proto, x = myokit.load('./BPS_f.mmt')

mod['multipliers']['i_na_multiplier'].set_rhs(0.6085)
mod['multipliers']['i_nal_multiplier'].set_rhs(1.8124)
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(1.6569)
mod['multipliers']['i_to_multiplier'].set_rhs(1.9992)
mod['multipliers']['i_kr_multiplier'].set_rhs(1.6219)
mod['multipliers']['i_ks_multiplier'].set_rhs(0.24112)
mod['multipliers']['i_k1_multiplier'].set_rhs(0.3345)
mod['multipliers']['i_NCX_multiplier'].set_rhs(1.9305)
mod['multipliers']['i_nak_multiplier'].set_rhs(1.6689)
mod['multipliers']['jrel_multiplier'].set_rhs(1.0267)
mod['multipliers']['jup_multiplier'].set_rhs(0.4958)

proto.schedule(5.3, 0.1, 1, 4000, 0)    
sim = myokit.Simulation(mod, proto)
sim.pre(1000*600)
dat_normal = sim.run(4000)
print(f'It took {time.time() - t} s')
t5 = dat_normal['engine.time']
v5 = dat_normal['membrane.v']

# 6 profile: 1907
t = time.time()
mod, proto, x = myokit.load('./BPS_f.mmt')

mod['multipliers']['i_na_multiplier'].set_rhs(0.965736015713675)
mod['multipliers']['i_nal_multiplier'].set_rhs(0.9546825899818911)
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(1.8079440006639957)
mod['multipliers']['i_to_multiplier'].set_rhs(1.0881264946860936)
mod['multipliers']['i_kr_multiplier'].set_rhs(1.0762597421581603)
mod['multipliers']['i_ks_multiplier'].set_rhs(0.37333859558451965)
mod['multipliers']['i_k1_multiplier'].set_rhs(0.5832940756929093)
mod['multipliers']['i_NCX_multiplier'].set_rhs(1.8293391385448716)
mod['multipliers']['i_nak_multiplier'].set_rhs(1.2254526053306092)
mod['multipliers']['jrel_multiplier'].set_rhs(1.0207900244234616)
mod['multipliers']['jup_multiplier'].set_rhs(1.2379786903273944)

proto.schedule(5.3, 0.1, 1, 4000, 0)    
sim = myokit.Simulation(mod, proto)
sim.pre(4000*100)
dat_normal = sim.run(4000)
print(f'It took {time.time() - t} s')
t6 = dat_normal['engine.time']
v6 = dat_normal['membrane.v']

# 7 profile: 4421
t = time.time()
mod, proto, x = myokit.load('./BPS_f.mmt')

mod['multipliers']['i_na_multiplier'].set_rhs(0.6248310482804874)
mod['multipliers']['i_nal_multiplier'].set_rhs(0.39330618377289456)
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(1.826962752393086)
mod['multipliers']['i_to_multiplier'].set_rhs(0.7170666948961162)
mod['multipliers']['i_kr_multiplier'].set_rhs(1.2306086370046445)
mod['multipliers']['i_ks_multiplier'].set_rhs(0.5412214206759218)
mod['multipliers']['i_k1_multiplier'].set_rhs(1.9265577704317045)
mod['multipliers']['i_NCX_multiplier'].set_rhs(1.221770282159352)
mod['multipliers']['i_nak_multiplier'].set_rhs(1.8356284029655854)
mod['multipliers']['jrel_multiplier'].set_rhs(0.9154710285268584)
mod['multipliers']['jup_multiplier'].set_rhs(1.0817307190363596)

proto.schedule(5.3, 0.1, 1, 4000, 0)    
sim = myokit.Simulation(mod, proto)
sim.pre(1000*600)
dat_normal = sim.run(4000)
print(f'It took {time.time() - t} s')
t7 = dat_normal['engine.time']
v7 = dat_normal['membrane.v']

# 8 profile: 4661
t = time.time()
mod, proto, x = myokit.load('./BPS_f.mmt')

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

proto.schedule(5.3, 0.1, 1, 4000, 0)    
sim = myokit.Simulation(mod, proto)
sim.pre(4000*1000)
dat_normal = sim.run(4000)
print(f'It took {time.time() - t} s')
t8 = dat_normal['engine.time']
v8 = dat_normal['membrane.v']

# 9 profile: 4679
t = time.time()
mod, proto, x = myokit.load('./BPS_f.mmt')

mod['multipliers']['i_na_multiplier'].set_rhs(0.7330343138643205)
mod['multipliers']['i_nal_multiplier'].set_rhs(1.5152621241373292)
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(1.9795063629450376)
mod['multipliers']['i_to_multiplier'].set_rhs(1.066005228488999)
mod['multipliers']['i_kr_multiplier'].set_rhs(1.7000107373968272)
mod['multipliers']['i_ks_multiplier'].set_rhs(0.423906340409529)
mod['multipliers']['i_k1_multiplier'].set_rhs(1.4230815362441478)
mod['multipliers']['i_NCX_multiplier'].set_rhs(1.117056744380932)
mod['multipliers']['i_nak_multiplier'].set_rhs(1.2014666596422858)
mod['multipliers']['jrel_multiplier'].set_rhs(0.8980163896135434)
mod['multipliers']['jup_multiplier'].set_rhs(1.53454371303696)

proto.schedule(5.3, 0.1, 1, 4000, 0)    
sim = myokit.Simulation(mod, proto)
sim.pre(1000*600)
dat_normal = sim.run(4000)
print(f'It took {time.time() - t} s')
t9 = dat_normal['engine.time']
v9 = dat_normal['membrane.v']




ax.plot(t1, v1, 'y', label = 'Case 451')

ax.plot(t2, v2, 'r', label = 'Case 900')

ax.plot(t3, v3, 'b', label = 'Case 1101')

ax.plot(t4, v4, 'k', label = 'Case 1461')

ax.plot(t5, v5, 'g', label = 'Case 1740')

ax.plot(t6, v6, 'c', label = 'Case 1907')

ax.plot(t7, v7, 'm', label = 'Case 4421')

ax.plot(t8, v8, color = 'orangered', label = 'Case 4661')

ax.plot(t9, v9, color = 'deeppink', label = 'Case 4679')



plt.legend(loc='best')
plt.show()