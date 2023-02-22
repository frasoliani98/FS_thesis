from math import log10
import matplotlib.pyplot as plt
from scipy.signal import find_peaks # pip install scipy
import seaborn as sns # pip install seaborn
from multiprocessing import Pool
import numpy as np
import pandas as pd
import myokit


ind_1 = pd.read_excel('Best_ind_3_tor.xlsx')
ind = ind_1.to_dict('index')
print(ind)


mod, proto, x = myokit.load('./tor_ord_endo2.mmt')
if ind is not None:
    for k, v in ind[0].items():
        mod['multipliers'][k].set_rhs(v)

proto.schedule(5.3, 0.1, 1, 1000, 0) 
sim = myokit.Simulation(mod, proto)
#sim.pre(1000*600)
dat = sim.run(20000)


# Get t, v, and cai for second to last AP#######################
i_stim = dat['stimulus.i_stim']
peaks = find_peaks(-np.array(i_stim), distance=100)[0]
start_ap = peaks[-3] #TODO change start_ap to be after stim, not during
end_ap = peaks[-2]

t = np.array(dat['engine.time'][start_ap:end_ap])
t = t - t[0]
max_idx = np.argmin(np.abs(t-900))
t = t[0:max_idx]
end_ap = start_ap + max_idx

v = np.array(dat['membrane.v'][start_ap:end_ap])
cai = np.array(dat['intracellular_ions.cai'][start_ap:end_ap])
i_ion = np.array(dat['membrane.i_ion'][start_ap:end_ap])



# Compute AP features
ap_features = {}
 
# Voltage/APD features#######################
mdp = min(v)
max_p = max(v)
max_p_idx = np.argmax(v)
apa = max_p - mdp
dvdt_max = np.max(np.diff(v[0:30])/np.diff(t[0:30]))    

ap_features['Vm_peak'] = max_p
ap_features['dvdt_max'] = dvdt_max


# Calcium/CaT features######################## 
max_cai = np.max(cai)
max_cai_idx = np.argmax(cai)
cat_amp = np.max(cai) - np.min(cai)
ap_features['cat_amp'] = cat_amp*1e5
max_cai_time = t[max_cai_idx]
ap_features['cat_peak'] = max_cai_time    

for cat_pct in [90]:
    cat_recov = max_cai - cat_amp * cat_pct / 100
    idx_catd = np.argmin(np.abs(cai[max_cai_idx:] - cat_recov))
    catd_val = t[idx_catd+max_cai_idx]            
    ap_features[f'cat{cat_pct}'] = catd_val     

Vm = list()
dvdt = list()
catampf = list()
catpeak = list()
catval = list()

Vm.append(max_p)
dvdt.append(dvdt_max)
catampf.append(cat_amp)
catpeak.append(max_cai_time)
catval.append(catd_val)

ap1 = pd.DataFrame(Vm, columns=["Vm Peak"])
ap2 = pd.DataFrame(dvdt, columns=["dVdt Max"])
ap3 = pd.DataFrame(catampf, columns=["Cat Amp"])
ap4 = pd.DataFrame(catpeak, columns=["Cat Peak"])
ap5 = pd.DataFrame(catval, columns=["Cat 90"])


df0 = ap1.join(ap2, how="outer")
df1 = df0.join(ap3, how="outer")
df2 = df1.join(ap4, how="outer")
df3 = df2.join(ap5, how="outer")

print(df3)
print(ap_features)
df3.to_excel('FS_ap_features_tor_ord.xlsx', sheet_name='Sheet1')

