from math import log10
import matplotlib.pyplot as plt
from scipy.signal import find_peaks # pip install scipy
import seaborn as sns # pip install seaborn
from multiprocessing import Pool
import numpy as np
import pandas as pd
import myokit

feature_targets = {'Vm_peak': [10, 33, 55],
                       'dvdt_max': [100, 347, 1000],
                       'apd40': [85, 198, 320],
                       'apd50': [110, 220, 430],
                       'apd90': [180, 271, 440],
                       #'triangulation': [50, 73, 150],
                       'RMP': [-95, -88, -80],
                       'cat_amp': [3E-4*1e5, 3.12E-4*1e5, 8E-4*1e5],
                       'cat_peak': [40, 58, 60],
                       'cat90': [350, 467, 500]}

ind_1 = pd.read_excel('Best_ind_ko54_ca.xlsx')
ind = ind_1.to_dict('index')
print(ind)


def get_ind_data(ind):

    mod, proto, x = myokit.load('./BPS3_ko54.mmt')
    if ind is not None:
        for k, v in ind[0].items():
            mod['multipliers'][k].set_rhs(v)

    return mod, proto

def get_normal_sim_dat(ind):
    """
        Runs simulation for a given individual. If the individuals is None,
        then it will run the baseline model

        Returns
        ------
            t, v, cai, i_ion
    """
    mod, proto = get_ind_data(ind)
    proto.schedule(5.3, 0.1, 1, 1000, 0) 
    sim = myokit.Simulation(mod, proto)
    sim.pre(1000 * 600) 
    dat = sim.run(50000)

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

    return (t, v, cai, i_ion)



t, v, cai, i_ion = get_normal_sim_dat(ind)
ap_features = {}
 

# Voltage/APD features#######################
mdp = min(v)
max_p = max(v)
max_p_idx = np.argmax(v)
apa = max_p - mdp
dvdt_max = np.max(np.diff(v[0:100])/np.diff(t[0:100]))    

ap_features['Vm_peak'] = max_p
ap_features['dvdt_max'] = dvdt_max

for apd_pct in [40, 50, 90]:
        repol_pot = max_p - apa * apd_pct/100
        idx_apd = np.argmin(np.abs(v[max_p_idx:] - repol_pot))
        apd_val = t[idx_apd+max_p_idx]        
        
        ap_features[f'apd{apd_pct}'] = apd_val
    
#ap_features['triangulation'] = ap_features['apd90'] - ap_features['apd40']
ap_features['RMP'] = np.mean(v[len(v)-50:len(v)]) 

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
df3.to_excel('FS_ap_features.xlsx', sheet_name='Sheet1')

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_xlabel('Time [ms]', fontsize=14)
ax.set_ylabel('Cai ', fontsize=14)
ax.set_xlim(0,1000)
#ax.set_xlim(0,1000)
ax.plot(t, cai, 'r')
plt.show()

