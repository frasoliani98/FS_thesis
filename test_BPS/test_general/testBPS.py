import random
from math import log10
import matplotlib.pyplot as plt
from scipy.signal import find_peaks # pip install scipy
import seaborn as sns # pip install seaborn
from multiprocessing import Pool
import numpy as np
import pandas as pd
from important_functions_BPS33 import run_EAD, detect_EAD


from deap import base, creator, tools # pip install deap
import myokit

feature_targets = {'Vm_peak': [10, 33, 55],
                       'dvdt_max': [100, 347, 1000],
                       #'apd40': [85, 198, 320],
                       #'apd50': [110, 220, 430],
                       #'apd90': [180, 271, 440],
                       #'triangulation': [50, 73, 150],
                       #'RMP': [-95, -88, -80],
                       'cat_amp': [3E-4*1e5, 3.12E-4*1e5, 8E-4*1e5],
                       'cat_peak': [40, 58, 60],
                       'cat90': [350, 467, 500]}

#tunable_parameters=['i_cal_pca_multiplier', 'i_kr_multiplier']

def closest(lst, K):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

def get_ind_data(ind):

    mod, proto, x = myokit.load('./BPS33.mmt')
    if ind is not None:
        for k, v in ind[0].items():
            mod['multipliers'][k].set_rhs(v)

    return mod, proto

def get_normal_sim_dat(ind, IC):
    """
        Runs simulation for a given individual. If the individuals is None,
        then it will run the baseline model

        Returns
        ------
            t, v, cai, i_ion
    """

    mod, proto = get_ind_data(ind)
    proto.schedule(5.3, 0.1, 1, 4000, 0) 
    sim = myokit.Simulation(mod, proto)
    
    if IC == None:
        IC = sim.state()

    sim.set_state(IC)
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

def get_feature_errors(t, v, cai):
    """
    Compares the simulation data for an individual to the baseline BPS values. The returned error value is a sum of the differences between the individual and baseline values.    
    Returns
    ------
        error
    """    
    ap_features = {}
    #t, v, cai, i_ion = get_normal_sim_dat(ind, IC)

    # Returns really large error value if cell AP is not valid 
    if ((min(v) > -60) or (max(v) < 0)):
        return 500000     
    
    # Voltage/APD features#######################
    mdp = min(v)
    max_p = max(v)
    max_p_idx = np.argmax(v)
    apa = max_p - mdp
    dvdt_max = np.max(np.diff(v[0:30])/np.diff(t[0:30]))    
    
    ap_features['Vm_peak'] = max_p
    ap_features['dvdt_max'] = dvdt_max

    '''
    for apd_pct in [40, 50, 90]:
        repol_pot = max_p - apa * apd_pct/100
        idx_apd = np.argmin(np.abs(v[max_p_idx:] - repol_pot))
        apd_val = t[idx_apd+max_p_idx]        
        
        ap_features[f'apd{apd_pct}'] = apd_val
    
    ap_features['triangulation'] = ap_features['apd90'] - ap_features['apd40']
    ap_features['RMP'] = np.mean(v[len(v)-50:len(v)])  
    '''  
    
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
    
    error = 0    
    
    '''
    for k, v in ap_features.items():
        error += (GA_CONFIG.feature_targets[k][1] - v)**2    
        
    error = 0    
    '''
    '''
    if GA_CONFIG.cost == 'function_1':
        for k, v in ap_features.items():
            error += (GA_CONFIG.feature_targets[k][1] - v)**2
    else:
        for k, v in ap_features.items():
            if ((v < GA_CONFIG.feature_targets[k][0]) or
                    (v > GA_CONFIG.feature_targets[k][2])):
                error += 1000
    '''

    for k, v in ap_features.items():
        if ((v < feature_targets[k][0]) or (v > feature_targets[k][2])):
            error += (feature_targets[k][1] - v)**2   
                
    return error

def check_physio_torord(t, v, path = './', filter = 'no'):    
    
    # Cut off the upstroke of the AP for profile
    t_ind = list(t[150:len(t)])
    v_ind = list(v[150:len(t)])    
    
    # Baseline tor-ord model & cut off upstroke
    base_df = pd.read_csv(path + 'baseline_torord_data.csv')
    t_base = list(base_df['t'])[150:len(t)]
    v_base = list(base_df['v'])[150:len(t)]    
    
    # Cut off the upstroke of the AP for the tor-ord data
    if filter == 'no':
        time, vol_10, vol_90 = get_torord_phys_data(path, filter)
        t = time[150:len(time)]
        v_10 = vol_10[150:len(time)]
        v_90 = vol_90[150:len(time)]    
    
    else:
        t, v_10, v_90 = get_torord_phys_data(path, filter)    
        
    result = 0 # valid AP
    #fail_time = 3000
    error = 0
    check_times = []
    data = {}    
    
    for i in list(range(0, len(t_ind))):
        t_dat = closest(t, t_ind[i]) # find the value closest to the ind's time within the exp data time list
        t_dat_base = closest(t_base, t_ind[i])
        t_dat_i = np.where(np.array(t)==t_dat)[0][0] #find the index of the closest value in the list
        t_dat_base_i = np.where(np.array(t_base)==t_dat_base)[0][0] #find the index of the closest value in the list
        v_model = v_ind[i]
        v_lowerbound = v_10[t_dat_i]
        v_upperbound = v_90[t_dat_i]
        v_torord = v_base[t_dat_base_i]        
        check_times.append(np.abs(t_ind[i] - t_dat))    

        if v_model < v_lowerbound or v_model > v_upperbound:
            result = 1 # not a valid AP
            #error += 10 - used in GA 5
            error += (v_model - v_torord)**2    

    data['result'] = result
    data['error'] = error
    data['check_times'] = check_times    

    return(data)

def get_torord_phys_data(path = './', filter = 'no'):
    data = pd.read_csv(path+'torord_physiologicData.csv')
    time = [x - 9.1666666669999994 for x in list(data['t'])] #shift action potential to match solutions
    t = time[275:len(data['v_10'])]
    v_10 = list(data['v_10'])[275:len(data['v_10'])]
    v_90 = list(data['v_90'])[275:len(data['v_10'])]    
    
    if filter != 'no':
        data = pd.DataFrame(data = {'t': t[1000:len(t)], 'v_10': v_10[1000:len(t)], 'v_90':v_90[1000:len(t)]})
        data_start = pd.DataFrame(data = {'t': t[150:1000], 'v_10': v_10[150:1000], 'v_90':v_90[150:1000]})        
        
        # FILTER V_10
        v_10_new = data.v_10.rolling(400, min_periods = 1, center = True).mean()
        v_10_start = data_start.v_10.rolling(100, min_periods = 1, center = True).mean()
        v_10_new = v_10_new.dropna()
        v_10 = list(v_10_start) + list(v_10_new)
        t = list(data_start['t']) + list(data['t'])        
        
        # FILTER V_90
        v_90_new = data.v_90.rolling(400, min_periods = 1, center = True).mean()
        v_90_start = data_start.v_90.rolling(200, min_periods = 1, center = True).mean()
        v_90_new = v_90_new.dropna()
        v_90 = list(v_90_start) + list(v_90_new)    
        
    return(t, v_10, v_90)

'''
lower_exp = log10(0.1)
upper_exp = log10(10)
initial_params = [10**random.uniform(lower_exp, upper_exp)
                    for i in range(0, len(
                        tunable_parameters))]
                

keys = [val for val in tunable_parameters]
par = (dict(zip(keys, initial_params)))
'''
ind = [{'i_cal_pca_multiplier': 0.156915875883899, 'i_kr_multiplier': 1.34345137421618, 'i_ks_multiplier': 1, 'i_nal_multiplier': 1, 'i_na_multiplier': 1, 'i_to_multiplier': 1, 'i_k1_multiplier': 1, 'i_NCX_multiplier': 1, 'i_nak_multiplier': 1, 'i_kb_multiplier': 1, 'jrel_multiplier' : 1, 'jup_multiplier' : 1 }]
#ind = [{'i_cal_pca_multiplier': 1, 'i_kr_multiplier': 1, 'i_ks_multiplier': 1, 'i_nal_multiplier': 1, 'i_na_multiplier': 1, 'i_to_multiplier': 1, 'i_k1_multiplier': 1, 'i_NCX_multiplier': 1, 'i_nak_multiplier': 1, 'i_kb_multiplier': 1, 'jup_multiplier' : 1 }]
#ind = [ind | par]

print(ind)


mod, proto = get_ind_data(ind)
proto.schedule(5.3, 0.1, 1, 4000, 0) 
sim = myokit.Simulation(mod, proto)
sim.pre(600 * 4000) #pre-pace for 100 beats
IC = sim.state()


tn, vn, cai, i_ion = get_normal_sim_dat(ind, IC)
feature_error = get_feature_errors(tn, vn, cai)
data = check_physio_torord(tn, vn, './', filter = 'yes')
morph_error = data['error']


t,v,t1,v1 = run_EAD(ind)
info, result = detect_EAD(t,v)
info1, result1 = detect_EAD(t1,v1)


ead_error = 0

if result == 1 or result1 == 1:
    ead_error = 100000
    #print('info' +str(info))
    #print('info1' +str(info1))

fitness = feature_error + morph_error + ead_error

data = [fitness, result]
print(feature_error)
print(morph_error)
print(ead_error) 100000
# Returns 
if feature_error == 500000:
    data = [500000, 1]

print(data)



mod, proto, x = myokit.load('./tor_ord_endo2.mmt')

proto.schedule(5.3, 0.1, 1, 1000, 0)    
sim = myokit.Simulation(mod, proto)
sim.pre(1000*600)
dat_normal = sim.run(1000)
t2 = dat_normal['engine.time']
v2 = dat_normal['membrane.v']

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Time [ms]', fontsize=14)
ax.set_ylabel('Voltage [mV]', fontsize=14)
fig.suptitle(f'9 EADs profiles', fontsize=14)


ax.plot(tn, vn, 'b--', label = 'Best fit')

ax.plot(t2, v2, 'k', label = 'Tor Ord')

plt.legend(loc='best')
plt.show()
