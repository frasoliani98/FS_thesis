import random
from math import log10
import matplotlib.pyplot as plt
from scipy.signal import find_peaks # pip install scipy
import seaborn as sns # pip install seaborn
from multiprocessing import Pool
import numpy as np
import pandas as pd

from deap import base, creator, tools # pip install deap
import myokit

feature_targets = {'Vm_peak': [10, 33, 55],
                       'dvdt_max': [100, 347, 1000],
                       'apd40': [85, 198, 320],
                       'apd50': [110, 220, 430],
                       'apd90': [180, 271, 440],
                       #'triangulation': [50, 73, 150],
                       #'RMP': [-95, -88, -80],
                       'cat_amp': [3E-4*1e5, 3.12E-4*1e5, 8E-4*1e5],
                       'cat_peak': [40, 58, 60],
                       'cat90': [350, 467, 500]}

def get_ind_data(ind):

    mod, proto, x = myokit.load('./BPS3.mmt')
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
    proto.schedule(5.3, 0.1, 1, 1000, 0) 
    sim = myokit.Simulation(mod, proto)
    sim.pre(1000*600)
    
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
    Compares the simulation data for an individual to the baseline Tor-ORd values. The returned error value is a sum of the differences between the individual and baseline values.    
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
    dvdt_max = np.max(np.diff(v[0:100])/np.diff(t[0:100]))    
    
    ap_features['Vm_peak'] = max_p
    ap_features['dvdt_max'] = dvdt_max

    
    for apd_pct in [40, 50, 90]:
        repol_pot = max_p - apa * apd_pct/100
        idx_apd = np.argmin(np.abs(v[max_p_idx:] - repol_pot))
        apd_val = t[idx_apd+max_p_idx]        
        
        ap_features[f'apd{apd_pct}'] = apd_val
    
    #ap_features['triangulation'] = ap_features['apd90'] - ap_features['apd40']
    #ap_features['RMP'] = np.mean(v[len(v)-50:len(v)])  

    
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

    return error, ap_features

ind = None
tb, vb, cai, i_ion = get_normal_sim_dat(ind, IC=None)
feature_error, apfeatures = get_feature_errors(tb, vb, cai)

ead_error = 0
fitness = feature_error + ead_error

print(apfeatures)
print(feature_error)
print(fitness)