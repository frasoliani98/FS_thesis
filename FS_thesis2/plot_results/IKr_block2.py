# %%
import random
from math import log10
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from important_functions import run_EAD_IKr, detect_EAD, baseline_run, plot_GA, baseline_EAD, baseline_run
import itertools

# %%

apd_90 = []
apd_90b = []

# READ THE DATA
# just load the data, then evrything is ok!
ind_1 = pd.read_excel('Best_ind_1.xlsx')
ind_1 = ind_1.to_dict('index')
ind_2 = pd.read_excel('Best_ind_2.xlsx')
ind_2 = ind_2.to_dict('index')
ind_3 = pd.read_excel('Best_ind_3.xlsx')
ind_3 = ind_3.to_dict('index')
ind_4 = pd.read_excel('Best_ind_4.xlsx')
ind_4 = ind_4.to_dict('index')
ind_5 = pd.read_excel('Best_ind_5.xlsx')
ind_5 = ind_5.to_dict('index')
ind_6 = pd.read_excel('Best_ind_6.xlsx')
ind_6 = ind_6.to_dict('index')
ind_7 = pd.read_excel('Best_ind_7.xlsx')
ind_7 = ind_7.to_dict('index')
ind_8 = pd.read_excel('Best_ind_8.xlsx')
ind_8 = ind_8.to_dict('index')

apd_val1 = list()
apd_val2 = list()
apd_val3 = list()
apd_val4 = list()
apd_val5 = list()
apd_val6 = list()
apd_val7 = list()
apd_val8 = list()
apd_val_tor = list()

# 0% = BASELINE INDS + BASELINE
# BASELINE TOR_ORD
tb, vb = baseline_run()
mdp = min(vb)
max_p = max(vb)
max_p_idx = np.argmax(vb)
apa = max_p - mdp  
repol_pot = max_p - apa * 90/100
idx_apd = np.argmin(np.abs(vb[max_p_idx:] - repol_pot))
apd_val = tb[idx_apd+max_p_idx]
apd_val_tor.append(apd_val) 


# APD baseline 1
t1b, v1b = plot_GA(ind_1)
mdp = min(v1b)
max_p = max(v1b)
max_p_idx = np.argmax(v1b)
apa = max_p - mdp  
repol_pot = max_p - apa * 90/100
idx_apd = np.argmin(np.abs(v1b[max_p_idx:] - repol_pot))
apd_val = t1b[idx_apd+max_p_idx]
apd_val1.append(apd_val)     



# APD baseline 2
t2b, v2b = plot_GA(ind_2)

mdp2 = min(v2b)
max_p2 = max(v2b)
max_p_idx2 = np.argmax(v2b)
apa2 = max_p2 - mdp2
repol_pot2 = max_p2 - apa2 * 90/100
idx_apd2 = np.argmin(np.abs(v2b[max_p_idx2:] - repol_pot2))
apd_val = t2b[idx_apd2+max_p_idx2]        
apd_val2.append(apd_val)  


# APD baseline 3
t3b, v3b = plot_GA(ind_3)
mdp3 = min(v3b)
max_p3 = max(v3b)
max_p_idx3 = np.argmax(v3b)
apa3 = max_p3 - mdp3 
repol_pot3 = max_p3 - apa3 * 90/100
idx_apd3 = np.argmin(np.abs(v3b[max_p_idx3:] - repol_pot3))
apd_val = t3b[idx_apd3+max_p_idx3]  

apd_val3.append(apd_val)  



# APD baseline 4
t4b, v4b = plot_GA(ind_4)
mdp4 = min(v4b)
max_p4 = max(v4b)
max_p_idx4 = np.argmax(v4b)
apa4 = max_p4 - mdp4
repol_pot4 = max_p4 - apa4 * 90/100
idx_apd4 = np.argmin(np.abs(v4b[max_p_idx4:] - repol_pot4))
apd_val = t4b[idx_apd4+max_p_idx4]        

apd_val4.append(apd_val)  


# APD baseline 5
t5b, v5b = plot_GA(ind_5)
mdp5 = min(v5b)
max_p5 = max(v5b)
max_p_idx5 = np.argmax(v5b)
apa5 = max_p5 - mdp5
repol_pot5 = max_p5 - apa5 * 90/100
idx_apd5 = np.argmin(np.abs(v5b[max_p_idx5:] - repol_pot5))
apd_val = t5b[idx_apd5+max_p_idx5]        

apd_val5.append(apd_val)  


# APD baseline 6
t6b, v6b = plot_GA(ind_6)
mdp6 = min(v6b)
max_p6 = max(v6b)
max_p_idx6 = np.argmax(v6b)
apa6 = max_p6 - mdp6
repol_pot6 = max_p6 - apa6 * 90/100
idx_apd6 = np.argmin(np.abs(v6b[max_p_idx6:] - repol_pot6))
apd_val = t6b[idx_apd6+max_p_idx6]        

apd_val6.append(apd_val)  


# APD baseline 7
t7b, v7b = plot_GA(ind_7)
mdp7 = min(v7b)
max_p7 = max(v7b)
max_p_idx7 = np.argmax(v7b)
apa7 = max_p7 - mdp7
    
repol_pot7 = max_p7 - apa7 * 90/100
idx_apd7 = np.argmin(np.abs(v7b[max_p_idx7:] - repol_pot7))
apd_val = t7b[idx_apd7+max_p_idx7]        

apd_val7.append(apd_val) 


# APD baseline 8
t8b, v8b = plot_GA(ind_8)
mdp8 = min(v8b)
max_p8 = max(v8b)
max_p_idx8 = np.argmax(v8b)
apa8 = max_p8 - mdp8
repol_pot8 = max_p8 - apa8 * 90/100
idx_apd8 = np.argmin(np.abs(v8b[max_p_idx8:] - repol_pot8))
apd_val = t8b[idx_apd8+max_p_idx8]        

apd_val8.append(apd_val) 





block = ([0.8, 0.6, 0.4, 0.2, 0.1])
adp_val1bl = list()

for i in block:

    # TOR BASELINE
    tb,vb = baseline_EAD(i)
    info, result = detect_EAD(tb,vb)


    mdp = min(vb)
    max_p = max(vb)
    max_p_idx = np.argmax(vb)
    apa = max_p - mdp
    dvdt_max = np.max(np.diff(vb[0:30])/np.diff(tb[0:30]))    
        
    repol_pot = max_p - apa * 90/100
    idx_apd = np.argmin(np.abs(vb[max_p_idx:] - repol_pot))
    apd_val = tb[idx_apd+max_p_idx]
    apd_val_tor.append(apd_val)
    ind_tor = pd.DataFrame(apd_val_tor, columns=["IND_BASE"]) 



    # IND1
    t1,v1 = run_EAD_IKr(ind_1, i, IC=None)
    info, result = detect_EAD(t1,v1)


    mdp = min(v1)
    max_p = max(v1)
    max_p_idx = np.argmax(v1)
    apa = max_p - mdp
    dvdt_max = np.max(np.diff(v1[0:30])/np.diff(t1[0:30]))    
        
    repol_pot = max_p - apa * 90/100
    idx_apd = np.argmin(np.abs(v1[max_p_idx:] - repol_pot))
    apd_val = t1[idx_apd+max_p_idx]
    apd_val1.append(apd_val)
    ind_1b = pd.DataFrame(apd_val1, columns=["IND1"])        


    print(ind_1b)

    
    # IND 2
    t2,v2 = run_EAD_IKr(ind_2, i, IC=None)
    info, result = detect_EAD(t2,v2)

    mdp = min(v2)
    max_p = max(v2)
    max_p_idx = np.argmax(v2)
    apa = max_p - mdp 
        
    repol_pot = max_p - apa * 90/100
    idx_apd = np.argmin(np.abs(v2[max_p_idx:] - repol_pot))
    apd_val = t2[idx_apd+max_p_idx]        
    apd_val2.append(apd_val)
    ind_2b = pd.DataFrame(apd_val2, columns=["IND2"])




    # IND 3
    t3,v3 = run_EAD_IKr(ind_3, i, IC=None)
    info, result = detect_EAD(t3,v3)



    mdp = min(v3)
    max_p = max(v3)
    max_p_idx = np.argmax(v3)
    apa = max_p - mdp
        
    repol_pot = max_p - apa * 90/100
    idx_apd = np.argmin(np.abs(v2[max_p_idx:] - repol_pot))
    apd_val = t3[idx_apd+max_p_idx]        
    apd_val3.append(apd_val)
    ind_3b = pd.DataFrame(apd_val3, columns=["IND3"])

    


    # IND 4
    t4,v4 = run_EAD_IKr(ind_4, i, IC=None)
    info, result = detect_EAD(t4,v4)



    mdp = min(v4)
    max_p = max(v4)
    max_p_idx = np.argmax(v4)
    apa = max_p - mdp  
        
    repol_pot = max_p - apa * 90/100
    idx_apd = np.argmin(np.abs(v4[max_p_idx:] - repol_pot))
    apd_val = t4[idx_apd+max_p_idx]        
    apd_val4.append(apd_val)
    ind_4b = pd.DataFrame(apd_val4, columns=["IND4"])

    


    # IND 5
    t5,v5 = run_EAD_IKr(ind_5, i, IC=None)
    info, result = detect_EAD(t5,v5)


    mdp = min(v5)
    max_p = max(v5)
    max_p_idx = np.argmax(v5)
    apa = max_p - mdp  
        
    repol_pot = max_p - apa * 90/100
    idx_apd = np.argmin(np.abs(v5[max_p_idx:] - repol_pot))
    apd_val = t5[idx_apd+max_p_idx]        
    apd_val5.append(apd_val)

    ind_5b = pd.DataFrame(apd_val5, columns=["IND5"])



    # IND 6
    t6,v6 = run_EAD_IKr(ind_6, i, IC=None)
    info, result = detect_EAD(t6,v6)


    mdp = min(v6)
    max_p = max(v6)
    max_p_idx = np.argmax(v6)
    apa = max_p - mdp  
        
    repol_pot = max_p - apa * 90/100
    idx_apd = np.argmin(np.abs(v6[max_p_idx:] - repol_pot))
    apd_val = t6[idx_apd+max_p_idx]        
    apd_val6.append(apd_val)
    ind_6b = pd.DataFrame(apd_val6, columns=["IND6"])




    # IND 7
    t7,v7 = run_EAD_IKr(ind_7, i, IC=None)
    info, result = detect_EAD(t7,v7)


    mdp = min(v7)
    max_p = max(v7)
    max_p_idx = np.argmax(v7)
    apa = max_p - mdp  
        
    repol_pot = max_p - apa * 90/100
    idx_apd = np.argmin(np.abs(v7[max_p_idx:] - repol_pot))
    apd_val = t7[idx_apd+max_p_idx]        
    apd_90.append(apd_val)
    apd_val7.append(apd_val)
    ind_7b = pd.DataFrame(apd_val7, columns=["IND7"])



    # IND 8
    t8,v8 = run_EAD_IKr(ind_8, i, IC=None)
    info, result = detect_EAD(t8,v8)


    mdp = min(v8)
    max_p = max(v8)
    max_p_idx = np.argmax(v8)
    apa = max_p - mdp  
        
    repol_pot = max_p - apa * 90/100
    idx_apd = np.argmin(np.abs(v8[max_p_idx:] - repol_pot))
    apd_val = t8[idx_apd+max_p_idx]        
    apd_val8.append(apd_val)

    ind_8b = pd.DataFrame(apd_val8, columns=["IND8"])


    if i == 0.1:
        df0 = ind_tor.join(ind_1b, how="outer")
        df1 = df0.join(ind_2b, how="outer")
        df2 = df1.join(ind_3b, how="outer")
        df3 = df2.join(ind_4b, how="outer")
        df4 = df3.join(ind_5b, how="outer")
        df5 = df4.join(ind_6b, how="outer")
        df6 = df5.join(ind_7b, how="outer")
        df7 = df6.join(ind_8b, how="outer")
        print(df7)

df7.to_excel('FS_IKr_block.xlsx', sheet_name='Sheet1')

        
    
# %%
