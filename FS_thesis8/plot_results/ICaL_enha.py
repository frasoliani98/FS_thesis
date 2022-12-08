# %%

import matplotlib.pyplot as plt

import pandas as pd
import itertools
from important_functions import run_EAD_IKr, detect_EAD, baseline_run, plot_GA, baseline_EAD, run_EAD, baseline_EAD_ICalenha, run_EAD_ICal
import itertools

# %%
# CODE TO PLOT THE RESULTS WITH THE SINGLE DIFFERENT TYPE OF BLOCKS AS I WANT

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Time [ms]', fontsize=14)
ax.set_ylabel('Voltage [mV]', fontsize=14)
fig.suptitle(f'Results ICaL*8', fontsize=14)

apd_90 = []
apd_90b = []

colors = itertools.cycle(['y', 'r', 'b', 'g', 'c', 'm', 'orangered', 'deeppink'])
labels = itertools.cycle(['Trial 1', 'Trial 2', 'Trial 3', 'Trial 4', 'Trial 5', 'Trial 6', 'Trial 7', 'Trial 8'])

block = 8

# IND1
ind_1 = pd.read_excel('Best_ind_1.xlsx')
ind_1 = ind_1.to_dict('index')
t1,v1 = run_EAD_ICal(ind_1, block, IC=None)
info, result = detect_EAD(t1,v1)
t1b, v1b = plot_GA(ind_1)

print(info)
ax.plot(t1, v1, color = next(colors), label = next(labels))

# IND 2
ind_2 = pd.read_excel('Best_ind_2.xlsx')
ind_2 = ind_2.to_dict('index')
t2,v2 = run_EAD_ICal(ind_2, block, IC=None)
info, result = detect_EAD(t2,v2)
t2b, v2b = plot_GA(ind_2)

print(info)
ax.plot(t2, v2, 'r', label = 'Trial 2')


# IND 3
ind_3 = pd.read_excel('Best_ind_3.xlsx')
ind_3 = ind_3.to_dict('index')
t3,v3 = run_EAD_ICal(ind_3, block, IC=None)
info, result = detect_EAD(t3,v3)
t3b, v3b = plot_GA(ind_3)

print(info)
ax.plot(t3, v3, 'b', label = 'Trial 3')



# IND 4
ind_4 = pd.read_excel('Best_ind_4.xlsx')
ind_4 = ind_4.to_dict('index')
t4,v4 = run_EAD_ICal(ind_4, block, IC=None)
info, result = detect_EAD(t4,v4)
t4b, v4b = plot_GA(ind_4)

print(info)
ax.plot(t4, v4, 'g', label = 'Trial 4')



# IND 5
ind_5 = pd.read_excel('Best_ind_5.xlsx')
ind_5 = ind_5.to_dict('index')
t5,v5 = run_EAD_ICal(ind_5, block, IC=None)
info, result = detect_EAD(t5,v5)
t5b, v5b = plot_GA(ind_5)

print(info)
ax.plot(t5, v5, 'c', label = 'Trial 5')


# IND 6
ind_6 = pd.read_excel('Best_ind_6.xlsx')
ind_6 = ind_6.to_dict('index')
t6,v6 = run_EAD_ICal(ind_6, block, IC=None)
info, result = detect_EAD(t6,v6)
t6b, v6b = plot_GA(ind_6)

print(info)
ax.plot(t6, v6, 'm', label = 'Trial 6')


# IND 7
ind_7 = pd.read_excel('Best_ind_7.xlsx')
ind_7 = ind_7.to_dict('index')
t7,v7 = run_EAD_ICal(ind_7, block, IC=None)
info, result = detect_EAD(t7,v7)
t7b, v7b = plot_GA(ind_7)

print(info)
ax.plot(t7, v7, color = 'orangered', label = 'Trial 7')


# IND 8
ind_8 = pd.read_excel('Best_ind_8.xlsx')
ind_8 = ind_8.to_dict('index')
t8,v8 = run_EAD_ICal(ind_8, block, IC=None)
info, result = detect_EAD(t8,v8)
t8b, v8b = plot_GA(ind_8)

print(info)
ax.plot(t8, v8, color = 'deeppink', label = 'Trial 8')

'''
t = itertools.cycle([t1, t2, t3, t4, t5, t6, t7, t8])
tb = itertools.cycle([t1b, t2b, t3b, t4b, t5b, t6b, t7b, t8b])
v = itertools.cycle([v1, v2, v3, v4, v5, v6, v7, v8])
vb = itertools.cycle([v1b, v2b, v3b, v4b, v5b, v6b, v7b, v8b])

for i in range(8):
    mdp = min(next(v))
    max_p = max(next(v))
    max_p_idx = np.argmax(next(v))
    apa = max_p - mdp  
        
    repol_pot = max_p - apa * 90/100
    idx_apd = np.argmin(np.abs(next(v)[max_p_idx:] - repol_pot))
    apd_val = next(t)[idx_apd+max_p_idx]        
    apd_90.append(apd_val)

    # APD baseline 
    mdp = min(next(vb))
    max_p = max(next(vb))
    max_p_idx = np.argmax(next(vb))
    apa = max_p - mdp  
        
    repol_pot = max_p - apa * 90/100
    idx_apd = np.argmin(np.abs(next(vb)[max_p_idx:] - repol_pot))
    apd_val = next(tb)[idx_apd+max_p_idx]        
    apd_90b.append(apd_val)

'''
# Baseline
tb, vb = baseline_EAD_ICalenha(block)
ax.plot(tb, vb, 'k--', label = 'Baseline')

'''


# IND1
ind_1 = pd.read_excel('Best_ind_1.xlsx')
ind_1 = ind_1.to_dict('index')
t1,v1 = run_EAD_IKr(ind_1, IC=None)
info, result = detect_EAD(t1,v1)

print(info)
ax.plot(t1, v1, 'y', label = 'Trial 1')

mdp = min(v1)
max_p = max(v1)
max_p_idx = np.argmax(v1)
apa = max_p - mdp
dvdt_max = np.max(np.diff(v1[0:30])/np.diff(t1[0:30]))    
    
repol_pot = max_p - apa * 90/100
idx_apd = np.argmin(np.abs(v1[max_p_idx:] - repol_pot))
apd_val = t1[idx_apd+max_p_idx]        
apd_90.append(apd_val)

# APD baseline 1
t1b, v1b = plot_GA(ind_1)
mdp = min(v1b)
max_p = max(v1b)
max_p_idx = np.argmax(v1b)
apa = max_p - mdp
dvdt_max = np.max(np.diff(v1b[0:30])/np.diff(t1b[0:30]))    
    
repol_pot = max_p - apa * 90/100
idx_apd = np.argmin(np.abs(v1b[max_p_idx:] - repol_pot))
apd_val = t1b[idx_apd+max_p_idx]        
apd_90b.append(apd_val)



# IND 2
ind_2 = pd.read_excel('Best_ind_2.xlsx')
ind_2 = ind_2.to_dict('index')
t2,v2 = run_EAD_IKr(ind_2, IC=None)
info, result = detect_EAD(t2,v2)

print(info)
ax.plot(t2, v2, 'r', label = 'Trial 2')

mdp = min(v2)
max_p = max(v2)
max_p_idx = np.argmax(v2)
apa = max_p - mdp 
    
repol_pot = max_p - apa * 90/100
idx_apd = np.argmin(np.abs(v2[max_p_idx:] - repol_pot))
apd_val = t2[idx_apd+max_p_idx]        
apd_90.append(apd_val)

# APD baseline 2
t2b, v2b = plot_GA(ind_2)

mdp = min(v2b)
max_p = max(v2b)
max_p_idx = np.argmax(v2b)
apa = max_p - mdp 
    
repol_pot = max_p - apa * 90/100
idx_apd = np.argmin(np.abs(v2b[max_p_idx:] - repol_pot))
apd_val = t2b[idx_apd+max_p_idx]        
apd_90b.append(apd_val)


# IND 3
ind_3 = pd.read_excel('Best_ind_3.xlsx')
ind_3 = ind_3.to_dict('index')
t3,v3 = run_EAD_IKr(ind_3, IC=None)
info, result = detect_EAD(t3,v3)

print(info)
ax.plot(t3, v3, 'b', label = 'Trial 3')

mdp = min(v3)
max_p = max(v3)
max_p_idx = np.argmax(v3)
apa = max_p - mdp
    
repol_pot = max_p - apa * 90/100
idx_apd = np.argmin(np.abs(v2[max_p_idx:] - repol_pot))
apd_val = t3[idx_apd+max_p_idx]        
apd_90.append(apd_val)

# APD baseline 3
t3b, v3b = plot_GA(ind_3)
mdp = min(v3b)
max_p = max(v3b)
max_p_idx = np.argmax(v3b)
apa = max_p - mdp 
    
repol_pot = max_p - apa * 90/100
idx_apd = np.argmin(np.abs(v3b[max_p_idx:] - repol_pot))
apd_val = t3b[idx_apd+max_p_idx]        
apd_90b.append(apd_val)



# IND 4
ind_4 = pd.read_excel('Best_ind_4.xlsx')
ind_4 = ind_4.to_dict('index')
t4,v4 = run_EAD_IKr(ind_4, IC=None)
info, result = detect_EAD(t4,v4)

print(info)
ax.plot(t4, v4, 'g', label = 'Trial 4')

mdp = min(v4)
max_p = max(v4)
max_p_idx = np.argmax(v4)
apa = max_p - mdp  
    
repol_pot = max_p - apa * 90/100
idx_apd = np.argmin(np.abs(v4[max_p_idx:] - repol_pot))
apd_val = t4[idx_apd+max_p_idx]        
apd_90.append(apd_val)

# APD baseline 4
t4b, v4b = plot_GA(ind_4)
mdp = min(v4b)
max_p = max(v4b)
max_p_idx = np.argmax(v4b)
apa = max_p - mdp

    
repol_pot = max_p - apa * 90/100
idx_apd = np.argmin(np.abs(v4b[max_p_idx:] - repol_pot))
apd_val = t4b[idx_apd+max_p_idx]        
apd_90b.append(apd_val)



# IND 5
ind_5 = pd.read_excel('Best_ind_5.xlsx')
ind_5 = ind_5.to_dict('index')
t5,v5 = run_EAD_IKr(ind_5, IC=None)
info, result = detect_EAD(t5,v5)

print(info)
ax.plot(t5, v5, 'c', label = 'Trial 5')

mdp = min(v5)
max_p = max(v5)
max_p_idx = np.argmax(v5)
apa = max_p - mdp  
    
repol_pot = max_p - apa * 90/100
idx_apd = np.argmin(np.abs(v5[max_p_idx:] - repol_pot))
apd_val = t5[idx_apd+max_p_idx]        
apd_90.append(apd_val)

# APD baseline 5
t5b, v5b = plot_GA(ind_5)
mdp = min(v5b)
max_p = max(v5b)
max_p_idx = np.argmax(v5b)
apa = max_p - mdp
    
repol_pot = max_p - apa * 90/100
idx_apd = np.argmin(np.abs(v5b[max_p_idx:] - repol_pot))
apd_val = t5b[idx_apd+max_p_idx]        
apd_90b.append(apd_val)



# IND 6
ind_6 = pd.read_excel('Best_ind_6.xlsx')
ind_6 = ind_6.to_dict('index')
t6,v6 = run_EAD_IKr(ind_6, IC=None)
info, result = detect_EAD(t6,v6)

print(info)
ax.plot(t6, v6, 'm', label = 'Trial 6')

mdp = min(v6)
max_p = max(v6)
max_p_idx = np.argmax(v6)
apa = max_p - mdp  
    
repol_pot = max_p - apa * 90/100
idx_apd = np.argmin(np.abs(v6[max_p_idx:] - repol_pot))
apd_val = t6[idx_apd+max_p_idx]        
apd_90.append(apd_val)

# APD baseline 6
t6b, v6b = plot_GA(ind_6)
mdp = min(v6b)
max_p = max(v6b)
max_p_idx = np.argmax(v6b)
apa = max_p - mdp
    
repol_pot = max_p - apa * 90/100
idx_apd = np.argmin(np.abs(v6b[max_p_idx:] - repol_pot))
apd_val = t6b[idx_apd+max_p_idx]        
apd_90b.append(apd_val)




# IND 7
ind_7 = pd.read_excel('Best_ind_7.xlsx')
ind_7 = ind_7.to_dict('index')
t7,v7 = run_EAD_IKr(ind_7, IC=None)
info, result = detect_EAD(t7,v7)

print(info)
ax.plot(t7, v7, color = 'orangered', label = 'Trial 7')

mdp = min(v7)
max_p = max(v7)
max_p_idx = np.argmax(v7)
apa = max_p - mdp  
    
repol_pot = max_p - apa * 90/100
idx_apd = np.argmin(np.abs(v7[max_p_idx:] - repol_pot))
apd_val = t7[idx_apd+max_p_idx]        
apd_90.append(apd_val)

# APD baseline 7
t7b, v7b = plot_GA(ind_7)
mdp = min(v7b)
max_p = max(v7b)
max_p_idx = np.argmax(v7b)
apa = max_p - mdp
    
repol_pot = max_p - apa * 90/100
idx_apd = np.argmin(np.abs(v7b[max_p_idx:] - repol_pot))
apd_val = t7b[idx_apd+max_p_idx]        
apd_90b.append(apd_val)



# IND 8
ind_8 = pd.read_excel('Best_ind_8.xlsx')
ind_8 = ind_8.to_dict('index')
t8,v8 = run_EAD_IKr(ind_8, IC=None)
info, result = detect_EAD(t8,v8)

print(info)
ax.plot(t8, v8, color = 'deeppink', label = 'Trial 8')

mdp = min(v8)
max_p = max(v8)
max_p_idx = np.argmax(v8)
apa = max_p - mdp  
    
repol_pot = max_p - apa * 90/100
idx_apd = np.argmin(np.abs(v8[max_p_idx:] - repol_pot))
apd_val = t8[idx_apd+max_p_idx]        
apd_90.append(apd_val)

# APD baseline 8
t8b, v8b = plot_GA(ind_8)
mdp = min(v8b)
max_p = max(v8b)
max_p_idx = np.argmax(v8b)
apa = max_p - mdp
    
repol_pot = max_p - apa * 90/100
idx_apd = np.argmin(np.abs(v8b[max_p_idx:] - repol_pot))
apd_val = t8b[idx_apd+max_p_idx]        
apd_90b.append(apd_val)




# Baseline
tb, vb = baseline_run()
ax.plot(tb, vb, 'k--', label = 'Baseline')


# APD baseline 1
apd_val = 300   
apd_90b.append(apd_val)
apd_90.append(apd_val)



# BLOCK 20%
tb1,vb1 = baseline_EAD(block = 0.8)
info, result = detect_EAD(tb1,vb1)

print(info)
ax.plot(tb1, vb1, 'y', label = 'Block 20%')

# APD
mdp = min(vb1)
max_p = max(vb1)
max_p_idx = np.argmax(vb1)
apa = max_p - mdp
    
repol_pot = max_p - apa * 90/100
idx_apd = np.argmin(np.abs(vb1[max_p_idx:] - repol_pot))
apd_val = tb1[idx_apd+max_p_idx]        
apd_90.append(apd_val)


# BLOCK 40%
tb1,vb1 = baseline_EAD(block = 0.6)
info, result = detect_EAD(tb1,vb1)

print(info)
ax.plot(tb1, vb1, 'r', label = 'Block 60%')

# APD
mdp = min(vb1)
max_p = max(vb1)
max_p_idx = np.argmax(vb1)
apa = max_p - mdp
    
repol_pot = max_p - apa * 90/100
idx_apd = np.argmin(np.abs(vb1[max_p_idx:] - repol_pot))
apd_val = tb1[idx_apd+max_p_idx]        
apd_90.append(apd_val)


# BLOCK 60%
tb1,vb1 = baseline_EAD(block = 0.4)
info, result = detect_EAD(tb1,vb1)

print(info)
ax.plot(tb1, vb1, 'g', label = 'Block 60%')

# APD
mdp = min(vb1)
max_p = max(vb1)
max_p_idx = np.argmax(vb1)
apa = max_p - mdp
    
repol_pot = max_p - apa * 90/100
idx_apd = np.argmin(np.abs(vb1[max_p_idx:] - repol_pot))
apd_val = tb1[idx_apd+max_p_idx]        
apd_90.append(apd_val)


# BLOCK 80%
tb1,vb1 = baseline_EAD(block = 0.2)
info, result = detect_EAD(tb1,vb1)

print(info)
ax.plot(tb1, vb1, 'b', label = 'Block 80%')

# APD
mdp = min(vb1)
max_p = max(vb1)
max_p_idx = np.argmax(vb1)
apa = max_p - mdp
    
repol_pot = max_p - apa * 90/100
idx_apd = np.argmin(np.abs(vb1[max_p_idx:] - repol_pot))
apd_val = tb1[idx_apd+max_p_idx]        
apd_90.append(apd_val)


# BLOCK 90%
tb1,vb1 = baseline_EAD(block = 0.1)
info, result = detect_EAD(tb1,vb1)

print(info)
ax.plot(tb1, vb1, 'm', label = 'Block 90%')

# APD
mdp = min(vb1)
max_p = max(vb1)
max_p_idx = np.argmax(vb1)
apa = max_p - mdp
    
repol_pot = max_p - apa * 90/100
idx_apd = np.argmin(np.abs(vb1[max_p_idx:] - repol_pot))
apd_val = tb1[idx_apd+max_p_idx]        
apd_90.append(apd_val)


# BLOCK 95%
#tb1,vb1 = baseline_EAD(block = 0.05)
#info, result = detect_EAD(tb1,vb1)

#print(info)
#ax.plot(tb1, vb1, 'c', label = 'Block 95%')





array1 = np.array(apd_90)
array2 = np.array(apd_90b)
subtracted_array = np.subtract(array1, array2)
apd_f = list(subtracted_array)
print(apd_90)
print(apd_f)
apd_90bf = pd.DataFrame(apd_90b, columns=["APD 90 Baseline"])
apd_90f = pd.DataFrame(apd_90, columns=["APD 90"])
apd_f = pd.DataFrame(apd_f, columns=["IKr Block"])
df1 = apd_90f.join(apd_90bf, how="outer")
df1 = df1.join(apd_f, how="outer")
df1.to_excel('FS_IKr_block.xlsx', sheet_name='Sheet1')
'''
plt.legend(loc='best')
plt.show()