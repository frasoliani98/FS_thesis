# %%
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
from important_functions_ca import plot_ca, plot_ca_baseline

# %%
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_xlabel('Time [ms]', fontsize=14)
#ax.set_ylabel('Voltage [mV]', fontsize=14)
ax.set_ylabel('Cai ', fontsize=14)
#axs[2].set_ylabel('IK1', fontsize=14)
#axs[3].set_ylabel('IKb', fontsize=14)
#axs[4].set_ylabel('IKs', fontsize=14)
ax.set_xlim(0,1000)
#axs[2].set_xlim(0,1000)
#axs[3].set_xlim(0,1000)
#axs[4].set_xlim(0,1000)

# IND 1
ind_1 = pd.read_excel('Best_ind_noCa_morph7.xlsx')
ind_1 = ind_1.to_dict('index')
tc, ICaL = plot_ca(ind_1)


ax.plot(tc, ICaL, 'r', label = 'Best Ind')
#axs[1].plot(tc, IKr, 'b')
#axs[2].plot(tc, IK1, 'r')
#axs[3].plot(tc, IKb, 'g')
#axs[4].plot(tc, IKs, 'm')

'''
# IND 2
ind_2 = pd.read_excel('Best_ind_2.xlsx')
ind_2 = ind_2.to_dict('index')
t2, v2 = run_EAD(ind_2)
info, result = detect_EAD(t2,v2)
print(info)

ax.plot(t2, v2, 'r', label = 'Trial 2')



# IND 4
ind_4 = pd.read_excel('Best_ind_4.xlsx')
ind_4 = ind_4.to_dict('index')
t4, v4 = run_EAD(ind_4)
info, result = detect_EAD(t4,v4)
print(info)

ax.plot(t4, v4, 'g', label = 'Trial 4')

# IND 5
ind_5 = pd.read_excel('Best_ind_5.xlsx')
ind_5 = ind_5.to_dict('index')
t5, v5 = run_EAD(ind_5)
info, result = detect_EAD(t5,v5)
print(info)

ax.plot(t5, v5, 'c', label = 'Trial 5')

# IND 6
ind_6 = pd.read_excel('Best_ind_6.xlsx')
ind_6 = ind_6.to_dict('index')
t6, v6 = run_EAD(ind_6)
info, result = detect_EAD(t6,v6)
print(info)

ax.plot(t6, v6, 'm', label = 'Trial 6')


# IND 7
ind_7 = pd.read_excel('Best_ind_7.xlsx')
ind_7 = ind_7.to_dict('index')
t7, v7 = run_EAD(ind_7)
info, result = detect_EAD(t7,v7)
print(info)

ax.plot(t7, v7, color = 'orangered', label = 'Trial 7')
'''

# Baseline
tcb, ICaLb = plot_ca_baseline()

ax.plot(tcb, ICaLb, 'k--', label = 'Baseline')
#axs[1].plot(tcb, IKr, 'k--')
#axs[2].plot(tcb, IK1, 'k--')
#axs[3].plot(tcb, IKb, 'k--')
#axs[4].plot(tcb, IKs, 'k--')


plt.legend(loc='best')
plt.show()