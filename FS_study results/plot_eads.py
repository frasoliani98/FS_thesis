# %%
import random
from math import log10
import matplotlib.pyplot as plt
from scipy.signal import find_peaks # pip install scipy
import seaborn as sns # pip install seaborn
from multiprocessing import Pool
import numpy as np
import pandas as pd
from important_functions_3 import run_EAD, detect_EAD, plot_GA, baseline_run, baseline_EAD_CaL
from deap import base, creator, tools # pip install deap
import myokit

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_xlabel('Time [ms]', fontsize=14)
ax.set_ylabel('Voltage [mV]', fontsize=14)



# IND 1
ind_1 = pd.read_excel('Best_ind_1.xlsx')
ind_1 = ind_1.to_dict('index')
t1, v1 = run_EAD(ind_1)
info, result = detect_EAD(t1,v1)
print(info)

ax.plot(t1, v1, 'y', label = 'Trial 1')



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


# Baseline
tb, vb = baseline_EAD_CaL()
info, result = detect_EAD(tb,vb)
ax.plot(tb, vb, 'k--', label = 'Baseline')
print(info)


plt.legend(loc='best')
plt.show()