# %%
import random
from math import log10
import matplotlib.pyplot as plt
from scipy.signal import find_peaks # pip install scipy
import seaborn as sns # pip install seaborn
from multiprocessing import Pool
import numpy as np
import pandas as pd
from eads_vs_noeads_functions import run_EAD, detect_EAD, plot_GA, baseline_run, baseline_EAD_CaL
from deap import base, creator, tools # pip install deap
import myokit


# Baseline+Best ind: ead vs no eads for intracalcium and intra potassium
# and ical and ikr

# 1) NO EADs

fig, axs = plt.subplots(5, 1, figsize=(8, 6))
plt.suptitle('NO EADs')
for ax in axs:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

axs[4].set_xlabel('Time [ms]', fontsize=14)
axs[0].set_ylabel('Voltage [mV]', fontsize=14)
axs[1].set_ylabel('Intra [Cai] ', fontsize=14)
axs[2].set_ylabel('Intra [Ki]', fontsize=14)
axs[3].set_ylabel('ICaL', fontsize=14)
axs[4].set_ylabel('IKr', fontsize=14)
axs[0].set_xlim(0,1000)
axs[1].set_xlim(0,1000)
axs[2].set_xlim(0,1000)
axs[3].set_xlim(0,1000)
axs[4].set_xlim(0,1000)


ind_1 = pd.read_excel('Best_ind_1.xlsx')
ind_1 = ind_1.to_dict('index')

t1, v1, cai, ki, ICal, IKr, tc = plot_GA(ind_1)
tb, vb, caib, kib, ICalb, IKrb, tcb  = baseline_run()



axs[0].plot(t1, v1, 'y')
axs[1].plot(tc, cai, 'b')
axs[2].plot(tc, ki, 'r')
axs[3].plot(tc, ICal, 'g')
axs[4].plot(tc, IKr, 'm')

axs[0].plot(tb, vb, 'k--')
axs[1].plot(tcb, caib, 'k--')
axs[2].plot(tcb, kib, 'k--')
axs[3].plot(tcb, ICalb, 'k--')
axs[4].plot(tcb, IKrb, 'k--')



# 2) EADs

fig, axs = plt.subplots(5, 1, figsize=(8, 6))
plt.suptitle('EADs')
for ax in axs:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

axs[4].set_xlabel('Time [ms]', fontsize=14)
axs[0].set_ylabel('Voltage [mV]', fontsize=14)
axs[1].set_ylabel('Intra [Cai] ', fontsize=14)
axs[2].set_ylabel('Intra [Ki]', fontsize=14)
axs[3].set_ylabel('ICaL', fontsize=14)
axs[4].set_ylabel('IKr', fontsize=14)
axs[0].set_xlim(0,1000)
axs[1].set_xlim(0,1000)
axs[2].set_xlim(0,1000)
axs[3].set_xlim(0,1000)
axs[4].set_xlim(0,1000)




t1, v1, cai, ki, ICal, IKr, tc = run_EAD(ind_1)
info, result = detect_EAD(t1,v1)
print(info)


tb, vb, caib, kib, ICalb, IKrb, tcb  = baseline_EAD_CaL()
info, result = detect_EAD(tb,vb)
print(info)

axs[0].plot(t1, v1, 'y')
axs[1].plot(tc, cai, 'b')
axs[2].plot(tc, ki, 'r')
axs[3].plot(tc, ICal, 'g')
axs[4].plot(tc, IKr, 'm')

axs[0].plot(tb, vb, 'k--')
axs[1].plot(tcb, caib, 'k--')
axs[2].plot(tcb, kib, 'k--')
axs[3].plot(tcb, ICalb, 'k--')
axs[4].plot(tcb, IKrb, 'k--')

plt.show()