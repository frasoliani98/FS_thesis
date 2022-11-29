#%% Plot population of models of the 4 cases:
# 1) BASELINE
# 2) CO-ESPRESSED POSITIVE
# 3) CO-EXPRESSED NEGATIVE
# 4) INDEPENDENT

# Libraries
import pandas as pd
import matplotlib.pyplot as plt
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


# LOAD DATA
data = pd.read_csv('IKr_data.csv')
data = pd.DataFrame(data)

#%%
data.head()

#%%


# 1) BASELINE

t_baseline_1 = data['t_1'].values.tolist()
v_baseline_1 = data['v_1'].values.tolist()
t_baseline_0_8 = data['t_0.8'].values.tolist()
v_baseline_0_8 = data['v_0.8'].values.tolist()
t_baseline_0_6 = data['t_0.6'].values.tolist()
v_baseline_0_6 = data['v_0.6'].values.tolist()
t_baseline_0_4 = data['t_0.4'].values.tolist()
v_baseline_0_4 = data['v_0.4'].values.tolist()
t_baseline_0_2 = data['t_0.2'].values.tolist()
v_baseline_0_2 = data['v_0.2'].values.tolist()
t_baseline_0_1 = data['t_0.1'].values.tolist()
v_baseline_0_1 = data['v_0.1'].values.tolist()
t_baseline_0_01 = data['t_0.01'].values.tolist()
v_baseline_0_01 = data['v_0.01'].values.tolist()





# PLOT BASELINE
fig, axs = plt.subplots(7, 1, figsize=(15, 15))

for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        

fig.suptitle(f'Baseline Case', fontsize=14)

for i in range(0, len(t_baseline_1)):
    axs[0].plot(eval(t_baseline_1[i]), eval(v_baseline_1[i]), 'b', label='Baseline 1')
    axs[1].plot(eval(t_baseline_0_8[i]), eval(v_baseline_0_8[i]), 'r', label='Baseline 0.8')
    axs[2].plot(eval(t_baseline_0_6[i]), eval(v_baseline_0_6[i]), 'g', label='Baseline 0.6')
    axs[3].plot(eval(t_baseline_0_4[i]), eval(v_baseline_0_4[i]), 'c', label='Baseline 0.4')
    axs[4].plot(eval(t_baseline_0_2[i]), eval(v_baseline_0_2[i]), 'm', label='Baseline 0.2')
    axs[5].plot(eval(t_baseline_0_1[i]), eval(v_baseline_0_1[i]), color = 'orangered', label='Baseline 0.1')
    axs[6].plot(eval(t_baseline_0_01[i]), eval(v_baseline_0_01[i]), color ='deeppink', label='Baseline 0.01')

axs[6].set_xlabel('Time [ms]', fontsize=14)
axs[3].set_ylabel('Voltage [mV]', fontsize=14)

plt.show()





# 2) CO-EXPRESSED POSITIVE
# t_initial_co_pos	v_initial_co_pos	

t_co_pos_1 = data['t_1_co_pos'].values.tolist()
v_co_pos_1 = data['v_1_co_pos'].values.tolist()
t_co_pos_0_8 = data['t_0.8_co_pos'].values.tolist()
v_co_pos_0_8 = data['v_0.8_co_pos'].values.tolist()
t_co_pos_0_6 = data['t_0.6_co_pos'].values.tolist()
v_co_pos_0_6 = data['v_0.6_co_pos'].values.tolist()
t_co_pos_0_4 = data['t_0.4_co_pos'].values.tolist()
v_co_pos_0_4 = data['v_0.4_co_pos'].values.tolist()
t_co_pos_0_2 = data['t_0.2_co_pos'].values.tolist()
v_co_pos_0_2 = data['v_0.2_co_pos'].values.tolist()
t_co_pos_0_1 = data['t_0.1_co_pos'].values.tolist()
v_co_pos_0_1 = data['v_0.1_co_pos'].values.tolist()
t_co_pos_0_01 = data['t_0.01_co_pos'].values.tolist()
v_co_pos_0_01 = data['v_0.01_co_pos'].values.tolist()


# PLOT COEXPRESSED POSITIVE
fig, axs = plt.subplots(7, 1, figsize=(15, 15))

for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)


fig.suptitle(f'Co-Expressed Positive Case', fontsize=14)

for i in range(0, len(t_co_pos_1)):
    axs[0].plot(eval(t_co_pos_1[i]), eval(v_co_pos_1[i]), 'b', label='Co pos 1')
    axs[1].plot(eval(t_co_pos_0_8[i]), eval(v_co_pos_0_8[i]), 'r', label='Co pos 0.8')
    axs[2].plot(eval(t_co_pos_0_6[i]), eval(v_co_pos_0_6[i]), 'g', label='Co pos 0.6')
    axs[3].plot(eval(t_co_pos_0_4[i]), eval(v_co_pos_0_4[i]), 'c', label='Co pos 0.4')
    axs[4].plot(eval(t_co_pos_0_2[i]), eval(v_co_pos_0_2[i]), 'm', label='Co pos 0.2')
    axs[5].plot(eval(t_co_pos_0_1[i]), eval(v_co_pos_0_1[i]), color = 'orangered', label='Co pos 0.1')
    axs[6].plot(eval(t_co_pos_0_01[i]), eval(v_co_pos_0_01[i]), color ='deeppink', label='Co pos 0.01')

axs[6].set_xlabel('Time [ms]', fontsize=14)
axs[3].set_ylabel('Voltage [mV]', fontsize=14)
plt.show()



# 3) CO-EXPRESSED NEGATIVE
# t_initial_co_neg	v_initial_co_neg	

t_co_neg_1 = data['t_1_co_neg'].values.tolist()
v_co_neg_1 = data['v_1_co_neg'].values.tolist()
t_co_neg_0_8 = data['t_0.8_co_neg'].values.tolist()
v_co_neg_0_8 = data['v_0.8_co_neg'].values.tolist()
t_co_neg_0_6 = data['t_0.6_co_neg'].values.tolist()
v_co_neg_0_6 = data['v_0.6_co_neg'].values.tolist()
t_co_neg_0_4 = data['t_0.4_co_neg'].values.tolist()
v_co_neg_0_4 = data['v_0.4_co_neg'].values.tolist()
t_co_neg_0_2 = data['t_0.2_co_neg'].values.tolist()
v_co_neg_0_2 = data['v_0.2_co_neg'].values.tolist()
t_co_neg_0_1 = data['t_0.1_co_neg'].values.tolist()
v_co_neg_0_1 = data['v_0.1_co_neg'].values.tolist()
t_co_neg_0_01 = data['t_0.01_co_neg'].values.tolist()
v_co_neg_0_01 = data['v_0.01_co_neg'].values.tolist()

# PLOT COEXPRESSED NEGATIVE
fig, axs = plt.subplots(7, 1, figsize=(15, 15))

for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        

fig.suptitle(f'Co-Expressed Negative Case', fontsize=14)

for i in range(0, len(t_co_neg_1)):
    axs[0].plot(eval(t_co_neg_1[i]), eval(v_co_neg_1[i]), 'b', label='Co neg 1')
    axs[1].plot(eval(t_co_neg_0_8[i]), eval(v_co_neg_0_8[i]), 'r', label='Co neg 0.8')
    axs[2].plot(eval(t_co_neg_0_6[i]), eval(v_co_neg_0_6[i]), 'g', label='Co neg 0.6')
    axs[3].plot(eval(t_co_neg_0_4[i]), eval(v_co_neg_0_4[i]), 'c', label='Co neg 0.4')
    axs[4].plot(eval(t_co_neg_0_2[i]), eval(v_co_neg_0_2[i]), 'm', label='Co neg 0.2')
    axs[5].plot(eval(t_co_neg_0_1[i]), eval(v_co_neg_0_1[i]), color = 'orangered', label='Co neg 0.1')
    axs[6].plot(eval(t_co_neg_0_01[i]), eval(v_co_neg_0_01[i]), color ='deeppink', label='Co neg 0.01')

axs[6].set_xlabel('Time [ms]', fontsize=14)
axs[3].set_ylabel('Voltage [mV]', fontsize=14)
plt.show()


# 4) INDEPENDENT
# t_initial_independ	v_initial_independ	

t_independ_1 = data['t_1_independ'].values.tolist()
v_independ_1 = data['v_1_independ'].values.tolist()
t_independ_0_8 = data['t_0.8_independ'].values.tolist()
v_independ_0_8 = data['v_0.8_independ'].values.tolist()
t_independ_0_6 = data['t_0.6_independ'].values.tolist()
v_independ_0_6 = data['v_0.6_independ'].values.tolist()
t_independ_0_4 = data['t_0.4_independ'].values.tolist()
v_independ_0_4 = data['v_0.4_independ'].values.tolist()
t_independ_0_2 = data['t_0.2_independ'].values.tolist()
v_independ_0_2 = data['v_0.2_independ'].values.tolist()
t_independ_0_1 = data['t_0.1_independ'].values.tolist()
v_independ_0_1 = data['v_0.1_independ'].values.tolist()
t_independ_0_01 = data['t_0.01_independ'].values.tolist()
v_independ_0_01 = data['v_0.01_independ'].values.tolist()


# PLOT INDEPENDENT
fig, axs = plt.subplots(7, 1, figsize=(15, 15))

for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)


fig.suptitle(f'Independent Case', fontsize=14)

for i in range(0, len(t_independ_1)):
    axs[0].plot(eval(t_independ_1[i]), eval(v_independ_1[i]), 'b', label='Indep 1')
    axs[1].plot(eval(t_independ_0_8[i]), eval(v_independ_0_8[i]), 'r', label='Indep 0.8')
    axs[2].plot(eval(t_independ_0_6[i]), eval(v_independ_0_6[i]), 'g', label='Indep 0.6')
    axs[3].plot(eval(t_independ_0_4[i]), eval(v_independ_0_4[i]), 'c', label='Indep 0.4')
    axs[4].plot(eval(t_independ_0_2[i]), eval(v_independ_0_2[i]), 'm', label='Indep 0.2')
    axs[5].plot(eval(t_independ_0_1[i]), eval(v_independ_0_1[i]), color = 'orangered', label='Indep 0.1')
    axs[6].plot(eval(t_independ_0_01[i]), eval(v_independ_0_01[i]), color ='deeppink', label='Indep 0.01')

axs[6].set_xlabel('Time [ms]', fontsize=14)
axs[3].set_ylabel('Voltage [mV]', fontsize=14)
plt.show()

# %%
