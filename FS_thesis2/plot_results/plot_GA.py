import random
from math import log10
import matplotlib.pyplot as plt
from scipy.signal import find_peaks # pip install scipy
from multiprocessing import Pool
import numpy as np
import pandas as pd
from plot_results.important_functions import baseline_run, plot_GA

from deap import base, creator, tools # pip install deap
import myokit

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Time [ms]', fontsize=14)
ax.set_ylabel('Voltage [mV]', fontsize=14)
ax.set_xlim(0,800)
fig.suptitle(f'Results from 8 trials', fontsize=14)

# IND 1
ind_1 = pd.read_excel('Best_ind_1.xlsx')
ind_1 = ind_1.to_dict('index')
t1, v1 = plot_GA(ind_1)
ax.plot(t1, v1, 'y', label = 'Trial 1')


# IND 2
ind_2 = pd.read_excel('Best_ind_2.xlsx')
ind_2 = ind_2.to_dict('index')

t2, v2 = plot_GA(ind_2)
ax.plot(t2, v2, 'r', label = 'Trial 2')

# IND 3
ind_3 = pd.read_excel('Best_ind_3.xlsx')
ind_3 = ind_3.to_dict('index')

t3, v3 = plot_GA(ind_3)
ax.plot(t3, v3, 'b', label = 'Trial 3')

# IND 4
ind_4 = pd.read_excel('Best_ind_4.xlsx')
ind_4 = ind_4.to_dict('index')

t4, v4 = plot_GA(ind_4)
ax.plot(t4, v4, 'g', label = 'Trial 4')

# IND 5
ind_5 = pd.read_excel('Best_ind_5.xlsx')
ind_5 = ind_5.to_dict('index')

t5, v5 = plot_GA(ind_5)
ax.plot(t5, v5, 'c', label = 'Trial 5')

# IND 6
ind_6 = pd.read_excel('Best_ind_6.xlsx')
ind_6 = ind_6.to_dict('index')

t6, v6 = plot_GA(ind_6)
ax.plot(t6, v6, 'm', label = 'Trial 6')


# IND 7
ind_7 = pd.read_excel('Best_ind_7.xlsx')
ind_7 = ind_7.to_dict('index')

t7, v7 = plot_GA(ind_7)
ax.plot(t7, v7, color = 'orangered', label = 'Trial 7')

# IND 8
ind_8 = pd.read_excel('Best_ind_8.xlsx')
ind_8 = ind_8.to_dict('index')

t8, v8 = plot_GA(ind_8)
ax.plot(t8, v8, color = 'deeppink', label = 'Trial 8')


# Baseline
tb, vb = baseline_run()
ax.plot(tb, vb, 'k--', label = 'Baseline')

plt.legend(loc='best')
plt.show()