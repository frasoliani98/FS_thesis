# %%
import random
from math import log10
import matplotlib.pyplot as plt
from scipy.signal import find_peaks # pip install scipy
import seaborn as sns # pip install seaborn
from multiprocessing import Pool
import numpy as np
import pandas as pd
from important_functions import run_EAD, detect_EAD
from deap import base, creator, tools # pip install deap
import myokit

# %%
gen = list(range(1,21))
best_err = pd.read_excel('FS_errors1.xlsx', sheet_name='Sheet1')
best_err = pd.DataFrame(best_err)
best_err_f1 = best_err['Best Error'].values.tolist()

best_err = pd.read_excel('FS_errors2.xlsx', sheet_name='Sheet1')
best_err = pd.DataFrame(best_err)
best_err_f2 = best_err['Best Error'].values.tolist()

best_err = pd.read_excel('FS_errors3.xlsx', sheet_name='Sheet1')
best_err = pd.DataFrame(best_err)
best_err_f3 = best_err['Best Error'].values.tolist()

best_err = pd.read_excel('FS_errors4.xlsx', sheet_name='Sheet1')
best_err = pd.DataFrame(best_err)
best_err_f4 = best_err['Best Error'].values.tolist()

best_err = pd.read_excel('FS_errors5.xlsx', sheet_name='Sheet1')
best_err = pd.DataFrame(best_err)
best_err_f5 = best_err['Best Error'].values.tolist()

best_err = pd.read_excel('FS_errors6.xlsx', sheet_name='Sheet1')
best_err = pd.DataFrame(best_err)
best_err_f6 = best_err['Best Error'].values.tolist()

best_err = pd.read_excel('FS_errors7.xlsx', sheet_name='Sheet1')
best_err = pd.DataFrame(best_err)
best_err_f7 = best_err['Best Error'].values.tolist()

best_err = pd.read_excel('FS_errors8.xlsx', sheet_name='Sheet1')
best_err = pd.DataFrame(best_err)
best_err_f8 = best_err['Best Error'].values.tolist()


fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Generations', fontsize=14)
ax.set_ylabel('Errors', fontsize=14)
fig.suptitle('Best error', fontsize=14)

ax.plot(gen, best_err_f1, 'yo', label = 'Trial 1')
ax.plot(gen, best_err_f2, 'ro', label = 'Trial 2')
ax.plot(gen, best_err_f3, 'bo', label = 'Trial 3')
ax.plot(gen, best_err_f4, 'go', label = 'Trial 4')
ax.plot(gen, best_err_f5, 'co', label = 'Trial 5')
ax.plot(gen, best_err_f6, 'mo', label = 'Trial 6')
ax.plot(gen, best_err_f7, 'o', color = 'orangered', label = 'Trial 7')
ax.plot(gen, best_err_f8, 'o', color = 'deeppink', label = 'Trial 8')
plt.legend(loc='best')


plt.show()

# %%
