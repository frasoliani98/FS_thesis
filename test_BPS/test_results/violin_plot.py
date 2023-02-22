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

# %%
# TOR ORD
ind_t1 = pd.read_excel('Best_ind_tor1.xlsx')
ind_t1 = pd.DataFrame(ind_t1)
indts = ind_t1

ind_t2 = pd.read_excel('Best_ind_tor2.xlsx')
ind_t2 = pd.DataFrame(ind_t2)
indts = pd.concat([indts, ind_t2])

ind_t3 = pd.read_excel('Best_ind_tor3.xlsx')
ind_t3 = pd.DataFrame(ind_t3)
indts = pd.concat([indts, ind_t3])

ind_t4 = pd.read_excel('Best_ind_tor4.xlsx')
ind_t4 = pd.DataFrame(ind_t4)
indts = pd.concat([indts, ind_t4])

ind_t5 = pd.read_excel('Best_ind_tor5.xlsx')
ind_t5 = pd.DataFrame(ind_t5)
indts = pd.concat([indts, ind_t5])

ind_t6 = pd.read_excel('Best_ind_tor6.xlsx')
ind_t6 = pd.DataFrame(ind_t6)
indts = pd.concat([indts, ind_t6])

ind_t7 = pd.read_excel('Best_ind_tor7.xlsx')
ind_t7 = pd.DataFrame(ind_t7)
indts = pd.concat([indts, ind_t7])

ind_t8 = pd.read_excel('Best_ind_tor8.xlsx')
ind_t8 = pd.DataFrame(ind_t8)
indts = pd.concat([indts, ind_t8])


# BPS
ind_1 = pd.read_excel('Best_ind_noCa_morph1.xlsx')
ind_1 = pd.DataFrame(ind_1)
inds = ind_1

ind_2 = pd.read_excel('Best_ind_noCa_morph2.xlsx')
ind_2 = pd.DataFrame(ind_2)
inds = pd.concat([inds, ind_2])

ind_3 = pd.read_excel('Best_ind_noCa_morph3.xlsx')
ind_3 = pd.DataFrame(ind_3)
inds = pd.concat([inds, ind_3])

ind_4 = pd.read_excel('Best_ind_noCa_morph4.xlsx')
ind_4 = pd.DataFrame(ind_4)
inds = pd.concat([inds, ind_4])

ind_5 = pd.read_excel('Best_ind_noCa_morph5.xlsx')
ind_5 = pd.DataFrame(ind_5)
inds = pd.concat([inds, ind_5])

ind_6 = pd.read_excel('Best_ind_noCa_morph12.xlsx')
ind_6 = pd.DataFrame(ind_6)
inds = pd.concat([inds, ind_6])

ind_7 = pd.read_excel('Best_ind_noCa_morph7.xlsx')
ind_7 = pd.DataFrame(ind_7)
inds = pd.concat([inds, ind_7])

ind_8 = pd.read_excel('Best_ind_noCamorph14.xlsx')
ind_8 = pd.DataFrame(ind_8)
inds = pd.concat([inds, ind_8])



# PLOT
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Conductance', fontsize=14)
ax.set_xticks([i for i in range(1, 3)])
ax.set_xticklabels(['GCaL', 'GKr'], fontsize=10)
pos = [1, 2]

plt.violinplot(inds, pos, widths=0.05)
plt.violinplot(indts, pos, widths=0.05)

legend_drawn_flag = True
plt.legend(['BPS model', 'Tor-ORd model'], loc='upper left')
plt.show()

print(inds)
