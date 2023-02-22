# %%
import random
from math import log10
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

# %%
# IND1
ind_1 = pd.read_excel('Best_ind_noCa_morph1.xlsx')
ind_1 = pd.DataFrame(ind_1)
inds = ind_1

# IND2
ind_2 = pd.read_excel('Best_ind_noCa_morph2.xlsx')
ind_2 = pd.DataFrame(ind_2)
inds = pd.concat([inds, ind_2])

#IND3
ind_3 = pd.read_excel('Best_ind_noCa_morph3.xlsx')
ind_3 = pd.DataFrame(ind_3)
inds = pd.concat([inds, ind_3])


#IND4
ind_4 = pd.read_excel('Best_ind_noCa_morph4.xlsx')
ind_4 = pd.DataFrame(ind_4)
inds = pd.concat([inds, ind_4])

#IND5
ind_5 = pd.read_excel('Best_ind_noCa_morph5.xlsx')
ind_5 = pd.DataFrame(ind_5)
inds = pd.concat([inds, ind_5])

#IND6
ind_6 = pd.read_excel('Best_ind_noCa_morph12.xlsx')
ind_6 = pd.DataFrame(ind_6)
inds = pd.concat([inds, ind_6])

#IND7
ind_7 = pd.read_excel('Best_ind_noCa_morph7.xlsx')
ind_7 = pd.DataFrame(ind_7)
inds = pd.concat([inds, ind_7])

#IND8
ind_8 = pd.read_excel('Best_ind_noCamorph14.xlsx')
ind_8 = pd.DataFrame(ind_8)
inds = pd.concat([inds, ind_8])



fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


curr_x = 0
colors = itertools.cycle(['y', 'r', 'b', 'g', 'c', 'm', 'orangered', 'deeppink'])
labels = itertools.cycle(['Trial 1', 'Trial 2', 'Trial 3', 'Trial 4', 'Trial 5', 'Trial 6', 'Trial 7', 'Trial 8'])
for k, conds in inds.items():
        for i, g in enumerate(conds):
            g = log10(g)
            x = curr_x + np.random.normal(0, .01)
            if curr_x == 0:
                ax.scatter(x, g, color=next(colors))

            if curr_x == 1:    
                ax.scatter(x, g, color=next(colors), label = next(labels))
            

        curr_x += 1
        

lower_bound = 0.1
upper_bound = 10

ax.hlines(0, -.5, (2-.5), colors='grey', linestyle='--')
ax.set_xticks([i for i in range(0, 2)])
ax.set_xticklabels(['GCaL','GKr'], fontsize=10)
ax.set_ylim(log10(lower_bound), 
                    log10(upper_bound))
ax.set_ylabel('Log10 Conductance', fontsize=14)
fig.suptitle(f'Best Conductances from 8 trials', fontsize=14)
plt.legend(loc='best')
plt.show()

# %%
