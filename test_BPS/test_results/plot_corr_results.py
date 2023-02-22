import random
from math import log10
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import seaborn as sns

ical = list()
ikr = list()

# READ THE DATA
# just load the data, then everything is ok!
ind_1 = pd.read_excel('Best_ind_noCa_morph1.xlsx')
ind_1 = pd.DataFrame(ind_1)
ical.append(ind_1['i_cal_pca_multiplier'][0])
ikr.append(ind_1['i_kr_multiplier'][0])


ind_2 = pd.read_excel('Best_ind_noCa_morph2.xlsx')
ind_2 = pd.DataFrame(ind_2)
ical.append(ind_2['i_cal_pca_multiplier'][0])
ikr.append(ind_2['i_kr_multiplier'][0])

ind_3 = pd.read_excel('Best_ind_noCa_morph3.xlsx')
ind_3 = pd.DataFrame(ind_3)
ical.append(ind_3['i_cal_pca_multiplier'][0])
ikr.append(ind_3['i_kr_multiplier'][0])

ind_4 = pd.read_excel('Best_ind_noCa_morph4.xlsx')
ind_4 = pd.DataFrame(ind_4)
ical.append(ind_4['i_cal_pca_multiplier'][0])
ikr.append(ind_4['i_kr_multiplier'][0])

ind_5 = pd.read_excel('Best_ind_noCa_morph5.xlsx')
ind_5 = pd.DataFrame(ind_5)
ical.append(ind_5['i_cal_pca_multiplier'][0])
ikr.append(ind_5['i_kr_multiplier'][0])

ind_6 = pd.read_excel('Best_ind_noCa_morph12.xlsx')
ind_6 = pd.DataFrame(ind_6)
ical.append(ind_6['i_cal_pca_multiplier'][0])
ikr.append(ind_6['i_kr_multiplier'][0])

ind_7 = pd.read_excel('Best_ind_noCa_morph7.xlsx')
ind_7 = pd.DataFrame(ind_7)
ical.append(ind_7['i_cal_pca_multiplier'][0])
ikr.append(ind_7['i_kr_multiplier'][0])

ind_8 = pd.read_excel('Best_ind_noCamorph14.xlsx')
ind_8 = pd.DataFrame(ind_8)
ical.append(ind_8['i_cal_pca_multiplier'][0])
ikr.append(ind_8['i_kr_multiplier'][0])

frames = [ind_1, ind_2, ind_3, ind_4, ind_5, ind_6, ind_7, ind_8,]
result = pd.concat(frames)
print(result)





# PLOT
fig, ax = plt.subplots(1, 1, figsize = (12, 6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

fig.suptitle('High Correlation between GKr and GCaL', fontsize=14)

sns.regplot(x = "i_kr_multiplier", y = "i_cal_pca_multiplier", data = result, x_estimator=np.mean)
ax.plot(ikr, ical, 'ro')
ax.set_xlabel('GKr Conductance', fontsize=14)
ax.set_ylabel('GCaL Conductance', fontsize=14)


plt.show()
