# %%

import matplotlib.pyplot as plt
import pandas as pd


# %%
# plot of individals and the baseline with different blocks: [0%, 20%, 40%, 60%, 80%, 100%]
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Percentage of ICaL Enhancements [%]', fontsize=14)
ax.set_ylabel('APD 90 [ms]', fontsize=14)
fig.suptitle(f'APD 90', fontsize=14)
ax.set_xticks([i for i in range(1, 7)])
ax.set_xticklabels(['0%', '200%', '400%', '600%', '800%', '1000%'], fontsize=10)
x = [1, 2, 3, 4, 5, 6]

# Baseline
ind_b = pd.read_excel('FS_ICaL_enha.xlsx', sheet_name='Sheet1')
ind_b = pd.DataFrame(ind_b)
ind_b = ind_b['IND_BASE'].values.tolist()


# IND1
ind_1 = pd.read_excel('FS_ICaL_enha.xlsx', sheet_name='Sheet1')
ind_1 = pd.DataFrame(ind_1)
ind_1 = ind_1['IND1'].values.tolist()

# IND2
ind_2 = pd.read_excel('FS_ICaL_enha.xlsx', sheet_name='Sheet1')
ind_2 = pd.DataFrame(ind_2)
ind_2 = ind_2['IND2'].values.tolist()

# IND3
ind_3 = pd.read_excel('FS_ICaL_enha.xlsx', sheet_name='Sheet1')
ind_3 = pd.DataFrame(ind_3)
ind_3 = ind_3['IND3'].values.tolist()

# IND4
ind_4 = pd.read_excel('FS_ICaL_enha.xlsx', sheet_name='Sheet1')
ind_4 = pd.DataFrame(ind_4)
ind_4 = ind_4['IND4'].values.tolist()

# IND5
ind_5 = pd.read_excel('FS_ICaL_enha.xlsx', sheet_name='Sheet1')
ind_5 = pd.DataFrame(ind_5)
ind_5 = ind_5['IND5'].values.tolist()

# IND6
ind_6 = pd.read_excel('FS_ICaL_enha.xlsx', sheet_name='Sheet1')
ind_6 = pd.DataFrame(ind_6)
ind_6 = ind_6['IND1'].values.tolist()

# IND7
ind_7 = pd.read_excel('FS_ICaL_enha.xlsx', sheet_name='Sheet1')
ind_7 = pd.DataFrame(ind_7)
ind_7 = ind_7['IND7'].values.tolist()

# IND8
ind_8 = pd.read_excel('FS_ICaL_enha.xlsx', sheet_name='Sheet1')
ind_8 = pd.DataFrame(ind_8)
ind_8 = ind_8['IND8'].values.tolist()



ax.plot(x, ind_b, 'k--', label = 'baseline')
ax.plot(x, ind_b, 'ko')

ax.plot(x, ind_1, 'y', label = 'Trial 1')
ax.plot(x, ind_1, 'yo')

ax.plot(x, ind_2, 'r', label = 'Trial 2')
ax.plot(x, ind_2, 'ro')

ax.plot(x, ind_3, 'b', label = 'Trial 3')
ax.plot(x, ind_3, 'bo')

ax.plot(x, ind_4, 'g', label = 'Trial 4')
ax.plot(x, ind_4, 'go')

ax.plot(x, ind_5, 'c', label = 'Trial 5')
ax.plot(x, ind_5, 'co')

ax.plot(x, ind_6, 'm', label = 'Trial 6')
ax.plot(x, ind_6, 'mo')

ax.plot(x, ind_7, 'g', label = 'Trial 7')
ax.plot(x, ind_7, 'go')

ax.plot(x, ind_8, 'y', label = 'Trial 8')
ax.plot(x, ind_8, 'yo')











plt.legend(loc='best')
plt.show()