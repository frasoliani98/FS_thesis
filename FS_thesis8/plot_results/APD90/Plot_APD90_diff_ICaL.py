# %%

import matplotlib.pyplot as plt
import pandas as pd


# %%
# plot of individals and the baseline with different blocks: [0%, 20%, 40%, 60%, 80%, 100%]
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Percentage of ICaL Enhancements [%]', fontsize=14)
ax.set_ylabel('APD 90 difference [ms]', fontsize=14)
fig.suptitle(f'APD 90 CHANGE', fontsize=14)
ax.set_xticks([i for i in range(1, 6)])
ax.set_xticklabels(['200%', '400%', '600%', '800%', '1000%'], fontsize=10)
x = [1, 2, 3, 4, 5]

apd90_diff_b = list()
apd90_diff_1 = list()
apd90_diff_2 = list()
apd90_diff_3 = list()
apd90_diff_4 = list()
apd90_diff_5 = list()
apd90_diff_6 = list()
apd90_diff_7 = list()
apd90_diff_8 = list()

# Baseline
ind_b = pd.read_excel('FS_ICaL_enha.xlsx', sheet_name='Sheet1')
ind_b = pd.DataFrame(ind_b)
ind_b = ind_b['IND_BASE'].values.tolist()

for i in ind_b:
    apd90_diff = i - ind_b[0]
    if apd90_diff != 0: #to get rid of the first value (difference with itself)
        apd90_diff_b.append(apd90_diff)



# IND1
ind_1 = pd.read_excel('FS_ICaL_enha.xlsx', sheet_name='Sheet1')
ind_1 = pd.DataFrame(ind_1)
ind_1 = ind_1['IND1'].values.tolist()

for i in ind_1:
    apd90_diff = i - ind_1[0]
    if apd90_diff != 0:
        apd90_diff_1.append(apd90_diff)

# IND2
ind_2 = pd.read_excel('FS_ICaL_enha.xlsx', sheet_name='Sheet1')
ind_2 = pd.DataFrame(ind_2)
ind_2 = ind_2['IND2'].values.tolist()

for i in ind_2:
    apd90_diff = i - ind_2[0]
    if apd90_diff != 0:
        apd90_diff_2.append(apd90_diff)

# IND3
ind_3 = pd.read_excel('FS_ICaL_enha.xlsx', sheet_name='Sheet1')
ind_3 = pd.DataFrame(ind_3)
ind_3 = ind_3['IND3'].values.tolist()

for i in ind_3:
    apd90_diff = i - ind_3[0]
    if apd90_diff != 0:
        apd90_diff_3.append(apd90_diff)

# IND4
ind_4 = pd.read_excel('FS_ICaL_enha.xlsx', sheet_name='Sheet1')
ind_4 = pd.DataFrame(ind_4)
ind_4 = ind_4['IND4'].values.tolist()

for i in ind_4:
    apd90_diff = i - ind_4[0]
    if apd90_diff != 0:
        apd90_diff_4.append(apd90_diff)


# IND5
ind_5 = pd.read_excel('FS_ICaL_enha.xlsx', sheet_name='Sheet1')
ind_5 = pd.DataFrame(ind_5)
ind_5 = ind_5['IND5'].values.tolist()

for i in ind_5:
    apd90_diff = i - ind_5[0]
    if apd90_diff != 0:
        apd90_diff_5.append(apd90_diff)

# IND6
ind_6 = pd.read_excel('FS_ICaL_enha.xlsx', sheet_name='Sheet1')
ind_6 = pd.DataFrame(ind_6)
ind_6 = ind_6['IND1'].values.tolist()

for i in ind_6:
    apd90_diff = i - ind_6[0]
    if apd90_diff != 0:
        apd90_diff_6.append(apd90_diff)

# IND7
ind_7 = pd.read_excel('FS_ICaL_enha.xlsx', sheet_name='Sheet1')
ind_7 = pd.DataFrame(ind_7)
ind_7 = ind_7['IND7'].values.tolist()

for i in ind_7:
    apd90_diff = i - ind_7[0]
    if apd90_diff != 0:
        apd90_diff_7.append(apd90_diff)

# IND8
ind_8 = pd.read_excel('FS_ICaL_enha.xlsx', sheet_name='Sheet1')
ind_8 = pd.DataFrame(ind_8)
ind_8 = ind_8['IND8'].values.tolist()

for i in ind_8:
    apd90_diff = i - ind_8[0]
    if apd90_diff != 0:
        apd90_diff_8.append(apd90_diff)



ax.plot(x, apd90_diff_b, 'k--', label = 'baseline')
ax.plot(x, apd90_diff_b, 'ko')

ax.plot(x, apd90_diff_1, 'y', label = 'Trial 1')
ax.plot(x, apd90_diff_1, 'yo')

ax.plot(x, apd90_diff_2, 'r', label = 'Trial 2')
ax.plot(x, apd90_diff_2, 'ro')

ax.plot(x, apd90_diff_3, 'b', label = 'Trial 3')
ax.plot(x, apd90_diff_3, 'bo')

ax.plot(x, apd90_diff_4, 'g', label = 'Trial 4')
ax.plot(x, apd90_diff_4, 'go')

ax.plot(x, apd90_diff_5, 'c', label = 'Trial 5')
ax.plot(x, apd90_diff_5, 'co')

ax.plot(x, apd90_diff_6, 'm', label = 'Trial 6')
ax.plot(x, apd90_diff_6, 'mo')

ax.plot(x, apd90_diff_7, 'g', label = 'Trial 7')
ax.plot(x, apd90_diff_7, 'go')

ax.plot(x, apd90_diff_8, 'y', label = 'Trial 8')
ax.plot(x, apd90_diff_8, 'yo')




plt.legend(loc='best')
plt.show()
