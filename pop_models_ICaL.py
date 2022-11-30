#%% Import libraries
import time
import myokit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pop_model_functions import run_EAD, detect_EAD


# %%
# NO EADS: BASELINE CASE
models = pd.read_csv('pop_models.csv')
labels = []

for i in range(2000):
    ical = models['i_cal_pca_multiplier'][i]
    ikr = models['i_kr_multiplier'][i]

    if ical>1 and ikr>1:
        labels.append('upregulated')
    
    if ical<1 and ikr<1:
        labels.append('downregulated')
    
    if (ical>1 and ikr<1) or (ical<1 and ikr>1):
        labels.append('independent')

models['label'] = labels

# check if it works
models_labeled = pd.DataFrame(models)
save_data_to = 'pop_models_labeled.csv'
models_labeled.to_csv(save_data_to, index = False)

df_labeled = models_labeled.filter(items= ['t','v','label'])
df_upreg = df_labeled[df_labeled["label"].str.contains("upregulated")]
df_downreg = df_labeled[df_labeled["label"].str.contains("downregulated")]
df_ind = df_labeled[df_labeled["label"].str.contains("independent")]

# PLOT VALUES
t_up = df_upreg['t'].values.tolist()
v_up = df_upreg['v'].values.tolist()

t_down = df_downreg['t'].values.tolist()
v_down = df_downreg['v'].values.tolist()

t_ind = df_ind['t'].values.tolist()
v_ind = df_ind['v'].values.tolist()


fig, axs = plt.subplots(3, 1, figsize = (15,15))
for ax in axs:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
palette1 = sns.color_palette("Paired", 2000).as_hex()
palette2 = sns.color_palette("Paired", 2000).as_hex()
palette3 = sns.color_palette("Paired", 2000).as_hex()

for i in range(len(t_up)):
    axs[0].plot(eval(t_up[i]), eval(v_up[i]), color = palette1[i], linewidth = 0.2)
    axs[1].plot(eval(t_down[i]), eval(v_down[i]), color = palette2[i], linewidth = 0.2)
    axs[2].plot(eval(t_ind[i]), eval(v_ind[i]), color = palette3[i], linewidth = 0.2)

fig.suptitle(f'NO EADS BASELINE', fontsize = 14)
axs[1].set_ylabel('Voltage [mV]', fontsize = 14)
axs[2].set_xlabel('Time [ms]', fontsize = 14)
axs[0].set_title(f'Co-Expressed Positive', fontsize = 14)
axs[1].set_title(f'Co-Expressed Negative', fontsize = 14)
axs[2].set_title(f'Independent', fontsize = 14)
axs[0].get_xaxis().set_visible(False)
axs[1].get_xaxis().set_visible(False)

plt.show()

# %%
# EADs
df_eads = models_labeled.filter(items= ['i_cal_pca_multiplier','i_ks_multiplier','i_kr_multiplier','i_nal_multiplier','i_na_multiplier','i_to_multiplier','i_k1_multiplier','i_NCX_multiplier','i_nak_multiplier','i_kb_multiplier','label'])
df_upreg_eads = df_eads[df_eads["label"].str.contains("upregulated")]
df_downreg_eads = df_eads[df_eads["label"].str.contains("downregulated")]
df_ind_eads = df_eads[df_eads["label"].str.contains("independent")]


# APPLY EADS
# 1) UP
ical_up = df_upreg_eads['i_cal_pca_multiplier'].values.tolist()
iks_up = df_upreg_eads['i_ks_multiplier'].values.tolist()
ikr_up = df_upreg_eads['i_kr_multiplier'].values.tolist()
inal_up = df_upreg_eads['i_nal_multiplier'].values.tolist()
ina_up = df_upreg_eads['i_na_multiplier'].values.tolist()
ito_up = df_upreg_eads['i_to_multiplier'].values.tolist()
ik1_up = df_upreg_eads['i_k1_multiplier'].values.tolist()
iNCX_up = df_upreg_eads['i_NCX_multiplier'].values.tolist()
inak_up = df_upreg_eads['i_nak_multiplier'].values.tolist()
ikb_up = df_upreg_eads['i_kb_multiplier'].values.tolist()

# 2) DOWN
ical_down = df_downreg_eads['i_cal_pca_multiplier'].values.tolist()
iks_down = df_downreg_eads['i_ks_multiplier'].values.tolist()
ikr_down = df_downreg_eads['i_kr_multiplier'].values.tolist()
inal_down = df_downreg_eads['i_nal_multiplier'].values.tolist()
ina_down = df_downreg_eads['i_na_multiplier'].values.tolist()
ito_down = df_downreg_eads['i_to_multiplier'].values.tolist()
ik1_down = df_downreg_eads['i_k1_multiplier'].values.tolist()
iNCX_down = df_downreg_eads['i_NCX_multiplier'].values.tolist()
inak_down = df_downreg_eads['i_nak_multiplier'].values.tolist()
ikb_down = df_downreg_eads['i_kb_multiplier'].values.tolist()

# 3) IND
ical_ind = df_ind_eads['i_cal_pca_multiplier'].values.tolist()
iks_ind = df_ind_eads['i_ks_multiplier'].values.tolist()
ikr_ind = df_ind_eads['i_kr_multiplier'].values.tolist()
inal_ind = df_ind_eads['i_nal_multiplier'].values.tolist()
ina_ind = df_ind_eads['i_na_multiplier'].values.tolist()
ito_ind = df_ind_eads['i_to_multiplier'].values.tolist()
ik1_ind = df_ind_eads['i_k1_multiplier'].values.tolist()
iNCX_ind = df_ind_eads['i_NCX_multiplier'].values.tolist()
inak_ind = df_ind_eads['i_nak_multiplier'].values.tolist()
ikb_ind = df_ind_eads['i_kb_multiplier'].values.tolist()


fig, axs = plt.subplots(3, 1, figsize = (15,15))
for ax in axs:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

palette1 = sns.color_palette("Paired", 2000).as_hex()
palette2 = sns.color_palette("Paired", 2000).as_hex()
palette3 = sns.color_palette("Paired", 2000).as_hex()

eads_up = 0
eads_down = 0
eads_ind = 0

for i in range(2000):
    # 3 RUN EADS
    t_up_eads, v_up_eads = run_EAD(ical_up[i], iks_up[i], ikr_up[i], inal_up[i], ina_up[i], ito_up[i], ik1_up[i], iNCX_up[i], inak_up[i], ikb_up[i])
    info_up, result_up = detect_EAD(t_up_eads, v_up_eads)
    if result_up == 1:
        eads_up = eads_up + 1
    axs[0].plot(t_up_eads, v_up_eads, color = palette1[i], linewidth = 0.2)
   
    t_down_eads, v_down_eads = run_EAD(ical_down[i], iks_down[i], ikr_down[i], inal_down[i], ina_down[i], ito_down[i], ik1_down[i], iNCX_down[i], inak_down[i], ikb_down[i])
    info_down, result_down = detect_EAD(t_down_eads, v_down_eads)
    if result_down == 1:
        eads_down = eads_down + 1
    axs[1].plot(t_down_eads, v_down_eads, color = palette2[i], linewidth = 0.2)
    
    t_ind_eads, v_ind_eads = run_EAD(ical_ind[i], iks_ind[i], ikr_ind[i], inal_ind[i], ina_ind[i], ito_ind[i], ik1_ind[i], iNCX_ind[i], inak_ind[i], ikb_ind[i])
    info_ind, result_ind = detect_EAD(t_ind_eads, v_ind_eads)
    if result_ind == 1:
        eads_ind = eads_ind + 1
    axs[2].plot(t_ind_eads, v_ind_eads, color = palette3[i], linewidth = 0.2)

fig.suptitle(f'EADS ICaL*8', fontsize = 14)
axs[1].set_ylabel('Voltage [mV]', fontsize = 14)
axs[2].set_xlabel('Time [ms]', fontsize = 14)
axs[0].set_title(f'Co-Expressed Positive', fontsize = 14)
axs[1].set_title(f'Co-Expressed Negative', fontsize = 14)
axs[2].set_title(f'Independent', fontsize = 14)
axs[0].get_xaxis().set_visible(False)
axs[1].get_xaxis().set_visible(False)

# Figure number EADs vs Cases ICaL
fig, ax = plt.subplots(1, 1, figsize = (15,15))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_xticks([i for i in range(1, 4)])
ax.set_xticklabels(['Co-Ex +', 'Co-Ex -', 'Ind'], fontsize=10)
x = [1, 2, 3]

ax.plot(x, eads_up, 'r')
ax.plot(x, eads_down, 'r')
ax.plot(x, eads_ind, 'r')

fig.suptitle(f'Number of EADs ICaL', fontsize = 14)
ax.set_ylabel('Voltage [mV]', fontsize = 14)
ax.set_xlabel('Time [ms]', fontsize = 14)



plt.show()