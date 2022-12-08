#%% Import libraries
import time
import myokit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pop_model_functions import run_EAD


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
    
    if ical>1 and ikr<1:
        labels.append('independent')

    if ical<1 and ikr>1:
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
    
palette1 = sns.color_palette("Paired", len(t_up)).as_hex()
palette2 = sns.color_palette("Paired", len(t_down)).as_hex()
palette3 = sns.color_palette("Paired", len(t_ind)).as_hex()

for i in range(len(t_up)):
    axs[0].plot(eval(t_up[i]), eval(v_up[i]), color = palette1[i], linewidth = 0.2)

for i in range(len(t_down)):
    axs[1].plot(eval(t_down[i]), eval(v_down[i]), color = palette2[i], linewidth = 0.2)

for i in range(len(t_ind)):
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

palette1 = sns.color_palette("Paired", len(ical_up)).as_hex()
palette2 = sns.color_palette("Paired", len(ical_down)).as_hex()
palette3 = sns.color_palette("Paired", len(ical_ind)).as_hex()

#palette1 = sns.color_palette("Paired", 5).as_hex()
#palette2 = sns.color_palette("Paired", 5).as_hex()
#palette3 = sns.color_palette("Paired", 5).as_hex()

enha = 6


for i in range(len(ical_up)):
#for i in range(5):
    # 3 RUN EAD
    t_up_eads, v_up_eads, t2up, v2up = run_EAD(ical_up[i], iks_up[i], ikr_up[i], inal_up[i], ina_up[i], ito_up[i], ik1_up[i], iNCX_up[i], inak_up[i], ikb_up[i], enha)

    axs[0].plot(t_up_eads, v_up_eads, color = palette1[i], linewidth = 0.2)
    axs[0].plot(t2up+1000, v2up, color = palette1[i], linewidth = 0.2)

for i in range(len(ical_down)):
#for i in range(5):
    t_down_eads, v_down_eads, t2down, v2down = run_EAD(ical_down[i], iks_down[i], ikr_down[i], inal_down[i], ina_down[i], ito_down[i], ik1_down[i], iNCX_down[i], inak_down[i], ikb_down[i], enha)
    
    axs[1].plot(t_down_eads, v_down_eads, color = palette2[i], linewidth = 0.2)
    axs[1].plot(t2down+1000, v2down, color = palette2[i], linewidth = 0.2)

for i in range(len(ical_ind)):
#for i in range(5):
    t_ind_eads, v_ind_eads, t2ind, v2ind = run_EAD(ical_ind[i], iks_ind[i], ikr_ind[i], inal_ind[i], ina_ind[i], ito_ind[i], ik1_ind[i], iNCX_ind[i], inak_ind[i], ikb_ind[i], enha)
    
    axs[2].plot(t_ind_eads, v_ind_eads, color = palette3[i], linewidth = 0.2)
    axs[2].plot(t2ind+1000, v2ind, color = palette3[i], linewidth = 0.2)



# %%
# PLOT
fig.suptitle(f'EADS ICaL*6', fontsize = 14)
axs[1].set_ylabel('Voltage [mV]', fontsize = 14)
axs[2].set_xlabel('Time [ms]', fontsize = 14)
axs[0].set_title(f'Co-Expressed Positive', fontsize = 14)
axs[1].set_title(f'Co-Expressed Negative', fontsize = 14)
axs[2].set_title(f'Independent', fontsize = 14)
axs[0].get_xaxis().set_visible(False)
axs[1].get_xaxis().set_visible(False)

plt.show()
