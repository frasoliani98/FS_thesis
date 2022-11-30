#%% Import libraries
import time
import myokit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pop_model_functions import run_EAD, detect_EAD, run_EAD_IKr


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


palette1 = sns.color_palette("Paired", 2000).as_hex()
palette2 = sns.color_palette("Paired", 2000).as_hex()
palette3 = sns.color_palette("Paired", 2000).as_hex()

eads_up_final = list()
eads_down_final = list()
eads_ind_final = list()
eads_start = 0

# Case 0% (no EADs)
eads_up_final.append(eads_start)
eads_down_final.append(eads_start)
eads_ind_final.append(eads_start)

# 1) 20% block = 0.8
eads_up20 = 0
eads_down20 = 0
eads_ind20 = 0
block = 0.8

for i in range(2000):
    # 3 RUN EADS
    
    t_up_eads, v_up_eads = run_EAD_IKr(ical_up[i], iks_up[i], ikr_up[i], inal_up[i], ina_up[i], ito_up[i], ik1_up[i], iNCX_up[i], inak_up[i], ikb_up[i], block)
    info_up, result_up = detect_EAD(t_up_eads, v_up_eads)
    if result_up == 1:
        eads_up20 = eads_up20 + 1
   
    t_down_eads, v_down_eads = run_EAD_IKr(ical_down[i], iks_down[i], ikr_down[i], inal_down[i], ina_down[i], ito_down[i], ik1_down[i], iNCX_down[i], inak_down[i], ikb_down[i], block)
    info_down, result_down = detect_EAD(t_down_eads, v_down_eads)
    if result_down == 1:
        eads_down20 = eads_down20 + 1
    
    t_ind_eads, v_ind_eads = run_EAD_IKr(ical_ind[i], iks_ind[i], ikr_ind[i], inal_ind[i], ina_ind[i], ito_ind[i], ik1_ind[i], iNCX_ind[i], inak_ind[i], ikb_ind[i], block)
    info_ind, result_ind = detect_EAD(t_ind_eads, v_ind_eads)
    if result_ind == 1:
        eads_ind20 = eads_ind20 + 1


eads_up_final.append(eads_up20)
eads_down_final.append(eads_down20)
eads_ind_final.append(eads_ind20)


# 2) 40% block = 0.6
eads_up40 = 0
eads_down40 = 0
eads_ind40 = 0
block = 0.6

for i in range(2000):
    # 3 RUN EADS
    
    t_up_eads, v_up_eads = run_EAD_IKr(ical_up[i], iks_up[i], ikr_up[i], inal_up[i], ina_up[i], ito_up[i], ik1_up[i], iNCX_up[i], inak_up[i], ikb_up[i], block)
    info_up, result_up = detect_EAD(t_up_eads, v_up_eads)
    if result_up == 1:
        eads_up40 = eads_up40 + 1
   
    t_down_eads, v_down_eads = run_EAD_IKr(ical_down[i], iks_down[i], ikr_down[i], inal_down[i], ina_down[i], ito_down[i], ik1_down[i], iNCX_down[i], inak_down[i], ikb_down[i], block)
    info_down, result_down = detect_EAD(t_down_eads, v_down_eads)
    if result_down == 1:
        eads_down40 = eads_down40 + 1
    
    t_ind_eads, v_ind_eads = run_EAD_IKr(ical_ind[i], iks_ind[i], ikr_ind[i], inal_ind[i], ina_ind[i], ito_ind[i], ik1_ind[i], iNCX_ind[i], inak_ind[i], ikb_ind[i], block)
    info_ind, result_ind = detect_EAD(t_ind_eads, v_ind_eads)
    if result_ind == 1:
        eads_ind40 = eads_ind40 + 1

eads_up_final.append(eads_up40)
eads_down_final.append(eads_down40)
eads_ind_final.append(eads_ind40)


# 3) 60% block = 0.4
eads_up60 = 0
eads_down60 = 0
eads_ind60 = 0
block = 0.4

for i in range(2000):
    # 3 RUN EADS
    
    t_up_eads, v_up_eads = run_EAD_IKr(ical_up[i], iks_up[i], ikr_up[i], inal_up[i], ina_up[i], ito_up[i], ik1_up[i], iNCX_up[i], inak_up[i], ikb_up[i], block)
    info_up, result_up = detect_EAD(t_up_eads, v_up_eads)
    if result_up == 1:
        eads_up60 = eads_up60 + 1
   
    t_down_eads, v_down_eads = run_EAD_IKr(ical_down[i], iks_down[i], ikr_down[i], inal_down[i], ina_down[i], ito_down[i], ik1_down[i], iNCX_down[i], inak_down[i], ikb_down[i], block)
    info_down, result_down = detect_EAD(t_down_eads, v_down_eads)
    if result_down == 1:
        eads_down60 = eads_down60 + 1
    
    t_ind_eads, v_ind_eads = run_EAD_IKr(ical_ind[i], iks_ind[i], ikr_ind[i], inal_ind[i], ina_ind[i], ito_ind[i], ik1_ind[i], iNCX_ind[i], inak_ind[i], ikb_ind[i], block)
    info_ind, result_ind = detect_EAD(t_ind_eads, v_ind_eads)
    if result_ind == 1:
        eads_ind60 = eads_ind60 + 1


eads_up_final.append(eads_up60)
eads_down_final.append(eads_down60)
eads_ind_final.append(eads_ind60)

# 4) 80% block = 0.2
eads_up80 = 0
eads_down80 = 0
eads_ind80 = 0
block = 0.8

for i in range(2000):
    # 3 RUN EADS
    
    t_up_eads, v_up_eads = run_EAD_IKr(ical_up[i], iks_up[i], ikr_up[i], inal_up[i], ina_up[i], ito_up[i], ik1_up[i], iNCX_up[i], inak_up[i], ikb_up[i], block)
    info_up, result_up = detect_EAD(t_up_eads, v_up_eads)
    if result_up == 1:
        eads_up80 = eads_up80 + 1
   
    t_down_eads, v_down_eads = run_EAD_IKr(ical_down[i], iks_down[i], ikr_down[i], inal_down[i], ina_down[i], ito_down[i], ik1_down[i], iNCX_down[i], inak_down[i], ikb_down[i], block)
    info_down, result_down = detect_EAD(t_down_eads, v_down_eads)
    if result_down == 1:
        eads_down80 = eads_down80 + 1
    
    t_ind_eads, v_ind_eads = run_EAD_IKr(ical_ind[i], iks_ind[i], ikr_ind[i], inal_ind[i], ina_ind[i], ito_ind[i], ik1_ind[i], iNCX_ind[i], inak_ind[i], ikb_ind[i], block)
    info_ind, result_ind = detect_EAD(t_ind_eads, v_ind_eads)
    if result_ind == 1:
        eads_ind80 = eads_ind80 + 1

eads_up_final.append(eads_up80)
eads_down_final.append(eads_down80)
eads_ind_final.append(eads_ind80)


# 5) 90% block = 0.1
eads_up90 = 0
eads_down90 = 0
eads_ind90 = 0
block90 = 0.1

for i in range(2000):
    # 3 RUN EADS
    
    t_up_eads, v_up_eads = run_EAD_IKr(ical_up[i], iks_up[i], ikr_up[i], inal_up[i], ina_up[i], ito_up[i], ik1_up[i], iNCX_up[i], inak_up[i], ikb_up[i], block)
    info_up, result_up = detect_EAD(t_up_eads, v_up_eads)
    if result_up == 1:
        eads_up90 = eads_up90 + 1
   
    t_down_eads, v_down_eads = run_EAD_IKr(ical_down[i], iks_down[i], ikr_down[i], inal_down[i], ina_down[i], ito_down[i], ik1_down[i], iNCX_down[i], inak_down[i], ikb_down[i], block)
    info_down, result_down = detect_EAD(t_down_eads, v_down_eads)
    if result_down == 1:
        eads_down90 = eads_down90 + 1
    
    t_ind_eads, v_ind_eads = run_EAD_IKr(ical_ind[i], iks_ind[i], ikr_ind[i], inal_ind[i], ina_ind[i], ito_ind[i], ik1_ind[i], iNCX_ind[i], inak_ind[i], ikb_ind[i], block)
    info_ind, result_ind = detect_EAD(t_ind_eads, v_ind_eads)
    if result_ind == 1:
        eads_ind90 = eads_ind90 + 1

eads_up_final.append(eads_up90)
eads_down_final.append(eads_down90)
eads_ind_final.append(eads_ind90)


# Figure number EADs vs Cases IKr blocks
fig, ax = plt.subplots(1, 1, figsize = (15,15))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_xticks([i for i in range(1, 4)])
ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '90%'], fontsize=10)
x = [1, 2, 3, 4, 5, 6]

ax.plot(x, eads_up_final, 'r', label = 'Co-Expressed Positive')
ax.plot(x, eads_up_final, 'ro')

ax.plot(x, eads_down_final, 'g', label = 'Co-Expressed Negative')
ax.plot(x, eads_down_final, 'go')

ax.plot(x, eads_ind_final, 'b', label = 'Independent')
ax.plot(x, eads_ind_final, 'bo')

fig.suptitle(f'Number of EADs ICaL', fontsize = 14)
ax.set_ylabel('Voltage [mV]', fontsize = 14)
ax.set_xlabel('Time [ms]', fontsize = 14)



plt.show()