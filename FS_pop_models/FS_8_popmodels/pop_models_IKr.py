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

blocks = [0.8, 0.6, 0.4, 0.2, 0.1]
eads_final_up = list()
eads_final_down = list()
eads_final_ind = list()
eads_start = 0

eads_final_up.append(eads_start)
eads_final_down.append(eads_start)
eads_final_ind.append(eads_start)

for block in blocks:
    eads_up = 0
    eads_down = 0
    eads_ind = 0
    
    for i in range(len(ical_up)):
    #for i in range(5):
        # 3 RUN EAD
        t_up_eads, v_up_eads, t2up, v2up = run_EAD_IKr(ical_up[i], iks_up[i], ikr_up[i], inal_up[i], ina_up[i], ito_up[i], ik1_up[i], iNCX_up[i], inak_up[i], ikb_up[i], block)
        info_up, result_up = detect_EAD(t_up_eads, v_up_eads)
        if result_up == 1:
            eads_up = eads_up + 1

        #axs[0].plot(t_up_eads, v_up_eads, color = palette1[i], linewidth = 0.2)
        #axs[0].plot(t2up+1000, v2up, color = palette1[i], linewidth = 0.2)

    for i in range(len(ical_down)):
    #for i in range(5):
        t_down_eads, v_down_eads, t2down, v2down = run_EAD_IKr(ical_down[i], iks_down[i], ikr_down[i], inal_down[i], ina_down[i], ito_down[i], ik1_down[i], iNCX_down[i], inak_down[i], ikb_down[i], block)
        info_down, result_down = detect_EAD(t_down_eads, v_down_eads)
        if result_down == 1:
            eads_down = eads_down + 1
        
        #axs[1].plot(t_down_eads, v_down_eads, color = palette2[i], linewidth = 0.2)
        #axs[1].plot(t2down+1000, v2down, color = palette1[i], linewidth = 0.2)

    for i in range(len(ical_ind)):
    #for i in range(5):
        t_ind_eads, v_ind_eads, t2ind, v2ind = run_EAD_IKr(ical_ind[i], iks_ind[i], ikr_ind[i], inal_ind[i], ina_ind[i], ito_ind[i], ik1_ind[i], iNCX_ind[i], inak_ind[i], ikb_ind[i], block)
        info_ind, result_ind = detect_EAD(t_ind_eads, v_ind_eads)
        if result_ind == 1:
            eads_ind = eads_ind + 1
        
        
        #axs[2].plot(t_ind_eads, v_ind_eads, color = palette3[i], linewidth = 0.2)
        #axs[2].plot(t2ind+1000, v2ind, color = palette1[i], linewidth = 0.2)

    eads_final_up.append(eads_up)
    eads_final_down.append(eads_down)
    eads_final_ind.append(eads_ind)

    print(eads_final_up)
    print(eads_final_down)
    print(eads_final_ind)

# %%
eads_f_up = [(i/len(ical_up))*100 for i in eads_final_up]
eads_f_down = [(i/len(ical_down))*100 for i in eads_final_down]
eads_f_ind = [(i/len(ical_ind))*100 for i in eads_final_ind]



# %%
# PLOT

fig, ax = plt.subplots(1, 1, figsize = (15,15))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_xticks([i for i in range(1, 7)])
ax.set_xticklabels(['0', '20', '40', '60', '80', '90'], fontsize=10)
ax.set_ylim([0, 100])
x = [1, 2, 3, 4, 5, 6]

ax.plot(x, eads_f_up, 'r--', label = 'Co-Expressed Positive')
ax.plot(x, eads_f_up, 'ro')

ax.plot(x, eads_f_down, 'm--', label = 'Co-Expressed Negative')
ax.plot(x, eads_f_down, 'mo')

ax.plot(x, eads_f_ind, 'c--', label = 'Independent')
ax.plot(x, eads_f_ind, 'co')

fig.suptitle(f'Number of EADs IKr Block', fontsize = 14)
ax.set_ylabel('EADs [%] ', fontsize = 14)
ax.set_xlabel('IKr Block [%]', fontsize = 14)


plt.legend(loc='best')
plt.show()





# %%
