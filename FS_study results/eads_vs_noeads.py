# %%
import random
from math import log10
import matplotlib.pyplot as plt
from scipy.signal import find_peaks # pip install scipy
import seaborn as sns # pip install seaborn
from multiprocessing import Pool
import numpy as np
import pandas as pd
from important_functions_2 import run_EAD, detect_EAD, plot_GA, baseline_run, baseline_EAD_CaL
from deap import base, creator, tools # pip install deap
import myokit

ind_1 = pd.read_excel('Best_ind_1.xlsx')
ind_1 = ind_1.to_dict('index')
t1, v1 = plot_GA(ind_1)
info, result = detect_EAD(t1,v1)
print(info)