# 2 BRANCHES
1) TOR-ORD: inside there are all the code and the results obtained from tor-ord
2) BPS: inside there are all the code and the results obtained from bps

# FS_results
1) FS_3: results from 1st version of the code
2) FS_4: results from 2nd version of the code
3) FS_5: results from 3rd version of the code: (3 different challenges: ICaL=8, ICaL+8, ICaL*8 -> but the results are the same between all the tests in these 3 different challenges)

      a) COST1: using of cost function_1 (mean squared differences) -> every time use EAD=100000 in order to be sure to give importance to EADs and to be                   able to get rid of them during the generations
      
      b) COST2: using of cost function_2 (check the boundaries) -> inside cost2 sometimes i used EAD=1000 and other times i used EAD=100000
      
      c) name of the files: 1st number indicate number of individuals while the second one indicate the number of generations used

# FS_thesis: valid for every version

"FS_data": excel file that will fill with the values of the conductances

"FS_errors": excel file that will fill with the errors

"EADs": excel file that will fill with the EADs

"important functions": functions used for the EADs (run_EAD and detect_EAD#

"tor_ord_endo" and "tor_ord_enndo2": models

"supercell_ga": just to try with the ideal cases or tests

# "FS_thesis3":
1) preprocessing and proto schedule 
2) old feature targets: {'dvdt_max': [80, 86, 92],
                          'apd10': [5, 15, 30],
                          'apd50': [200, 220, 250],
                          'apd90': [250, 270, 300],
                          'cat_amp': [2.8E-4, 3.12E-4, 4E-4],
                          'cat10': [80, 100, 120],
                          'cat50': [200, 220, 240],
                          'cat90': [450, 470, 490]}

# "FS_thesis4":
1) just 2 tunable parameters (Ikr and ICaL) instead of 5 tunable parameters
2) error goes to 0
3) the new feature targets: {'Vm_peak': [10, 33, 55],
                               'dvdt_max': [100, 347, 1000],
                               'apd40': [85, 198, 320],
                               'apd50': [110, 220, 430],
                               'apd90': [180, 271, 440],
                               'triangulation': [50, 73, 150],
                               'RMP': [-95, -88, -80],
                               'cat_amp': [3E-4*1e5, 3.12E-4*1e5, 8E-4*1e5],
                               'cat_peak': [40, 58, 60],
                               'cat90': [350, 467, 500]}


# "FS_thesis4_2": 
1) little error 1e-5 (inside "get_feature_error" line " ap_features['cat_amp'] = cat_amp*1e-5" )
2) computation of EADs whith run_EAD and detect_EAD

# "FS_thesis5": Kristin code to save EADs:
1) use "concat" instead of "append"
2) adjustments on return 500000 condition 
3) excel file EADs
4) Graph EADs vs Generations

# "FS_thesis6": Alex suggestion:
1) just consider ICaL and IKr as blockers -> changed the interval of lower and upper tunable parameters from 0.1 to 1 instead of 0.1 to 10


# "FS_thesis7": Improvement (mix) of the error function:
KRISTIN ADDING NEW FUNCTIONS ABOUT THE ERROR COMPUTATION:
1) def closest
2) def check_physio_torord
3) def get_torord_phys_data
4) Modify computation of fitness error inside get_feature_error
