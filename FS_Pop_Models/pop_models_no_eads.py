#%% Import functions and tools

import time
import myokit
import pandas as pd
from multiprocessing import Pool
from signal import signal, SIGPIPE, SIG_DFL  
signal(SIGPIPE,SIG_DFL)

def change_ind_data(ind, cond_combo):
    vals = list(ind.values())
    labs = list(ind.keys())
    
    for i in list(range(0, len(vals))):
        if labs[i]=='i_cal_pca_multiplier':
            if vals[i]>1:
                vals[i] = cond_combo[0]

        if labs[i]=='i_kr_multiplier':
            vals[i] = cond_combo[1]
    
    
    new_ind = dict(zip(labs, vals))

    return new_ind

def get_ind_data(ind, path = './', model = 'tor_ord_endo2.mmt'):

    mod, proto, x = myokit.load(path)
    #mod, proto, x = myokit.load(path+model)
    if ind is not None:
        for k, v in ind[0].items():
            mod['multipliers'][k].set_rhs(v)

    return mod, proto

def run_model(ind, beats, cl = 1000, prepace = 100, location = 0, ion = 0, value = 0, I0 = 0, sim =0, return_sim = 0, path = '../', model = 'tor_ord_endo2.mmt'): 

    if I0 != 0:
        sim.reset()
        sim.set_state(I0)
        if location !=0 and ion !=0 and value !=0:
            sim.set_constant('multipliers.'+ion, value)

    else:
        mod, proto = get_ind_data(ind, path, model = model)
        if location !=0 and ion !=0 and value !=0:
            mod[location][ion].set_rhs(value)
        proto.schedule(5.3, 0.1, 1, cl, 0) 
        sim = myokit.Simulation(mod,proto)

    sim.pre(cl * prepace) #pre-pace for 100 beats
    dat = sim.run(beats*cl) 
    IC = sim.state()

    if return_sim != 0:
        return(dat, sim, IC)
    else:
        return(dat, IC) 

def define_pop_experiment(ind, cond, purturb, p, model = 'tor_ord_endo2.mmt'):

    data = {}
    
    # Cond Analysis

    dat_initial, sim, IC_initial = run_model([ind], 5, prepace = 600, return_sim=1,  path = p, model = model)

    data['t_initial'] = list(dat_initial['engine.time'])
    data['v_initial'] = list(dat_initial['membrane.v'])
    
    for i in list(range(0,len(purturb))):
        #dat, sim, IC = run_model([ind], 5, location = 'multipliers', ion=cond, value = ind[cond]*purturb[i], prepace = 600, sim = sim, I0 = IC_initial, path = p, model = model)
        vals = run_model([ind], 5, location = 'multipliers', ion=cond, value = ind[cond]*purturb[i], prepace = 600, sim = sim, I0 = IC_initial, path = p, model = model)
        
        if len(vals) == 2:
            dat, IC = vals[0], vals[1]
        else:
            dat, sim, IC = vals[0], vals[1], vals[2]
        
        # SAVE DATA
        data['t_'+str(purturb[i])] = list(dat['engine.time'])
        data['v_'+str(purturb[i])] = list(dat['membrane.v'])

    return data

def get_ind(vals = [1,1,1,1,1,1,1,1,1,1]):
    tunable_parameters=['i_cal_pca_multiplier','i_ks_multiplier','i_kr_multiplier','i_nal_multiplier','i_na_multiplier','i_to_multiplier','i_k1_multiplier','i_NCX_multiplier','i_nak_multiplier','i_kb_multiplier']
    ind = dict(zip(tunable_parameters, vals))
    return(ind)

def load_data(i, path):

    data = pd.read_csv(path)
    #ind = eval(data['ind'][i])
    ind = get_ind(vals=data.filter(like = 'multiplier').to_numpy().tolist()[i])
    return(ind)

def collect_pop_data(args):
    i, cond_combo_co_pos, cond_combo_co_neg, cond_combo_ind, cond, purturb, p, p_models = args
    #ind = load_data(i, p_models+'pop_models.csv')
    ind = load_data(i, p_models)

    coex_pos = change_ind_data(ind)
    coex_neg = change_ind_data(ind)
    independ = change_ind_data(ind)


   #try:
    data_base = define_pop_experiment(ind, cond, purturb, p)
    data_co_pos = define_pop_experiment(coex_pos, cond, purturb, p)
    data_co_neg = define_pop_experiment(coex_neg, cond, purturb, p)
    data_independ = define_pop_experiment(independ, cond, purturb, p)
    
    data_co_pos = { k+'_co_pos': v for k, v in data_co_pos.items() }
    coex_pos = { k+'_co_pos': v for k, v in coex_pos.items() }

    data_co_neg = { k+'_co_neg': v for k, v in data_co_neg.items() }
    coex_neg = { k+'_co_neg': v for k, v in coex_neg.items() }

    data_independ = { k+'_independ': v for k, v in data_independ.items() }
    independ = { k+'_independ': v for k, v in independ.items() }

    data = {**ind, **coex_pos, **coex_neg, **independ, **data_base, **data_co_pos, **data_co_neg, **data_independ}
    
    #except:

    #    data = {**ind, **coex_pos, **coex_neg, **independ}

    return(data)


# %% Apply IKr block to every model in Population for positive coexpressed case, negative coexpressed case, and independent case  - TO RUN WITH MULTIPROCESSING

print(time.time())
time1 = time.time()

if __name__ == "__main__":

    num_models = 10                               # Length of the population
    cond_combo_co_pos = [1.18089138576701, 1.87248414989593]                      # Add coexpressed positive values here found by the GA in the form of a list
    cond_combo_co_neg = [0.846817930973797, 0.534049914417476]                    # Add coexpressed negative values here found by the GA in the form of a list
    
    cond_combo_ind = [1.09824337075312, 0.679309672538978]                        # Add independent values here found by the GA in the form of a list 

    cond = 'i_kr_multiplier'                        # This specifies the conductance you would like to manipulate
    ikr_blocks = [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.01] # This is the list of ikr blocks that will be applied to each model in the population

    path_to_torord_model = './tor_ord_endo2.mmt'                     # Add the path to where your tor_ord model is here 
    #path_to_population_models = './home/francescosoliani/Documents/projects/FS_thesis/FS_Pop_Models/'
    path_to_population_models = './pop_models.csv'     


    args = [(i, cond_combo_co_pos, cond_combo_co_neg, cond_combo_ind, cond, ikr_blocks, path_to_torord_model, path_to_population_models) for i in range(num_models)]

    p = Pool() 
    result = p.map(collect_pop_data, args) 
    p.close()
    p.join()

time2 = time.time()
print('processing time: ', (time2-time1)/60, ' Minutes')
print(time.time())

# %% TO RUN WITHOUT MULTIPROCESSING
"""
num_models = 2
#num_models = 1360  # this is the length of the population
profile = [0.6156318475551733, 1.869782283373008, 1.5662322563788678, 1.9829696059987776, 0.3021585355211869, 1.978334512908048, 0.9794589996934674, 0.690019164383329, 0.8446114217171479, 0.6018711836557888]
args = [(i, profile, 'i_kr_multiplier', 'cond', [1, 0.8, 0.6, 0.4, 0.2, 0], '../../../', '../../../../../../Research Data/GA6/') for i in range(num_models)]

time1 = time.time()

result = list(map(collect_pop_data, args))

time2 = time.time()
print('processing time: ', (time2-time1)/60, ' Minutes')
"""


#%% Save Data
df_data = pd.DataFrame(result)

save_data_to = 'IKr_data.csv'  # Specify the path you would like to save all your data to here -- I would reccomend saving this outside of your folder to avoid issues with GitHub... This will generate a very large file which GitHub will not like!
df_data.to_csv(save_data_to, index = False)  