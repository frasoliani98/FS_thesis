import random
from math import log10
import matplotlib.pyplot as plt
from scipy.signal import find_peaks # pip install scipy
import seaborn as sns # pip install seaborn
from multiprocessing import Pool
import numpy as np
import pandas as pd
from important_functions_BPS3 import run_EAD, detect_EAD

from deap import base, creator, tools # pip install deap
import myokit

#%%
class Ga_Config():
    def __init__(self,
             population_size,
             max_generations,
             params_lower_bound,
             params_upper_bound,
             tunable_parameters,
             mate_probability,
             mutate_probability,
             gene_swap_probability,
             gene_mutation_probability,
             tournament_size,
             cost,
             feature_targets):
        self.population_size = population_size
        self.max_generations = max_generations
        self.params_lower_bound = params_lower_bound
        self.params_upper_bound = params_upper_bound
        self.tunable_parameters = tunable_parameters
        self.mate_probability = mate_probability
        self.mutate_probability = mutate_probability
        self.gene_swap_probability = gene_swap_probability
        self.gene_mutation_probability = gene_mutation_probability
        self.tournament_size = tournament_size
        self.cost = cost
        self.feature_targets = feature_targets


def get_ind_data(ind):
    #import os 
    #dir_path = os.path.dirname(os.path.realpath(__file__))
    #mod, proto, x = myokit.load(r'C:\Users\user\Desktop\Thesis\GeneticAlgorithms\tor-ord-GA\tor_ord_endo.mmt')

    mod, proto, x = myokit.load('./BPS3_ko54.mmt')
    if ind is not None:
        for k, v in ind[0].items():
            mod['multipliers'][k].set_rhs(v)

    return mod, proto


def run_ga(toolbox):
    """
    Runs an instance of the genetic algorithm.

    Returns
    -------
        final_population : List[Individuals]
    """
    print('Evaluating initial population.')

    # 3. Calls _initialize_individuals and returns initial population
    population = toolbox.population(GA_CONFIG.population_size)


    # 4. Calls _evaluate_fitness on every individual in the population
    fitnesses = toolbox.map(toolbox.evaluate, population)
    
    eads = []
    eads_number = list()
    eads_f = list()

    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = (fit[0],)
        ind_ead = fit[1]
        eads.append(ind_ead)

    df_eads = pd.DataFrame(eads, columns = ['EADs'])
    df_eads.to_excel('EADs.xlsx', sheet_name='Sheet1', index=False)

    # Graph EADs vs generation
    eads_count1 = pd.read_excel('EADs.xlsx', sheet_name='Sheet1')
    eads_count = pd.DataFrame(eads_count1)
    eads_f = eads_count['EADs'].values.tolist()
    #print(eads_f)

    eads_f = eads_f.count(1)
    eads_number.append(eads_f)
    print(eads_number)
    
    #for ind, fit in zip(population, fitnesses):
    #    ind.fitness.values = (fit,)

    # Note: visualize individual fitnesses with: population[0].fitness
    gen_fitnesses = [ind.fitness.values[0] for ind in population]

    print(f'\tAvg fitness is: {np.mean(gen_fitnesses)}')
    print(f'\tBest fitness is {np.min(gen_fitnesses)}')

    avg_err = list()
    best_err = list()
    

    # Store initial population details for result processing.
    final_population = [population]

    # Excel Errors initial population
    avg_error = np.mean(gen_fitnesses)
    avg_err.append(avg_error)
    best_error = np.min(gen_fitnesses)
    best_err.append(best_error)
    df_avg_err = pd.DataFrame(avg_err, columns=["Avg Error"])
    df_best_err = pd.DataFrame(best_err, columns=["Best Error"])
    dfe = df_avg_err.join(df_best_err, how="outer")
    dfe.to_excel('FS_errors.xlsx', sheet_name='Sheet1')

    # Excel
    tempdata = [i[0] for i in population]
    df_data = pd.DataFrame(tempdata)
    df_error = pd.DataFrame(gen_fitnesses, columns=["error"])
    df = df_data.join(df_error, how="outer")

    for generation in range(1, GA_CONFIG.max_generations):
        
        old_population = population
        old_eads = eads

        print('Generation {}'.format(generation))

        # 5. DEAP selects the individuals 
        selected_offspring = toolbox.select(population, len(population))

        offspring = [toolbox.clone(i) for i in selected_offspring]
        #print(offspring)
        # 6. Mate the individualse by calling _mate()
        for i_one, i_two in zip(offspring[::2], offspring[1::2]):
            if random.random() < GA_CONFIG.mate_probability:
                toolbox.mate(i_one, i_two)
                del i_one.fitness.values
                del i_two.fitness.values

        # 7. Mutate the individualse by calling _mutate()
        for i in offspring:
            if random.random() < GA_CONFIG.mutate_probability:
                toolbox.mutate(i)
                del i.fitness.values


        # 8. Evaluating the offspring of the current generation
        updated_idx = [i for i in list(range(0, len(offspring))) if not offspring[i].fitness.values]
        updated_individuals = [i for i in offspring if not i.fitness.values]
        fitnesses = toolbox.map(toolbox.evaluate, updated_individuals)

        eads = []
        for ind, fit in zip(updated_individuals, fitnesses):
            ind.fitness.values = (fit[0],)
            ind_ead = fit[1]
            eads.append(ind_ead)

        #for ind, fit in zip(updated_individuals, fitnesses):
        #    ind.fitness.values = (fit,)

        population = offspring

        gen_fitnesses = [ind.fitness.values[0] for ind in population]

        # Excel
        tempdata = [i[0] for i in population]
        df_data = pd.DataFrame(tempdata)
        df_error = pd.DataFrame(gen_fitnesses, columns=["error"])
        df1 = df_data.join(df_error, how="outer")
        df = pd.concat([df, df1])


        print(f'\tAvg fitness is: {np.mean(gen_fitnesses)}')
        print(f'\tBest fitness is {np.min(gen_fitnesses)}')

        avg_error = np.mean(gen_fitnesses)
        avg_err.append(avg_error)
        best_error = np.min(gen_fitnesses)
        best_err.append(best_error)


        final_population.append(population)

        for i in list(range(0, GA_CONFIG.population_size)):
            if updated_idx.count(i) == 0:
                for x in list(range(0, len(old_population))):
                    if list(old_population[x][0].values())==list(population[i][0].values()):
                        idx = x
                eads.insert(i, old_eads[idx])
        
        # Save EADs
        new_eads = pd.DataFrame(eads, columns = ['EADs'])
        df_eads = pd.concat([df_eads, new_eads], ignore_index=True)

        

        # Graph EADs vs Generation
        eads_f = new_eads['EADs'].values.tolist()
        #print(eads_f)

        eads_f = eads_f.count(1)
        eads_number.append(eads_f)
        print(eads_number)
        

    
    # EADs to Excel
    df_eads.to_excel('EADs.xlsx', sheet_name='Sheet1', index=False)

    # Plot EADs
    gen = list(range(1,GA_CONFIG.max_generations+1))
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(gen, eads_number, 'ro', label= 'Num of EADs')
    ax.plot(gen, eads_number, 'b')
    ax.set_xlabel('Generations',fontsize=14)
    ax.set_ylabel('Number of EADs', fontsize=14)
    fig.suptitle('EADs during Generations ', fontsize=14)

    ax.legend()
    plt.show()

    # Average and Best errors to excel
    df_avg_err = pd.DataFrame(avg_err, columns=["Avg Error"])
    df_best_err = pd.DataFrame(best_err, columns=["Best Error"])
    dfe = df_avg_err.join(df_best_err, how="outer")
    dfe.to_excel('FS_errors.xlsx', sheet_name='Sheet1')

    #Conductances with error to excel
    df.to_excel('FS_data.xlsx', sheet_name='Sheet1')
    
    
    # Plot Errors
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(gen, avg_err, 'ro', label= 'Average Error')
    ax.plot(gen, best_err, 'bo', label= 'Best Error')
    ax.set_xlabel('Generations',fontsize=14)
    ax.set_ylabel('Errors', fontsize=14)
    fig.suptitle('AVG and BEST Err BPS model ', fontsize=14)

    ax.legend()
    plt.show()
    
    return final_population


def _initialize_individuals():
    """
    Creates the initial population of individuals. The initial 
    population 

    Returns:
        An Individual with conductance parameters 
    """
    # Builds a list of parameters using random upper and lower bounds.
    
    
    lower_exp = log10(GA_CONFIG.params_lower_bound)
    upper_exp = log10(GA_CONFIG.params_upper_bound)
    initial_params = [10**random.uniform(lower_exp, upper_exp)
                      for i in range(0, len(
                          GA_CONFIG.tunable_parameters))]
                  
    
    keys = [val for val in GA_CONFIG.tunable_parameters]
    return dict(zip(keys, initial_params))


def _evaluate_fitness(ind):
    """
    Calls cost functions and adds costs together

    Returns
    -------
        fitness : number 
    """
    mod, proto = get_ind_data(ind)
    proto.schedule(5.3, 0.1, 1, 1000, 0) 
    sim = myokit.Simulation(mod, proto)

    try:
        sim.pre(600 * 1000) 
    except:
        data = [500000, 1]
        return data

    IC = sim.state()

    t, v, cai, i_ion = get_normal_sim_dat(ind, IC)
    feature_error = get_feature_errors(t, v, cai)
    data = check_physio_torord(t, v, './', filter = 'yes')
    morph_error = data['error']


    # Returns 
    if feature_error == 500000:
        data = [500000, 1]
        return data

    
    t,v,t1,v1 = run_EAD(ind)
    info, result = detect_EAD(t,v)
    info1, result1 = detect_EAD(t1,v1)


    ead_error = 0
    
    
    '''
    if result == 0:
        df_noeads = pd.DataFrame(info)
        df_noeads.to_excel('FS_EADs.xlsx', sheet_name='Sheet1')

    if result == 1:
        ead_error = 1000
        #total_EADs += 1
        print(info)
        df_eads = pd.DataFrame(info)
        #df_error = pd.DataFrame(gen_fitnesses, columns=["error"])
        #df = df_data.join(df_error, how="outer")
        df_eads.to_excel('FS_EADs.xlsx', sheet_name='Sheet1')

        
        #print("Total number of EADs: " +str(total_EADs))
    '''

    if result == 1 or result1 == 1:
        ead_error = 100000
        #ead_error = 1000
        #print('info' +str(info))
        #print('info1' +str(info1))

    fitness = feature_error + morph_error + ead_error

    data = [fitness, result]

    return data


def _mate(i_one, i_two):
    """Performs crossover between two individuals.

    There may be a possibility no parameters are swapped. This probability
    is controlled by `GA_CONFIG.gene_swap_probability`. Modifies
    both individuals in-place.

    Args:
        i_one: An individual in a population.
        i_two: Another individual in the population.
    """
    for key, val in i_one[0].items():
        if random.random() < GA_CONFIG.gene_swap_probability:
            i_one[0][key],\
                i_two[0][key] = (
                    i_two[0][key],
                    i_one[0][key])


def _mutate(individual):
    """Performs a mutation on an individual in the population.

    Chooses random parameter values from the normal distribution centered
    around each of the original parameter values. Modifies individual
    in-place.

    Args:
        individual: An individual to be mutated.
    """
    keys = [k for k, v in individual[0].items()]

    for key in keys:
        if random.random() < GA_CONFIG.gene_mutation_probability:
            new_param = -1

            while ((new_param < GA_CONFIG.params_lower_bound) or
                   (new_param > GA_CONFIG.params_upper_bound)):
                new_param = np.random.normal(
                        individual[0][key],
                        individual[0][key] * .1)

            individual[0][key] = new_param


def get_feature_errors(t, v, cai):
    """
    Compares the simulation data for an individual to the baseline Tor-ORd values. The returned error value is a sum of the differences between the individual and baseline values.    
    Returns
    ------
        error
    """    
    ap_features = {}
    #t, v, cai, i_ion = get_normal_sim_dat(ind, IC)

    # Returns really large error value if cell AP is not valid 
    if ((min(v) > -60) or (max(v) < 0)):
        return 500000     
    
    # Voltage/APD features#######################
    mdp = min(v)
    max_p = max(v)
    max_p_idx = np.argmax(v)
    apa = max_p - mdp
    dvdt_max = np.max(np.diff(v[0:100])/np.diff(t[0:100]))    
    
    ap_features['Vm_peak'] = max_p
    ap_features['dvdt_max'] = dvdt_max

    
    for apd_pct in [40, 50, 90]:
        repol_pot = max_p - apa * apd_pct/100
        idx_apd = np.argmin(np.abs(v[max_p_idx:] - repol_pot))
        apd_val = t[idx_apd+max_p_idx]        
        
        ap_features[f'apd{apd_pct}'] = apd_val
    
    #ap_features['triangulation'] = ap_features['apd90'] - ap_features['apd40']
    ap_features['RMP'] = np.mean(v[len(v)-50:len(v)])  

    
    # Calcium/CaT features######################## 
    '''
    max_cai = np.max(cai)
    max_cai_idx = np.argmax(cai)
    cat_amp = np.max(cai) - np.min(cai)
    ap_features['cat_amp'] = cat_amp*1e5
    max_cai_time = t[max_cai_idx]
    ap_features['cat_peak'] = max_cai_time    
    
    for cat_pct in [90]:
        cat_recov = max_cai - cat_amp * cat_pct / 100
        idx_catd = np.argmin(np.abs(cai[max_cai_idx:] - cat_recov))
        catd_val = t[idx_catd+max_cai_idx]        
        
        ap_features[f'cat{cat_pct}'] = catd_val     
    '''
    error = 0    
    

    '''
    if GA_CONFIG.cost == 'function_1':
        for k, v in ap_features.items():
            error += (GA_CONFIG.feature_targets[k][1] - v)**2
    else:
        for k, v in ap_features.items():
            if ((v < GA_CONFIG.feature_targets[k][0]) or
                    (v > GA_CONFIG.feature_targets[k][2])):
                error += 1000
    
    '''
    for k, v in ap_features.items():
        if ((v < GA_CONFIG.feature_targets[k][0]) or (v > GA_CONFIG.feature_targets[k][2])):
            error += (GA_CONFIG.feature_targets[k][1] - v)**2 

    return error


def get_normal_sim_dat(ind, IC):
    """
        Runs simulation for a given individual. If the individuals is None,
        then it will run the baseline model

        Returns
        ------
            t, v, cai, i_ion
    """

    mod, proto = get_ind_data(ind)
    proto.schedule(5.3, 0.1, 1, 1000, 0) 
    sim = myokit.Simulation(mod, proto)
    
    if IC == None:
        IC = sim.state()

    sim.set_state(IC)
    dat = sim.run(50000)

    # Get t, v, and cai for second to last AP#######################
    i_stim = dat['stimulus.i_stim']
    peaks = find_peaks(-np.array(i_stim), distance=100)[0]
    start_ap = peaks[-3] #TODO change start_ap to be after stim, not during
    end_ap = peaks[-2]

    t = np.array(dat['engine.time'][start_ap:end_ap])
    t = t - t[0]
    max_idx = np.argmin(np.abs(t-900))
    t = t[0:max_idx]
    end_ap = start_ap + max_idx

    v = np.array(dat['membrane.v'][start_ap:end_ap])
    cai = np.array(dat['intracellular_ions.cai'][start_ap:end_ap])
    i_ion = np.array(dat['membrane.i_ion'][start_ap:end_ap])

    return (t, v, cai, i_ion)


def plot_generation(inds,
                    gen=None,
                    is_top_ten=True,
                    lower_bound=.1,
                    upper_bound=10):
    if gen is None:
        gen = len(inds) - 1

    pop = inds[gen]
    

    pop.sort(key=lambda x: x.fitness.values[0])
    best_ind = pop[0]

    if is_top_ten:
        pop = pop[0:10]

    keys = [k for k in pop[0][0].keys()]
    empty_arrs = [[] for i in range(len(keys))]
    all_ind_dict = dict(zip(keys, empty_arrs))

    fitnesses = []

    for ind in pop:
        for k, v in ind[0].items():
            all_ind_dict[k].append(v)

        fitnesses.append(ind.fitness.values[0])

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    curr_x = 0

    for k, conds in all_ind_dict.items():
        for i, g in enumerate(conds):
            g = log10(g)
            x = curr_x + np.random.normal(0, .01)
            g_val = 1 - fitnesses[i] / max(fitnesses)
            axs[0].scatter(x, g, color=(0, g_val, 0))
            #if i < 10:
            #    axs[0].scatter(x-.1, g, color='r')


        curr_x += 1


    curr_x = 0

    axs[0].hlines(0, -.5, (len(keys)-.5), colors='grey', linestyle='--')
    axs[0].set_xticks([i for i in range(0, len(keys))])
    #axs[0].set_xticklabels(['GCaL', 'GKs', 'GKr', 'GNaL', 'Jup'], fontsize=10)
    axs[0].set_xticklabels(['GCaL','GKr'], fontsize=10)
    axs[0].set_ylim(log10(lower_bound), 
                    log10(upper_bound))
    axs[0].set_ylabel('Log10 Conductance', fontsize=14)

    t, v, cai, i_ion = get_normal_sim_dat(best_ind, IC=None)
    axs[1].plot(t, v, 'b--', label='Best Fit')

    t, v, cai, i_ion = get_normal_sim_dat(None, IC=None)
    axs[1].plot(t, v, 'k', label='Original BPS')

    axs[1].set_ylabel('Voltage (mV)', fontsize=14)
    axs[1].set_xlabel('Time (ms)', fontsize=14)

    axs[1].legend()

    fig.suptitle(f'Generation {gen+1}', fontsize=14)

    plt.show()

    tempdata = best_ind
    df = pd.DataFrame(tempdata)
    df.to_excel("Best_ind.xlsx", sheet_name='Sheet1', index=False)


def start_ga(pop_size, max_generations):
    feature_targets = {'Vm_peak': [10, 33, 55],
                       'dvdt_max': [100, 347, 1000],
                       'apd40': [85, 198, 320],
                       'apd50': [110, 220, 430],
                       'apd90': [180, 271, 440],
                       #'triangulation': [50, 73, 150],
                       'RMP': [-95, -88, -80],
                       #'cat_amp': [3E-4*1e5, 3.12E-4*1e5, 8E-4*1e5],
                       #'cat_peak': [40, 58, 60],
                       #'cat90': [350, 467, 500]
                       }

    # 1. Initializing GA hyperparameters
    global GA_CONFIG
    GA_CONFIG = Ga_Config(population_size=pop_size,
                          max_generations=max_generations,
                          params_lower_bound=0.1,
                          params_upper_bound=10,
                          tunable_parameters=['i_cal_pca_multiplier',
                                              'i_kr_multiplier'],
                          
                          #tunable_parameters=['i_cal_pca_multiplier',
                                              #'i_ks_multiplier',
                                              #'i_kr_multiplier',
                                              #'i_nal_multiplier',
                                              #'jup_multiplier'],
                          mate_probability=0.9,
                          mutate_probability=0.9,
                          gene_swap_probability=0.2,
                          gene_mutation_probability=0.2,
                          tournament_size=2,
                          cost='function_2',
                          feature_targets=feature_targets)

    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))

    creator.create('Individual', list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register('init_param',
                     _initialize_individuals)
    toolbox.register('individual',
                     tools.initRepeat,
                     creator.Individual,
                     toolbox.init_param,
                     n=1)
    toolbox.register('population',
                     tools.initRepeat,
                     list,
                     toolbox.individual)

    toolbox.register('evaluate', _evaluate_fitness)
    toolbox.register('select',
                     tools.selTournament,
                     tournsize=GA_CONFIG.tournament_size)
    toolbox.register('mate', _mate)
    toolbox.register('mutate', _mutate)

    # To speed things up with multi-threading
    p = Pool()
    toolbox.register("map", p.map)
    #toolbox.register("map", map)

    # Use this if you don't want multi-threading
    #toolbox.register("map", map)

    # 2. Calling the GA to run
    final_population = run_ga(toolbox)

    return final_population


def closest(lst, K):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]


def check_physio_torord(t, v, path = './', filter = 'no'):    
    
    # Cut off the upstroke of the AP for profile
    t_ind = list(t[150:len(t)])
    v_ind = list(v[150:len(t)])    
    
    # Baseline tor-ord model & cut off upstroke
    base_df = pd.read_csv(path + 'baseline_torord_data.csv')
    t_base = list(base_df['t'])[150:len(t)]
    v_base = list(base_df['v'])[150:len(t)]    
    
    # Cut off the upstroke of the AP for the tor-ord data
    if filter == 'no':
        time, vol_10, vol_90 = get_torord_phys_data(path, filter)
        t = time[150:len(time)]
        v_10 = vol_10[150:len(time)]
        v_90 = vol_90[150:len(time)]    
    
    else:
        t, v_10, v_90 = get_torord_phys_data(path, filter)    
        
    result = 0 # valid AP
    #fail_time = 3000
    error = 0
    check_times = []
    data = {}    
    
    for i in list(range(0, len(t_ind))):
        t_dat = closest(t, t_ind[i]) # find the value closest to the ind's time within the exp data time list
        t_dat_base = closest(t_base, t_ind[i])
        t_dat_i = np.where(np.array(t)==t_dat)[0][0] #find the index of the closest value in the list
        t_dat_base_i = np.where(np.array(t_base)==t_dat_base)[0][0] #find the index of the closest value in the list
        v_model = v_ind[i]
        v_lowerbound = v_10[t_dat_i]
        v_upperbound = v_90[t_dat_i]
        v_torord = v_base[t_dat_base_i]        
        check_times.append(np.abs(t_ind[i] - t_dat))    

        if v_model < v_lowerbound or v_model > v_upperbound:
            result = 1 # not a valid AP
            #error += 10 - used in GA 5
            error += (v_model - v_torord)**2    

    data['result'] = result
    data['error'] = error
    data['check_times'] = check_times    

    return(data)


def get_torord_phys_data(path = './', filter = 'no'):
    data = pd.read_csv(path+'torord_physiologicData.csv')
    time = [x - 9.1666666669999994 for x in list(data['t'])] #shift action potential to match solutions
    t = time[275:len(data['v_10'])]
    v_10 = list(data['v_10'])[275:len(data['v_10'])]
    v_90 = list(data['v_90'])[275:len(data['v_10'])]    
    
    if filter != 'no':
        data = pd.DataFrame(data = {'t': t[1000:len(t)], 'v_10': v_10[1000:len(t)], 'v_90':v_90[1000:len(t)]})
        data_start = pd.DataFrame(data = {'t': t[150:1000], 'v_10': v_10[150:1000], 'v_90':v_90[150:1000]})        
        
        # FILTER V_10
        v_10_new = data.v_10.rolling(400, min_periods = 1, center = True).mean()
        v_10_start = data_start.v_10.rolling(100, min_periods = 1, center = True).mean()
        v_10_new = v_10_new.dropna()
        v_10 = list(v_10_start) + list(v_10_new)
        t = list(data_start['t']) + list(data['t'])        
        
        # FILTER V_90
        v_90_new = data.v_90.rolling(400, min_periods = 1, center = True).mean()
        v_90_start = data_start.v_90.rolling(200, min_periods = 1, center = True).mean()
        v_90_new = v_90_new.dropna()
        v_90 = list(v_90_start) + list(v_90_new)    
        
    return(t, v_10, v_90)



#%%
def main():
    all_individuals = start_ga(pop_size=100, max_generations=20)

    plot_generation(all_individuals, gen=None, is_top_ten=False)

if __name__ == '__main__':
    main()