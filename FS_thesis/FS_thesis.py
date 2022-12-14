# %%

import random
from math import log10
from scipy.signal import find_peaks 
from multiprocessing import Pool
import numpy as np
import pandas as pd
from deap import base, creator, tools 
import myokit
from important_functions import detect_EAD, run_EAD
import matplotlib.pyplot as plt

# %%
# GA

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

def _initialize_individual():
    # Builds a list of parameters using random upper and lower bounds.
    lower_exp = log10(GA_CONFIG.params_lower_bound)
    upper_exp = log10(GA_CONFIG.params_upper_bound)
    initial_params = [10**random.uniform(lower_exp, upper_exp)
                      for i in range(0, len(
                          GA_CONFIG.tunable_parameters))]

    keys = [val for val in GA_CONFIG.tunable_parameters]
    return dict(zip(keys, initial_params))

def get_ind_data(ind):
    #import os 
    #dir_path = os.path.dirname(os.path.realpath(__file__))
    #mod, proto, x = myokit.load(r'C:\Users\user\Desktop\Thesis\GeneticAlgorithms\tor-ord-GA\tor_ord_endo.mmt')

    mod, proto, x = myokit.load('./tor_ord_endo2.mmt')
    if ind is not None:
        for k, v in ind[0].items():
            mod['multipliers'][k].set_rhs(v)

    return mod, proto

def _evaluate_fitness(ind):
    mod, proto = get_ind_data(ind)
    proto.schedule(5.3, 0.1, 1, 1000, 0) 
    sim = myokit.Simulation(mod,proto)
    sim.pre(1000 * 100) #pre-pace for 100 beats
    IC = sim.state()

    feature_error = get_feature_errors(ind, IC=IC)  #Run get feature errors with prepaced simulation object

    # Returns 
    if feature_error == 500000:
        return feature_error


    #ind = {'i_cal_pca_multiplier': 1, 'i_kr_multiplier': 1}
    t, v = run_EAD(ind, IC=IC)
    info, result = detect_EAD(t, v)
    #print(info)
    
    ead_error = 0
    if result == 1:
        ead_error = 1000

    fitness = feature_error + ead_error


    return fitness

def get_feature_errors(ind, IC):
    """
    Compares the simulation data for an individual to the baseline Tor-ORd values. 
    The returned error value is a sum of the differences between the individual and baseline values.

    Returns
    ------
        error
    """

    ap_features = {}
    t, v, cai, i_ion = get_normal_sim_dat(ind, IC)

    # Returns really large error value if cell AP is not valid 
    if ((min(v) > -60) or (max(v) < 0)):
        return 500000 

    # Voltage/APD features#######################
    mdp = min(v)
    max_p = max(v)
    max_p_idx = np.argmax(v)
    apa = max_p - mdp
    dvdt_max = np.max(np.diff(v[0:30])/np.diff(t[0:30]))

    ap_features['dvdt_max'] = dvdt_max

    for apd_pct in [10, 50, 90]:
        repol_pot = max_p - apa * apd_pct/100
        idx_apd = np.argmin(np.abs(v[max_p_idx:] - repol_pot))
        apd_val = t[idx_apd+max_p_idx]

        ap_features[f'apd{apd_pct}'] = apd_val

    # Calcium/CaT features######################## 
    max_cai = np.max(cai)
    max_cai_idx = np.argmax(cai)
    cat_amp = np.max(cai) - np.min(cai)
    ap_features['cat_amp'] = cat_amp

    for cat_pct in [10, 50, 90]:
        cat_recov = max_cai - cat_amp * cat_pct / 100
        idx_catd = np.argmin(np.abs(cai[max_cai_idx:] - cat_recov))
        catd_val = t[idx_catd+max_cai_idx]

        ap_features[f'cat{cat_pct}'] = catd_val 

    error = 0

    if GA_CONFIG.cost == 'function_1':
        for k, v in ap_features.items():
            error += (GA_CONFIG.feature_targets[k][1] - v)**2
    else:
        for k, v in ap_features.items():
            if ((v < GA_CONFIG.feature_targets[k][0]) or
                    (v > GA_CONFIG.feature_targets[k][2])):
                error += 1000 

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
    
    # in order to resolve the error on the plot_generations
    # where IC wasnt declared
    if IC == None:
        IC = sim.state()

    sim.set_state(IC)
    dat = sim.run(5000) # set time in ms

    # Get t, v, and cai for second to last AP#######################
    i_stim = dat['stimulus.i_stim']
    peaks = find_peaks(-np.array(i_stim), distance=100)[0]
    start_ap = peaks[-3] #TODo change start_ap to be after stim, not during
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

def start_ga(pop_size, max_generations):
    feature_targets = {'dvdt_max': [80, 86, 92],
                       'apd10': [5, 15, 30],
                       'apd50': [200, 220, 250],
                       'apd90': [250, 270, 300],
                       'cat_amp': [2.8E-4, 3.12E-4, 4E-4],
                       'cat10': [80, 100, 120],
                       'cat50': [200, 220, 240],
                       'cat90': [450, 470, 490]}

    # 1. Initializing GA hyperparameters
    global GA_CONFIG
    GA_CONFIG = Ga_Config(population_size=pop_size,
                          max_generations=max_generations,
                          params_lower_bound=0.1,
                          params_upper_bound=2,
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
                          cost='function_1',
                          feature_targets=feature_targets)

    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))

    creator.create('Individual', list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register('init_param',
                     _initialize_individual)
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
    #p = Pool()
    toolbox.register("map", map)

    # Use this if you don't want multi-threading
    #toolbox.register("map", map)

    # 2. Calling the GA to run
    final_population = run_ga(toolbox)

    return final_population

def run_ga(toolbox):
    """
    Runs an instance of the genetic algorithm.

    Returns
    -------
        final_population : List[Individuals]
    """
    print('Evaluating initial population.')

    # 3. Calls _initialize_individuals GA_CONFIG.population size number of times and returns the initial population
    population = toolbox.population(GA_CONFIG.population_size)


    # 4. Calls _evaluate_fitness on every individual in the population
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = (fit,)
        
    # Note: visualize individual fitnesses with: population[0].fitness


    # Store initial population details for result processing.
    final_population = [population]
    avg_err = list()
    best_err = list()
    
    for generation in range(0, GA_CONFIG.max_generations):
        print('Generation {}'.format(generation))
        #print("Generation number is: " +str(generation))
        # 5. DEAP selects the individuals 
        selected_offspring = toolbox.select(population, len(population))
        #print(selected_offspring)

        offspring = [toolbox.clone(i) for i in selected_offspring]
        print(offspring)

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

        # All individuals who were updated, either through crossover or
        # mutation, will be re-evaluated.
        
        # 8. Evaluating the offspring of the current generation
        updated_individuals = [i for i in offspring if not i.fitness.values]
        fitnesses = toolbox.map(toolbox.evaluate, updated_individuals)
        for ind, fit in zip(updated_individuals, fitnesses):
            ind.fitness.values = (fit,)

        population = offspring
        gen_fitnesses = [ind.fitness.values[0] for ind in population]
        #print("Fitness values are: " + str(gen_fitdata))
        tempdata = [i[0] for i in population]

        """ for tempdic, temperror in zip(tempdata, gen_fitdata):
            tempdic["error"] = temperror """

        df_data = pd.DataFrame(tempdata)
        df_error = pd.DataFrame(gen_fitnesses, columns=["error"])
        df1 = df_data.join(df_error, how="outer")
        
        if generation == 0:
            df = df1
        
        if generation > 0:
            df = pd.concat([df, df1])
        

        print(f'\tAvg fitness is: {np.mean(gen_fitnesses)}')
        print(f'\tBest fitness is {np.min(gen_fitnesses)}')


        avg_error = np.mean(gen_fitnesses)
        avg_err.append(avg_error)
        best_error = np.min(gen_fitnesses)
        best_err.append(best_error)

        

        final_population.append(population)

    print("avg_error: " +str(avg_err))
    print("best_err: " +str(best_err))

    # Average and Best errors to excel
    df_avg_err = pd.DataFrame(avg_err, columns=["Avg Error"])
    df_best_err = pd.DataFrame(best_err, columns=["Best Error"])
    dfe = df_avg_err.join(df_best_err, how="outer")
    dfe.to_excel('FS_errors.xlsx', sheet_name='Sheet1')

    #Conductances with error to excel
    df.to_excel('FS_data.xlsx', sheet_name='Sheet1')

    return final_population


# Apply IKr Block
#1. i need to take the individuals from the last generation
def ikr_block(best_ind, IC):
    # and then i have to rerun the code ?
    mod, proto = get_ind_data(best_ind)
    mod['multipliers']['i_cal_pca_multiplier'].set_rhs(8)
    mod['multipliers']['i_kr_multiplier'].set_rhs(0.5)
    proto.schedule(5.3, 0.1, 1, 1000, 0) 
    sim = myokit.Simulation(mod, proto)
    
    '''
    # Do we need IC?
    if IC == None:
        IC = sim.state()
    
    sim.pre(1000 * 100)
    sim.set_state(IC)
    '''

    dat = sim.run(20000)

    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    axs[0].plot(dat['engine.time'], dat['membrane.v'])
    axs[1].plot(dat['engine.time'], dat['membrane.i_ion'])

    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    axs[1].set_xlabel('Time (ms)', fontsize=14)
    axs[0].set_ylabel('Voltage (mV)', fontsize=14)
    axs[1].set_ylabel('Current (pA/pF)', fontsize=14)

    axs[1].set_ylim(-1, 4)
    plt.show()

def plot_generation(inds,
                    gen=None,
                    is_top_ten=True,
                    lower_bound=.1,
                    upper_bound=2):
    
    #print("Individuals: " +str(inds))
    
    if gen is None:
        gen = len(inds) - 1

    pop = inds[gen]

    print("Populations: " +str(pop))

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

    # all_ind_dict: contains currents and then values of them
    # and do it for every ind in population

    # APPLY THE BLOCK TO THE BEST INDIVIDUALS
    #  OF THE LAST GENERATION
    ikr_block(best_ind, IC=None)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    curr_x = 0

    for k, conds in all_ind_dict.items():
        for i, g in enumerate(conds):
            # g = log10(g)
            x = curr_x + np.random.normal(0, .01)
            g_val = 1 - fitnesses[i] / max(fitnesses)
            axs[0].scatter(x, g, color=(0, g_val, 0))
            #if i < 10:
            #    axs[0].scatter(x-.1, g, color='r')


        curr_x += 1

    # curr_x -> variable in order to shift by 1 on the graph
    # in order to distinguish between Ca and Kr

    curr_x = 0


    axs[0].hlines(0, -.5, (len(keys)-.5), colors='grey', linestyle='--')
    axs[0].set_xticks([i for i in range(0, len(keys))])
    #axs[0].set_xticklabels(['GCaL', 'GKs', 'GKr', 'GNaL', 'Jup'], fontsize=10)
    axs[0].set_xticklabels(['GCaL', 'GKr'], fontsize=10)
    #axs[0].set_ylim(log10(lower_bound), 
    #                log10(upper_bound))
    axs[0].set_ylim(lower_bound, 
                    upper_bound)
    axs[0].set_ylabel('Conductances', fontsize=14)                                
    #axs[0].set_ylabel('Log10 Conductance', fontsize=14)

    t, v, cai, i_ion = get_normal_sim_dat(best_ind, IC=None)
    axs[1].plot(t, v, 'b--', label='Best Fit')

    t, v, cai, i_ion = get_normal_sim_dat(None, IC=None)
    axs[1].plot(t, v, 'k', label='Original Tor-ORd')

    axs[1].set_ylabel('Voltage (mV)', fontsize=14)
    axs[1].set_xlabel('Time (ms)', fontsize=14)

    axs[1].legend()

    fig.suptitle(f'Generation {gen}', fontsize=14)

    # Plot of avg and mean errors (load data from excel)
    num_gen = np.array(range(1, (GA_CONFIG.max_generations)+1))

    avg_best_errs = pd.read_excel('FS_errors.xlsx')
    avg_err_plot = pd.DataFrame(avg_best_errs, columns= ['Avg Error'])
    best_err_plot = pd.DataFrame(avg_best_errs, columns= ['Best Error'])
    #print(avg_best_errs)
    

    plt.figure(figsize=(12, 6))

    '''
    x = list()
    curr_x = 1

    for i in avg_err_plot:
        x = curr_x + np.random.normal(0, .01)
        curr_x += 1
        plt.scatter(x, i, color='r')
        #plt.scatter(curr_x, j, color='b')
    
    print("Var x: " +str(x))
    '''

    plt.plot(num_gen, avg_err_plot, "bo", label="Average Error")
    plt.plot(num_gen, avg_err_plot)
    plt.plot(num_gen, best_err_plot, "ro", label="Best Error")
    plt.plot(num_gen, best_err_plot)
    #plt.spines['right'].set_visible(False)
    #plt.spines['top'].set_visible(False)
    plt.legend()

    plt.xlabel('Number of Generations', fontsize=14)
    plt.ylabel('Errors', fontsize=14)
    plt.xlim(0,(GA_CONFIG.max_generations)+0.5)
    #plt.ylim(min(best_err_plot), max(avg_err_plot))
    plt.title('Average and Best error')

    
    plt.show()
    
    
# %% Plot
def main():
    all_individuals = start_ga(pop_size=5, max_generations=8)

    plot_generation(all_individuals,
                    gen=None,
                    is_top_ten=False,
                    lower_bound=GA_CONFIG.params_lower_bound,
                    upper_bound=GA_CONFIG.params_upper_bound)
    
    # these value have to be the return of plot generation

if __name__ == '__main__':
    main()


# %%



    