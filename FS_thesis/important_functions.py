import myokit
import numpy as np
from scipy.signal import find_peaks # pip install scipy

def get_ind_data(ind):
    mod, proto, x = myokit.load('./tor_ord_endo2.mmt')
    if ind is not None:
        for k, v in ind[0].items():
            mod['multipliers'][k].set_rhs(v)

    return mod, proto

def get_last_ap(dat, AP):

    # Get t, v, and cai for second to last AP#######################
    i_stim = dat['stimulus.i_stim']

    # This is here so that stim for EAD doesnt interfere with getting the whole AP
    for i in list(range(0,len(i_stim))):
        if abs(i_stim[i])<50:
            i_stim[i]=0

    peaks = find_peaks(-np.array(i_stim), distance=100)[0]
    start_ap = peaks[AP] #TODO change start_ap to be after stim, not during
    end_ap = peaks[AP+1]

    t = np.array(dat['engine.time'][start_ap:end_ap])
    t = t - t[0]
    max_idx = np.argmin(np.abs(t-995))
    t = t[0:max_idx]
    end_ap = start_ap + max_idx

    v = np.array(dat['membrane.v'][start_ap:end_ap])
    cai = np.array(dat['intracellular_ions.cai'][start_ap:end_ap])
    i_ion = np.array(dat['membrane.i_ion'][start_ap:end_ap])

    #convert to list for accurate storage in dataframe
    #t = list(t)
    #v = list(v)

    return (t, v)

def run_model(ind, beats, cl = 1000, location = 0, ion = 0, value = 0, I0 = 0): 
    mod, proto = get_ind_data(ind)
    if location !=0 and ion !=0 and value !=0:
        mod[location][ion].set_rhs(value)
    proto.schedule(5.3, 0.1, 1, cl, 0) 
    sim = myokit.Simulation(mod,proto)

    if I0 != 0:
        sim.set_state(I0)
    else:
        sim.pre(cl * 100) #pre-pace for 100 beats

    dat = sim.run(beats*cl) 
    IC = sim.state()
    return(dat, IC)

def detect_EAD(t, v):
    #find slope
    slopes = []
    for i in list(range(0, len(v)-1)):
        m = (v[i+1]-v[i])/(t[i+1]-t[i])
        # dvdt -> m is the slope
        slopes.append(round(m, 2))
        # add value of m rounded to slopes at each iteration

    #print("Slopes are: " +str(slopes))
    
    # Find rises
    pos_slopes = np.where(slopes > np.float64(0.0))[0].tolist()
    #print("pos slopes are: " +str(pos_slopes))
    pos_slopes_idx = np.where(np.diff(pos_slopes)!=1)[0].tolist()
    # tolist : to convert it to a list
    # pos_slopes_indx -> to identify when we have a rise
    # So in this way we will be able to detect when we have the change in the
    # derivatives -> understand where are the rises and descents (slopes)
    pos_slopes_idx.append(len(pos_slopes)) #list must end with last index
    # append all the indexes found 
    #print("Position slopes idx: " +str(pos_slopes_idx))

    # pull out groups of rises (indexes)
    pos_groups = []
    pos_groups.append(pos_slopes[0:pos_slopes_idx[0]+1])
    for x in list(range(0,len(pos_slopes_idx)-1)):
        g = pos_slopes[pos_slopes_idx[x]+1:pos_slopes_idx[x+1]+1]
        pos_groups.append(g)
    # print("new pos groups: " +str(pos_groups))
    # just to print the indexes where we have the rises

    #pull out groups of rises (voltages and times)
    vol_pos = []
    tim_pos = []
    for y in list(range(0,len(pos_groups))):
        vol = []
        tim = []
        for z in pos_groups[y]:
            vol.append(v[z])
            tim.append(t[z])
        vol_pos.append(vol)
        tim_pos.append(tim) 

    # vol_pos and tim_pos append values of voltage (v) and time (t)
    # from the respective indexes computed in pos_groups()

    #Find EAD given the conditions (voltage>-70 & time>100)
    EADs = []
    EAD_vals = []
    for k in list(range(0, len(vol_pos))):
        if np.mean(vol_pos[k]) > -70 and np.mean(tim_pos[k]) > 100:
            EAD_vals.append(tim_pos[k])
            EAD_vals.append(vol_pos[k])
            EADs.append(max(vol_pos[k])-min(vol_pos[k]))
    
    # EADs as max-min in order to compute the amplitude of the repolarization
    # time>100 in order to avoid to detect the initial depolarization phase
    # voltage>-70 in order to avoid to detect something which has already completed
    # the repolarization

    #Report EAD 
    if len(EADs)==0:
        info = "no EAD"
        result = 0
    else:
        info = "EAD: " + str(round(max(EADs))) +' mV'
        # i think that take the max because is the first that occurs (the max one)
        result = 1
    
    return info, result

def run_EAD(ind): 

    ## EAD CHALLENGE: Istim = -.1
    mod, proto = get_ind_data(ind)
    proto.schedule(5.3, 0.1, 1, 1000, 0) 
    proto.schedule(0.1, 3004, 1000-100, 1000, 1) #EAD amp is about 4mV from this
    sim = myokit.Simulation(mod,proto)
    sim.pre(100*1000)
    dat = sim.run(5000)

    # Get t, v, and cai for second to last AP#######################
    t, v = get_last_ap(dat, -2)

    return t,v

def calc_APD(t, v, apd_pct):
    t = [i-t[0] for i in t]
    mdp = min(v)
    max_p = max(v)
    max_p_idx = np.argmax(v)
    apa = max_p - mdp
    repol_pot = max_p - apa * apd_pct/100
    idx_apd = np.argmin(np.abs(v[max_p_idx:] - repol_pot))
    apd_val = t[idx_apd+max_p_idx]
    return(apd_val) 

def get_normal_sim_dat(mod, proto):
    """
        Runs simulation for a given individual. If the individuals is None,
        then it will run the baseline model
        Returns
        ------
            t, v, cai, i_ion
    """
    proto.schedule(5.3, 0.1, 1, 1000, 0) 
    sim = myokit.Simulation(mod,proto)
    sim.pre(1000 * 100) #pre-pace for 100 beats
    dat = sim.run(5000)
    IC = sim.state()

    # Get t, v, and cai for second to last AP#######################
    #t, v, cai, i_ion = get_last_ap(dat, -2)
    t, v = get_last_ap(dat, -2)

    return (t, v, IC) 