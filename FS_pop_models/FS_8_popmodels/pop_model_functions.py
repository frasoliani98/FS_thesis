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
        #info = 0
        info = "no EAD"
        result = 0
    else:
        #info = round(max(EADs))
        info = "EAD: " + str(round(max(EADs))) +' mV'
        # i think that take the max because is the first that occurs (the max one)
        
        result = 1
    
    return info, result

def run_EAD(ical, iks, ikr, inal, ina, ito, ik1, iNCX, inak, ikb, enha):

    mod, proto = get_ind_data(ind = None)

    # SET MULTIPLIERS
    mod['multipliers']['i_cal_pca_multiplier'].set_rhs(ical)
    mod['multipliers']['i_ks_multiplier'].set_rhs(iks)
    mod['multipliers']['i_kr_multiplier'].set_rhs(ikr)
    mod['multipliers']['i_nal_multiplier'].set_rhs(inal)
    mod['multipliers']['i_na_multiplier'].set_rhs(ina)
    mod['multipliers']['i_to_multiplier'].set_rhs(ito)
    mod['multipliers']['i_k1_multiplier'].set_rhs(ik1)
    mod['multipliers']['i_NCX_multiplier'].set_rhs(iNCX)
    mod['multipliers']['i_nak_multiplier'].set_rhs(inak)
    mod['multipliers']['i_kb_multiplier'].set_rhs(ikb)
    

    # EADS) Add or multiply 8
    mod['multipliers']['i_cal_pca_multiplier'].set_rhs(
        enha*mod['multipliers']['i_cal_pca_multiplier'].value())

    proto.schedule(5.3, 0.1, 1, 1000, 0)
    sim = myokit.Simulation(mod, proto)
    dat = sim.run(50000)

    # Get t, v, and cai for second to last AP#######################
    t, v = get_last_ap(dat, -2)
    t1, v1 = get_last_ap(dat, -3)

    return t, v, t1, v1

def run_EAD_IKr(ical, iks, ikr, inal, ina, ito, ik1, iNCX, inak, ikb, block):

    mod, proto = get_ind_data(ind = None)

    # SET MULTIPLIERS
    mod['multipliers']['i_cal_pca_multiplier'].set_rhs(ical)
    mod['multipliers']['i_ks_multiplier'].set_rhs(iks)
    mod['multipliers']['i_kr_multiplier'].set_rhs(ikr)
    mod['multipliers']['i_nal_multiplier'].set_rhs(inal)
    mod['multipliers']['i_na_multiplier'].set_rhs(ina)
    mod['multipliers']['i_to_multiplier'].set_rhs(ito)
    mod['multipliers']['i_k1_multiplier'].set_rhs(ik1)
    mod['multipliers']['i_NCX_multiplier'].set_rhs(iNCX)
    mod['multipliers']['i_nak_multiplier'].set_rhs(inak)
    mod['multipliers']['i_kb_multiplier'].set_rhs(ikb)
    

    # EADS) Add or multiply 8
    mod['multipliers']['i_kr_multiplier'].set_rhs(
        block*mod['multipliers']['i_kr_multiplier'].value())
    mod['multipliers']['i_kb_multiplier'].set_rhs(
        0.5*mod['multipliers']['i_kb_multiplier'].value())
    

    proto.schedule(5.3, 0.1, 1, 1000, 0)
    sim = myokit.Simulation(mod, proto)
    dat = sim.run(50000)

    # Get t, v, and cai for second to last AP#######################
    t, v = get_last_ap(dat, -2)
    t1, v1 = get_last_ap(dat, -3)
    

    return t, v, t1, v1
