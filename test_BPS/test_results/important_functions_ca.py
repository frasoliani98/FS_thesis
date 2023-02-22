import myokit
def get_ind_data(ind):

    mod, proto, x = myokit.load('./BPS3.mmt')
    if ind is not None:
        for k, v in ind[0].items():
            mod['multipliers'][k].set_rhs(v)

    return mod, proto

def plot_ca(ind):
    ## EAD CHALLENGE: Istim = -.1
    mod, proto = get_ind_data(ind)
    proto.schedule(5.3, 0.1, 1, 1000, 0)
    sim = myokit.Simulation(mod, proto)
    
    sim.pre(1000*600)
    dat = sim.run(5000)
    cai = dat['intracellular_ions.cai']
    #ki = dat['intracellular_ions.ki']
    tc = dat['engine.time']
    #Ical = dat['ICal.ICal']
    #ICal = dat['ICaL.ICaL']
    #IK1 = dat['IK1.IK1']
    #IKb = dat['IKb.IKb']
    #IKs = dat['IKs.IKs']
    #INa = dat['INa.INa']
    #INaCa = dat['INaCa.INaCa_i']
    #INaK = dat['INaK.INaK']
    #INaL = dat['INaL.INaL']
    #INab = dat['INab.INab']


    return tc, cai

def plot_ca_baseline():
    ## EAD CHALLENGE: Istim = -.1
    mod, proto, x = myokit.load('./BPS3.mmt')
    proto.schedule(5.3, 0.1, 1, 1000, 0)
    sim = myokit.Simulation(mod, proto)
    
    sim.pre(1000*600)
    dat = sim.run(5000)
    cai = dat['intracellular_ions.cai']
    #ki = dat['intracellular_ions.ki']
    tc = dat['engine.time']
    #Ical = dat['ICal.ICal']
    #ICal = dat['ICaL.ICaL']
    #IK1 = dat['IK1.IK1']
    #IKb = dat['IKb.IKb']
    #IKs = dat['IKs.IKs']
    #INa = dat['INa.INa']
    #INaCa = dat['INaCa.INaCa_i']
    #INaK = dat['INaK.INaK']
    #INaL = dat['INaL.INaL']
    #INab = dat['INab.INab']


    return tc, cai
