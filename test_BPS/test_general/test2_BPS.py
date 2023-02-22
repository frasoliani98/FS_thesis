import myokit

def get_ind_data(ind):

    mod, proto, x = myokit.load('./BPS.mmt')
    if ind is not None:
        for k, v in ind[0].items():
            mod['multipliers'][k].set_rhs(v)

    return mod, proto

ind = [{'i_cal_pca_multiplier': 1, 'i_kr_multiplier': 1, 'i_ks_multiplier': 1, 'i_nal_multiplier': 1, 'i_na_multiplier': 1, 'i_to_multiplier': 1, 'i_k1_multiplier': 1, 'i_NCX_multiplier': 1, 'i_nak_multiplier': 1, 'i_kb_multiplier': 1, 'jrel_multiplier' : 1, 'jup_multiplier' : 1, }]
print(ind)
mod, proto = get_ind_data(ind)
proto.schedule(5.3, 0.1, 1, 4000, 0) 
sim = myokit.Simulation(mod, proto)
sim.pre(4000 * 100) #pre-pace for 100 beats

a = 100
print(a)
