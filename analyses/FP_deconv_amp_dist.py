from neuroanalysis.data import Trace, TraceList
from multipatch_analysis.database import database as db
import multipatch_analysis.connection_strength as cs 
from multipatch_analysis.database.database import TableGroup
import matplotlib.pyplot as plt
import numpy as np
import time
from neuroanalysis.fitting import fit_psp
import multipatch_analysis.FPFitting_DB_library as FPF_lib
import datetime
import os
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import random

def query_specific_experiment(expt_id, pre_cell=None, post_cell=None):
    """Query and experiment id.
    
    Inputs
    ------
    expt_id: float (3 decimal positions)
        id of the experiment, corresponds to the db.Experiment.acq_timestamp
    pre_cell: int
        identifies the pre synaptic cell, if pre_cell is None, finds all pre synaptic cells in experiment
    post_cell: int
        identifies the post synaptic cell, if post_cell is None, finds all post synaptic cells in experiment

    Returns
    -------
    expt_stuff: list of tuples.  
        Each tuple contains the following information for a pair in the experiment
        pair:  db pair object 
            contains information about the synapse between a pre and post synaptic pair
        uid: float (3 decimal positions)
            id of the experiment, corresponds to the db.Experiment.acq_timestamp
        pre_cell_id: int
            identifies the pre synaptic cell, if pre_cell is None, finds all pre synaptic cells in experiment 
        post_cell_id: int
            identifies the post synaptic cell, if post_cell is None, finds all post synaptic cells in experiment
        pre_cell_cre: string
            cre line of the pre synaptic cell 
        post_cell_cre: string
            cre line of the post synaptic cell
    """
    print("QUERYING (expt_id=%f)" % expt_id)

    #do the query
    session = db.Session() #create session
    pre_cell = db.aliased(db.Cell)
    post_cell = db.aliased(db.Cell)
    expt_stuff = session.query(db.Pair, db.Experiment.acq_timestamp, pre_cell.ext_id, post_cell.ext_id,pre_cell.cre_type, post_cell.cre_type)\
                        .join(db.Experiment)\
                        .join(pre_cell, db.Pair.pre_cell_id==pre_cell.id)\
                        .join(post_cell, db.Pair.post_cell_id==post_cell.id).filter(db.Experiment.acq_timestamp==expt_id).all()
    # make sure query returned something
    if len(expt_stuff) <=0:
        print('No pairs found for expt_id=%f', expt_id)
        return
    
    return expt_stuff

def event_qc(events, criterion):
    mask = events[criterion] == True
    # need more stringent qc for dynamics:
    mask &= np.abs(events['baseline_current']) < 500e-12
    return mask 

def get_first_pulse_deconvolved_amp(uid, pre_cell_id, post_cell_id, get_data=False):
    session = db.Session()
    #note requerying the pair because the Session was going lazy
    expt = db.experiment_from_timestamp(uid, session=session)
    pair = expt.pairs[(pre_cell_id, post_cell_id)]
    
    # 1. Get a list of all presynaptic spike times and the amplitudes of postsynaptic responses    
    clamp_mode='ic'
    raw_events = cs.get_amps(session, pair, clamp_mode=clamp_mode, get_data=get_data)
    if ((pair.connection_strength.synapse_type == 'in') & (clamp_mode == 'ic')) or ((pair.connection_strength.synapse_type == 'ex') & (clamp_mode == 'vc')):
        amplitude_field = 'neg_dec_amp'
        criterion='in_qc_pass'
    elif ((pair.connection_strength.synapse_type == 'ex') & (clamp_mode == 'ic')) or ((pair.connection_strength.synapse_type == 'in') & (clamp_mode == 'vc')):  
        amplitude_field = 'pos_dec_amp'
        criterion='ex_qc_pass'
    else:
        raise Exception('huh?')
    pass_qc_mask = event_qc(raw_events, criterion)
    events = raw_events[pass_qc_mask]
    if pair.connection_strength is None:
        print ("\t\tSKIPPING: pair_id %s, uid %s, is not yielding pair.connection_strength" % (pair.id, uid))
        return None, None, None
    print("TRYING: %0.3f, cell ids:%s %s" % (uid, pre_cell_id, post_cell_id))
    if len(events)==0:
        print ("\tno events: %0.3f, cell ids:%s %s" % (uid, pre_cell_id, post_cell_id))
        session.close()
        return None, None, None
    
    rec_times = 1e-9 * (events['rec_start_time'].astype(float) - float(events['rec_start_time'][0]))
    spike_times = events['max_dvdt_time'] + rec_times

    # 2. Initialize model parameters:
    #    - release model with depression, facilitation
    #    - number of synapses, distribution of per-vesicle amplitudes estimated from first pulse CV
    #    - measured distribution of background noise
    #    - parameter space to be searched
    raw_bg_events = cs.get_baseline_amps(session, pair, clamp_mode='ic')
    bg_qc_mask = event_qc(raw_bg_events, criterion)
    bg_events = raw_bg_events[bg_qc_mask]
    mean_bg_amp = bg_events[amplitude_field].mean()
    amplitudes = events[amplitude_field] - mean_bg_amp
    pulse_ids = events['id']

    first_pulse_mask = events['pulse_number'] == 1
    first_pulse_amps = amplitudes[first_pulse_mask]
    first_pulse_times = rec_times[first_pulse_mask]
    first_pulse_ids = pulse_ids[first_pulse_mask]
    if get_data==True:
        data = events['data'][first_pulse_mask]
        session.close()
        return first_pulse_times, first_pulse_amps, first_pulse_ids, data
    else:
        return first_pulse_times, first_pulse_amps, first_pulse_ids

def plot_expt_dec_amp_dist(uid, pre=None, post=None, show_plot=False, block_plot=True):
    # get all pair recordings in experiment
    expt_stuff=query_specific_experiment(uid)

    #plot amplitudes
    n=0
    xlabels=[]
    for (p, uid, pre_cell_id, post_cell_id, pre_cell_cre, post_cell_cre) in expt_stuff:
        first_pulse_times, first_pulse_amps, pulse_ids=get_first_pulse_deconvolved_amp(uid, pre_cell_id, post_cell_id, get_data=False)
        if (np.any(first_pulse_times)==None) or (np.any(first_pulse_amps)==None):
            continue
        if ((pre_cell_id == pre) & (post_cell_id == post)):
            plt.plot(n*np.ones(len(first_pulse_amps)), first_pulse_amps*1e3, '.', ms=20)
        else:
            plt.plot(n*np.ones(len(first_pulse_amps)), first_pulse_amps*1e3, '.', ms=10)
        xlabels.append('%s %i, %s %i' %(pre_cell_cre, pre_cell_id, post_cell_cre, post_cell_id))
        n=n+1
    plt.ylabel('mV')
    plt.title('First Pulse Deconvolution Amplitude\n%.3f' % uid)
    plt.xticks(range(0,n), xlabels, rotation=90)
    plt.tight_layout()
    if show_plot==True:
        if block_plot==True:
            plt.show()
        else:
            plt.show(block=False)


#--- All experiments
# session = db.Session() #create session
# experiments = session.query(db.Experiment.acq_timestamp).all()

#--- Some good PV examples
pv = [(1533244490.755, 6, 4),
    (1530559621.966, 7, 6),
    (1527020350.517, 6, 7),
    (1525903868.115, 6, 5),
    (1525903868.115, 7, 6),
    (1525812130.134, 7, 6),
    (1523398687.484, 2, 1),
    (1523398687.484, 2, 3),
    (1523398687.484, 3, 2),
    (1521667891.153, 3, 4),
    (1519425770.224, 6, 7),
    (1519425770.224, 7, 2),
    (1518564082.242, 4, 6),
    (1517356361.727, 7, 6),
    (1517356063.393, 3, 4),
    (1517348193.989, 1, 6),
    (1517348193.989, 6, 8),
    (1517266234.424, 5, 4),
    (1517266234.424, 6, 5),
    (1517269966.858, 2, 4),
    (1517269966.858, 2, 8),
    (1517269966.858, 7, 8),
    (1516820219.855, 3, 4),
    (1516744107.347, 3, 6),
    (1513976168.029, 2, 1),
    (1511913615.871, 5, 4),
    (1510268339.831, 4, 5),
    (1510268339.831, 4, 7),
    (1510268339.831, 5, 4),
    (1510268339.831, 7, 4),
    (1508189280.042, 4, 8),
    (1507235159.776, 5, 6),
    (1505421643.243, 6, 7),
    (1505421643.243, 7, 6),
    (1505515553.146, 3, 8),
    (1505515553.146, 6, 5),
    (1505515553.146, 6, 7),
    (1505515553.146, 7, 3),
    (1500668871.652, 1, 7),
    (1496703451.204, 4, 1),
    (1495574744.012, 4, 2),
    (1493159586.902, 3, 5),
    (1493159586.902, 7, 8),
    (1492018873.073, 8, 4),
    (1491587831.024, 8, 3),
    (1491252584.113, 1, 5),
    (1491347329.805, 3, 6),
    (1491347329.805, 6, 2),
    (1490043541.218, 2, 7),
    (1490043541.218, 6, 2),
    (1490043541.218, 7, 2),
    (1490043541.218, 7, 6),
    (1484952266.115, 8, 4),
    (1539801888.714, 1, 8),
    (1539801888.714, 7, 1)]


for uid, pre, post in pv:
    plt.figure(figsize=(15,10))
    plt.subplot(221)
    #wrapper around get_first_pulse_deconvolved_amp that plots the deconvolved amplitude distributions for the experiment
    plot_expt_dec_amp_dist(uid, pre=pre, post=post, show_plot=False, block_plot=False)  
    first_pulse_times, first_pulse_amps, pulse_ids, data=get_first_pulse_deconvolved_amp(uid, pre, post, get_data=True)
    plt.subplot(222)
    plt.plot(first_pulse_times, first_pulse_amps*1e3, '.', ms=20)
    for ii, (p,t,a) in enumerate(zip(pulse_ids, first_pulse_times, first_pulse_amps*1e3)):
        plt.annotate(str(ii), xy=(t,a), textcoords='data')
    plt.ylabel('mV')
    plt.xlabel('time (unknown units)')
    plt.title('First Pulse Deconvolution Amplitude\n%.3f, pre %i, post %i' %(uid, pre, post))
    
    #extract pulse wave forms
#    pulse_waveforms=get_pulse_waveforms(uid, pre, post, pulse_ids)  
    plt.subplot(2,1,2)
    for ii, v_trace in enumerate(data):
        time=np.arange(0, len(v_trace)/20000., 1./20000 )
        l=plt.plot(time, v_trace)
        c=l[0].get_color()
        r=random.randint(0,len(time)-1)
        plt.annotate(str(ii), xy=(time[r], v_trace[r]), textcoords='data', color=c)


    plt.plot()
    plt.tight_layout()
    plt.show()
#    plt.savefig('/home/corinnet/workspace/aiephys/decon_images/%0.3f_pre%i_post%i.png' %(uid, pre, post))
    plt.close()
 

