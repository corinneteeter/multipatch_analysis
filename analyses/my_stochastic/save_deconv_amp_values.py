# Extracting and saving deconvolved amplitudes from database for easy remote use.

from neuroanalysis.data import Trace, TraceList
from multipatch_analysis.database import database as db
import multipatch_analysis.connection_strength as cs 
from multipatch_analysis.database.database import TableGroup
from multipatch_analysis.pulse_response_strength import PulseResponseStrength, BaselineResponseStrength
import matplotlib.pyplot as plt
import numpy as np
import time
from neuroanalysis.fitting import fit_psp
import datetime
import os
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import random
import pandas


def query_specific_experiment(expt_id, pre_cell_ext_id=None, post_cell_ext_id=None):
    """Query and experiment id.
    
    Inputs
    ------
    expt_id: float (3 decimal positions)
        id of the experiment, corresponds to the db.Experiment.acq_timestamp
    pre_cell_ext_id: int
        identifies the pre synaptic cell, if pre_cell is None, finds all pre synaptic cells in experiment
    post_cell_ext_id: int
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
    if pre_cell_ext_id and post_cell_ext_id:
        expt_stuff = session.query(db.Pair, db.Experiment.acq_timestamp, pre_cell.ext_id, post_cell.ext_id, pre_cell.cre_type, post_cell.cre_type)\
                        .join(db.Experiment)\
                        .join(pre_cell, db.Pair.pre_cell_id==pre_cell.id)\
                        .join(post_cell, db.Pair.post_cell_id==post_cell.id).filter(db.Experiment.acq_timestamp == expt_id)\
                                                                            .filter(pre_cell.ext_id == pre_cell_ext_id)\
                                                                            .filter(post_cell.ext_id == post_cell_ext_id).all()
    else:
        expt_stuff = session.query(db.Pair, db.Experiment.acq_timestamp, pre_cell.ext_id, post_cell.ext_id, pre_cell.cre_type, post_cell.cre_type)\
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

def join_pulse_response_to_expt(query):
    pre_rec = db.aliased(db.Recording)
    post_rec = db.aliased(db.Recording)
    joins = [
        (post_rec, db.PulseResponse.recording),
        (db.PatchClampRecording,),
        (db.MultiPatchProbe,),
        (db.SyncRec,),
        (db.Experiment,),
        (db.StimPulse, db.PulseResponse.stim_pulse),
        (pre_rec, db.StimPulse.recording),
    ]
    for join_args in joins:
        query = query.join(*join_args)

    return query, pre_rec, post_rec

def stolen_get_amps(session, pair, clamp_mode='ic', get_data=False):
    """Select records from pulse_response_strength table
    This is a version grabbed from cs.get_amps altered to get
    additional info.  Note that if cs.get_baseline_amps is also
    called there could be some unexpected results/correspondense
    problems if cs.get_amps or cs.get_baseline is altered.
    """
    cols = [
        PulseResponseStrength.id,
        PulseResponseStrength.pos_amp,
        PulseResponseStrength.neg_amp,
        PulseResponseStrength.pos_dec_amp,
        PulseResponseStrength.neg_dec_amp,
        PulseResponseStrength.pos_dec_latency,
        PulseResponseStrength.neg_dec_latency,
        PulseResponseStrength.crosstalk,
        db.PulseResponse.ex_qc_pass,
        db.PulseResponse.in_qc_pass,
        db.PatchClampRecording.clamp_mode,
        db.PatchClampRecording.baseline_potential,
        db.PatchClampRecording.baseline_current,
        db.StimPulse.pulse_number,
        db.StimSpike.max_dvdt_time,
        db.PulseResponse.start_time.label('response_start_time'),
        db.MultiPatchProbe.induction_frequency.label('stim_freq'),
    ]
    if get_data:
        cols.append(db.PulseResponse.data)

    q = session.query(*cols)
    q = q.join(db.PulseResponse)
    
    q, pre_rec, post_rec = join_pulse_response_to_expt(q)
    q = q.join(db.StimSpike)
    # note that these are aliased so needed to be joined outside the join_pulse_response_to_expt() function
    q = q.add_columns(post_rec.start_time.label('rec_start_time'))
    q = q.add_columns(post_rec.id.label('post_rec_id'))  #this will identify the train the pulse is in
        
    filters = [
        (pre_rec.electrode==pair.pre_cell.electrode,),
        (post_rec.electrode==pair.post_cell.electrode,),
        (db.PatchClampRecording.clamp_mode==clamp_mode,),
        (db.PatchClampRecording.qc_pass==True,),
    ]
    for filter_args in filters:
        q = q.filter(*filter_args)
    
    # should result in chronological order
    q = q.order_by(db.PulseResponse.id)

    df = pandas.read_sql_query(q.statement, q.session.bind)
    recs = df.to_records()
    return recs

def get_deconvolved_amp(uid, pre_cell_id, post_cell_id, get_data=False, get_only_qc_pass=True):
    """
    Input
    -----
    Returns
    -------
    """
    session = db.Session()
    #note requerying the pair because the Session was going lazy
    expt = db.experiment_from_timestamp(uid, session=session)
    pair = expt.pairs[(pre_cell_id, post_cell_id)]
    
    # Get a list of all presynaptic spike times and the amplitudes of postsynaptic responses    
    clamp_mode='ic'
    raw_events = stolen_get_amps(session, pair, clamp_mode=clamp_mode, get_data=get_data)
    
    # set up qc filter
    if ((pair.connection_strength.synapse_type == 'in') & (clamp_mode == 'ic')) or ((pair.connection_strength.synapse_type == 'ex') & (clamp_mode == 'vc')):
        amplitude_field = 'neg_dec_amp'
        criterion='in_qc_pass'
    elif ((pair.connection_strength.synapse_type == 'ex') & (clamp_mode == 'ic')) or ((pair.connection_strength.synapse_type == 'in') & (clamp_mode == 'vc')):  
        amplitude_field = 'pos_dec_amp'
        criterion='ex_qc_pass'
    else:
        raise Exception("the qc filter doesnt make sense?")
    pass_qc_mask = event_qc(raw_events, criterion)
    # return all or just the passing pulses
    if get_only_qc_pass:
        included_events = raw_events[pass_qc_mask]
        pass_qc_out = pass_qc_mask[pass_qc_mask] 
    else:
        included_events = raw_events
        pass_qc_out = pass_qc_mask
    
    # jump out of loop if there is there are no events that passed qc
    print("TRYING: %0.3f, cell ids:%s %s" % (uid, pre_cell_id, post_cell_id))
    if len(included_events) == 0:
        print ("\tno qc passing events: %0.3f, cell ids:%s %s" % (uid, pre_cell_id, post_cell_id))
        session.close()
        return None

    # jump out of loop if the there is no connection strength
    if pair.connection_strength is None:
        print ("\t\tSKIPPING: pair_id %s, uid %s, is not yielding pair.connection_strength" % (pair.id, uid))
        return None
    
    # get the background values.  Note thate they all pass qc.
    raw_bg_events = cs.get_baseline_amps(session, pair, clamp_mode='ic')
    bg_qc_mask = event_qc(raw_bg_events, criterion)
    bg_events = raw_bg_events[bg_qc_mask]
    mean_bg_amp = bg_events[amplitude_field].mean()

    # get spike times and amplitudes
    # I believe the following must be putting the events in reference to the start of the experiment. 
    # the 1e-9 is present because the time stamp is turned into nanoseconds
    rec_times = 1e-9 * (included_events['rec_start_time'].astype(float) - float(included_events['rec_start_time'][0]))
    #stim_spike table says 'max_dvdt_time' it is relative to beginning of recording.  Is the recording a sweep (train)?
    spike_times_relative_to_experiment = included_events['max_dvdt_time'] + rec_times
    amplitudes = included_events[amplitude_field] - mean_bg_amp

    #Example of screening for first pulse
    # first_pulse_mask = included_events['pulse_number'] == 1
    # first_pulse_amps = amplitudes[first_pulse_mask]
    # first_pulse_times = rec_times[first_pulse_mask]
    # first_pulse_ids = pulse_ids[first_pulse_mask]

    # make sure all the arrays are the same length
    if (len(spike_times_relative_to_experiment) != len(amplitudes)) | \
        (len(amplitudes) != len(included_events['id'])) | \
        (len(amplitudes) != len(included_events['pulse_number'])) | \
        (len(amplitudes) != len(included_events['post_rec_id'])) | \
        (len(amplitudes) != len(included_events['stim_freq'])):
        raise Exception('arrays are a different length')

    out_dict={'spike_times_relative_to_experiment': spike_times_relative_to_experiment,
              'deconvolved_amps': amplitudes,
              'pulse_database_ids': included_events['id'],
              'pulse_num_within_train': included_events['pulse_number'],
              'post_syn_rec_id': included_events['post_rec_id'], 
              'only_qc_pass_returned': get_only_qc_pass,
              'stim_freq': included_events['stim_freq']}
              
              #make sure these are all coded appropriately
    if get_data:
        out_dict['data_traces'] = included_events['data']
        if len(amplitudes) != len(out_dict['data_traces']):
            raise Exception('arrays are a different length')
        session.close()
    else:
        out_dict['data_traces'] = None

    if get_only_qc_pass is False:
        out_dict['qc_mask'] = pass_qc_out
        if len(amplitudes) != len(out_dict['qc_mask']):
            raise Exception('arrays are a different length')
    else:
        out_dict['qc_mask'] = None
    return out_dict

def plot_expt_dec_amp_dist(uid, pre=None, post=None, show_plot=False, block_plot=True):
    # get all pair recordings in experiment
    expt_stuff=query_specific_experiment(uid)

    #plot amplitudes
    n=0
    xlabels=[]
    for (p, uid, pre_cell_id, post_cell_id, pre_cell_cre, post_cell_cre) in expt_stuff:

        deconvolved = get_deconvolved_amp(uid, pre_cell_id, post_cell_id, get_data=False)
        spike_times_relative_to_experiment = deconvolved['spike_times_relative_to_experiment'] 
        amplitudes = deconvolved['deconvolved_amps']
        pulse_ids = deconvolved['pulse_database_ids']
        pulse_numbers = deconvolved['pulse_num_within_train']
        pass_qc = deconvolved['qc_mask']
        if deconvolved is None:
            continue
        if ((pre_cell_id == pre) & (post_cell_id == post)):
            plt.plot(n*np.ones(len(amplitudes)), amplitudes*1e3, '.', ms=20)
        else:
            plt.plot(n*np.ones(len(amplitudes)), amplitudes*1e3, '.', ms=10)
        xlabels.append('%s %i, %s %i' %(pre_cell_cre, pre_cell_id, post_cell_cre, post_cell_id))
        n=n+1
    plt.ylabel('Deconvolution Arbitrary Units')
    plt.title('Deconvolution Amplitude\n%.3f' % uid)
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

#--- Some good PV examples.  These are excellent fits
connections = [
    #pv to pv
    (1533244490.755, 6, 4),
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
    (1539801888.714, 7, 1),
    # tlx 3 to sst
    (1486504558.948, 3, 5),
    (1486511976.478, 7, 2),
    (1490218204.397, 8, 5),
    (1490304528.810, 8, 7),
    (1490304528.810, 5, 7),
    (1490819652.306, 6, 5),
    (1492030010.631, 6, 2),
    (1492204582.392, 2, 7),
    (1492036074.933, 7, 5),
    (1492212394.505, 5, 4),
    (1492212394.505, 3, 4),
    (1496356414.741, 3, 6),
    (1496356414.741, 3, 4),
    (1497653267.742, 3, 4),
    (1498258681.855, 3, 1),
    (1499898590.696, 2, 8),
    (1499898590.696, 1, 8),
    (1508362350.796, 6, 3),
    (1513809172.285, 3, 1),
    (1526677592.224, 5, 4),
    (1502914763.893, 2, 8),
    (1504300627.314, 4, 8),
    (1504300627.314, 2, 8),
    (1498255210.715, 1, 5),
    (1522789170.885, 8, 5),
    (1535145422.740, 4, 6),
    (1535150219.310, 5, 3)]


import pandas as pd
#set up dictionary for group output
d={'uid':[],
   'pre_id': [],
   'post_id':[],
   'pre_cre':[],
   'post_cre':[],
   'spike_times_relative_to_experiment':[],
   'pulse_num_within_train':[],
   'deconvolved_amps':[],
   'pulse_database_ids':[],
   'qc_mask':[],
   'stim_freq': [],
   'post_syn_rec_id': []}

for uid, pre, post in connections:
    expt_stuff= query_specific_experiment(uid, pre_cell_ext_id=pre, post_cell_ext_id=post)
    if len(expt_stuff) > 1:
        raise Exception('There should only be a list of 1 experiment returned')

    deconvolved = get_deconvolved_amp(uid, pre, post, get_data=False)

    d['uid'].append(uid)
    d['pre_id'].append(pre)
    d['post_id'].append(post)
    d['pre_cre'].append(expt_stuff[0][4])
    d['post_cre'].append(expt_stuff[0][5])
    d['spike_times_relative_to_experiment'].append( deconvolved['spike_times_relative_to_experiment'] )
    d['deconvolved_amps'].append(deconvolved['deconvolved_amps'])
    d['pulse_database_ids'].append(deconvolved['pulse_database_ids'])
    d['qc_mask'].append(deconvolved['qc_mask'])
    d['pulse_num_within_train'].append(deconvolved['pulse_num_within_train'])
    d['stim_freq'].append(deconvolved['stim_freq'])
    d['post_syn_rec_id'].append(deconvolved['post_syn_rec_id'])
data=pd.DataFrame.from_dict(d)
data.to_csv('pv_tlx_to_sst.csv')

    
    # plt.figure(figsize=(15,10))
    # sample=range(0,len(deconvolved['deconvolved_amps']))
    # plt.plot(deconvolved['spike_times_relative_to_experiment'][sample], deconvolved['deconvolved_amps'][sample]*1e3, '.-', ms=20)
    # # for ii, (p,t,a) in enumerate(zip(pulse_ids[sample], spike_times_relative_to_experiment[sample], amplitudes[sample]*1e3)):
    # #     plt.annotate(str(ii), xy=(t,a), textcoords='data')
    # plt.ylabel('Deconvolution Arbitrary Units')
    # plt.xlabel('time (unknown units)')
    # plt.title('Deconvolution Amplitude\n%.3f, pre %i, post %i' %(uid, pre, post))
    # plt.show()

    
    #extract pulse wave forms if you want to plot them
# #    pulse_waveforms=get_pulse_waveforms(uid, pre, post, pulse_ids)  
#     plt.subplot(2,1,2)
#     for ii, v_trace in enumerate(data):
#         time=np.arange(0, len(v_trace)/20000., 1./20000 )
#         l=plt.plot(time, v_trace)
#         c=l[0].get_color()
#         r=random.randint(0,len(time)-1)
#         plt.annotate(str(ii), xy=(time[r], v_trace[r]), textcoords='data', color=c)


#    plt.tight_layout()
#    plt.show()
#    plt.savefig('/home/corinnet/workspace/aiephys/decon_images/%0.3f_pre%i_post%i.png' %(uid, pre, post))
#    plt.close()
 
