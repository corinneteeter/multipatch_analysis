"""get_deconvolved_first_pulse_amps(pair) will return deconvolved 
first pulse amplitudes of qc'd current clamp pulses for a specified pair.
See __main__ for use example. 
"""

import multipatch_analysis.database as db
import multipatch_analysis.connection_strength as cs 
import pandas
import numpy as np

def event_qc(events, criterion):
    '''QC

    Inputs
    ------
    events: class 'numpy.recarray'
        information on pulses as returned from local get_amps() (as opposed to cs.get_amps)
    criterion: string
        specifies use of excitatory or inhibitory qc critera
        opitions: 'in_qc_pass' or 'ex_qc_pass'
    
    Returns
    -------
    mask: numpy.ndarray
        an array of booleans which specify whether to use the pulse based on the qc criterea;
        i.e. if True, use the corresponding pulse, if False, mask it

    '''
    mask = events[criterion] == True
    # need more stringent qc for dynamics:
    mask &= np.abs(events['baseline_current']) < 500e-12
    return mask 

def join_pulse_response_to_expt(query):
    """
    The specifics of the following are undocumented for now.
    Inputs
    ------
    query: 'querysqlalchemy.orm.query.Query' class

    Returns
    -------
    query: 'sqlalchemy.orm.query.Query' class 
    pre_rec: 'sqlalchemy.orm.util.AliasedClass' class 
    post_rec: 'sqlalchemy.orm.util.AliasedClass' class
    """
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

def get_amps(session, pair, clamp_mode='ic', get_data=False):
    """Select records from pulse_response_strength table
    This is a version grabbed from cs.get_amps altered to get
    additional info.  Note that if cs.get_baseline_amps is also
    called there could be some unexpected results/correspondense
    problems if cs.get_amps or cs.get_baseline is altered.  Also
    note that this function returns more information than is 
    needed for the use case in this script.
    
    Inputs
    ------
    session: 'sqlalchemy.orm.session.Session' class
    pair: 'multipatch_analysis.database.database.Pair' class
        data structure from database for a specific pre and post synaptic connection
    clamp_mode: string
        specifies whether to return current clamp or voltage clamp data
        option: 'ic' (current clamp) or 'vc' (voltage clamp)
    get_data: boolean
        if True: return data waveforms 

    Returns
    -------
    recs: numpy.recarray
        contains various results for pulses which are not specifically documented here;  
        look at function for more info
    """

    cols = [
        db.PulseResponseStrength.id,
        db.PulseResponseStrength.pos_amp,
        db.PulseResponseStrength.neg_amp,
        db.PulseResponseStrength.pos_dec_amp,
        db.PulseResponseStrength.neg_dec_amp,
        db.PulseResponseStrength.pos_dec_latency,
        db.PulseResponseStrength.neg_dec_latency,
        db.PulseResponseStrength.crosstalk,
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

def get_deconvolved_first_pulse_amps(pair):
    """
    Get the amplitudes of deconvolved first pulses.  Note
    this is only set up for current clamp at the moment.
    Input
    -----
    pair: 'multipatch_analysis.database.database.Pair' class
        data structure from database for a specific pre and post synaptic connection 

    Returns
    -------
    first_pulse_amps: numpy.ndarray
        deconvolved amplitudes of first pulses from current clamp

    """
    session = db.Session()
    # Get a list of all presynaptic spike times and the amplitudes of postsynaptic responses    
    clamp_mode='ic'
    raw_events = get_amps(session, pair, clamp_mode=clamp_mode, get_data=False) #note this is the local version not the one in cs.get_amps
    
    # set up qc filter
    if ((pair.connection_strength.synapse_type == 'in') & (clamp_mode == 'ic')) or ((pair.connection_strength.synapse_type == 'ex') & (clamp_mode == 'vc')):
        amplitude_field = 'neg_dec_amp'
        criterion='in_qc_pass'
    elif ((pair.connection_strength.synapse_type == 'ex') & (clamp_mode == 'ic')) or ((pair.connection_strength.synapse_type == 'in') & (clamp_mode == 'vc')):  
        amplitude_field = 'pos_dec_amp'
        criterion='ex_qc_pass'
    else:
        raise Exception("the qc filter doesnt make sense?")
    
    # get qc'd events
    pass_qc_mask = event_qc(raw_events, criterion)
    included_events = raw_events[pass_qc_mask]
    pass_qc_out = pass_qc_mask[pass_qc_mask] 

    # return None if there is there are no events that passed qc
    if len(included_events) == 0:
        print ("\tno qc passing events for pair.id: %s" % pair.id)
        session.close()
        return None

    # return None if the there is no connection strength
    if pair.connection_strength is None:
        print ("\t\tSKIPPING: pair.id %s, is not yielding pair.connection_strength" % pair.id)
        return None
    
    # get the background values.  Note thate they all pass qc.
    raw_bg_events = cs.get_baseline_amps(session, pair, clamp_mode='ic')
    bg_qc_mask = event_qc(raw_bg_events, criterion)
    bg_events = raw_bg_events[bg_qc_mask]
    mean_bg_amp = bg_events[amplitude_field].mean()

    # get spike times and amplitudes
    # the 1e-9 is present because the time stamp is turned into nanoseconds
    rec_times = 1e-9 * (included_events['rec_start_time'].astype(float) - float(included_events['rec_start_time'][0]))
    amplitudes = included_events[amplitude_field] - mean_bg_amp

    # get first pulses
    first_pulse_mask = included_events['pulse_number'] == 1
    first_pulse_amps = amplitudes[first_pulse_mask]
 
    return first_pulse_amps

if __name__ == "__main__":

    """Example"""

    db.Session()
    pair=db.experiment_from_timestamp(1533244490.755).pairs[(6, 4)]
    fpda = get_deconvolved_first_pulse_amps(pair)
    print(fpda)