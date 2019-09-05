"""This file views the stimuli available in each pair 
"""

from multipatch_analysis.database import default_db as db
import numpy as np
import pandas as pd

def set_recovery(delay):
    """If a *delay* is within 5 ms of an expected recovery value, set it to that.  
    Else do not alter it."""
    if not delay:
        return None
    expected_delays = np.array([.125, .250, .500, 1.000, 2.000, 4.000])
    value = expected_delays[np.isclose(delay, expected_delays, atol=.005)] #allow a 5 ms variation in recovery
    if len(value) == 0:
        return delay 
    elif len(value) == 1:
        return value[0]
    else: 
        raise Exception("a number can't be within 5 ms of two specified delays")


def load_pair_pulse_responses(pair):
    print("Loading:", pair)
    
#    q = db.query(db.PulseResponse, db.PulseResponseFit, db.StimPulse, db.PatchClampRecording, db.MultiPatchProbe, db.Synapse, db.PulseResponse.data)
    q = db.query(db.PatchClampRecording, db.MultiPatchProbe)
    q = q.join(db.PulseResponseFit, db.PulseResponse.pulse_response_fit)
    q = q.join(db.StimPulse, db.PulseResponse.stim_pulse)
    q = q.join(db.Recording, db.PulseResponse.recording)
    q = q.join(db.PatchClampRecording, db.PatchClampRecording.recording_id==db.Recording.id)
    q = q.join(db.MultiPatchProbe, db.MultiPatchProbe.patch_clamp_recording_id==db.PatchClampRecording.id)
    q = q.join(db.Synapse, db.Synapse.pair_id==db.PulseResponse.pair_id)
    q = q.filter(db.PulseResponse.pair_id==pair.id)

    pr_recs = q.all()

    # # group records by (clamp mode, ind_freq, rec_delay), and then by pulse number
    sorted_recs = {}
    for rec in pr_recs:
        # if a recovery delay is within 5 ms of expected value round it to that value
        delay = set_recovery(rec.multi_patch_probe.recovery_delay)
        stim_key = rec.patch_clamp_recording.clamp_mode+', '+str(rec.multi_patch_probe.induction_frequency)+', '+str(delay)
#        stim_key = (rec.patch_clamp_recording.clamp_mode, rec.multi_patch_probe.induction_frequency)
        sorted_recs.setdefault(stim_key, 0) #initialize key if it is not present 
        sorted_recs[stim_key] += 1 # count this instance of this key
        # pn = rec.stim_pulse.pulse_number
        # sorted_recs[stim_key].setdefault(pn, [])
        # sorted_recs[stim_key][pn].append(rec)
    return sorted_recs
    
def max_value_dict(dict1, dict2):
    """Make a global dictionary were each stimulus will have the maximum number 
    present in any one stimulus"""
    for key1 in dict1.keys():
        if key1 in dict2:
            dict2[key1] = max(dict2[key1], dict1[key1]) 
        else:
            dict2[key1] = dict1[key1]
    return dict2

def put_stim_in_table(df, pair, pair_dict):

    possible_stimuli =[
        'ic, 10.0, 0.25', 
        'ic, 20.0, 0.25', 
        'ic, 25.0, 0.25', 
        'ic, 50.0, 0.053', 
        'ic, 50.0, 0.103', 
        'ic, 50.0, 0.125', 
        'ic, 50.0, 0.203', 
        'ic, 50.0, 0.25', 
        'ic, 50.0, 0.403', 
        'ic, 50.0, 0.5', 
        'ic, 50.0, 0.803', 
        'ic, 50.0, 1.0', 
        'ic, 50.0, 1.603', 
        'ic, 50.0, 2.0', 
        'ic, 50.0, 3.203', 
        'ic, 50.0, 4.0', 
        'ic, 100.0, 0.25', 
        'ic, 150.0, 0.25', 
        'ic, 200.0, 0.25', 
        'ic, None, None', 
        'vc, 10.0, 0.25', 
        'vc, 20.0, 0.25', 
        'vc, 25.0, 0.25', 
        'vc, 50.0, 0.053', 
        'vc, 50.0, 0.103', 
        'vc, 50.0, 0.125', 
        'vc, 50.0, 0.203', 
        'vc, 50.0, 0.25', 
        'vc, 50.0, 0.403', 
        'vc, 50.0, 0.5', 
        'vc, 50.0, 0.803', 
        'vc, 50.0, 1.0', 
        'vc, 50.0, 1.603', 
        'vc, 50.0, 2.0', 
        'vc, 50.0, 3.203', 
        'vc, 50.0, 4.0', 
        'vc, 100.0, 0.25', 
        'vc, 150.0, 0.25', 
        'vc, 200.0, 0.25', 
        'vc, None, None']
    if df.empty:
        df=pd.DataFrame(columns=['expt', 'pre_cell', 'post_cell', 'pre_cre', 
                                'post_cre', 'pre_layer', 'post_layer'] + 
                                possible_stimuli)

    df =df.append(pair_dict, ignore_index=True)
    
    return df


# One example: <pair 1492018873.073 8 5>
# expt = db.experiment_from_timestamp(1492018873.073)
# pair = expt.pairs['8','4']
# stuff = load_pair_pulse_responses(pair)

# Figuring out what will be in the data set
q = db.pair_query(synapse=True)
#all_pairs = q.all()
all_pairs = q.all()
print(len(all_pairs), 'pairs')
columns={}
df = pd.DataFrame()
for ii, pair in enumerate(all_pairs):
    print(ii, 'out of', len(all_pairs) )
    stuff = load_pair_pulse_responses(pair)
    # populate information on pair to stimulus dictionary
    stuff['expt'] = str(pair.experiment.acq_timestamp)
    stuff['pre_cell'] = pair.pre_cell.ext_id
    stuff['post_cell'] = pair.post_cell.ext_id
    stuff['pre_cre'] = pair.pre_cell.cre_type
    stuff['post_cre'] = pair.post_cell.cre_type
    stuff['pre_layer'] = pair.pre_cell.target_layer
    stuff['post_layer'] = pair.post_cell.target_layer
    columns = max_value_dict(stuff, columns)
    df = put_stim_in_table(df, pair, stuff)
#     #stuff = {**stuff, **load_pair_pulse_responses(pair)} easy way to see all the keys that are available

print('FINAL RESULT')
print(columns)

#df.to_csv('num_stim_09_05_2019.csv', sep='#')

# import json
# with open('max_num_of_stim_in_pair.json', 'w') as outfile:
#     json.dump(sorted(columns), outfile)

 