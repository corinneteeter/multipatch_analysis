from multipatch_analysis.database import default_db as db
import numpy as np
from lib import set_recovery, possible_stimuli
import pandas as pd
import random
import logging
import os

LOG_FILENAME = '09_05_2019.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)

fit_params = ['amp', 'latency', 'rise_time', 'decay_tau']
pair_description = ['expt','pre_cell', 'post_cell','pre_cre','post_cre','pre_layer', 'post_layer']

def assign_wo_replacement(values, num):
    repeat = int(np.ceil(num/len(values)))
    #make a longerlist
    out=[]
    for ii in range(repeat):
        random.shuffle(values)
        out =  out + values
    return out[:num]


def load_pair_pulse_responses(pair):
    """For a given *pair* return a dictionary of the individual pulse response fit parameters
    output dict hierarchy looks like stimulus_type >> fit parameter >> pulse number
    """
    print("Loading:", pair)
    
#    q = db.query(db.PulseResponse, db.PulseResponseFit, db.StimPulse, db.PatchClampRecording, db.MultiPatchProbe, db.Synapse, db.PulseResponse.data)
    q = db.query(db.PulseResponse, db.PulseResponseFit, db.StimPulse, db.PatchClampRecording, db.MultiPatchProbe, db.Synapse, db.PulseResponse.data)
    q = q.join(db.PulseResponseFit, db.PulseResponse.pulse_response_fit)
    q = q.join(db.StimPulse, db.PulseResponse.stim_pulse)
    q = q.join(db.Recording, db.PulseResponse.recording)
    q = q.join(db.PatchClampRecording, db.PatchClampRecording.recording_id==db.Recording.id)
    q = q.join(db.MultiPatchProbe, db.MultiPatchProbe.patch_clamp_recording_id==db.PatchClampRecording.id)
    q = q.join(db.Synapse, db.Synapse.pair_id==db.PulseResponse.pair_id)
    q = q.filter(db.PulseResponse.pair_id==pair.id)

    pr_recs = q.all()

    # group records by (clamp mode, ind_freq, rec_delay), and then by pulse number
    sorted_recs = {}
    for rec in pr_recs:
        delay = set_recovery(rec.multi_patch_probe.recovery_delay)
        stim_key = rec.patch_clamp_recording.clamp_mode+', '+str(rec.multi_patch_probe.induction_frequency)+', '+str(delay)
#        stim_key = (rec.patch_clamp_recording.clamp_mode, rec.multi_patch_probe.induction_frequency, rec.multi_patch_probe.recovery_delay)
        sorted_recs.setdefault(stim_key, {})
        pn = rec.stim_pulse.pulse_number
        
        for param_key, value in zip(fit_params,
                                [rec.pulse_response_fit.fit_amp,
                                rec.pulse_response_fit.fit_latency,
                                rec.pulse_response_fit.fit_rise_time,
                                rec.pulse_response_fit.fit_decay_tau]):
            # create keys if not already present
            sorted_recs[stim_key].setdefault(param_key, {})
            sorted_recs[stim_key][param_key].setdefault(pn, [])
            # fill in the value
            sorted_recs[stim_key][param_key][pn].append(value)
    
    return sorted_recs
    
# One example: <pair 1492018873.073 8 5>
# expt = db.experiment_from_timestamp(1492018873.073)
# pair = expt.pairs['8','4']
# pr_dict = load_pair_pulse_responses(pair)

# Figuring out what will be in the data set
q = db.pair_query(synapse=True)
all_pairs = q.all()
print(len(all_pairs), 'pairs')

save_folder = 'data09_05_2019/'

for ii, pair in enumerate(all_pairs):
    print(ii, 'out of', len(all_pairs) )
    save_file_name = str(pair.experiment.acq_timestamp)+'_'+str(pair.pre_cell.ext_id)+'_'+str(pair.post_cell.ext_id)+'.csv'

    if save_file_name in os.listdir(save_folder):
        print('skipping', pair, 'already processed')
        continue

    pr_dict = load_pair_pulse_responses(pair)

#----------------------------------------------------------------
# --Turn the dictionary into format needed for an auto encoder---
#----------------------------------------------------------------

    rows_per_pair = 15 # maximum number of rows per neuron

    # check how many combinations will be possible
    possible_combinations = 1 #initializing value
    for stim_key in pr_dict:
        possible_combinations *= len(pr_dict[stim_key]['amp'][1]) # of all possible comination of stimuli

    if possible_combinations < rows_per_pair:
        print('warning', pair, 'only has', possible_combinations, 'possible combinations of stimuli and thus will be under represented')
        logging.debug('warning '+str(pair)+ ' only has '+ str(possible_combinations)+ ' possible combinations of stimuli and thus will be under represented')      
        rows_per_pair = possible_combinations

    giant_matrix = {}
    for pd_key in pair_description:
        giant_matrix.setdefault(pd_key, [])

    for (pd_key, des) in zip(pair_description, [pair.experiment.acq_timestamp,
                                                pair.pre_cell.ext_id,
                                                pair.post_cell.ext_id,
                                                pair.pre_cell.cre_type,
                                                pair.post_cell.cre_type,
                                                pair.pre_cell.target_layer,
                                                pair.post_cell.target_layer]):
        for jj in range(rows_per_pair):
            giant_matrix[pd_key] = giant_matrix[pd_key] + [des]

    bail_out_flag = None
    for stim_key in possible_stimuli:
        for param_key in fit_params:
            for pulse_num in range(1,13): 
                giant_matrix.setdefault((stim_key, param_key, pulse_num), [])

                if stim_key in pr_dict.keys():
                    try: 
                        values = pr_dict[stim_key][param_key][pulse_num] #assigning values to a variable for ease
                        vec = assign_wo_replacement(values, rows_per_pair)
                        giant_matrix[(stim_key, param_key, pulse_num)] = giant_matrix[(stim_key, param_key, pulse_num)] + vec
                    except:
                        print('something went wrong with', pair)
                        logging.debug('something went wrong with', pair)
                        bail_out_flag = 1
                else:
                    for jj in range(rows_per_pair):
                        giant_matrix[(stim_key, param_key, pulse_num)] = giant_matrix[(stim_key, param_key, pulse_num)] + [None]
                        
    if bail_out_flag is None:
        df=pd.DataFrame().from_dict(giant_matrix)
        print('matrix size', df.shape)
        df.to_csv(save_folder+save_file_name, sep = '#')
