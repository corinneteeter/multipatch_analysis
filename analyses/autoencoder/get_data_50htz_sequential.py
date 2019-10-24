"""
Creates a file to run though an autoencoder
The columns 'expt','pre_cell', 'post_cell' identify the pair.
The columns 'pre_cre','post_cre','pre_layer', 'post_layer' are descriptive features of the pair.

The predecessor to this code is get_data_randomize.py where the parameters were randomized within a column within a pair. 
"""

from aisynphys.database import default_db as db
import numpy as np
from lib import set_recovery, specify_excitation
import pandas as pd
import random
import logging
import os
import sys

def load_pair_pulse_responses(pair):
    """For a given *pair* return a dictionary of the individual pulse response fit parameters.
    The output dict hierarchy looks like stimulus_type >> fit parameter >> pulse number.  Note that 
    the recoveries can vary in timing within a stimulus frequency.  Thus recovery are specifically
    set to a number if they are within a range.  See *set_recovery* for more information.
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
        stim_key = rec.patch_clamp_recording.clamp_mode+', '+str(rec.multi_patch_probe.induction_frequency)
#        stim_key = (rec.patch_clamp_recording.clamp_mode, rec.multi_patch_probe.induction_frequency, rec.multi_patch_probe.recovery_delay)
        sorted_recs.setdefault(stim_key, {})
        pn = rec.stim_pulse.pulse_number
        
        # note this has to match up with the *fit_param_keys* above
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

#this is here to deal with VSC stupid path stuff
os.chdir(sys.path[0])

date='10_22_2019'

LOG_FILENAME = date + '.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)

pair_description = ['species','expt','pre_cell', 'post_cell','pre_cre','post_cre','pre_layer', 'post_layer', 'pre_ex', 'post_ex', 'stp_induction_50hz']
fit_params = ['amp', 'latency', 'rise_time', 'decay_tau']

# One example: <pair 1492018873.073 8 5>
# expt = db.experiment_from_timestamp(1492018873.073)
# pair = expt.pairs['8','4']
# pr_dict = load_pair_pulse_responses(pair)


# Useful for debugging specific pair issues
#expt = db.experiment_from_timestamp(1554236360.106)    
#pair = expt.pairs['1','3']
#all_pairs = [pair]

# Figuring out what will be in the data set
q = db.pair_query(synapse=True)
all_pairs = q.all()
print(len(all_pairs), 'pairs')

save_folder = 'data' + date
if not os.path.exists(save_folder):
    os.makedirs(save_folder)



for ii, pair in enumerate(all_pairs):
    # the following pair does not have an entry in the dynamics table
    if pair.experiment.acq_timestamp == 1552517188.758:
        continue

    print(ii, 'out of', len(all_pairs) )
    save_file_name = str(pair.experiment.acq_timestamp)+'_'+str(pair.pre_cell.ext_id)+'_'+str(pair.post_cell.ext_id)+'.csv'

    #skip a pair if it is already in the output directory so that pairs are not reprocessed on restart
    if save_file_name in os.listdir(save_folder):
        print('skipping', pair, 'already processed')
        continue

    # pair dictionary with a stimulus_type >> fit parameter >> pulse number hierarchy
    pr_dict = load_pair_pulse_responses(pair)

#----------------------------------------------------------------
# --Turn the dictionary into format needed for an auto encoder---
#----------------------------------------------------------------

    # initiate a dictionary 
    giant_matrix = {}
    # initiate keys/columns for the descrptive attributes such as species etc.
    for pd_key in pair_description:
        giant_matrix.setdefault(pd_key, [])

    bail_out_flag = None  # if gets set to true in the code an error message is logged and 
                          # the output matrix for the pair is not saved to a csv
    
    first_time_through = True
    has_at_least_one_value = False
    for stim_key in pr_dict.keys():
        if stim_key != 'ic, 50.0':
            # the stimulus key is not one we are interested in, skip it
            continue

        else:
            has_at_least_one_value = True  # at least one relevant stimulus was found
            if first_time_through:
                num_of_rows = len(pr_dict[stim_key][fit_params[0]][1])

                # specify whether the cre lines are excitatory or inhibitory
                pre_ex = specify_excitation(pair.pre_cell.cre_type)
                post_ex = specify_excitation(pair.post_cell.cre_type)

                #create the number of description rows needed
                for (pd_key, db_pd) in zip(pair_description, [pair.experiment.slice.species,
                                            pair.experiment.acq_timestamp,
                                            pair.pre_cell.ext_id,
                                            pair.post_cell.ext_id,
                                            pair.pre_cell.cre_type,
                                            pair.post_cell.cre_type,
                                            pair.pre_cell.target_layer,
                                            pair.post_cell.target_layer,
                                            pre_ex,
                                            post_ex, 
                                            pair.dynamics.stp_induction_50hz]):
                    for jj in range(num_of_rows):
                        giant_matrix[pd_key] = giant_matrix[pd_key] + [db_pd]
                first_time_through = False  


            for param_key in fit_params:
                for pulse_num in range(1,9):  #NOTE: not using recovery pulses 
                    giant_matrix.setdefault((stim_key, param_key, pulse_num), [])
                    if len(pr_dict[stim_key][param_key][pulse_num]) != num_of_rows:
                        print(pair, 'has inconsistent vector lengths; should be', num_of_rows, 'but ', stim_key, param_key, pulse_num, 'report', len(pr_dict[stim_key][param_key][pulse_num]))
                        logging.error(str(pair)+' has inconsistent vector lengths; should be ' +str(num_of_rows), 
                                        'but '+ stim_key+' '+param_key+' '+ pulse_num+ 'report'+ str(len(pr_dict[stim_key][param_key][pulse_num])))
                        bail_out_flag = 1                                            
                    try: 
                        values = pr_dict[stim_key][param_key][pulse_num] #assigning values to a variable for ease
                        giant_matrix[(stim_key, param_key, pulse_num)] = giant_matrix[(stim_key, param_key, pulse_num)] + values
                    except:
                        print('something went wrong with', pair)
                        logging.debug('something went wrong with'+ str(pair)+': couldnt load values')
                        bail_out_flag = 1

                # append a none if the pair does not have this possible stimulus.
                # else:
                #     for jj in range(num_of_rows):
                #        giant_matrix[(stim_key, param_key, pulse_num)] = giant_matrix[(stim_key, param_key, pulse_num)] + [None]

    # if nothing went wrong save the data out in the matrix form
    if bail_out_flag is None: 
        if has_at_least_one_value is True:
            df=pd.DataFrame().from_dict(giant_matrix)
            print('matrix size', df.shape)
            if df.shape[0] == 0:
                logging.info('not saving pair'+ str(pair)+ ': empty df')
            elif df.drop(pair_description, axis =1).isnull().all().all():
                logging.info('not saving pair'+ str(pair)+ ': empty all data is None')
            else:
                df.to_csv(os.path.join(save_folder, save_file_name), sep = '#')
        else:
            logging.info('not saving pair ' +str(pair)+ ': no 50hz stimuli')
