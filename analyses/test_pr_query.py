from multipatch_analysis.database import database as db
import multipatch_analysis.connection_strength as cs 
from multipatch_analysis.database.database import TableGroup

uid=1527020350.517
pre_cell_id=6
post_cell_id=7

# query a pair
session = db.Session()
expt = db.experiment_from_timestamp(uid, session=session)
pair = expt.pairs[(pre_cell_id, post_cell_id)]

# get pulse ids from connection strength
raw_events = cs.get_amps(session, pair, clamp_mode='ic')
print(raw_events['id'])

# get pulse ids from pulse_responses table
pulse_stim_response_ids=[]
for pr in pair.pulse_responses:
    pulse_stim_response_ids.append(pr.stim_pulse_id)
print(pulse_stim_response_ids)
