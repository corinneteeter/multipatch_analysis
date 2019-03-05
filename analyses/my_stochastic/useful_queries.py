# Some useful queries
from neuroanalysis.data import Trace, TraceList
from multipatch_analysis.database import database as db
import multipatch_analysis.connection_strength as cs 
from multipatch_analysis.database.database import TableGroup

#do the query
session = db.Session() #create session
pre_cell = db.aliased(db.Cell)
post_cell = db.aliased(db.Cell)

# get info for a specific experiment but all probed
# expt_stuff = session.query(db.Pair, db.Experiment.acq_timestamp, pre_cell.ext_id, post_cell.ext_id,pre_cell.cre_type, post_cell.cre_type)\
#                     .join(db.Experiment)\
#                     .join(pre_cell, db.Pair.pre_cell_id==pre_cell.id)\
#                     .join(post_cell, db.Pair.post_cell_id==post_cell.id).filter(db.Experiment.acq_timestamp==expt_id).all()

# get info for particular connected cre lines
expt_stuff = session.query(db.Pair, db.Experiment.acq_timestamp, pre_cell.ext_id, post_cell.ext_id, pre_cell.cre_type, post_cell.cre_type)\
                    .join(db.Experiment)\
                    .join(pre_cell, db.Pair.pre_cell_id==pre_cell.id)\
                    .join(post_cell, db.Pair.post_cell_id==post_cell.id)\
                    .filter(db.Pair.synapse==True)\
                    .filter(pre_cell.cre_type=='tlx3')\
                    .filter(post_cell.cre_type=='sst')\
                    .all()

for exp in expt_stuff:
    print ('(%.3f, %i, %i),' % (exp[1], exp[2], exp[3]))
print len(expt_stuff)