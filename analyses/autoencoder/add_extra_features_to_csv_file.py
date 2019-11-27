import numpy as np
import pandas as pd
import os
import sys
from aisynphys.database import default_db as db
from lib import specify_excitation, specify_class

"""THIS CODE IS BEING USED FOR SEVERAL DIFFERENT FILES THAT HAVE DIFFERENT MISSING ATTRIBUTES.  MAKE SURE FILE NAMES ARE CORRECT
AND DONT FORGET TO CHECK HOW THE COLUMNS ARE BEING ADJUSTED ~L51 (SHIFTING HAS TO MATCH HOW MANY COLUMNS ARE BEING ADDED)
"""


#this is here to deal with VSC stupid path stuff
os.chdir(sys.path[0])

#df = pd.read_csv('data/ae_data_50hz_sequential_10_24_2019.csv', sep = '#', dtype={'expt': 'str'}, low_memory=False)
df = pd.read_csv('data/autoencoder_data_09_06_2019.csv', sep = '#', dtype={'expt': 'str'}, low_memory=False) 

#fix issue with string/float experimental ids
def fix_expt_id(ts):
    """
    input
    -----
    ts: string
    """
    if len(ts) > 14:
        ts_np = np.float(ts)
        ts_np = np.around(ts_np, decimals = 3)
        ts = str(ts_np)
        print(ts, 'after string fix')
    if len(ts) < 14: #add zeros
        n = 14 - len(ts)
        for i in range(n):
            ts = ts+'0'
        print(ts, 'after string fix')
    return ts

df['expt_fixed'] = ''
df['expt_fixed'] = df['expt'].apply(fix_expt_id)
df = df.assign(pair = df['expt_fixed'] + '_' + df['pre_cell'].map(str)+ '_' + df['post_cell'].map(str))
df = df.drop(columns = ['expt', 'expt_fixed', 'pre_cell', 'post_cell'])
df['stp_induction_50hz'] = np.nan
df['pre_ex'] = ''
df['post_ex'] = ''
df['pre_class'] = ''
df['post_class'] = ''
df['species'] = ''

# reorder columns for convenience.  Note that this is dependent on how many columns have been added.
cols = df.columns.tolist()
cols = cols[-7:] + cols[:-7]  #make sure this number is dependent on how many columns are being shifted
df = df[cols]
pairs = df['pair'].unique()

for id in pairs:
    ts, pre, post = id.split('_')
    print(ts, pre, post)
 
    try:
        #convert to float do deal with string float conversions happening as csv files
        expt = db.experiment_from_timestamp(np.float(ts))
        pair = expt.pairs[pre,post]
        species =  expt.slice.species

        pre_ex = specify_excitation(pair.pre_cell.cre_type)
        post_ex = specify_excitation(pair.pre_cell.cre_type)

        pre_class = specify_class(species,  pair.pre_cell.cre_type, pair.pre_cell.target_layer)
        post_class = specify_class(species,  pair.post_cell.cre_type, pair.post_cell.target_layer)

        try:
            stp_induction_50hz = pair.dynamics.stp_induction_50hz
        except:
            stp_induction_50hz = np.nan
    except:
        print ('yikes', ts, 'is missing')
        species = 'unknown'
        pre_ex = 'U'
        post_ex = 'U'
        pre_class = 'unknown'
        post_class ='unknown'
        stp_induction_50hz = np.nan


    df['species'][df['pair']==id] = species
    df['pre_ex'][df['pair']==id] = pre_ex
    df['post_ex'][df['pair']==id] = post_ex
    df['pre_class'][df['pair']==id] = pre_class
    df['post_class'][df['pair']==id] = post_class
    df['stp_induction_50hz'][df['pair']==id] = stp_induction_50hz



#df.to_csv('data/ae_data_50hz_sequential_10_24_2019UPDATE.csv', sep = '#', index=False)
df.to_csv('data/ae_data_09_06_2019UPDATE.csv', sep = '#', index=False)
