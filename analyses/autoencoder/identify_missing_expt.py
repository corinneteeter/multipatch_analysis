import numpy as np
import pandas as pd
import os
import sys
from aisynphys.database import default_db as db
from lib import specify_excitation, specify_class


#this is here to deal with VSC stupid path stuff                                 
os.chdir(sys.path[0])

df = pd.read_csv('data/autoencoder_data_09_06_2019.csv', sep = '#', dtype={'expt': 'str'}, low_memory=False)
# import pdb; pdb.set_trace()                                                                                   
expts = df['expt'].unique()
for ts in expts:

    try:
	#convert to float do deal with string float conversions happening as csv files
        expt = db.experiment_from_timestamp(np.float(ts))
    except:
        print ('yikes', ts, 'is missing')
	# make the string the appropriate length
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

        q = db.query(db.Pipeline)
        q = q.filter(db.Pipeline.job_id == ts)
        pipeline = q.all()
        for p in pipeline:
            print(p.error)
