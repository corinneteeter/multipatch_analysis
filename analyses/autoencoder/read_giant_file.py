import pandas as pd
import time
import os

start_time = time.time()
df=pd.read_csv('autoencoder_data_09_06_2019.csv', sep='#')
df.astype({'expt': str}) # change the type to a string so can see experiment id
print((time.time() - start_time), 'seconds to load') 
mask = ~df.isnull()

import pdb; pdb.set_trace()
print (df.shape)
