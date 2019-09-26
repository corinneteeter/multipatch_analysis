import os
import pandas as pd

dir='data09_05_2019'
master_df=pd.DataFrame()
files = os.listdir(dir)
for ii, file in enumerate(files):
    if ii % 10==0:
        print('done with', ii, 'out of', len(files))
    df=pd.read_csv(os.path.join(dir, file), sep ='#').drop(['Unnamed: 0'], axis=1)
#    df=pd.read_csv(os.path.join(dir, file), sep ='#')
    master_df = pd.concat([master_df, df])
    
master_df.astype({'expt': str})
print('total size', master_df.shape)
master_df.to_csv('autoencoder_data_09_06_2019.csv', sep='#', index=False)
