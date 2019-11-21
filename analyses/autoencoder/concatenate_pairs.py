""" Takes the individual pair files with the desired extracted data for auto encoder 
and puts them in one file.  Specify the directory you want to grab data from. NOTE 
THAT THIS CODE MAY BE DATA SPECIFIC AS NOTED IN CODE.  PAY ATTENTION OS DIR AND SAVE 
STATEMENTS FOR CONSISTENT NAMING
"""
import os
import pandas as pd
import os
import sys

#this is here to deal with VSC stupid path stuff
os.chdir(sys.path[0])

data_folder_name = 'data_vc_50hz_11_10_2019'
dir='data/' + data_folder_name
master_df=pd.DataFrame()
files = os.listdir(dir)
for ii, file in enumerate(files):
    print(file)
    if ii % 10==0:
        print('done with', ii, 'out of', len(files))
    df=pd.read_csv(os.path.join(dir, file), sep ='#', dtype={'expt': 'str'}).drop(['Unnamed: 0'], axis=1) #drop the unnamed index column

    # remove rows that have all nans in the data columns
    #!!!!!!!!!!!NOTE THAT NEXT LINE IS SPEDIFIC FOR COLUMN NAME!!!!!!!!!!!!!!!!!!!!!!!!
    #df = df.dropna(subset=df.filter(regex="^\('ic", axis = 1).head().columns, how ='all', axis='rows') #drop rows that have all Nones in value columns
    
    # remove rows that have a nan in the data columns
    #!!!!!!!!!!!NOTE THAT NEXT LINE IS SPEDIFIC FOR COLUMN NAME!!!!!!!!!!!!!!!!!!!!!!!!
    df = df.dropna(subset=df.filter(regex="^\('ic", axis = 1).head().columns, how ='any', axis='rows') #drop rows that have all Nones in value columns

    #--------------------------------------------------------
    #------make a new column with unique id pair id ---------
    #-----this may not be necessary in future if ------------
    #-----if build into data creation------------------------
    #--------------------------------------------------------
    df = df.assign(pair_identifier = df['expt'] + '_' + df['pre_cell'].map(str)+ '_' + df['post_cell'].map(str))
    df = df.drop(columns = ['expt', 'pre_cell', 'post_cell'])

    # reorder columns for convenience
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    #-----------------------------------------------------
    
    # concatinate the df for this pair onto the master df
    master_df = pd.concat([master_df, df])
      
#master_df.astype({'expt': str}) #no longer present
print('total size', master_df.shape)
master_df.to_csv('data/ae_' + data_folder_name +'.csv', sep='#', index=False)
