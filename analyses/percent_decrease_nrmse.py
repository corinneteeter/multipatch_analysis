"""Calculates how much better or worse fitting paradigm did
in terms of nrsme.
"""

import multipatch_analysis.database as db
import multipatch_analysis.connection_strength as cs
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from weighted_error_lib import *



session=db.Session()
out = session.query(db.AvgFirstPulseFit, db.AvgFirstPulseFit2, 
                    db.AvgFirstPulseFit3, db.AvgFirstPulseFit4).join(
                    db.Pair).filter(db.Pair.synapse == True
                    ).filter(db.AvgFirstPulseFit.pair_id == db.AvgFirstPulseFit2.pair_id
                    ).filter(db.AvgFirstPulseFit.pair_id == db.AvgFirstPulseFit3.pair_id
                    ).filter(db.AvgFirstPulseFit.pair_id == db.AvgFirstPulseFit4.pair_id
                    ).all()

#fit1: fixed window weighted to zero intended to weight out cross talk 
#fit2: no crosstalk removal
#fit3: dynamic cross talk window where region where cross talk exists in any individual sweep is zeroed
#fit4: 3 but upweighted baseline
#fit5: did not work

# can't use nrmse to compare fits accross paradigms because it uses the weighting in 
# paradigm to calcute rmse and std


norm_nrmse = []
# calculated a universal nrmse
for fits in out:
    nrmse_list =[]
    # use same weighting paradigm for all fitting using paradigm  
    weight = np.ones(len(fits[2].ic_weight))
    weight[np.where(fits[2].ic_weight==0)] = 0
    for fit in fits: # fit 1 though 5
        wrmse, wstd, wnrmse = weighted_nrmse(fit.ic_avg_psp_data, fit.ic_avg_psp_fit, weight)
        nrmse_list.append(wrmse)

    norm_nrmse.append(nrmse_list)

labels= ['1', '2', '3', '4']
df = pd.DataFrame.from_records(norm_nrmse, columns = labels)

df['1 to 2']=(df['1'] - df['2'])/df['1']
df['1 to 3']=(df['1'] - df['3'])/df['1']
df['1 to 4']=(df['1'] - df['4'])/df['1']
df['lowest_nrmse']= df[['1', '2', '3', '4']].min(axis=1)

# sns.distplot(df['1'], rug=True, label='1')
# sns.distplot(df['2'], rug=True, label='2')
# sns.distplot(df['3'], rug=True, label='3')
# sns.distplot(df['4'], rug=True, label='4')
# sns.distplot(df['lowest_nrmse'], rug=True, label='best')
# plt.legend()
# plt.show()


print('decrease in nrmse for changing weighting from 1 to 2: mean', df['1 to 2'].mean(), 'median', df['1 to 2'].median())
print('decrease in nrmse for changing weighting from 1 to 3: mean', df['1 to 3'].mean(), 'median', df['1 to 3'].median())
print('decrease in nrmse for changing weighting from 1 to 4: mean', df['1 to 4'].mean(), 'median', df['1 to 4'].median())