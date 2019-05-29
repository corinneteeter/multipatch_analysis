"""Calculates how much better or worse current clamp fitting paradigm did
in terms of nrsme.
"""

import multipatch_analysis.database as db
import multipatch_analysis.connection_strength as cs
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from weighted_error_lib import *
import pdb
from neuroanalysis.data import Trace, TraceList


def convert_saved_str_to_array(the_string):
    """Convert 'string arrays' into numpy arrays. String arrays were created by saving 
    the database queries in save_fit_varation_tables.py.  Form looks like '[20. 20. 0.\n 1.]'.
    """
    return np.fromstring(the_string.replace('\n', '').replace('[', '').replace(']',''), sep=' ')

def plot_individual_responses(pulse_ids, pair, (ax_pre, ax_post)):
    """
    using matplotlib
    """
    s = db.Session()
    for pulse_id in pulse_ids:
        pulse_response = s.query(db.PulseResponse).join(db.Pair).filter(
            db.PulseResponse.stim_pulse_id == pulse_id).filter(
                db.Pair.id == pair.id).all()
        
        if len(pulse_response) != 1:
            raise Exception('There should only be one response per pulse_id')
        
        pulse_response= pulse_response[0]  #grap the pulse reponse out of the list

        #Align individual responses
        start_time = pulse_response.start_time  
        spike_time = pulse_response.stim_pulse.spikes[0].max_dvdt_time      
        time_before_spike = 10.e-3  
        post_trace = Trace(data=pulse_response.data, t0= start_time-spike_time+time_before_spike, sample_rate=db.default_sample_rate).time_slice(start=0, stop=None) #start of the data is the spike time
        pre_trace =Trace(data=pulse_response.stim_pulse.data, t0= start_time-spike_time+time_before_spike, sample_rate=db.default_sample_rate).time_slice(start=0, stop=None)

        ax_pre.plot(pre_trace.time_values*1.e3, pre_trace.data*1.e3, 'k', alpha=.8)
        ax_post.plot(post_trace.time_values*1.e3, post_trace.data*1.e3, 'k', alpha=.8)



df=pd.read_csv('fits_all.csv')

fit_df=df[['pair_id',
           'ic_weight', 'ic_weight.1', 'ic_weight.2', 'ic_weight.3',
           'ic_avg_psp_fit', 'ic_avg_psp_fit.1', 'ic_avg_psp_fit.2', 'ic_avg_psp_fit.3',
           'ic_avg_psp_data', 'ic_avg_psp_data.1', 'ic_avg_psp_data.2', 'ic_avg_psp_data.3',
           'ic_amp', 'ic_amp.1', 'ic_amp.2', 'ic_amp.3']]
fit_df.columns = ['pair_id',
                  'w1', 'w2', 'w3', 'w4',
                  'fit1', 'fit2', 'fit3', 'fit4',
                  'data1', 'data2', 'data3', 'data4',
                  'amp1', 'amp2', 'amp3', 'amp4']

# convert all string arrays to numpy arrays
for name in fit_df.columns[1:13]:
    fit_df[name]=fit_df[name].apply(convert_saved_str_to_array)
    

#fit1: fixed window weighted to zero intended to weight out cross talk 
#fit2: no crosstalk removal
#fit3: dynamic cross talk window where region where cross talk exists in any individual sweep is zeroed
#fit4: 3 but upweighted baseline


# can't use nrmse to compare fits accross paradigms because it uses the weighting in 
# paradigm to calcute rmse and std


universal_nrmse = []
# calculated a universal nrmse
for index, row in fit_df.iterrows():
    
    nrmse_list =[]
    # use same weighting paradigm for all fitting using paradigm  
    weight = np.ones(len(row.w3))
#    weight[np.where(row.w3==0)] = 0
    
    for data, fit in (['data1', 'fit1'], ['data2', 'fit2'], ['data3', 'fit3'], ['data4', 'fit4']) : # fit 1 though 4
        wrmse, wstd, wnrmse = weighted_nrmse(row[data], row[fit], weight)
        nrmse_list.append(wrmse)
    universal_nrmse.append(nrmse_list)

fit_df=pd.concat([fit_df, pd.DataFrame.from_records(universal_nrmse, columns = ['nrmse1', 'nrmse2', 'nrmse3', 'nrmse4'])], axis=1)


#labels= ['1', '2', '3', '4']
#df = pd.DataFrame.from_records(universal_nrmse, columns = labels)

fit_df['1 to 2']=(fit_df['nrmse1'] - fit_df['nrmse2'])/fit_df['nrmse1']
fit_df['1 to 3']=(fit_df['nrmse1'] - fit_df['nrmse3'])/fit_df['nrmse1']
fit_df['1 to 4']=(fit_df['nrmse1'] - fit_df['nrmse4'])/fit_df['nrmse1']
fit_df['lowest_nrmse']= fit_df[['nrmse1', 'nrmse2', 'nrmse3', 'nrmse4']].min(axis=1)


# sns.distplot(fit_df['nrmse1'], rug=True, label='1')
# sns.distplot(fit_df['nrmse2'], rug=True, label='2')
# sns.distplot(fit_df['nrmse3'], rug=True, label='3')
# sns.distplot(fit_df['nrmse4'], rug=True, label='4')
# sns.distplot(fit_df['lowest_nrmse'], rug=True, label='best')
# plt.legend()
# plt.show()


print('decrease in nrmse for changing weighting from 1 to 2: mean', fit_df['1 to 2'].mean(), 'median', fit_df['1 to 2'].median())
print('decrease in nrmse for changing weighting from 1 to 3: mean', fit_df['1 to 3'].mean(), 'median', fit_df['1 to 3'].median())
print('decrease in nrmse for changing weighting from 1 to 4: mean', fit_df['1 to 4'].mean(), 'median', fit_df['1 to 4'].median())

#Now we must choose which universal nrmse/fit is the best.
fit_df[['abs_amp1', 'abs_amp2', 'abs_amp3', 'abs_amp4']]= fit_df[['amp1', 'amp2', 'amp3', 'amp4']].abs()
fit_df['min_abs_amp']=fit_df[['abs_amp1', 'abs_amp2', 'abs_amp3', 'abs_amp4']].min(axis=1)

#sns.distplot(fit_df['min_abs_amp'],label='min amp')
#plt.legend()
#plt.show()

value = 1 * fit_df['min_abs_amp'].mean()
small_amp_df=fit_df[fit_df['min_abs_amp'] < .1 * value]
small_amp_df = small_amp_df.sort_values(by=['min_abs_amp'])
print 'looking at', len(small_amp_df), 'abs amplitudes smaller than', value * 1.e3, 'mV'

i=0

for index, row in small_amp_df.iterrows():
    i+=1

    plt.figure(figsize=(16,11))
    ax1=plt.subplot(411)
    ax2=plt.subplot(4,1,4)
    ax3 = plt.subplot2grid((4, 1), (1, 0), rowspan=2)

    index_to_time = 1.e3/db.default_sample_rate
    cross_talk_region = np.where(row.w3==0)

    s = db.Session()
    out=s.query(db.Pair, db.AvgFirstPulseFit).filter(db.Pair.id == row['pair_id']).filter(db.AvgFirstPulseFit.pair_id == row['pair_id']).all()
    if len(out) > 1:
        raise Exception('there should not be more than one pair returned')

    pair= out[0][0]
    pulse_ids=out[0][1].ic_pulse_ids



    plot_individual_responses(pulse_ids, pair, (ax1,ax2))
    ax1.annotate('pre-synaptic spikes', xy=(.75, .9), xycoords = 'axes fraction', fontsize=12)
    ax1.axvspan(cross_talk_region[0][0]*index_to_time, cross_talk_region[0][-1]*index_to_time, facecolor = 'k', alpha=.2)
    ax2.annotate('individual responses', xy=(.75, .9), xycoords = 'axes fraction', fontsize=12)
    ax2.axvspan(cross_talk_region[0][0]*index_to_time, cross_talk_region[0][-1]*index_to_time, facecolor = 'k', alpha=.2)


    time=np.arange(len(row['data1']))*index_to_time #ms
    ax3.plot(time, row['data1']*1.e3, lw=2, label='average response')
    ax3.plot(time, row['fit1']*1.e3, label=('fit 1, amp %.2f, nrmse %.03f' % ((row['amp1']*1.e3), row['nrmse1'])))
    ax3.plot(time, row['fit2']*1.e3, label=('fit 2, amp %.2f, nrmse %.03f' % ((row['amp2']*1.e3), row['nrmse2'])))
    ax3.plot(time, row['fit3']*1.e3, label=('fit 3, amp %.2f, nrmse %.03f' % ((row['amp3']*1.e3), row['nrmse3'])))
    ax3.plot(time, row['fit4']*1.e3, label=('fit 4, amp %.2f, nrmse %.03f' % ((row['amp4']*1.e3), row['nrmse4'])))

    #pdb.set_trace()
    ax3.axvspan(cross_talk_region[0][0]*index_to_time, cross_talk_region[0][-1]*index_to_time, facecolor = 'k', alpha=.2)

    ax3.set_ylabel('voltage (mV)')
    ax3.set_xlabel('time (ms)')

    ax3.legend()

    ax3.annotate(('%i out of %i fits with at \nleast one fit amp < %.2f, \n    %.3f, %i to %i, \n    %s to %s' % (i, 
                len(small_amp_df), 
                (value * 1.e3), 
                pair.experiment.acq_timestamp, 
                pair.pre_cell.ext_id,
                pair.post_cell.ext_id,
                pair.pre_cell.cre_type,
                pair.post_cell.cre_type)),
                xy=(.1, .8), xycoords = 'axes fraction', va='center', fontsize=12)


    plt.tight_layout()
    plt.show()
    s.close()
    plt.close()
