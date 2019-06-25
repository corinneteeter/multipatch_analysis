"""prototying better spike detector.  This code plots single pulses and
the derivatives to show detection.
"""
import multipatch_analysis.database as db
import multipatch_analysis.connection_strength as cs
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from weighted_error_lib import * 
from neuroanalysis.data import Trace, TraceList
import pdb
#import ipfx.feature_extractor as fe
import ipfx.spike_detector as sd
from scipy.optimize import curve_fit

#import pyqtgraph as pg #this is here to be able to use pyqt debugger 
#pg.dbg() #will open console if exception happens and you are running in interactive mode


# cells with currently poorly identified spikes 
cell_ids = [[1544582617.589, 1, 8, 5656957, 5654136, 5654045],  #this is a good text bc two fail but the others are sort of sad looking.
        #[1544582617.589, 1, 6, 5654136, 5654045], 
#        [1497417667.378, 5, 2, 7483977, 7483912],
        [1491942526.646, 8, 1, 6693052, 6693000],  #this presynaptic cell is sick.  Spiking is ambiguious, very interesting examples
        #[1521004040.059, 5, 6],
#        [1534293227.896, 7, 8, 7271530], #this one stuff spikes at end
#         [1540356446.981, 8, 6],
#         [1550101654.271, 1, 6], # these spike toward the end and are found correctly
        [1516233523.013, 6, 7],  #very interesting example: a voltage deflection happens very early but cant be seen in dvvdt due to to onset being to early.  Think about if there is a way to fix this.  Maybe and initial pulse window.  
        [1534297702.068, 7, 2]
            ]


s = db.Session()
for cell_id in cell_ids:
    pair = db.experiment_from_timestamp(cell_id[0]).pairs[cell_id[1], cell_id[2]]
    synapse = pair.synapse
    synapse_type = pair.connection_strength.synapse_type
    ic_pulse_ids = pair.avg_first_pulse_fit.ic_pulse_ids
    pulse_responses = pair.pulse_responses
    pre_syn_times = []
    pre_syn_voltages = []
    for pr in pulse_responses:
        if pr.stim_pulse_id in cell_id[3:]: # ic_pulse_ids:# 
            
            print(synapse, synapse_type, ', pass ex qc', pr.ex_qc_pass)
            print(synapse, synapse_type, ', pass in qc', pr.in_qc_pass)
            #Align individual responses
            start_time = pr.start_time  
            legacy_spike_time = pr.stim_pulse.spikes[0].max_dvdt_time      
            time_before_spike = 10.e-3  
            t0 = start_time-legacy_spike_time+time_before_spike
            pulse_start_time = pr.stim_pulse.onset_time - legacy_spike_time+time_before_spike
            pulse_end_time = pr.stim_pulse.onset_time + pr.stim_pulse.duration -legacy_spike_time+time_before_spike

            post_voltage = Trace(data=pr.data, t0= t0, sample_rate=db.default_sample_rate).time_slice(start=0, stop=None) 
            pre_voltage = Trace(data=pr.stim_pulse.data, t0 = t0, sample_rate=db.default_sample_rate).time_slice(start=0, stop=None)

#            pre_inj_current = Trace(data=pr.stim_pulse.data, t0= start_time-legacy_spike_time+time_before_spike, sample_rate=db.default_sample_rate).time_slice(start=0, stop=None)
            pre_syn_times.append(pre_voltage.time_values)
            pre_syn_voltages.append(pre_voltage.data)
            time = pre_syn_times[-1]
            dvdt = np.diff(pre_syn_voltages[-1])
            dvdt_time = time[1:]
            dvvdt = np.diff(dvdt)
            dvvdt_time = time[2:]

            #-----------------------------------------------------
            #----------spike detection----------------------------
            #-----------------------------------------------------
            # look for max of dvvdt in region during pulse and eliminating region where current injection artifact 
            pulse_window =np.where((dvvdt_time > (pulse_start_time + .0003)) & (dvvdt_time < (pulse_end_time - .0002)))[0]
            max_dvvdt = np.max(dvvdt[pulse_window]) #this 
                
            max_index = np.where(dvvdt==max_dvvdt)
            if len(max_index) > 1:
                raise Exception('should only be one max')
            else: 
                max_index=max_index[0][0]

            if max_dvvdt > .5e-3:
                spike_index = max_index + 2 #converting to voltage space  

            else:
                spike_index = None
            # Note that there is slop in the time indexing due to dvdt and dvvdt but it would be surprising if it ever mattered
            # --now look in window where pulse is terminated to see if it spikes (hard to differentiate)----
            # note that if it does not spike the trace will be similar to an exponential starting at the 
            # min of dv/dt in the pulse termination window.---
            pulse_end_window = np.where((dvvdt_time > (pulse_end_time - .0002)) & (dvvdt_time < (pulse_end_time + .0005)))[0] 
            #find location of minimum dv/dt
            min_dvdt = np.min(dvdt[pulse_end_window-1]) #-1 because now using dvdt
            min_index = np.where(dvdt==min_dvdt) #note that this index is relative to whole trace not just pulse termination window      
            
            if len(min_index) > 1:
                raise Exception('should only be one min')
            else: 
                min_index=min_index[0][0]

            def derivative(t, tau, Vo): 
                """function describing the deriviative of the voltage.  If there
                is no spike one would expect this to fall off as the RC of the cell. """
                return -(Vo/tau)*np.exp(-t/tau)

            ttofit=time[(min_index+1):] #note the plus one because time trace of derivative needs to be one shorter
            dvtofit=dvdt[min_index:]
            popt, pcov = curve_fit(derivative, ttofit, dvtofit, maxfev=10000)

            fit = derivative(ttofit, *popt)
            #TODO need to catch a runtime error here "RuntimeError: Optimal parameters not found: Number of calls to function has reached maxfev = 600.""

            mse = (np.sum((dvtofit-fit)**2))/len(fit)*1e10 #mean squared error
            if (mse > 500.) &  (spike_index == None):
                print("Unidentified spike found in pulse termination window") 
                # find min dvvdt in pulse termination window.
                min_dvvdt = np.min(dvvdt[pulse_end_window]) #-1 because now using dvdt
                min_index = np.where(dvvdt==min_dvvdt) #note that this index is relative to whole trace not just pulse termination window      
                if len(min_index) > 1:
                    raise Exception('should only be one min')
                else: 
                    min_index=min_index[0][0]
                
                spike_index = min_index + 2 #converting to voltage space

                plt.figure()
                plt.plot(ttofit, dvtofit, 'r')
                plt.plot(ttofit, fit, 'k--', label=('mse: %f' % (mse)))
                plt.legend()
            
            #------------------------------------------------------

            # plotting
            plt.figure(figsize =  (8, 5))
            
            ax1 = plt.subplot(1,1, 1)
            ln1=ax1.plot(time*1.e3, pre_syn_voltages[-1]*1.e3, 'b', lw=2, label="data")
            ax1.tick_params(axis='y', colors='b')
            ax1.set_ylabel('voltage (mV)', color='b') 
            ax1.axvline(10, linestyle= '-', color = 'k',lw=.5, alpha=.5) #light line denoting previous alignment
            ax1.axvspan(dvvdt_time[pulse_end_window[0]]*1.e3, dvvdt_time[pulse_end_window[-1]]*1.e3, facecolor = 'k', alpha=.2)
            ax1.axvspan(dvvdt_time[pulse_window[0]]*1.e3, dvvdt_time[pulse_window[-1]]*1.e3, facecolor = 'm', alpha=.1)

            ax1.set_xlabel('time (ms)')

            ax2=ax1.twinx()
            ln2=ax2.plot(dvdt_time*1.e3, dvdt, 'r', label="dv/dt")
            ax2.set_yticks([])
            
            ax3=ax1.twinx()
            ln3=ax3.plot(dvvdt_time*1.e3, dvvdt, 'g', label="dvv/dt")

            # add legend
            lns = ln1+ln2+ln3
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs)
            ax1.get_xlim()
            ax1.set_xlim((5., ax1.get_xlim()[1]))


            if spike_index:
                ax1.axvline(time[spike_index]*1.e3, linestyle= '--', color = 'k')
            plt.title("%.3f, %i, %i, %i" % (cell_id[0], cell_id[1], cell_id[2], pr.stim_pulse_id))

            relative_spike_time = time[spike_index]
            new_DB_spike_time = start_time + relative_spike_time - t0
            plt.show()
            
        #spikes = sd.detect_putative_spikes(pre_syn_voltages[-1], time, start=None, end=None, filter=None, dv_cutoff=20.)           



#    out=s.query(db.Pair, db.AvgFirstPulseFit).filter(db.Pair.id == row['pair_id']).filter(db.AvgFirstPulseFit.pair_id == row['pair_id']).all()
#if len(out) > 1:
#    raise Exception('there should not be more than one pair returned')
#if len(out) < 1:
#    print('skipping pair_id', row['pair_id'])
#    continue
#    pair= out[0][0]
#    pulse_ids=out[0][1].ic_pulse_ids
