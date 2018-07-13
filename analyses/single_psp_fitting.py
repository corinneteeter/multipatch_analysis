'''This code is meant to explore fitting single PSPs'''

from __future__ import print_function, division
from multipatch_analysis.experiment_list import cached_experiments
from multipatch_analysis.connection_detection import MultiPatchSyncRecAnalyzer, MultiPatchExperimentAnalyzer
from multipatch_analysis.synaptic_dynamics import DynamicsAnalyzer
import pickle
import numpy as np
import matplotlib.pyplot as plt
from neuroanalysis.baseline import float_mode
from neuroanalysis.fitting import fit_psp
from neuroanalysis.data import TraceList


desired_organism = 'human'

def plot_trace_views(num_subplot, trace_view, title=None):
    t, v = make_plotting_arrays(trace_view)
    plt.subplot(num_subplot[0], 1, num_subplot[1])
    if title:
        plt.title(title)
    plt.plot(t,v)

def make_plotting_arrays(trace_view):
    voltage_trace = trace_view.data
    dt = trace_view.dt
    time_trace = np.arange(0, len(voltage_trace)*dt+dt, dt) #create time values a bit longer so not to have fill issues
    time_trace = time_trace[0:len(voltage_trace)] # resize data_trace incase it is too long
    return time_trace, voltage_trace

def calculate_base_line(baseline_trace_view):
    '''return the average (voltage) for the data in a trace_view object'''
    return np.average(baseline.baseline_trace_view)


expts = cached_experiments()
selected_expt_list = expts.select(organism=desired_organism)

uid_skip=[]
for connection in selected_expt_list.connection_summary():        
    cells = connection['cells'] #(pre, post) synaptic *Cell* objects
    expt = connection['expt'] #*Experiment* object
    print (expt.uid)

    # cycle though connections in experiment; expt.connections is a list of tuple pairs
    for (pre_cell_id, post_cell_id) in expt.connections:
    
        # convert cell ID to headstage ID
        pre_electrode_id = pre_cell_id - 1
        post_electrode_id = post_cell_id - 1

        # how much padding to include when extracting events
        pre_pad = 10e-3, 
        post_pad = 50e-3
        
        # loop though sweeps in recording and pull out the ones you want
        no_skipping_sweep_count=0 #hack to avoid plotting when no sweeps in the experiment are recorded
        first_pulse_list=[]
        for srec in expt.data.contents:
            sweep_id=srec._sweep_id
            print ('sweep_id', sweep_id)
            if pre_electrode_id not in srec.devices or post_electrode_id not in srec.devices:
                print("Skipping %s electrode ids %d, %d; no skipping because pre or post synaptic electrode id is not in srec.devices" % (expt.uid, pre_electrode_id, post_electrode_id))
                continue
            pre_rec = srec[pre_electrode_id]
            post_rec = srec[post_electrode_id]
            if post_rec.clamp_mode != 'ic':
                print("Skipping %s electrode ids %d, %d; rec.clamp_mode != current clamp" % (expt.uid, pre_electrode_id, post_electrode_id))
                continue

            analyzer = MultiPatchSyncRecAnalyzer.get(srec)
            
            # get information about the spikes and make sure there is a spike on the first pulse
            spike_data = analyzer.get_spike_responses(pre_rec, post_rec, pre_pad=pre_pad, align_to='spike')
            if 0 in [pulse['pulse_n'] for pulse in spike_data]: # confirm pulse number starts at one, not zero
                raise Exception("Skipping %s electrode ids %d, %d; ; should have not have zero" % (expt.uid, pre_electrode_id, post_electrode_id)) 
            if 1 not in [pulse['pulse_n'] for pulse in spike_data]: # skip this sweep if not a spike on the first pulse
                print("Skipping %s electrode ids %d, %d; no spike on first pulse" % (expt.uid, pre_electrode_id, post_electrode_id))                
                continue

            else: # selects the first pulse data and excludes data that doesn't pass the excitatory qc
                for pulse in spike_data:
                    if pulse['pulse_n'] == 1 and pulse['ex_qc_pass'] == True:
                        first_pulse_list.append(pulse)
                                            
        if len(first_pulse_list)>0:    
        
            # get average trace baseline subtracted average for average fit
            bsub_trace_list=[]
            for sweep in first_pulse_list:
                sweep_trace=sweep['response']
                sweep_baseline_float_mode=float_mode(sweep['baseline'].data)
                bsub_trace_list.append(sweep_trace.copy(data=sweep_trace.data-sweep_baseline_float_mode)) #Trace object with baseline subtracted data via float mode method 

            average=TraceList(bsub_trace_list).mean()
            plt.figure()
            plt.plot(average.data)
            plt.show()
            
#             
##            # plotting to see what the spikes look like
##            plt.figure("aligned_spikes")
##            plt.subplot(3,1,1)
##            plt.plot(first_pulse_dict['command'].data)
##            plt.title('command')
##            plt.subplot(3,1,2)
##            plt.plot(first_pulse_dict['pre_rec'].data)
##            plt.title('pre synaptic response')
##            plt.subplot(3,1,3)
##            plt.plot(first_pulse_dict['response'].data)
##            plt.title('post synaptic response')
#
#            # ----Note that we may want to put a induction frequency filter here---
#            
#            no_skipping_sweep_count=no_skipping_sweep_count+1 #records if a sweep made it though filters
#            
#        if no_skipping_sweep_count>0:
#            response_voltage=first_pulse_dict['response'].data
##            # weight parts of the trace during fitting
#            dt = first_pulse_dict['response'].dt
#            pulse_ind=first_pulse_dict['pulse_ind']-first_pulse_dict['rec_start'] #get pulse indicies 
#            weight = np.ones(len(response_voltage))*10.  #set everything to ten initially
#            weight[pulse_ind:pulse_ind+int(3e-3/dt)] = 0.   #area around stim artifact
#            weight[pulse_ind+int(3e-3/dt):pulse_ind+int(15e-3/dt)] = 30.  #area around steep PSP rise 
#            
#            psp_fits = fit_psp(first_pulse_dict['response'], 
#                               xoffset=(.525, -float('inf'), float('inf')),
#                               sign='any', 
#                               weight=weight) 
#            
#            time_values=first_pulse_dict['response'].time_values
#            fig=plt.figure(figsize=(20,8))
#            a1=fig.add_subplot(2,1,1)
#            a1.plot(time_values, first_pulse_dict['pre_rec'].data)
#            a2=a1.twinx()
#            a2.plot(time_values, first_pulse_dict['command'].data)
#            plt.title('uid %s, pre/post electrodes %d, %d' % (expt.uid, pre_electrode_id, post_electrode_id) + ', nrmse,' + str(psp_fits.nrmse()))
#            
#            ax=fig.add_subplot(2,1,2)
#            ax2=ax.twinx()
#            ax.plot(time_values, psp_fits.data*1.e3, 'b', label='data')
#            ax.plot(time_values, psp_fits.best_fit*1.e3, 'g', lw=5, label='current best fit')
#            ax2.plot(time_values, weight, 'r', label='weighting')
#            ax.legend()
#            plt.tight_layout()
#            plt.show(block=False)
#            
#        plt.show()
#
#            #---------------------------------------------------------------------------------

            
