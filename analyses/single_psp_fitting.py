'''This code is meant to explore fitting single PSPs'''

from __future__ import print_function, division
from multipatch_analysis.experiment_list import cached_experiments
from multipatch_analysis.connection_detection import MultiPatchSyncRecAnalyzer, MultiPatchExperimentAnalyzer
from multipatch_analysis.synaptic_dynamics import DynamicsAnalyzer
import pickle
import numpy as np
import matplotlib.pyplot as plt

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
        
        # loop though sweeps in recording
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
            
            #--for visualization to see what spike traces look like.  Can delete when done--- 
            else: # selects the first pulse data
                for pulse in spike_data:
                    if pulse['pulse_n'] == 1:
                        first_pulse_dict = pulse
            
            #plotting to see what the spikes look like; can delete when happy
            plt.figure()
            plt.subplot(3,1,1)
            plt.plot(first_pulse_dict['command'].data)
            plt.title('command')
            plt.subplot(3,1,2)
            plt.plot(first_pulse_dict['pre_rec'].data)
            plt.title('pre synaptic response')
            plt.subplot(3,1,3)
            plt.plot(first_pulse_dict['response'].data)
            plt.title('post synaptic response')
            plt.annotate('uid %s, pre/post electrodes %d, %d, sweep %d' % (expt.uid, pre_electrode_id, post_electrode_id, sweep_id),  
                         xy = (.5, .97), ha = 'center', fontsize = 16, xycoords = 'figure fraction')
            plt.show()
            #---------------------------------------------------------------------------------

            # ----Note that we may want to put a induction frequency filter here---
            
#            # get the first pulses: dont need this since get_spike_responses() returns necessary responses
#            post_cell_trace_view, baseline_trace_view, pre_cell_trace_view, command_trace_view = analyzer.get_train_response(pre_rec,
#                                                                                                                            post_rec, 
#                                                                                                                            start_pulse=0, 
#                                                                                                                            stop_pulse=0, 
#                                                                                                                            padding=(-10e-3, 50e-3)) #notice that instead of skipping traces without a spike at first pulse you could set which pule to use here 
#            
#            # plot what is coming out of the get train responses.
#            plt.figure()
#            plot_trace_views([4,1], post_cell_trace_view, title = 'Post synaptic response?')
#            plot_trace_views([4,4], baseline_trace_view, title = 'Baseline')
#            plot_trace_views([4,2], pre_cell_trace_view, title = 'Pre synaptic resonse')
#            plot_trace_views([4,3], command_trace_view, title = 'Current injection')
#            plt.tight_layout()
#            plt.annotate('uid %s, pre/post electrodes %d, %d, sweep %d' % (expt.uid, pre_electrode_id, post_electrode_id, sweep_id),  
#                         xy = (.5, .97), ha = 'center', fontsize = 16, xycoords = 'figure fraction')
#            plt.show()
#            print ('hi')
            
