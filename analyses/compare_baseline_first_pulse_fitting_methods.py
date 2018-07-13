'''See if fitting average psp yields the same results as fitting baseline subtracted and then averaged psps 
'''

from __future__ import print_function, division
from multipatch_analysis.experiment_list import cached_experiments
from multipatch_analysis.connection_detection import MultiPatchSyncRecAnalyzer, MultiPatchExperimentAnalyzer
from multipatch_analysis.synaptic_dynamics import DynamicsAnalyzer
import pickle
import numpy as np
import matplotlib.pyplot as plt
from neuroanalysis.baseline import float_mode
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
        
        # initialize trace_view lists (want trace view lists so can use averaging methods which take care of sampling issues etc.)
        floatp_bl_sub_v_waveform_trace_list=TraceList()
        mean_bl_sub_v_waveform_trace_list=TraceList()
        non_altered_v_waveform_trace_list=TraceList()
        plt.figure()
        no_skipping_sweep_count=0 #hack to avoid plotting when no sweeps in the experiment are recorded
        
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

            else: # selects the first pulse data
                for pulse in spike_data:
                    if pulse['pulse_n'] == 1:
                        first_pulse_dict = pulse
            
            # only use pulses that pass qc for excitatory.
            if first_pulse_dict['ex_qc_pass'] != True:
                print("Skipping %s electrode ids %d, %d; doesnt pass excitatory qc" % (expt.uid, pre_electrode_id, post_electrode_id))                
                continue
            
            # get baseline via float mode method
            sweep_baseline_float_mode=float_mode(first_pulse_dict['baseline'].data)
            sweep_baseline_mean=np.mean(first_pulse_dict['baseline'].data)
            
            # create traces with altered data for different baseline conditions
            response = first_pulse_dict['response'] #for ease of using trace copy
            floatp_bl_sub_v_waveform_trace=response.copy(data=response.data-sweep_baseline_float_mode) #Trace object with baseline subtracted data via float mode method 
            mean_bl_sub_v_waveform_trace=response.copy(data=response.data-sweep_baseline_mean) #Trace object with baseline subtracted data via mean
            
            # plot individual sweeps
            plt.subplot(3,1,1)
            plt.plot(floatp_bl_sub_v_waveform_trace.data)
            plt.title('float_mode baseline subtraction')
            plt.subplot(3,1,2)
            plt.plot(mean_bl_sub_v_waveform_trace.data)
            plt.title('average baseline subtraction')
            plt.subplot(3,1,3)
            plt.plot(first_pulse_dict['response'].data)
            plt.title('non altered data')
                       
            no_skipping_sweep_count=no_skipping_sweep_count+1 #records if a sweep made it though filters
            # put traces in a TraceList so can use mean method
            floatp_bl_sub_v_waveform_trace_list.append(floatp_bl_sub_v_waveform_trace)
            mean_bl_sub_v_waveform_trace_list.append(mean_bl_sub_v_waveform_trace)
            non_altered_v_waveform_trace_list.append(first_pulse_dict['response'])
            
        # if there are sweeps recorded for a neuron do more analysis
        if no_skipping_sweep_count>0:
            plt.subplot(3,1,1)
            plt.plot(mean_bl_sub_v_waveform_trace_list.mean().data, 'k', lw=10)
            plt.title('float mode baseline subtracted')
            plt.subplot(3,1,2)
            plt.plot(mean_bl_sub_v_waveform_trace_list.mean().data, 'k', lw=10)
            plt.title('mean baseline subtracted')
            plt.subplot(3, 1, 3)
            plt.plot(non_altered_v_waveform_trace_list.mean().data, 'k', lw=10)
            plt.title('data')
            plt.tight_layout()
            plt.annotate('uid %s, pre/post electrodes %d, %d' % (expt.uid, pre_electrode_id, post_electrode_id),  
                         xy = (.5, .97), ha = 'center', fontsize = 16, xycoords = 'figure fraction')
            plt.show()
            # start here
    #        psp_fits = fit_psp(avg_trace, 
    #                   xoffset=(14e-3, -float('inf'), float('inf')),
    #                   sign=amp_sign, 
    #                   weight=weight)   
        else:
            plt.close()
            
       
            
            
            