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



def measure_amp(trace, baseline=(6e-3, 8e-3), response=(13e-3, 17e-3)):
    '''This function is altered from PSP_amp_vs_time.py that was used to measure amplitude.
    Returns the largest deflection during a response window from the baseline average to 
    define psp amplitude. 
    
    input:
    ------
    trace: trace object
    baseline: tuple with start and end time points (seconds) to define baseline
    response: tuple with start and end time points (seconds) to define the region of the amplitude
    '''
    baseline = trace.time_slice(*baseline).data.mean()
    max_first_peak_region = trace.time_slice(*response).data.max()
    min_first_peak_region = trace.time_slice(*response).data.min()
    bsub=np.array([max_first_peak_region-baseline, min_first_peak_region-baseline])
    amp=bsub[np.where(abs(bsub)==max(abs(bsub)))[0][0]] #find the absolute maximum deflection from base_line
    return amp
    
def calculate_base_line(baseline_trace_view):
    '''return the average (voltage) for the data in a trace_view object'''
    return np.average(baseline.baseline_trace_view)


expts = cached_experiments()
selected_expt_list = expts.select(organism=desired_organism)

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
        pre_pad = 10e-3 
        post_pad = 50e-3
        
        # loop though sweeps in recording and pull out the ones you want
        no_skipping_sweep_count=0 #hack to avoid plotting when no sweeps in the experiment are recorded
        first_pulse_list=[]
        command_trace_list=[]
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
                        pulse['sweep_id']=sweep_id
                        first_pulse_list.append(pulse)
                                            
        # if there are sweeps do the analysis
        if len(first_pulse_list)>0:    
        
            # ---get baseline subtracted average for average fit
            bsub_trace_list=[]
            for sweep in first_pulse_list:
                sweep_trace=sweep['response']
                sweep_baseline_float_mode=float_mode(sweep['baseline'].data)
                bsub_trace_list.append(sweep_trace.copy(data=sweep_trace.data-sweep_baseline_float_mode, t0=0)) #Trace object with baseline subtracted data via float mode method. Note t0 is realigned to 0 
                command_trace_list.append(sweep['command'].copy(t0=0)) #get command traces so can see the average pulse for cross talk region estimation
            
            # take average of baseline subtracted data
            avg_voltage=TraceList(bsub_trace_list).mean()
            avg_dt=avg_voltage.dt
            avg_command=TraceList(command_trace_list).mean()  # pulses are slightly different in reference spike
            
            #fit the average base_line subtracted data
            weight = np.ones(len(avg_voltage.data))*10.  #set everything to ten initially
            weight[int((pre_pad-3e-3)/avg_dt):int(pre_pad/avg_dt)] = 0.   #area around stim artifact note that since this is spike aligned there will be some blur in where the cross talk is
            weight[int((pre_pad+1e-3)/avg_dt):int((pre_pad+5e-3)/avg_dt)] = 30.  #area around steep PSP rise 
            ave_psp_fit = fit_psp(avg_voltage, 
                               xoffset=(pre_pad+2e-3, pre_pad, pre_pad+5e-3), #since these are spike aligned the psp should not happen before the spike that happens at pre_pad by definition 
                               sign='any', 
                               weight=weight) 

            # if the synapse is not small skip
            if ave_psp_fit.best_values['amp']>.1e-3 or (expt.uid=='1494361299.85' and  pre_electrode_id==7 and post_electrode_id==0):
                continue
            
            # plot average fit
            plt.figure()
            c1=plt.subplot(2,1,1)
            c1.plot(avg_command.time_values, avg_command.data)
            c2=c1.twinx()
            c2.plot(avg_voltage.time_values, weight, 'k')
            plt.title('uid %s, pre/post electrodes %d, %d, average command, %s individual sweeps' % (expt.uid, pre_electrode_id, post_electrode_id, len(first_pulse_list)))
            ax1=plt.subplot(2,1,2)
            ax1.plot(avg_voltage.time_values, avg_voltage.data, label='data')
            ax1.plot(avg_voltage.time_values, ave_psp_fit.best_fit, 'g', lw=3, label='fit, amp= %g' % ave_psp_fit.best_values['amp'])
            ax2=ax1.twinx()
            ax2.plot(avg_voltage.time_values, weight, 'k', label='weight')
            plt.title('mean baseline subtracted spike aligned, nrmse=%.3g' % ave_psp_fit.nrmse())
            plt.show()

            #-----------single psp fitting------------------------
                
            amplitude_comparison=[] #initalize matrix for comparison of amplitudes via different algorithms
            for first_pulse_dict in first_pulse_list:
                response_trace=first_pulse_dict['response'].copy(t0=0) #reset time traces so can use fixed xoffset from average fit
    #            # weight parts of the trace during fitting
                dt = response_trace.dt
                pulse_ind=first_pulse_dict['pulse_ind']-first_pulse_dict['rec_start'] #get pulse indicies 
                weight = np.ones(len(response_trace.data))*10.  #set everything to ten initially
                weight[pulse_ind:pulse_ind+int(3e-3/dt)] = 0.   #area around stim artifact
                weight[pulse_ind+int(3e-3/dt):pulse_ind+int(15e-3/dt)] = 30.  #area around steep PSP rise 
                weight[len(avg_voltage):] = 0 # give decay zero weight
                
                # fit single psps while allowing all parameters to vary
                single_psp_fit_free = fit_psp(response_trace, 
                                   xoffset=(.011, -float('inf'), float('inf')),
                                   sign='any', 
                                   weight=weight) 

                avg_xoffset=ave_psp_fit.best_values['xoffset']
                
                # fit single psps while fixing xoffset, rise_time, and decay_tau
                single_psp_fit_fixed = fit_psp(response_trace, 
                                               xoffset=(avg_xoffset,'fixed'),
                                               rise_time=(ave_psp_fit.best_values['rise_time'], 'fixed'),
                                               decay_tau=(ave_psp_fit.best_values['decay_tau'], 'fixed'),
                                               sign='any', 
                                               weight=weight)
                
                # fit single psps while using a small jitter for xoffset, and rise_time, and fixing decay_tau
                xoff_min=max(avg_xoffset-.5e-3, pre_pad) #do not allow minimum jitter to go below the spike in the case in which xoffset of average is at the spike (which is at the location of pre_pad)
                single_psp_fit_small_bounds = fit_psp(response_trace, 
                                               xoffset=(avg_xoffset, xoff_min, avg_xoffset+.5e-3),
                                               rise_time=(ave_psp_fit.best_values['rise_time'], 0., ave_psp_fit.best_values['rise_time']),
                                               decay_tau=(ave_psp_fit.best_values['decay_tau'], 'fixed'),
                                               sign='any', 
                                               weight=weight)
             
                if np.isclose(single_psp_fit_small_bounds.best_values['rise_time'], 0.):
                    warnings.warn('the rise time is hitting zero')
                # plot single fits
                if True: #single_psp_fit_fixed.best_values['amp']<0:
                    time_values=response_trace.time_values
                    fig=plt.figure(figsize=(20,8))
                    a1=fig.add_subplot(2,1,1)
                    a1.plot(time_values, first_pulse_dict['pre_rec'].data)
                    a2=a1.twinx()
                    a2.plot(time_values, first_pulse_dict['command'].data)
                    plt.title('uid %s, pre/post electrodes %d, %d' % (expt.uid, pre_electrode_id, post_electrode_id))
                    
                    ax=fig.add_subplot(2,1,2)
                    ax.plot(time_values, single_psp_fit_free.data, 'b', label='data')
                    ax.plot(time_values, single_psp_fit_free.best_fit, 'g', lw=5, label='free fit, amp=%.3g' % single_psp_fit_free.best_values['amp'])
                    ax.plot(time_values, single_psp_fit_fixed.best_fit, 'c', lw=5, label='fixed fit, amp=%.3g' % (single_psp_fit_fixed.best_values['amp']))                
                    ax.plot(avg_voltage.time_values, ave_psp_fit.best_fit+first_pulse_dict['baseline'].mean(), 'r', label='average fit, amp=%.3g' % ave_psp_fit.best_values['amp']) #note adding base line so average can be on same scale
                    ax.plot(time_values, single_psp_fit_small_bounds.best_fit, 'm', lw=5, label='small_bounds, amp=%.3g' % single_psp_fit_small_bounds.best_values['amp'])                
                    ax2=ax.twinx()
                    ax2.plot(time_values, weight, 'k', label='weight')
                    ax.legend()
                    ax2.legend()
                    plt.title('sweep id=%d, nrmse=%.3g' % (first_pulse_dict['sweep_id'], single_psp_fit_free.nrmse()))
                    plt.tight_layout()
                    plt.show()

                # find the deflection found by measuring min or max individual psp
                measured_deflection = measure_amp(response_trace, baseline=(6e-3, 8e-3), response=(12e-3, 16e-3)) # measured deflection 
                
                amplitude_comparison.append([single_psp_fit_fixed.best_values['amp'], single_psp_fit_free.best_values['amp'], measured_deflection])
                
            plt.figure()
            fixed=[a[0] for a in amplitude_comparison]
            measured=[a[2] for a in amplitude_comparison]
            plt.plot(fixed, measured, '.')
            x,y=plt.xlim()
            plt.plot([x,y], [x,y], 'r')
            plt.xlabel('fit via fixing several parameters')
            plt.ylabel('measured via min/max data')
            plt.title('uid %s, pre/post electrodes %d, %d' % (expt.uid, pre_electrode_id, post_electrode_id))
            plt.show()