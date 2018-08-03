"""
This is an updated version of PSP_amp_vs_time.py made to just 
look at manuscript data. It has been reorganized to be more efficient.
This file assumes that neurons are qc'd etc.  This also has updated 
functions and utilizes the fitting for amplitudes.
"""

from __future__ import print_function, division

from collections import OrderedDict

import argparse
import sys
import os
import pickle
import pyqtgraph.multiprocess as mp
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import pandas as pd

from neuroanalysis.baseline import float_mode
import datetime
from multipatch_analysis.connection_detection import MultiPatchSyncRecAnalyzer, MultiPatchExperimentAnalyzer
from multipatch_analysis.constants import INHIBITORY_CRE_TYPES
from multipatch_analysis.constants import EXCITATORY_CRE_TYPES
from multipatch_analysis.connection_detection import MultiPatchExperimentAnalyzer
from multipatch_analysis.synaptic_dynamics import DynamicsAnalyzer
from multipatch_analysis.experiment_list import cached_experiments
from neuroanalysis.ui.plot_grid import PlotGrid
from neuroanalysis.data import TraceList, PatchClampRecording
from neuroanalysis.filter import bessel_filter
from neuroanalysis.event_detection import exp_deconvolve
import allensdk.core.json_utilities as ju
relative_path=os.path.dirname(os.getcwd())
sys.path.insert(1, os.path.join(relative_path))
import scipy.stats as stats
import pickle
import statsmodels.api as sm
from neuroanalysis.fitting import fit_psp

# mark to True if want to use pyqtgraph (conflicts with matplotlib) data
use_qt_plot=False
if use_qt_plot==True:
    import pyqtgraph as pg

Stephs_data=np.array([
    # [('unknown', 'unknown'),	('1501090950.86', 8, 1)],
    # [('unknown', 'unknown'),	('1501101571.17', 1, 5)],
    # [('unknown', 'unknown'),	('1501101571.17', 1, 7)],
    # [('unknown', 'unknown'),	('1501101571.17', 7, 5)],
    # [('unknown', 'unknown'),	('1501104688.89', 7, 3)],
    [('unknown', 'unknown'),	('1501621744.85', 1, 6)],
    [('unknown', 'unknown'),	('1501621744.85', 6, 1)],
    [('unknown', 'unknown'),	('1501627688.56', 3, 8)],
    [('unknown', 'unknown'),	('1501627688.56', 4, 7)],
    [('unknown', 'unknown'),	('1501627688.56', 8, 3)],
    [('unknown', 'unknown'),	('1501792378.34', 2, 8)],
    [('unknown', 'unknown'),	('1501792378.34', 8, 2)],
 #   [('rorb', 'rorb'),	('1498687063.99', 7, 1)],
    [('rorb', 'rorb'),	('1502301827.80', 6, 8)],
    [('rorb', 'rorb'),	('1502301827.80', 8, 6)],
    [('rorb', 'rorb'),	('1523470754.85', 3, 4)],
    [('rorb', 'rorb'),	('1523470754.85', 4, 3)],
    [('rorb', 'rorb'),	('1523470754.85', 4, 6)],
    [('rorb', 'rorb'),	('1523470754.85', 4, 7)],
    [('rorb', 'rorb'),	('1523470754.85', 6, 4)],
    [('rorb', 'rorb'),	('1523470754.85', 7, 3)],
    [('rorb', 'rorb'),	('1523470754.85', 7, 4)],
    [('rorb', 'rorb'),	('1523470754.85', 7, 6)],
    [('rorb', 'rorb'),	('1523479910.95', 2, 3)],
    # [('sim1', 'sim1'),	('1487107236.82', 7, 5)],
    # [('sim1', 'sim1'),	('1487107236.82', 7, 2)],
    # [('sim1', 'sim1'),	('1487367784.96', 6, 2)],
    # [('sim1', 'sim1'),	('1487376645.68', 1, 7)],
    # [('sim1', 'sim1'),	('1490642434.41', 5, 3)],
    # [('sim1', 'sim1'),	('1490642434.41', 3, 5)],
    # [('sim1', 'sim1'),	('1490642434.41', 7, 3)],
    # [('sim1', 'sim1'),	('1490651407.27', 2, 5)],
    # [('sim1', 'sim1'),	('1490651901.46', 4, 8)],
    # [('sim1', 'sim1'),	('1497468556.18', 8, 2)],
    # [('sim1', 'sim1'),	('1497468556.18', 8, 3)],
    # [('sim1', 'sim1'),	('1497468556.18', 8, 6)],
    # [('sim1', 'sim1'),	('1497468556.18', 2, 8)],
    # [('sim1', 'sim1'),	('1497469151.70', 1, 2)],
    # [('sim1', 'sim1'),	('1497469151.70', 1, 8)],
    # [('sim1', 'sim1'),	('1497469151.70', 8, 5)],
    # [('sim1', 'sim1'),	('1497469151.70', 8, 1)],
    # [('sim1', 'sim1'),	('1497473076.69', 7, 4)],
    # [('tlx3', 'tlx3'),	('1485904693.10', 8, 2)],
    # [('tlx3', 'tlx3'),	('1492460382.78', 6, 2)],
    # [('tlx3', 'tlx3'),	('1492460382.78', 4, 6)],
    # [('tlx3', 'tlx3'),	('1492468194.97', 6, 5)],
    # [('tlx3', 'tlx3'),	('1492545925.15', 2, 4)],
    # [('tlx3', 'tlx3'),	('1492545925.15', 8, 5)],
    # [('tlx3', 'tlx3'),	('1492545925.15', 4, 2)],
    # [('tlx3', 'tlx3'),	('1492545925.15', 8, 6)],
    # [('tlx3', 'tlx3'),	('1492546902.92', 2, 6)],
    # [('tlx3', 'tlx3'),	('1492546902.92', 2, 8)],
    # [('tlx3', 'tlx3'),	('1492546902.92', 4, 8)],
    # [('tlx3', 'tlx3'),	('1492546902.92', 8, 2)],
    # [('tlx3', 'tlx3'),	('1492637310.55', 5, 4)],
    # [('tlx3', 'tlx3'),	('1492810479.48', 1, 7)],
    # [('tlx3', 'tlx3'),	('1492812013.49', 5, 3)],
    # [('tlx3', 'tlx3'),	('1494881995.55', 7, 1)],
    [('tlx3', 'tlx3'),	('1502920642.09', 7, 8)],
    [('ntsr1', 'ntsr1'),('1504737622.52', 8, 2)],
    [('ntsr1', 'ntsr1'),('1529443918.26', 1, 6)]])


Steph_uids=[l[1] for l in Stephs_data]

def measure_amp(trace, baseline=(6e-3, 8e-3), response=(13e-3, 17e-3), plot=False):
    '''This function is altered from PSP_amp_vs_time.py that was used to measure amplitude.
    Returns the largest deflection during a response window from the baseline average to 
    define psp amplitude. 
    
    input
    -----
    trace: trace object
    baseline: tuple with start and end time points (seconds) to define baseline
    response: tuple with start and end time points (seconds) to define the region of the amplitude
    
    Returns
    -------
    amp: the largest deflection (negative or positive)in the region of the psp
    
    '''
    baseline = trace.time_slice(*baseline).data.mean()
    max_first_peak_region = trace.time_slice(*response).data.max()
    min_first_peak_region = trace.time_slice(*response).data.min()
    bsub=np.array([max_first_peak_region-baseline, min_first_peak_region-baseline])
    amp=bsub[np.where(abs(bsub)==max(abs(bsub)))[0][0]] #find the absolute maximum deflection from base_line

    return amp

def bin_data(the_list, data_list, bin_size=10):
    '''bins sec_since_t0 series data in sec_since_t0 bins with a specified size.
    Time must be bigger than 0.
    inputs
        the_list: list of arrays
            each array contains the times during the recording
        data_list: list of arrays
            data corresponding to the_list
        bin_size:
            specifies the size of the sec_since_t0 bin
    returns:
        data_in_bins: list of numpy arrays
            each array corresponds to a sec_since_t0 bin. Values in array are 
            values in the sec_since_t0 bin
        the_bins: list
            values in list denote bin edges
        middle_of_bins: list
            values in list correspond to the center of sec_since_t0 bins     
    '''
    max_value=max([max(tt) for tt in the_list])
    # make sure sec_since_t0 and bin_size make sense
    if min([min(tt) for tt in the_list])<0: 
        raise Exception('sec_since_t0 values should not be negative')
    if max_value<bin_size:
        raise Exception('bin size is bigger than max sec_since_t0')
    
    #specify the sec_since_t0 bins 
    the_bins=np.arange(0, max_value+bin_size, bin_size) # this could potentially be broken depending on what the bin_size is
    
    if the_bins[-1] < max_value:
        raise Exception('Your largest sec_since_t0 bin is less than max sec_since_t0.  Tweak your bin size.') 
    
    middle_of_bins=np.mean(np.array([np.append(the_bins, 0), np.append(0, the_bins)]), axis=0)[1:-1]  #sec_since_t0 bin is defined by middle of bin
    data_in_bins=[np.array([]) for ii in range(len(middle_of_bins))] #initialize a data structure to receive data in bins
    
    #assign data to correct sec_since_t0 bins
    for tt, ff in zip(the_list, data_list):
        assert len(tt)==len(ff)
        digits=np.digitize(tt, the_bins)-1 #note,digitize will assign values less than smallest timebin to a 0 index so -1 is used here
        for ii in range(len(digits)-1):  #note I just added this -1 for the sweep indexing so not sure if it makes sense 
#            print (ii, len(digits))
#            print (data_in_bins[digits[ii]], ff[ii])
            data_in_bins[digits[ii]]=np.append(data_in_bins[digits[ii]], ff[ii])

    return data_in_bins, the_bins, middle_of_bins

def test_bin_data():
    '''meant to tests the bin_data() module but really just using a simple input so one can watch what is happening
    '''
    time_list=[np.array([1,2,3,4,5]), np.array([0, 3, 4.5]), np.array([1.1, 1.2, 2.3, 2.5, 2.8, 4, 4.3, 4.9])]
    data_list=[np.array([1,2,3,4,5]), np.array([6,7,8]), np.array([9,10,11,12,13,14,15, 16])]
    data, time_bins, time_mid_points=bin_data(time_list, data_list, bin_size=2.1)

def average_via_bins(time_list, data_list, bin_size=10):
    '''takes list of sec_since_t0 arrays and corresponding list of data arrays and returns the average
    by placing the data in sec_since_t0 bins
    '''
    data,time_bins, time_bin_middle=bin_data(time_list, data_list, bin_size)
    average_data=[]
    std_err_data=[]
    for bins in data:
        average_data.append(np.mean(bins))
        std_err_data.append(stats.sem(bins))
    assert len(average_data)==len(time_bin_middle), "data length doesn't match sec_since_t0 length"
    return time_bin_middle, average_data, std_err_data



class fit_first_pulse():
    def __init__(self, expt, pre_syn_electrode_id, post_syn_electrode_id, pre_pad=10e-3, post_pad=50e-3):    
        """
        Inputs
        ------
        pre_pad: float
        Amount of time (s) before a spike to be included in the waveform.
        post_pad: float 
            Amount of time (s) after a spike to be included in the waveform.
        """

        self.expt=expt
        self.pre_syn_electrode_id = pre_syn_electrode_id
        self.post_syn_electrode_id = post_syn_electrode_id
        self.pre_pad = pre_pad
        self.post_pad = post_pad

    def get_spike_aligned_first_pulses(self):
        """Get all the first pulses recorded in current clamp and align them by spikes
        Returns
        -------
        first_pulse_list
        """
            
        # loop though sweeps in recording and pull out the ones you want
        first_pulse_list=[]
        command_trace_list=[]
        for sweep_rec in self.expt.data.contents:
            sweep_id=sweep_rec._sweep_id
            print ("SWEEP ID: %d, %s, electrode ids %d, %d, devices: %s" % (sweep_id, expt.uid, pre_syn_electrode_id, post_syn_electrode_id, sweep_rec.devices))
            if pre_syn_electrode_id not in sweep_rec.devices or post_syn_electrode_id not in sweep_rec.devices:
                print("Skipping %s electrode ids %d, %d; pre or post synaptic electrode id is not in sweep_rec.devices" % (expt.uid, pre_syn_electrode_id, post_syn_electrode_id))
                continue
            pre_rec = sweep_rec[pre_syn_electrode_id]
            post_rec = sweep_rec[post_syn_electrode_id]
            if post_rec.clamp_mode != 'ic':
                #print("Skipping %s electrode ids %d, %d; rec.clamp_mode != current clamp" % (expt.uid, pre_syn_electrode_id, post_syn_electrode_id))
                continue

            analyzer = MultiPatchSyncRecAnalyzer.get(sweep_rec)
            
            # get information about the spikes and make sure there is a spike on the first pulse
            spike_data = analyzer.get_spike_responses(pre_rec, post_rec, pre_pad=self.pre_pad, align_to='spike')
            if 0 in [pulse['pulse_n'] for pulse in spike_data]: # confirm pulse number starts at one, not zero
                raise Exception("Skipping %s electrode ids %d, %d; ; should have not have zero" % (expt.uid, pre_syn_electrode_id, post_syn_electrode_id)) 
            if 1 not in [pulse['pulse_n'] for pulse in spike_data]: # skip this sweep if not a spike on the first pulse
                #print("Skipping %s electrode ids %d, %d; no spike on first pulse" % (expt.uid, pre_syn_electrode_id, post_syn_electrode_id))                
                continue
            else: # appends sweep to the first pulse data 
                for pulse in spike_data:
                    if pulse['pulse_n'] == 1:
                        pulse['sweep_id']=sweep_id
                        first_pulse_list.append(pulse)
        
        return first_pulse_list

    def get_baseline_sub_average(self, first_pulse_list):
        """Substract the baseline for each individual fit and then
        take the average.
        
        input
        -----
        first_pulse_list
        """        

        bsub_trace_list=[]
        command_trace_list=[]
        for sweep in first_pulse_list:
            sweep_trace=sweep['response']
            sweep_baseline_float_mode=float_mode(sweep['baseline'].data)
            bsub_trace_list.append(sweep_trace.copy(data=sweep_trace.data-sweep_baseline_float_mode, t0=0)) #Trace object with baseline subtracted data via float mode method. Note t0 is realigned to 0 
            command_trace_list.append(sweep['command'].copy(t0=0)) #get command traces so can see the average pulse for cross talk region estimation
        
        # take average of baseline subtracted data
        self.avg_voltage=TraceList(bsub_trace_list).mean()
        self.avg_dt=self.avg_voltage.dt
        self.avg_command=TraceList(command_trace_list).mean() # pulses are slightly different in reference spike
        return self.avg_voltage, self.avg_dt, self.avg_command 

    def fit_avg(self):
        #fit the average base_line subtracted data
        weight = np.ones(len(self.avg_voltage.data))*10.  #set everything to ten initially
        weight[int((self.pre_pad-3e-3)/self.avg_dt):int(self.pre_pad/self.avg_dt)] = 0.   #area around stim artifact note that since this is spike aligned there will be some blur in where the cross talk is
        weight[int((self.pre_pad+1e-3)/self.avg_dt):int((self.pre_pad+5e-3)/self.avg_dt)] = 30.  #area around steep PSP rise 

        self.ave_psp_fit = fit_psp(self.avg_voltage, 
                        xoffset=(self.pre_pad+2e-3, self.pre_pad, self.pre_pad+5e-3), #since these are spike aligned the psp should not happen before the spike that happens at pre_pad by definition 
                        sign='any', 
                        weight=weight) 

        return self.ave_psp_fit, weight

    def fit_single(self, first_pulse_dict):
        response_trace=first_pulse_dict['response'].copy(t0=0) #reset time traces so can use fixed xoffset from average fit
#            # weight parts of the trace during fitting
        dt = response_trace.dt
        pulse_ind=first_pulse_dict['pulse_ind']-first_pulse_dict['rec_start'] #get pulse indicies 
        weight = np.ones(len(response_trace.data))*10.  #set everything to ten initially
        weight[pulse_ind:pulse_ind+int(3e-3/dt)] = 0.   #area around stim artifact
        weight[pulse_ind+int(3e-3/dt):pulse_ind+int(15e-3/dt)] = 30.  #area around steep PSP rise 
        weight[len(self.avg_voltage):] = 0 # give decay zero weight
        
        # fit single psps while using a small jitter for xoffset, and rise_time, and fixing decay_tau
        avg_xoffset=self.ave_psp_fit.best_values['xoffset']
        xoff_min=max(avg_xoffset-.5e-3, self.pre_pad) #do not allow minimum jitter to go below the spike in the case in which xoffset of average is at the spike (which is at the location of pre_pad)
        single_psp_fit_small_bounds = fit_psp(response_trace, 
                                        xoffset=(avg_xoffset, xoff_min, avg_xoffset+.5e-3),
                                        rise_time=(self.ave_psp_fit.best_values['rise_time'], 0., self.ave_psp_fit.best_values['rise_time']),
                                        decay_tau=(self.ave_psp_fit.best_values['decay_tau'], 'fixed'),
                                        sign='any', 
                                        weight=weight)
        return single_psp_fit_small_bounds, weight

    def measure_amp_single(self, first_pulse_dict):
        response_trace=first_pulse_dict['response'].copy(t0=0) #reset time traces so can use fixed xoffset from average fit
        dt = response_trace.dt
        pulse_ind=first_pulse_dict['pulse_ind']-first_pulse_dict['rec_start'] #get pulse indicies 

        psp_region_start_ind=pulse_ind+int(3e-3/dt)
        psp_region_end_ind=pulse_ind+int(15e-3/dt)

        baseline = first_pulse_dict['baseline'].mean()  #baseline voltage
        # get the maximum value in a region around where psp amp should be
        max_first_peak_region = response_trace.data[psp_region_start_ind:psp_region_end_ind].max()
        # get the minimum value in a region around where psp amp should be
        min_first_peak_region = response_trace.data[psp_region_start_ind:psp_region_end_ind].min()
        # subtract the baseline value from min and max values
        bsub=np.array([max_first_peak_region-baseline, min_first_peak_region-baseline])
        #find the absolute maximum deflection from base_line
        relative_amp=bsub[np.where(abs(bsub)==max(abs(bsub)))[0][0]] 
        # if plot==True:
        #     plt.figure()
        #     plt.baseline(response_trace.data)
        #     plt.show()
        return relative_amp, baseline

    def plot_fit(self, fit, voltage, command, weight, title, 
                measured_baseline=0,
                measured_amp=False,
                show_plot=False,
                save_name=False):

        # plot average fit
        plt.figure(figsize=(14,14))
        c1=plt.subplot(2,1,1)
        ln1=c1.plot(command.time_values*1.e3, command.data*1.e3, label='command')
        c1.set_ylabel('voltage (mV)')
        c2=c1.twinx()
        ln2=c2.plot(voltage.time_values*1.e3, weight, 'k', label='weight')
        c2.set_ylabel('weight')
        plt.title('uid %s, pre/post electrodes %d, %d, individual sweeps: %s' % (self.expt.uid, 
                                    self.pre_syn_electrode_id, 
                                    self.post_syn_electrode_id, 
                                    len(first_pulse_list)))

        lines_plot_1 = ln1+ln2
        label_plot_1 = [l.get_label() for l in lines_plot_1]
        c1.legend(lines_plot_1, label_plot_1)

        ax1=plt.subplot(2,1,2)
        ln3=ax1.plot(voltage.time_values*1.e3, (voltage.data-measured_baseline)*1.e3, label='data')
        ln4=ax1.plot(voltage.time_values*1.e3, (fit.best_fit-measured_baseline)*1.e3, 'g', lw=3, label='fit')
        ax1.set_ylabel('voltage (mV)')
        ax2=ax1.twinx()
        ln5=ax2.plot(voltage.time_values*1.e3, weight, 'k', label='weight')
        ax2.set_ylabel('weight')
        ax1.set_xlabel('time (ms)')
        lines_plot_2 = ln3+ln4+ln5
        label_plot_2 = [l.get_label() for l in lines_plot_2]
        ax1.legend(lines_plot_2, label_plot_2)
        if measured_amp:
            ax1.plot(voltage.time_values*1.e3, np.ones(len(voltage.time_values))*measured_amp*1.e3, 'r--')
            plt.title(title + ' nrmse=%.3g, fit amp:%.3g, measured amp:%3g' % (ave_psp_fit.nrmse(), ave_psp_fit.best_values['amp']*1.e3, measured_amp*1.e3))
        else:
            plt.title(title + ' nrmse=%.3g, fit amp:%.3g' % (ave_psp_fit.nrmse(), ave_psp_fit.best_values['amp']*1e-3))

        plt.tight_layout()
        if show_plot:
            plt.show()
        if save_name:
            plt.savefig(save_name)



if __name__ == '__main__':

    if use_qt_plot == True:
        app = pg.mkQApp()
        pg.dbg()

    path='/home/corinnet/workspace/aiephys/rundown_results'
    if not os.path.exists(path):
        os.makedirs(path)

    # load experiments
    expts = cached_experiments()
    dictionary={}
    synapses = []
    # cycle through all expt in connections summary (note the experiment summary is being cycled though a lot here)
    for connection in expts.connection_summary(): # expts.connection_summary() is list of dictionaries
        cells = connection['cells'] #(pre, post) synaptic "Cell" objects
        expt = connection['expt'] #"Experiment" object
#            print (expt.uid) #unique id for experiment
        pre_syn_cell_id=cells[0].cell_id
        post_syn_cell_id=cells[1].cell_id
        pre_syn_electrode_id=cells[0].electrode.device_id
        post_syn_electrode_id=cells[1].electrode.device_id

        # skip connection if not in Stephs set 
        if (expt.uid, pre_syn_cell_id, post_syn_cell_id) not in Steph_uids:
            continue
        else:
            print ("RUNNING: %s, cell ids:%s %s, electrode ids: %s %s" % (expt.uid, pre_syn_cell_id, post_syn_cell_id, pre_syn_electrode_id, post_syn_electrode_id))

        # initialize fitting class
        fitting=fit_first_pulse(expt, pre_syn_electrode_id, post_syn_electrode_id)
        # get spike aligned first pulses
        first_pulse_list=fitting.get_spike_aligned_first_pulses()

        # skip neuron if there is no data
        if not len(first_pulse_list)>0:
            continue
        
        # Get the average of the baseline subtracted first pulses
        avg_voltage, dt, avg_command=fitting.get_baseline_sub_average(first_pulse_list)
        ave_psp_fit, weight_for_average=fitting.fit_avg()

        # set up output directory and naming convention output files
        name_string=expt.uid+'_'+str(pre_syn_cell_id)+'_'+str(post_syn_cell_id)+'_'+cells[0].cre_type+'_'+cells[0].cre_type
        connection_path=os.path.join(path, name_string) #path to connection specific directory        
        if not os.path.exists(connection_path):
            os.makedirs(connection_path)

        # fit the average of the first pulses
        fitting.plot_fit(ave_psp_fit, avg_voltage, avg_command, 
                        weight_for_average, 
                        'Mean baseline subtracted spike aligned,', 
                        save_name=os.path.join(connection_path, 'AVG_'+name_string+'.png'))

        #fit individual pulses
        out_data=[]
        for first_pulse_dict in first_pulse_list:
            # fit single pulse
            single_psp_fit, weight_for_single=fitting.fit_single(first_pulse_dict)
            # get measured baseline and individual psp amplitude
            measured_amp, baseline_value=fitting.measure_amp_single(first_pulse_dict)
            
            # plot the fit and compare with measured
            fitting.plot_fit(single_psp_fit, 
                            first_pulse_dict['response'], 
                            first_pulse_dict['command'], 
                            weight_for_single, 
                            'SWEEP:'+str(first_pulse_dict['sweep_id'])+', ', 
                            measured_baseline=baseline_value, 
                            measured_amp=measured_amp,
                            save_name=os.path.join(connection_path, 'Sweep_'+str(first_pulse_dict['sweep_id'])+'_'+name_string+'.png'))
            
            # put data in output list
            out_data.append([expt.uid, 
                            pre_syn_cell_id, 
                            post_syn_cell_id,
                            first_pulse_dict['sweep_id'],
                            measured_amp,
                            single_psp_fit.best_values['amp'],
                            first_pulse_dict['response'].t0,
                            cells[0].cre_type,
                            cells[1].cre_type]) 
    
        out_df=pd.DataFrame(out_data)
        out_df.columns=['uid','pre_cell_id','post_cell_id', 'sweep_id', 'measured_amp', 'fit_amp', 't0', 'pre_cre', 'post_cre']
        out_df.to_csv(os.path.join(connection_path,'rundown08_03_2018.csv')) 

            



    
        

# # some other options to potentially be able to choose from            
# #            responses = analyzer.train_responses
# #            pulse_offset = analyzer.pulse_offsets
# #            response = analyzer.pulse_responses
    
#     #        plt.figure()
#     #        for trace in amp_responses.responses:
#     #            plt.plot(trace.time_values, trace.data)
#     #            plt.title('responses')
#     #        plt.figure()
#     #        for trace in amp_responses.baselines:
#     #            plt.plot(trace.time_values, trace.data)
#     #            plt.title('baselines')         
#     #        plt.show(block=False)
    
#             # figure out whether the trough or individual_first_peaks of the average synaptic trace is bigger and if that corresponds to the excitation of the neurons.  
#             # i.e. if it is an excitatory synapse we would expect the max defection to be positive
#             average = amp_responses.bsub_mean() #returns average synaptic response with the average baseline subtracted
#             max_peak = measure_amp(average, min_or_max='max', baseline=(6e-3, 8e-3), response=(12e-3, 16e-3))
#             min_peak = measure_amp(average, min_or_max='min', baseline=(6e-3, 8e-3), response=(12e-3, 16e-3))
#             ave_deflection=max(abs(max_peak), abs(min_peak)) #individual_first_peaks of average trace
#             max_min = "max" if abs(max_peak)> abs(min_peak) else "min"  #find whether the individual_first_peaks or trough of the first pulse average is larger 
#             correct_syn_amp_dir = True
#             if max_min == "min" and expt.cells[pre_id].cre_type in EXCITATORY_CRE_TYPES: 
#                 print ("Whoa this synapse looks inhibitory when cre line would say it should be excitatory!!!" )  
#                 correct_syn_amp_dir = False
#             if max_min == "max" and expt.cells[pre_id].cre_type in INHIBITORY_CRE_TYPES: 
#                 print ("Whoa this synapse looks excitatory when cre line would say it should be inhibitory!!!" )    
#                 correct_syn_amp_dir = False  
       
#             # find the individual_first_peaks or trough of every potential event and plot their amplitude over sec_since_t0 of the experiment
#             individual_first_peaks=[]
#             individual_first_baselines=[]
#             individual_start_times=[]
#             individual_holding_potentials=[]
#             sweep_numbers=[]
#             ordered=sorted(amp_responses.responses, key=lambda rr:rr.start_time) #order the traces by individual_start_times during the experiment
#             for jj, rr in enumerate(ordered):
#                 individual_first_peaks.append(measure_amp(rr, min_or_max=max_min, baseline=(6e-3, 8e-3), response=(12e-3, 16e-3)))
#                 individual_first_baselines.append(measure_amp(rr, min_or_max=max_min, baseline=(0e-3, 2e-3), response=(6e-3, 10e-3)))
#                 individual_start_times.append(rr.start_time)  
#                 individual_holding_potentials.append(rr.parent.parent.holding_potential)
#                 sweep_numbers.append(float(rr.parent.parent._sweep._sweep_id))
#                 #print ('for each first pulse of a synapse: individual_first_peaks', individual_first_peaks[-1], 'individual_first_baselines', individual_first_baselines[-1], 'individual_start_times', individual_start_times[-1], 'holding potential', individual_holding_potentials[-1], 'sweep_numbers', sweep_numbers[-1])      

#     #        for trace in amp_responses.responses:
#     #            plt.plot(trace.time_values, trace.data)
#     #            plt.title('responses')
#     #        plt.figure()
#     #        for trace in amp_responses.baselines:
#     #            plt.plot(trace.time_values, trace.data)
#     #            plt.title('baselines')         
#     #        plt.show()
    
#             # check if holding potential is within a desired range
#             holding=np.mean(individual_holding_potentials) # average holding potential across plots
#             #print ('holding potential is', holding)
#             if holding>-0.072 and holding<-0.068:
#                 holding_good_flag=True
#             else:
#                 holding_good_flag=False
#             #print ('\tholding potential flag set to ', holding_good_flag)

#             mean_base=np.mean(individual_first_baselines) # average individual_first_baselines across pulses of a synapse
#             sweep_numbers=np.array(sweep_numbers)
#             sec_since_t0=np.array(individual_start_times) - individual_start_times[0] # remap individual_start_times basis to be in reference to start of experiment
#             sec_since_t0=np.array([td.total_seconds() for td in sec_since_t0])
#             #TODO: !!!!from what I can tell, individual_first_peaks are already subtracting there individual baselines so I am not sure why it is being resubtracted below!!!
#             peak_minus_base_average=np.array(individual_first_peaks)-mean_base # take each individual_first_peaks and put it in reference to the average individual_first_baselines
#             smoothed=ndi.gaussian_filter(peak_minus_base_average, 2) # 
#             t_slope, t_intercept, _,_,_=stats.linregress(sec_since_t0, peak_minus_base_average)
#             sn_slope, sn_intercept, _,_,_=stats.linregress(sweep_numbers, peak_minus_base_average)
            
#             # can look at the output of the f-test in statsmodels
#             # https://blog.datarobot.com/ordinary-least-squares-in-python
#             # actually states what the summary numbers mean
#             # https://en.wikipedia.org/wiki/Lack-of-fit_sum_of_squares
#             # does a reasonable job of reporting what the numbers mean
#             sm_lr=sm.OLS(peak_minus_base_average, sm.add_constant(sec_since_t0)) #Note statsmodels takes x,y in reverse order
#             out=sm_lr.fit()
#             t_f_pvalue=out.f_pvalue
#             res=out.predict()
#             t_max_residual=np.max([np.max(res), np.absolute(np.min(res))])
            
#             sm_lr=sm.OLS(peak_minus_base_average, sm.add_constant(sweep_numbers)) #Note statsmodels takes x,y in reverse order
#             out=sm_lr.fit()
#             sn_f_pvalue=out.f_pvalue
#             res=out.predict()
#             sn_max_residual=np.max([np.max(res), np.absolute(np.min(res))])
            
#             # Slope is very small here due to small amplitude.
#             #    If voltage is fit in mV it will make scatter and 
#             #    slope bigger.  Will f_pvalue stay the same?
#             #print(out.summary())
#             plt.figure(figsize=(16, 10))
#             plt.subplot(3,1,1)
#             plt.scatter(sec_since_t0, peak_minus_base_average)
#             plt.plot(sec_since_t0, out.predict())
#             plt.ylim([min(peak_minus_base_average), max(peak_minus_base_average)])
#             plt.xlabel('time (s)')
#             plt.ylabel('amplitude (V)')
#             plt.title(expt.uid+'_'+str(pre_id)+'_'+str(post_id)+' fit, slope:'+str(t_slope)+', intercept:'+str(t_intercept)+'\n, f_pvalue: '+str(t_f_pvalue)+', max. res.:'+str(t_max_residual))
#             plt.subplot(3,1,2)
#             residuals=peak_minus_base_average-out.predict()
#             plt.scatter(sec_since_t0, residuals) #plotting residuals
#             plt.ylim([min(residuals), max(residuals)])
#             plt.title('residuals')
#             plt.xlabel('time (s)')
#             plt.ylabel('residual (V)')
#             plt.subplot(3,1,3)
#             plt.hist(residuals)
#             plt.title('residuals')
#             plt.ylabel('n')
#             plt.xlabel('residuals (V)')    
#             plt.tight_layout()
#             plt.savefig(os.path.join('/home/corinnet/Desktop/human_rundown', expt.uid+'_'+str(pre_id)+'_'+str(post_id)+'.png'))
#             plt.close()
            
#             def update_plots_and_dicts(qc_key):
#                 '''updates and values for different qc groupings
#                 inputs: string
#                     options: 'correct_amp_good_HP','correct_amp_bad_HP', 'wrong_amp_good_HP', 'wrong_amp_bad_HP'
#                 '''
#                 if use_qt_plot==True:
#                     ave_psp_plot.plot(average.time_values, average.data, pen=pg.mkPen(color=colors[qc_key])) #plot average of first pulse in each epoch of spikes of individual synapses
#     #                time_vs_psp_plot.plot(np.array(sec_since_t0), smoothed, pen=pg.mkPen(color=colors[qc_key])) # (i, len(synapses)*1.3))
#                     time_vs_psp_plot.plot(np.array(sec_since_t0), peak_minus_base_average, pen=pg.mkPen(color=colors[qc_key])) # (i, len(synapses)*1.3))
#                     time_vs_psp_plot.plot(np.array(sec_since_t0), t_slope*np.array(sec_since_t0)+t_intercept, pen=pg.mkPen(color=colors[qc_key], style=pg.QtCore.Qt.DashLine)) # (i, len(synapses)*1.3))
#     #                sweep_vs_psp_plot.plot(sweep_numbers, smoothed, pen=pg.mkPen(color=colors[qc_key]))
#                     sweep_vs_psp_plot.plot(sweep_numbers, peak_minus_base_average, pen=pg.mkPen(color=colors[qc_key]))
#                     sweep_vs_psp_plot.plot(sweep_numbers, sn_slope*sweep_numbers+sn_intercept, pen=pg.mkPen(color=colors[qc_key],style=pg.QtCore.Qt.DashLine)) # (i, len(synapses)*1.3))
#                 slopes['sweep_numbers'][qc_key].append(sn_slope)
#                 slopes['sec_since_t0'][qc_key].append(t_slope)
#                 intercepts['sweep_numbers'][qc_key].append(sn_intercept)
#                 intercepts['sec_since_t0'][qc_key].append(t_intercept)      
#                 f_pvalues['sweep_numbers'][qc_key].append(sn_f_pvalue)
#                 f_pvalues['sec_since_t0'][qc_key].append(t_f_pvalue)        
#                 max_residuals['sweep_numbers'][qc_key].append(sn_max_residual)
#                 max_residuals['sec_since_t0'][qc_key].append(t_max_residual)        
            
#             #TODO: check to see if this is a bug to only record in the optimal state.
#             # record values for different qc states
#             if correct_syn_amp_dir == True and holding_good_flag ==True:
#                 print('recording synapse')
#                 #these are appending array to be used futher before dictionary output
#                 time_list.append(sec_since_t0)  
#                 filtered.append(smoothed)   
#                 raw.append(peak_minus_base_average) 
#                 sweep_number_list.append(sweep_numbers) 
                
#                 #following appends a single value to a list and gets used in output dictionary
#                 num_of_synapses=num_of_synapses+1 
#                 update_plots_and_dicts('correct_amp_good_HP')
#                 PSPs_amp_start.append(peak_minus_base_average[0])
#                 PSPs_amp_ave.append(np.mean(peak_minus_base_average))
#                 PSPs_amp_end.append(peak_minus_base_average[-1])
#                 length_of_experiment.append(sec_since_t0[-1])
         
#             else: 
#                 if correct_syn_amp_dir==True and holding_good_flag==False:
#                     update_plots_and_dicts('correct_amp_bad_HP')
#                 elif correct_syn_amp_dir==False and holding_good_flag==True:
#                     update_plots_and_dicts('wrong_amp_good_HP')

#                 elif correct_syn_amp_dir==False and holding_good_flag==False:
#                     update_plots_and_dicts('wrong_amp_bad_HP')
#                 else:
#                     print(correct_syn_amp_dir)
#                     print(holding_good_flag)
#                     raise Exception("This flag combo doesn't exist")
            
#             print('done with one synapse')
#             if use_qt_plot == True:
#                 app.processEvents()

        
#         #because times of events aren't all at the same sec_since_t0, sec_since_t0 binning is needed to get average sec_since_t0 course
#         time_points, time_avg_data, time_std_err=average_via_bins(time_list, raw, bin_size=60)
#         if use_qt_plot == True:
#             time_vs_psp_plot.plot(time_points, time_avg_data, pen=pg.mkPen(color='w', width=5)) #plots average of the data
        
#         sweeps, sweep_avg_data, sweep_std_err=average_via_bins(sweep_number_list, raw, bin_size=5)
#         if use_qt_plot == True:
#             sweep_vs_psp_plot.plot(sweeps, sweep_avg_data, pen=pg.mkPen(color='w', width=5)) #plots average of the data
        
#         dictionary[title_str]={'time_points': time_points, 
#                                'time_avg_data':time_avg_data, 
#                                'time_std_err':time_std_err, 
#                                'num_of_synapses':num_of_synapses,
#                                'sweeps':sweeps,
#                                'sweep_avg_data':sweep_avg_data,
#                                'sweep_std_err':sweep_std_err,
#                                'slopes':slopes,
#                                'intercepts':intercepts,
#                                'PSPs_amp_start':PSPs_amp_start,
#                                'PSPs_amp_ave':PSPs_amp_ave,
#                                'PSPs_amp_end':PSPs_amp_end,
#                                'length_of_experiment':length_of_experiment,
#                                'uids_skipped':np.unique(uid_skip),
#                                'f_pvalues':f_pvalues,
#                                'max_residuals':max_residuals}
        
#     ju.write("PSP_vs_time_output_data/7_30_2018.json", dictionary)
#     if not os.path.isdir("PSP_vs_time_output_data"):
#         os.mkdir("PSP_vs_time_output_data")
#     ju.write("PSP_vs_time_output_data/6_13_2017_human.json", dictionary)
    

#     plt.figure()
#     for key in dictionary.keys():
#         plt.errorbar(dictionary[key]['time_points'], dictionary[key]['time_avg_data'],  yerr=dictionary[key]['time_std_err'], label=key+', n='+str(dictionary[key]['num_of_synapses']))
#     plt.title('average individual_first_baselines-line subtracted first pulse synaptic deflection')
#     plt.legend(loc=4)
#     plt.ylabel('voltage (V)')
#     plt.xlabel('sec_since_t0 since first recorded synapse (s)')

#     plt.figure()
#     for key in dictionary.keys():
#         plt.errorbar(dictionary[key]['sweeps'], dictionary[key]['sweep_avg_data'],  yerr=dictionary[key]['sweep_std_err'], label=key+', n='+str(dictionary[key]['num_of_synapses']))
#     plt.title('average individual_first_baselines-line subtracted first pulse synaptic deflection')
#     plt.legend(loc=4)
#     plt.ylabel('voltage (V)')
#     plt.xlabel('sweep number')    
    
    
#     plt.show(block=False)

# #        app.processEvents()    
# #        pg.QtGui.QApplication.exec_()  
    
#     plt.show()
        
    
  
    
