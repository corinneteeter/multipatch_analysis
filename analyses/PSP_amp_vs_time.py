"""
Question: how does PSP height change during the duration of the experiment.
The code saves output to a file to be plotted elsewhere.
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

import datetime
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

# mark to True if want to use pyqtgraph (conflicts with matplotlib) data
use_qt_plot=False
if use_qt_plot==True:
    import pyqtgraph as pg

colors={}
colors['correct_amp_good_HP']=(0, 128, 0) #green
colors['correct_amp_bad_HP']=(138,43,226) #purple
colors['wrong_amp_good_HP']=(0,191,255) #cyan
colors['wrong_amp_bad_HP']=(255, 0, 0) #red


def measure_amp(trace, min_or_max, baseline=(6e-3, 8e-3), response=(13e-3, 17e-3)):
    '''get the max or min of the data in the trace object and subtract out the baseline
    at the specified times
    '''
    baseline = trace.time_slice(*baseline).data.mean()
    if min_or_max=='max':
        individual_first_peaks = trace.time_slice(*response).data.max()
    elif min_or_max=='min':
        individual_first_peaks = trace.time_slice(*response).data.min()
    else:
        raise Exception('Are you looking for min or max?')
    return individual_first_peaks - baseline

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
    
def check_synapse(expt, cells):
    '''checks if a synapse meets the requirements and if so, it appends it to the synapse 
    dictionary.  A synapses list must be initialized before calling this function
    inputs:
        expt: object
            object obtained from cached_experiments.connection_summary[*]['expt']
        cells: object
            object obtained from cached_experiments.connection_summary[*]['cells']    
    output:
        returns nothing but appends info to the synapse list
    '''
    try: #needed because pyqt is breaking on datetime sometimes
        if expt.expt_info['solution']=='2mM Ca & Mg':
            synapses.append((expt, cells[0].cell_id, cells[1].cell_id))
    except:
        pass
 

 
if __name__ == '__main__':

    if use_qt_plot == True:
        app = pg.mkQApp()
        pg.dbg()

    # load file that specifies ids to use (made by first_pulse_feature.py code)
    file=open('/home/corinnet/workspace/aiephys/multipatch_analysis/analyses/pulse_expt_ids.pkl')
    Stephs_data=pickle.load(file)
    file.close()
    Steph_uids=np.array([])
    for key in Stephs_data.keys():
        print(key)
        Steph_uids=np.append(Steph_uids, np.array([d[2] for d in Stephs_data[key]]))

    # Load experiment index
    expts = cached_experiments()
#    expts.select(calcium='high')  #this is throwing datetime errors

    # connections to test
    connection_list=[['rorb', 'rorb'],
                     ['tlx3', 'tlx3'],
                     ['ntsr1', 'ntsr1'],
                     ['L23pyr', 'L23pyr'],
                     ['sim1','sim1']]

    dictionary={}
    
    # for each type of connection, i.e. ntsr1 to ntsr1, or L23pyr to L23pyr etc.
    for synapic_pairs in connection_list: 
        print(synapic_pairs)
        #----------adding suitable synapses from experiment cache to list of synapses to analyze-----------
        synapses = []
        uid_skip=[]
        for connection in expts.connection_summary(): # expts.connection_summary() is list of dictionaries
            cells = connection['cells'] #(pre, post) synaptic "Cell" objects
            expt = connection['expt'] #"Experiment" object
            print (expt.uid) #unique id for experiment
            
            # skip any experiment that is not in the predefined list
            if expt.uid == '1523470754.85':
                pass
            if expt.uid not in Steph_uids:
                uid_skip.append(expt.uid)
                print('here skipping ')
                continue
            
            # if the pre and post synaptic cells are the same type:
            #    and the experiment passes 
            #    add this connection to the list of synapses to analyze  
            pre_synaptic=synapic_pairs[0]
            post_synaptic=synapic_pairs[1]
            
            # L23pyr and cre lines kept in a different key so two equalities are needed
            if pre_synaptic=='L23pyr':
                if cells[0].target_layer=='2/3' and cells[1].target_layer=='2/3':
                    check_synapse(expt, cells) #this adds the synapse to analysis list if passes inner criteria
            else:
                if cells[0].cre_type == pre_synaptic and cells[1].cre_type == post_synaptic: #this adds the synapse to analysis list if passes inner criteria
                    check_synapse(expt, cells)
        #------------------------------------------------------------------------------------            
        title_str= pre_synaptic+' to '+post_synaptic  #used in plotting and output dictionary
        
        # setup qt plotting
        if use_qt_plot==True: 
            time_vs_psp_plot = pg.plot(labels={'left': 'individual_first_peaks of synaptic deflection (V)', 
                                'bottom': 'sec_since_t0 since first recorded synapse (s)', 
                                'top':(title_str+' connections: progression of synaptic defection over an experiment')})    
            ave_psp_plot = pg.plot(labels={'top':('average individual_first_baselines-line subtracted first pulse synaptic deflection ('+ title_str+ ')'), 
                              'bottom': 'sec_since_t0 (s)', 
                              'left':'voltage (V)'}) 
            sweep_vs_psp_plot = pg.plot(labels={'left': 'individual_first_peaks of synaptic deflection (V)', 
                                'bottom': 'sweep number', 
                                'top':(title_str+' connections: progression of synaptic defection over an experiment')})  
        
        

        
        # initializing output dictionary
        slopes={}
        slopes['sec_since_t0']={}
        slopes['sweep_numbers']={}
        slopes['sec_since_t0']['correct_amp_good_HP']=[]
        slopes['sec_since_t0']['correct_amp_bad_HP']=[]
        slopes['sec_since_t0']['wrong_amp_good_HP']=[]
        slopes['sec_since_t0']['wrong_amp_bad_HP']=[]
        slopes['sweep_numbers']['correct_amp_good_HP']=[]
        slopes['sweep_numbers']['correct_amp_bad_HP']=[]
        slopes['sweep_numbers']['wrong_amp_good_HP']=[]
        slopes['sweep_numbers']['wrong_amp_bad_HP']=[]
        intercepts={}
        intercepts['sec_since_t0']={}
        intercepts['sweep_numbers']={}
        intercepts['sec_since_t0']['correct_amp_good_HP']=[]
        intercepts['sec_since_t0']['correct_amp_bad_HP']=[]
        intercepts['sec_since_t0']['wrong_amp_good_HP']=[]
        intercepts['sec_since_t0']['wrong_amp_bad_HP']=[]
        intercepts['sweep_numbers']['correct_amp_good_HP']=[]
        intercepts['sweep_numbers']['correct_amp_bad_HP']=[]
        intercepts['sweep_numbers']['wrong_amp_good_HP']=[]
        intercepts['sweep_numbers']['wrong_amp_bad_HP']=[]
        f_pvalues={}
        f_pvalues['sec_since_t0']={}
        f_pvalues['sweep_numbers']={}
        f_pvalues['sec_since_t0']['correct_amp_good_HP']=[]
        f_pvalues['sec_since_t0']['correct_amp_bad_HP']=[]
        f_pvalues['sec_since_t0']['wrong_amp_good_HP']=[]
        f_pvalues['sec_since_t0']['wrong_amp_bad_HP']=[]
        f_pvalues['sweep_numbers']['correct_amp_good_HP']=[]
        f_pvalues['sweep_numbers']['correct_amp_bad_HP']=[]
        f_pvalues['sweep_numbers']['wrong_amp_good_HP']=[]
        f_pvalues['sweep_numbers']['wrong_amp_bad_HP']=[]
        max_residuals={}
        max_residuals['sec_since_t0']={}
        max_residuals['sweep_numbers']={}
        max_residuals['sec_since_t0']['correct_amp_good_HP']=[]
        max_residuals['sec_since_t0']['correct_amp_bad_HP']=[]
        max_residuals['sec_since_t0']['wrong_amp_good_HP']=[]
        max_residuals['sec_since_t0']['wrong_amp_bad_HP']=[]
        max_residuals['sweep_numbers']['correct_amp_good_HP']=[]
        max_residuals['sweep_numbers']['correct_amp_bad_HP']=[]
        max_residuals['sweep_numbers']['wrong_amp_good_HP']=[]
        max_residuals['sweep_numbers']['wrong_amp_bad_HP']=[]

        # initialize lists that will be and array for each synapse
        raw=[]
        filtered=[]
        time_list=[]
        sweep_number_list=[]
        PSPs_amp_start=[]
        PSPs_amp_ave=[]
        PSPs_amp_end=[]
        length_of_experiment=[]
        
        num_of_synapses=0
        # for each individual synapse
        for i,syn in enumerate(synapses): #[expt, pre_id, post_id], created by check synapse
            expt, pre_id, post_id = syn
            analyzer = DynamicsAnalyzer(expt, pre_id, post_id, align_to='spike')
            
            # collect all first pulse responses
            amp_responses = analyzer.amp_group
            if len(amp_responses) == 0:
                print("Skipping %s %d %d; no responses" % (expt.uid, pre_id, post_id))
                continue               
            
# some other options to potentially be able to choose from            
#            responses = analyzer.train_responses
#            pulse_offset = analyzer.pulse_offsets
#            response = analyzer.pulse_responses
    
    #        plt.figure()
    #        for trace in amp_responses.responses:
    #            plt.plot(trace.time_values, trace.data)
    #            plt.title('responses')
    #        plt.figure()
    #        for trace in amp_responses.baselines:
    #            plt.plot(trace.time_values, trace.data)
    #            plt.title('baselines')         
    #        plt.show(block=False)
    
            # figure out whether the trough or individual_first_peaks of the average synaptic trace is bigger and if that corresponds to the excitation of the neurons.  
            # i.e. if it is an excitatory synapse we would expect the max defection to be positive
            average = amp_responses.bsub_mean() #returns average synaptic response with the average baseline subtracted
            max_peak = measure_amp(average, min_or_max='max', baseline=(6e-3, 8e-3), response=(12e-3, 16e-3))
            min_peak = measure_amp(average, min_or_max='min', baseline=(6e-3, 8e-3), response=(12e-3, 16e-3))
            ave_deflection=max(abs(max_peak), abs(min_peak)) #individual_first_peaks of average trace
            max_min = "max" if abs(max_peak)> abs(min_peak) else "min"  #find whether the individual_first_peaks or trough of the first pulse average is larger 
            correct_syn_amp_dir = True
            if max_min == "min" and expt.cells[pre_id].cre_type in EXCITATORY_CRE_TYPES: 
                print ("Whoa this synapse looks inhibitory when cre line would say it should be excitatory!!!" )  
                correct_syn_amp_dir = False
            if max_min == "max" and expt.cells[pre_id].cre_type in INHIBITORY_CRE_TYPES: 
                print ("Whoa this synapse looks excitatory when cre line would say it should be inhibitory!!!" )    
                correct_syn_amp_dir = False  
       
            # find the individual_first_peaks or trough of every potential event and plot their amplitude over sec_since_t0 of the experiment
            individual_first_peaks=[]
            individual_first_baselines=[]
            individual_start_times=[]
            individual_holding_potentials=[]
            sweep_numbers=[]
            ordered=sorted(amp_responses.responses, key=lambda rr:rr.start_time) #order the traces by individual_start_times during the experiment
            for jj, rr in enumerate(ordered):
                individual_first_peaks.append(measure_amp(rr, min_or_max=max_min, baseline=(6e-3, 8e-3), response=(12e-3, 16e-3)))
                individual_first_baselines.append(measure_amp(rr, min_or_max=max_min, baseline=(0e-3, 2e-3), response=(6e-3, 10e-3)))
                individual_start_times.append(rr.start_time)  
                individual_holding_potentials.append(rr.parent.parent.holding_potential)
                sweep_numbers.append(float(rr.parent.parent._sweep._sweep_id))
                #print ('for each first pulse of a synapse: individual_first_peaks', individual_first_peaks[-1], 'individual_first_baselines', individual_first_baselines[-1], 'individual_start_times', individual_start_times[-1], 'holding potential', individual_holding_potentials[-1], 'sweep_numbers', sweep_numbers[-1])      

    #        for trace in amp_responses.responses:
    #            plt.plot(trace.time_values, trace.data)
    #            plt.title('responses')
    #        plt.figure()
    #        for trace in amp_responses.baselines:
    #            plt.plot(trace.time_values, trace.data)
    #            plt.title('baselines')         
    #        plt.show()
    
            # check if holding potential is within a desired range
            holding=np.mean(individual_holding_potentials) # average holding potential across plots
            #print ('holding potential is', holding)
            if holding>-0.072 and holding<-0.068:
                holding_good_flag=True
            else:
                holding_good_flag=False
            #print ('\tholding potential flag set to ', holding_good_flag)

            mean_base=np.mean(individual_first_baselines) # average individual_first_baselines across pulses of a synapse
            sweep_numbers=np.array(sweep_numbers)
            sec_since_t0=np.array(individual_start_times) - individual_start_times[0] # remap individual_start_times basis to be in reference to start of experiment
            sec_since_t0=np.array([td.total_seconds() for td in sec_since_t0])
            #TODO: !!!!from what I can tell, individual_first_peaks are already subtracting there individual baselines so I am not sure why it is being resubtracted below!!!
            peak_minus_base_average=np.array(individual_first_peaks)-mean_base # take each individual_first_peaks and put it in reference to the average individual_first_baselines
            smoothed=ndi.gaussian_filter(peak_minus_base_average, 2) # 
            t_slope, t_intercept, _,_,_=stats.linregress(sec_since_t0, peak_minus_base_average)
            sn_slope, sn_intercept, _,_,_=stats.linregress(sweep_numbers, peak_minus_base_average)
            
            # can look at the output of the f-test in statsmodels
            # https://blog.datarobot.com/ordinary-least-squares-in-python
            # actually states what the summary numbers mean
            # https://en.wikipedia.org/wiki/Lack-of-fit_sum_of_squares
            # does a reasonable job of reporting what the numbers mean
            sm_lr=sm.OLS(peak_minus_base_average, sm.add_constant(sec_since_t0)) #Note statsmodels takes x,y in reverse order
            out=sm_lr.fit()
            t_f_pvalue=out.f_pvalue
            res=out.predict()
            t_max_residual=np.max([np.max(res), np.absolute(np.min(res))])
            
            sm_lr=sm.OLS(peak_minus_base_average, sm.add_constant(sweep_numbers)) #Note statsmodels takes x,y in reverse order
            out=sm_lr.fit()
            sn_f_pvalue=out.f_pvalue
            res=out.predict()
            sn_max_residual=np.max([np.max(res), np.absolute(np.min(res))])
            
            # Slope is very small here due to small amplitude.
            #    If voltage is fit in mV it will make scatter and 
            #    slope bigger.  Will f_pvalue stay the same?
            #print(out.summary())
            plt.figure(figsize=(16, 10))
            plt.subplot(3,1,1)
            plt.scatter(sec_since_t0, peak_minus_base_average)
            plt.plot(sec_since_t0, out.predict())
            plt.ylim([min(peak_minus_base_average), max(peak_minus_base_average)])
            plt.xlabel('time (s)')
            plt.ylabel('amplitude (V)')
            plt.title(expt.uid+'_'+str(pre_id)+'_'+str(post_id)+' fit, slope:'+str(t_slope)+', intercept:'+str(t_intercept)+'\n, f_pvalue: '+str(t_f_pvalue)+', max. res.:'+str(t_max_residual))
            plt.subplot(3,1,2)
            residuals=peak_minus_base_average-out.predict()
            plt.scatter(sec_since_t0, residuals) #plotting residuals
            plt.ylim([min(residuals), max(residuals)])
            plt.title('residuals')
            plt.xlabel('time (s)')
            plt.ylabel('residual (V)')
            plt.subplot(3,1,3)
            plt.hist(residuals)
            plt.title('residuals')
            plt.ylabel('n')
            plt.xlabel('residuals (V)')    
            plt.tight_layout()
            plt.savefig(os.path.join('/home/corinnet/Desktop/human_rundown', expt.uid+'_'+str(pre_id)+'_'+str(post_id)+'.png'))
            plt.close()
            
            def update_plots_and_dicts(qc_key):
                '''updates and values for different qc groupings
                inputs: string
                    options: 'correct_amp_good_HP','correct_amp_bad_HP', 'wrong_amp_good_HP', 'wrong_amp_bad_HP'
                '''
                if use_qt_plot==True:
                    ave_psp_plot.plot(average.time_values, average.data, pen=pg.mkPen(color=colors[qc_key])) #plot average of first pulse in each epoch of spikes of individual synapses
    #                time_vs_psp_plot.plot(np.array(sec_since_t0), smoothed, pen=pg.mkPen(color=colors[qc_key])) # (i, len(synapses)*1.3))
                    time_vs_psp_plot.plot(np.array(sec_since_t0), peak_minus_base_average, pen=pg.mkPen(color=colors[qc_key])) # (i, len(synapses)*1.3))
                    time_vs_psp_plot.plot(np.array(sec_since_t0), t_slope*np.array(sec_since_t0)+t_intercept, pen=pg.mkPen(color=colors[qc_key], style=pg.QtCore.Qt.DashLine)) # (i, len(synapses)*1.3))
    #                sweep_vs_psp_plot.plot(sweep_numbers, smoothed, pen=pg.mkPen(color=colors[qc_key]))
                    sweep_vs_psp_plot.plot(sweep_numbers, peak_minus_base_average, pen=pg.mkPen(color=colors[qc_key]))
                    sweep_vs_psp_plot.plot(sweep_numbers, sn_slope*sweep_numbers+sn_intercept, pen=pg.mkPen(color=colors[qc_key],style=pg.QtCore.Qt.DashLine)) # (i, len(synapses)*1.3))
                slopes['sweep_numbers'][qc_key].append(sn_slope)
                slopes['sec_since_t0'][qc_key].append(t_slope)
                intercepts['sweep_numbers'][qc_key].append(sn_intercept)
                intercepts['sec_since_t0'][qc_key].append(t_intercept)      
                f_pvalues['sweep_numbers'][qc_key].append(sn_f_pvalue)
                f_pvalues['sec_since_t0'][qc_key].append(t_f_pvalue)        
                max_residuals['sweep_numbers'][qc_key].append(sn_max_residual)
                max_residuals['sec_since_t0'][qc_key].append(t_max_residual)        
            
            #TODO: check to see if this is a bug to only record in the optimal state.
            # record values for different qc states
            if correct_syn_amp_dir == True and holding_good_flag ==True:
                print('recording synapse')
                #these are appending array to be used futher before dictionary output
                time_list.append(sec_since_t0)  
                filtered.append(smoothed)   
                raw.append(peak_minus_base_average) 
                sweep_number_list.append(sweep_numbers) 
                
                #following appends a single value to a list and gets used in output dictionary
                num_of_synapses=num_of_synapses+1 
                update_plots_and_dicts('correct_amp_good_HP')
                PSPs_amp_start.append(peak_minus_base_average[0])
                PSPs_amp_ave.append(np.mean(peak_minus_base_average))
                PSPs_amp_end.append(peak_minus_base_average[-1])
                length_of_experiment.append(sec_since_t0[-1])
         
            else: 
                if correct_syn_amp_dir==True and holding_good_flag==False:
                    update_plots_and_dicts('correct_amp_bad_HP')
                elif correct_syn_amp_dir==False and holding_good_flag==True:
                    update_plots_and_dicts('wrong_amp_good_HP')

                elif correct_syn_amp_dir==False and holding_good_flag==False:
                    update_plots_and_dicts('wrong_amp_bad_HP')
                else:
                    print(correct_syn_amp_dir)
                    print(holding_good_flag)
                    raise Exception("This flag combo doesn't exist")
            
            print('done with one synapse')
            if use_qt_plot == True:
                app.processEvents()

        
        #because times of events aren't all at the same sec_since_t0, sec_since_t0 binning is needed to get average sec_since_t0 course
        time_points, time_avg_data, time_std_err=average_via_bins(time_list, raw, bin_size=60)
        if use_qt_plot == True:
            time_vs_psp_plot.plot(time_points, time_avg_data, pen=pg.mkPen(color='w', width=5)) #plots average of the data
        
        sweeps, sweep_avg_data, sweep_std_err=average_via_bins(sweep_number_list, raw, bin_size=5)
        if use_qt_plot == True:
            sweep_vs_psp_plot.plot(sweeps, sweep_avg_data, pen=pg.mkPen(color='w', width=5)) #plots average of the data
        
        dictionary[title_str]={'time_points': time_points, 
                               'time_avg_data':time_avg_data, 
                               'time_std_err':time_std_err, 
                               'num_of_synapses':num_of_synapses,
                               'sweeps':sweeps,
                               'sweep_avg_data':sweep_avg_data,
                               'sweep_std_err':sweep_std_err,
                               'slopes':slopes,
                               'intercepts':intercepts,
                               'PSPs_amp_start':PSPs_amp_start,
                               'PSPs_amp_ave':PSPs_amp_ave,
                               'PSPs_amp_end':PSPs_amp_end,
                               'length_of_experiment':length_of_experiment,
                               'uids_skipped':np.unique(uid_skip),
                               'f_pvalues':f_pvalues,
                               'max_residuals':max_residuals}
        
    ju.write("PSP_vs_time_output_data/7_30_2018.json", dictionary)
    if not os.path.isdir("PSP_vs_time_output_data"):
        os.mkdir("PSP_vs_time_output_data")
    ju.write("PSP_vs_time_output_data/6_13_2017_human.json", dictionary)
    

    plt.figure()
    for key in dictionary.keys():
        plt.errorbar(dictionary[key]['time_points'], dictionary[key]['time_avg_data'],  yerr=dictionary[key]['time_std_err'], label=key+', n='+str(dictionary[key]['num_of_synapses']))
    plt.title('average individual_first_baselines-line subtracted first pulse synaptic deflection')
    plt.legend(loc=4)
    plt.ylabel('voltage (V)')
    plt.xlabel('sec_since_t0 since first recorded synapse (s)')

    plt.figure()
    for key in dictionary.keys():
        plt.errorbar(dictionary[key]['sweeps'], dictionary[key]['sweep_avg_data'],  yerr=dictionary[key]['sweep_std_err'], label=key+', n='+str(dictionary[key]['num_of_synapses']))
    plt.title('average individual_first_baselines-line subtracted first pulse synaptic deflection')
    plt.legend(loc=4)
    plt.ylabel('voltage (V)')
    plt.xlabel('sweep number')    
    
    
    plt.show(block=False)

#        app.processEvents()    
#        pg.QtGui.QApplication.exec_()  
    
    plt.show()
        
    
  
    
