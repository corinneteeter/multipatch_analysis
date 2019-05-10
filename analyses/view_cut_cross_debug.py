import multipatch_analysis.database as db
import multipatch_analysis.connection_strength as cs 
import pandas
#import matplotlib.pyplot as plt 
import numpy as np
import pyqtgraph as pg
from neuroanalysis.data import Trace, TraceList
import copy
pg.mkQApp()
    
pre_plot = pg.plot()
post_plot = pg.plot()
cut_plot = pg.plot()


s=db.Session()
out = s.query(db.AvgFirstPulseFit, db.Pair).join(
    db.Pair).join(db.Experiment).filter(db.Experiment.acq_timestamp==1547248417.685
    ).filter(db.Pair.pre_cell_id==18385).filter(db.Pair.post_cell_id==18383).all()

afpf, pair = out[0]

# Get the individual plots that went into this data
synapse_type = pair.connection_strength.synapse_type
avg_cut_post_response = []
pulse_cut_starts = []
pulse_cut_ends = []
for pulse_id in afpf.ic_pulse_ids:
    pulse_response = s.query(db.PulseResponse).join(db.Pair).filter(
        db.PulseResponse.stim_pulse_id == pulse_id).filter(
            db.Pair.id == pair.id).all()
    
    if len(pulse_response) != 1:
        raise Exception('There should only be one response per pulse_id')
    
    pulse_response= pulse_response[0]  #grap the pulse reponse out of the list
    
    start_time = pulse_response.start_time  
    spike_time = pulse_response.stim_pulse.spikes[0].max_dvdt_time 
    pulse_onset = pulse_response.stim_pulse.onset_time 
    pulse_duration = pulse_response.stim_pulse.duration
            
    time_before_spike = 10.e-3  
    t0 = start_time - spike_time + time_before_spike 
        
    pulse_cut_start = pulse_onset - spike_time + time_before_spike
    pulse_cut_end = pulse_cut_start + pulse_duration

    pre_trace = Trace(data=pulse_response.stim_pulse.data, t0=t0, sample_rate=db.default_sample_rate).time_slice(start=0, stop=None)
    pre_trace_cut = copy.deepcopy(pre_trace)
    pre_trace_cut.time_slice(pulse_cut_start, pulse_cut_end).data[:] = np.nan
    
    post_trace = Trace(data=pulse_response.data, t0=t0, sample_rate=db.default_sample_rate).time_slice(start=0, stop=None) 
    post_trace_cut = copy.deepcopy(post_trace)
    post_trace_cut.time_slice(pulse_cut_start, pulse_cut_end).data[:] = np.nan

    # plot pre synaptic data                                                                                                                                                                          
    PCI = pg.PlotCurveItem(pre_trace.time_values*1.e3, pre_trace.data*1.e3, pen=.5)
    pre_plot.addItem(PCI)
    PCI = pg.PlotCurveItem(pre_trace_cut.time_values*1.e3, pre_trace_cut.data*1.e3, pen='g')
    pre_plot.addItem(PCI)

    # plot post synaptic data
    PCI = pg.PlotCurveItem(post_trace.time_values*1.e3, post_trace.data*1.e3, pen=.5)
    post_plot.addItem(PCI)
    PCI = pg.PlotCurveItem(post_trace_cut.time_values*1.e3, post_trace_cut.data*1.e3, pen='g')
    post_plot.addItem(PCI)

    pulse_cut_starts.append(pulse_cut_start)
    pulse_cut_ends.append(pulse_cut_end)
    avg_cut_post_response.append(post_trace_cut)

    
pre_plot.setLabels(bottom='time (mS)', left='voltage (mV)',
                    top=('Pre Synaptic Spikes: %.3f, %i to %i, n=%i, %s to %s' % (pair.experiment.acq_timestamp,
                                                        pair.pre_cell.ext_id,
                                                        pair.post_cell.ext_id,
                                                        len(afpf.ic_pulse_ids),
                                                        pair.pre_cell.cre_type,
                                                        pair.post_cell.cre_type)))

# add average data and fit to plot

# this doesnt work
# avg_post_cut_trace = TraceList(avg_cut_post_response).mean()
# avg_post_cut_trace = pg.PlotCurveItem(avg_post_cut_trace.time_values*1.e3, avg_post_cut_trace.data*1.e3)
# avg_post_cut_trace.setPen('b', width=4)

data_trace = afpf.ic_avg_psp_data
data_trace = Trace(data=(data_trace), sample_rate=db.default_sample_rate)
cut_data_trace = copy.deepcopy(data_trace)
cut_data_trace.time_slice(min(pulse_cut_starts), max(pulse_cut_ends)).data[:] = np.nan

data_trace = pg.PlotCurveItem(data_trace.time_values*1.e3, data_trace.data*1.e3)
data_trace.setPen('y', width=4)

cut_data_trace = pg.PlotCurveItem(cut_data_trace.time_values*1.e3, cut_data_trace.data*1.e3)
cut_data_trace.setPen('b', width=4)

fit_trace = afpf.ic_avg_psp_fit
fit_trace = Trace(data=(fit_trace), sample_rate=db.default_sample_rate)
fit_trace = pg.PlotCurveItem(fit_trace.time_values*1.e3, fit_trace.data*1.e3, pen='r')
fit_trace.setPen('r', width=4)

post_plot.addItem(avg_post_cut_trace)
post_plot.addItem(data_trace)
post_plot.addItem(fit_trace)
post_plot.setLabels(bottom='time (mS)', left='voltage (mV)', 
                    top=('Post Synaptic Response: %.3f, %i to %i, nrmse=%.1f, amp=%.2f, n=%i, %s to %s' % (pair.experiment.acq_timestamp, 
                                                        pair.pre_cell.ext_id,
                                                        pair.post_cell.ext_id,
                                                        afpf.ic_nrmse, 
                                                        afpf.ic_amp*1.e3, 
                                                        len(afpf.ic_pulse_ids),
                                                        pair.pre_cell.cre_type,
                                                        pair.post_cell.cre_type)))