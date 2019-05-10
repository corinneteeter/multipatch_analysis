"""Create data point and click.
"""

import multipatch_analysis.database as db
import multipatch_analysis.connection_strength as cs 
import pandas
#import matplotlib.pyplot as plt 
import numpy as np
import pyqtgraph as pg
from neuroanalysis.data import Trace, TraceList
pg.mkQApp()

def get_point(scatterplot, points):
    """Function called by *scatter.sigClicked.connect*.
    Plot the data and corresponding fit waveform associated 
    with the selected data point. I am not sure how *scatterplot*
    and *points* are magically being fed into function.
    Input
    -----
    scatterplot: pyqtgraph.graphicsItems.ScatterPlotItem.ScatterPlotItem object
        object corresponding to the scatter plot selecting from (not sure why
        this is necessary)?
    points: list of pyqtgraph.graphicsItems.ScatterPlotItem.SpotItem objects
        the objects correspond to the points selected from the graph
    
    """

    if len(points) != 1:
        print(len(points),'points selected.  Only plotting first one')
    afpf, pair = points[0].data()  #this accesses the data binded in *setData* for the selected points
    return afpf, pair

# def plot_afpf_trace():


#     # create the plot window and add the waveforms to it
#     pre_plot = pg.plot()
#     post_plot = pg.plot()
#     cut_plot = pg.plot()
    

#         # plot post synaptic data
#         PCI = pg.PlotCurveItem(post_trace.time_values*1.e3, post_trace.data*1.e3, pen=.5)
#         post_plot.addItem(PCI)

#         # plot pre synaptic data                                                                                                                                                                          
#         PCI = pg.PlotCurveItem(pre_trace.time_values*1.e3, pre_trace.data*1.e3, pen=.5)
#         pre_plot.addItem(PCI)

#     pre_plot.setLabels(bottom='time (mS)', left='voltage (mV)',
#                         top=('Pre Synaptic Spikes: %.3f, %i to %i, n=%i, %s to %s' % (pair.experiment.acq_timestamp,
#                                                             pair.pre_cell.ext_id,
#                                                             pair.post_cell.ext_id,
#                                                             len(afpf.ic_pulse_ids),
#                                                             pair.pre_cell.cre_type,
#                                                             pair.post_cell.cre_type)))
        

#     # add average data and fit 
#     data_trace = afpf.ic_avg_psp_data
#     data_trace = Trace(data=(data_trace), sample_rate=db.default_sample_rate)
#     data_trace = pg.PlotCurveItem(data_trace.time_values*1.e3, data_trace.data*1.e3)
#     data_trace.setPen('y', width=4)
    
#     fit_trace = afpf.ic_avg_psp_fit
#     fit_trace = Trace(data=(fit_trace), sample_rate=db.default_sample_rate)
#     fit_trace = pg.PlotCurveItem(fit_trace.time_values*1.e3, fit_trace.data*1.e3, pen='r')
#     fit_trace.setPen('r', width=4)

#     post_plot.addItem(data_trace)
#     post_plot.addItem(fit_trace)
#     post_plot.setLabels(bottom='time (mS)', left='voltage (mV)', 
#                         top=('Post Synaptic Response: %.3f, %i to %i, nrmse=%.1f, amp=%.2f, n=%i, %s to %s' % (pair.experiment.acq_timestamp, 
#                                                             pair.pre_cell.ext_id,
#                                                             pair.post_cell.ext_id,
#                                                             afpf.ic_nrmse, 
#                                                             afpf.ic_amp*1.e3, 
#                                                             len(afpf.ic_pulse_ids),
#                                                             pair.pre_cell.cre_type,
#                                                             pair.post_cell.cre_type)))

# #         point.setBrush(pg.mkBrush('y'))
# #         point.setSize(15)
# #   trace.setPen('y', width=2)
# # print('Clicked:' '%s' % pair)    

def get_pulses(afpf, pair):
    # Get the individual plots that went into this data
    s = db.Session()
    pulse_cut_starts = []
    pulse_cut_ends = []
    pre_syn_traceList = []
    post_syn_traceList = []
    for pulse_id in afpf.ic_pulse_ids:
        pulse_response = s.query(db.PulseResponse).join(db.Pair).filter(
            db.PulseResponse.stim_pulse_id == pulse_id).filter(
                db.Pair.id == pair.id).all()
        
        if len(pulse_response) != 1:
            raise Exception('There should only be one response per pulse_id')
        
        pulse_response= pulse_response[0]  #grap the pulse reponse out of the list
        print(pulse_response.id) 
        
        start_time = pulse_response.start_time  
        spike_time = pulse_response.stim_pulse.spikes[0].max_dvdt_time 
        pulse_onset = pulse_response.stim_pulse.onset_time 
        pulse_duration = pulse_response.stim_pulse.duration
                
        time_before_spike = 10.e-3  
        t0 = start_time - spike_time + time_before_spike 
            
        pulse_cut_start = pulse_onset - spike_time + time_before_spike
        pulse_cut_end = pulse_cut_start + pulse_duration

        pre_syn_traceList.append(Trace(data=pulse_response.stim_pulse.data, t0=t0, sample_rate=db.default_sample_rate).time_slice(start=0, stop=None))
        # pre_trace_cut = copy.deepcopy(pre_trace)
        # pre_trace_cut.time_slice(pulse_cut_start, pulse_cut_end).data[:] = np.nan
        
        post_syn_traceList.append(Trace(data=pulse_response.data, t0=t0, sample_rate=db.default_sample_rate).time_slice(start=0, stop=None)) 
        # post_trace_cut = copy.deepcopy(post_trace)
        # post_trace_cut.time_slice(pulse_cut_start, pulse_cut_end).data[:] = np.nan

        pulse_cut_starts.append(pulse_cut_start)
        pulse_cut_ends.append(pulse_cut_end)
    



session=db.Session()
out = session.query(db.AvgFirstPulseFit, db.Pair).join(
    db.Pair).filter(db.Pair.synapse == True
    ).all()
#    filter(db.AvgFirstPulseFit.ic_nrmse < 1).all()
#afpfs=session.query(db.SingleFirstPulseFit).join(db.PulseResponse).join(db.Pair).filter(db.Pair.synapse == True).all()

nrmse = []
amp = []
pre_cre = []
post_cre = []
cre_meld = []
for afpf, pair in out:
    nrmse.append(afpf.ic_nrmse)
    amp.append(afpf.ic_amp)
scatter = pg.ScatterPlotItem(symbol='o', brush='b', pen='w', size=12) #create scatter plot item
# note that data will be refered to as points in the referenced function
scatter.setData(np.array(amp)*1.e3, nrmse, data=out) #set x, y, and data associated with the scatter points

# Make clickable scatter plot
s_plot=pg.plot() #create the plot window
s_plot.addItem(scatter) #add the scatter plot data from above
s_plot.setLabels(left='nrmse', bottom = 'fit amplitude (mV)')
print(len(out))

afpf, pair = scatter.sigClicked.connect(get_point)

print(afpf, pair)
