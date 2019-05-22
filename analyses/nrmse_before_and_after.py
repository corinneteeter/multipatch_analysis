"""plot nrmse before and after altering weighting to not use region where any of the sweeps might have artifact."""

import multipatch_analysis.database as db
import multipatch_analysis.connection_strength as cs 
import pandas
#import matplotlib.pyplot as plt 
import numpy as np
import pyqtgraph as pg
from neuroanalysis.data import Trace, TraceList
pg.mkQApp()
from weighted_error_lib import *


def plot_individual_responses(pulse_ids, pair, plot):
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

        # plot post synaptic data
        PCI = pg.PlotCurveItem(post_trace.time_values*1.e3, post_trace.data*1.e3, pen=.5)
        plot.addItem(PCI)

        # # plot pre synaptic data                                                                                                                                                                          
        # PCI = pg.PlotCurveItem(pre_trace.time_values*1.e3, pre_trace.data*1.e3, pen=.5)
        # pre_plot.addItem(PCI)

def plot_afpf_trace(scatterplot, points):
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
    afpf_before, afpf_after, xx, pair = points[0].data()  #this accesses the data binded in *setData* for the selected points

    # create the plot window and add the waveforms to it
    post_plot = pg.plot()
    #pre_plot = pg.plot()
    
    # Get the individual plots that went into this data
    s = db.Session()
    synapse_type = pair.connection_strength.synapse_type
    s.close()

    # note that I am only doing this once since they should be the case in both instances
    plot_individual_responses(afpf_before.ic_pulse_ids, pair, post_plot)

    # pre_plot.setLabels(bottom='time (mS)', left='voltage (mV)',
    #                     top=('Pre Synaptic Spikes: %.3f, %i to %i, n=%i, %s to %s' % (pair.experiment.acq_timestamp,
    #                                                         pair.pre_cell.ext_id,
    #                                                         pair.post_cell.ext_id,
    #                                                         len(afpf_before.ic_pulse_ids),
    #                                                         pair.pre_cell.cre_type,
    #                                                         pair.post_cell.cre_type)))
        

    # add average data and fits before and after the update

    # data average should be the same before and after updates 
    data_trace = afpf_before.ic_avg_psp_data
    data_trace = Trace(data=(data_trace), sample_rate=db.default_sample_rate)
    data_trace = pg.PlotCurveItem(data_trace.time_values*1.e3, data_trace.data*1.e3)
    data_trace.setPen('y', width=4)
    
    fit_trace_before = afpf_before.ic_avg_psp_fit
    fit_trace_before = Trace(data=(fit_trace_before), sample_rate=db.default_sample_rate)
    fit_trace_before = pg.PlotCurveItem(fit_trace_before.time_values*1.e3, fit_trace_before.data*1.e3)
    fit_trace_before.setPen('r', width=4)

    fit_weight_before = afpf_before.ic_weight
    fit_weight_before = Trace(data=fit_weight_before, sample_rate=db.default_sample_rate)
    fit_weight_before = pg.PlotCurveItem(fit_weight_before.time_values*1.e3, (fit_weight_before.data/30.)*abs(afpf_before.ic_measured_amp*1.e3) + afpf_before.ic_measured_baseline*1.e3)
    fit_weight_before.setPen('r', width=1)

    fit_trace_after = afpf_after.ic_avg_psp_fit
    fit_trace_after = Trace(data=(fit_trace_after), sample_rate=db.default_sample_rate)
    fit_trace_after = pg.PlotCurveItem(fit_trace_after.time_values*1.e3, fit_trace_after.data*1.e3)
    fit_trace_after.setPen('g', width=4)

    fit_weight_after = afpf_after.ic_weight
    fit_weight_after = Trace(data=fit_weight_after, sample_rate=db.default_sample_rate)
    fit_weight_after = pg.PlotCurveItem(fit_weight_after.time_values*1.e3, (fit_weight_after.data/30.)*abs(afpf_before.ic_measured_amp*1.e3) + afpf_before.ic_measured_baseline*1.e3)
    fit_weight_after.setPen('g', width=1)

    post_plot.addItem(data_trace)
    post_plot.addItem(fit_trace_before)
    post_plot.addItem(fit_trace_after)
    post_plot.addItem(fit_weight_before)
    post_plot.addItem(fit_weight_after)
    post_plot.setLabels(bottom='time (mS)', left='voltage (mV)', 
                        top=('Post Synaptic: \n%.3f, %i to %i, nrmse_before=%.1f, nrmse_after=%.2f, %s to %s' % (pair.experiment.acq_timestamp, 
                                                            pair.pre_cell.ext_id,
                                                            pair.post_cell.ext_id,
                                                            afpf_before.ic_nrmse, 
                                                            afpf_after.ic_nrmse, 
                                                            pair.pre_cell.cre_type,
                                                            pair.post_cell.cre_type)))

 
# Dp query

# db.Session()
# pair=db.experiment_from_timestamp(1547248417.685).pairs[(2, 7)]
# afpf = afpf.fit_first_pulses(pair)
# print(afpf)

session=db.Session()
out = session.query(db.AvgFirstPulseFit, db.AvgFirstPulseFit2, db.AvgFirstPulseFit3, db.Pair).join(
    db.Pair).filter(db.Pair.synapse == True
    ).filter(db.AvgFirstPulseFit.pair_id == db.AvgFirstPulseFit2.pair_id       
    ).filter(db.AvgFirstPulseFit.pair_id == db.AvgFirstPulseFit3.pair_id       
    ).all()



#    filter(db.AvgFirstPulseFit.ic_nrmse < 1).all()
#afpfs=session.query(db.SingleFirstPulseFit).join(db.PulseResponse).join(db.Pair).filter(db.Pair.synapse == True).all()


nrmse_before = []
nrmse_after = []

for afpf_before, afpf_after, afpf_for_weight, pair in out:
    # note that you cant compare across these
    # nrmse_before.append(afpf_before.ic_nrmse)
    # nrmse_after.append(afpf_after.ic_nrmse)

    # use same weighting paradigm for all fitting using paradigm  
    weight = np.ones(len(afpf_for_weight.ic_weight))
    weight[np.where(afpf_for_weight.ic_weight==0)] = 0
    nrmse_before.append(weighted_nrmse(afpf_before.ic_avg_psp_data, afpf_before.ic_avg_psp_fit, weight))
    nrmse_after.append(weighted_nrmse(afpf_after.ic_avg_psp_data, afpf_after.ic_avg_psp_fit, weight))


scatter = pg.ScatterPlotItem(symbol='o', brush='b', pen='w', size=12) #create scatter plot item
# note that data will be refered to as points in the referenced function
scatter.setData(nrmse_before, nrmse_after, data=out) #set x, y, and data associated with the scatter points


# Make clickable scatter plot
s_plot=pg.plot() #create the plot window
s_plot.addItem(scatter) #add the scatter plot data from above
s_plot.setLabels(left='nrmse_after', bottom = 'nrmse_before')

scatter.sigClicked.connect(plot_afpf_trace)

    #         point.setBrush(pg.mkBrush('y'))
    #         point.setSize(15)
    #   trace.setPen('y', width=2)
    # print('Clicked:' '%s' % pair)   
