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

# db.Session()
# pair=db.experiment_from_timestamp(1547248417.685).pairs[(2, 7)]
# afpf = afpf.fit_first_pulses(pair)
# print(afpf)


session=db.Session()
afpfs=session.query(db.AvgFirstPulseFit).join(db.Pair).filter(db.Pair.synapse == True).filter(db.AvgFirstPulseFit.ic_nrmse < 1).all()
#afpfs=session.query(db.SingleFirstPulseFit).join(db.PulseResponse).join(db.Pair).filter(db.Pair.synapse == True).all()



nrmse=[]
amp=[]
for afpf in afpfs:
    nrmse.append(afpf.ic_nrmse)
    amp.append(afpf.ic_amp)
    
scatter = pg.ScatterPlotItem(symbol='o', brush='b', pen='w', size=12) #create scatter plot item
scatter.setData(amp, nrmse, data=afpfs) #set x, y, and data associated with the scatter points


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

    print(points)
    print(scatterplot)
    if len(points) != 1:
        print(len(points),'points selected.  Only plotting first one')
    
    afpf = points[0].data()  #this accesses the data binded in *setData* for the selected points
    data_trace = afpf.ic_avg_psp_data
    data_trace = Trace(data=(data_trace), sample_rate=db.default_sample_rate)
    data_trace = pg.PlotCurveItem(data_trace.time_values, data_trace.data, pen='y')

    fit_trace = afpf.ic_avg_psp_fit
    fit_trace = Trace(data=(fit_trace), sample_rate=db.default_sample_rate)
    fit_trace = pg.PlotCurveItem(fit_trace.time_values, fit_trace.data, pen='r')

    # create the plot window and add the waveforms to it
    fit_plot = pg.plot()
    fit_plot.addItem(data_trace)
    fit_plot.addItem(fit_trace)


#         point.setBrush(pg.mkBrush('y'))
#         point.setSize(15)
#   trace.setPen('y', width=2)
# print('Clicked:' '%s' % pair)    


#pg.plot(nrmse, amp, pen=None, symbol='o')
s_plot=pg.plot() #create the plot window
s_plot.addItem(scatter) #add the scatter plot data from above
print len(afpfs)

scatter.sigClicked.connect(plot_afpf_trace)
