"""test psp_fit function using data from test directory
"""
import os
import numpy as np
from pprint import pprint
import json
import neuroanalysis.data
from multipatch_analysis.connection_detection import fit_psp

plotting=False # specifies whether to make plots of fitting results

test_data_dir='test_psp_fit' # directory containing test data

test_data_files=[os.path.join(test_data_dir,f) for f in os.listdir(test_data_dir)] #list of test files
for file in sorted(test_data_files):
    print 'file', file
    test_dict=json.load(open(file)) # load test data
    avg_trace=neuroanalysis.data.Trace(data=np.array(test_dict['input']['data']), dt=test_dict['input']['dt']) # create Trace object
    psp_fits = fit_psp(avg_trace, 
                       sign=test_dict['input']['amp_sign'], 
                       stacked=test_dict['input']['stacked'] 
                        )                        
    
    change_flag=False
    if test_dict['out']['best_values']!=psp_fits.best_values:     
        print '  the best values dont match'
        print '\tsaved', test_dict['out']['best_values']
        print '\tobtained', psp_fits.best_values
        change_flag=True
        
    if test_dict['out']['best_fit']!=psp_fits.best_fit.tolist():
        print '  the best fit traces dont match'
        print '\tsaved', test_dict['out']['best_fit']
        print '\tobtained', psp_fits.best_fit.tolist()
        change_flag=True
    
    if test_dict['out']['nrmse']!=float(psp_fits.nrmse()):
        print '  the nrmse doesnt match'
        print '\tsaved', test_dict['out']['nrmse']
        print '\tobtained', float(psp_fits.nrmse())
        change_flag=True
        
    if plotting:
        import matplotlib.pylab as mplt
        fig=mplt.figure(figsize=(20,8))
        ax=fig.add_subplot(1,1,1)
        ax2=ax.twinx()
        ax.plot(avg_trace.time_values, psp_fits.data*1.e3, 'b', label='data')
        ax.plot(avg_trace.time_values, psp_fits.best_fit*1.e3, 'g', lw=5, label='current best fit')
        ax2.plot(avg_trace.time_values, test_dict['input']['weight'], 'r', label='weighting')
        if change_flag is True:
            ax.plot(avg_trace.time_values, np.array(test_dict['out']['best_fit'])*1.e3, 'k--', lw=5, label='original best fit')
            mplt.annotate('CHANGE', xy=(.5, .5), xycoords='figure fraction', fontsize=40)
        ax.legend()
        mplt.title(file + ', nrmse =' + str(psp_fits.nrmse()))
        mplt.show()