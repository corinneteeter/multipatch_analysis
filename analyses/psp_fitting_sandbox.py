'''this code meant to try out different fitting against test data and then update if necessary
'''

from neuroanalysis.fitting import fit_psp
import os
import numpy as np
from pprint import pprint
import json
import neuroanalysis.data

test_data_dir = os.path.join(os.path.dirname(neuroanalysis.__file__), '..', 'test_data', 'test_psp_fit')               

"""Plots the results of the current fitting with the save fits and denotes 
when there is a change. 
"""
plotting=True # specifies whether to make plots of fitting results


#Note that the traces in this test data start 10 ms before a pulse   
test_data_files=[os.path.join(test_data_dir,f) for f in os.listdir(test_data_dir)] #list of test files

continue_flag = 0
for file in sorted(test_data_files):
#    for file in ['test_psp_fit/1492546902.92_2_6stacked.json']: order of parameters affects this fit
    print 'file', file
    #use if statement below to look at specific connections
    if '1492545925.15_4_2' not in os.path.basename(file) and continue_flag==0:
        continue
    else: #set continue_flag = 1 when hit the file you are looking for.
        continue_flag=1 
    test_dict=json.load(open(file)) # load test data
    avg_trace=neuroanalysis.data.Trace(data=np.array(test_dict['input']['data']), dt=test_dict['input']['dt']) # create Trace object

    psp_fits = fit_psp(avg_trace, 
                       xoffset=(np.arange(10e-3,20e-3, 1e-3).tolist(), -float('inf'), float('inf')),
                       weight=np.array(test_dict['input']['weight']),
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
        print os.path.basename(file)
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
        mplt.title(os.path.basename(file) + ', nrmse now=' + str(psp_fits.nrmse())+' nrmse before='+str(test_dict['out']['nrmse']))
        mplt.show()