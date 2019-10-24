import numpy as np

# These are the possbile stimuli found in the db14 on 8/29/2019
# in the format [clamp type, stimulus frequecy, recover time].
# Note that the recovery times are rounded to numbers as in 
# the *set_recovery* module below.
possible_stimuli =[
    'ic, 10.0, 0.25', 
    'ic, 20.0, 0.25', 
    'ic, 25.0, 0.25', 
    'ic, 50.0, 0.053', 
    'ic, 50.0, 0.103', 
    'ic, 50.0, 0.125', 
    'ic, 50.0, 0.203', 
    'ic, 50.0, 0.25', 
    'ic, 50.0, 0.403', 
    'ic, 50.0, 0.5', 
    'ic, 50.0, 0.803', 
    'ic, 50.0, 1.0', 
    'ic, 50.0, 1.603', 
    'ic, 50.0, 2.0', 
    'ic, 50.0, 3.203', 
    'ic, 50.0, 4.0', 
    'ic, 100.0, 0.25', 
    'ic, 150.0, 0.25', 
    'ic, 200.0, 0.25', 
    'ic, None, None', 
    'vc, 10.0, 0.25', 
    'vc, 20.0, 0.25', 
    'vc, 25.0, 0.25', 
    'vc, 50.0, 0.053', 
    'vc, 50.0, 0.103', 
    'vc, 50.0, 0.125', 
    'vc, 50.0, 0.203', 
    'vc, 50.0, 0.25', 
    'vc, 50.0, 0.403', 
    'vc, 50.0, 0.5', 
    'vc, 50.0, 0.803', 
    'vc, 50.0, 1.0', 
    'vc, 50.0, 1.603', 
    'vc, 50.0, 2.0', 
    'vc, 50.0, 3.203', 
    'vc, 50.0, 4.0', 
    'vc, 100.0, 0.25', 
    'vc, 150.0, 0.25', 
    'vc, 200.0, 0.25', 
    'vc, None, None']


def set_recovery(delay):
    """If a *delay* is within 5 ms of an expected recovery value, set it to that.  
    Else do not alter it."""
    if not delay:
        return None
    expected_delays = np.array([.125, .250, .500, 1.000, 2.000, 4.000])
    value = expected_delays[np.isclose(delay, expected_delays, atol=.005)] #allow a 5 ms variation in recovery
    if len(value) == 0:
        return delay 
    elif len(value) == 1:
        return value[0]
    else: 
        raise Exception("a number can't be within 5 ms of two specified delays")

# specify excitatory and inhibitory cre-lines
def specify_excitation(cre):
    excitation_specification= {'unknown': 'U',
                           'sim1,pvalb': 'U',    
                           'pvalb,sim1': 'U',
                           'pvalb': 'I',
                           'sst': 'I', 
                           'vip': 'I',
                           'pvalb,sst': 'I',    
                           'sim1': 'E',
                           'tlx3': 'E',              
                           'nr5a1': 'E',  
                           'rorb': 'E',
                           'ntsr1': 'E',
                           'rbp4': 'E',
                           'slc17a8': 'E',
                           'cux2': 'E',
                           'fam84b': 'E'}

    if cre not in excitation_specification.keys():
        raise Exception('provided cre line not recognized')
    else:
        return excitation_specification[cre]
