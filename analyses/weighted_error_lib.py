"""functions to to weighted error mesurements"""
import numpy as np


def weighted_std(values, weights):
    """Return the weighted standard deviation of *values*.
    Source: http://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return variance**0.5

def weighted_rmse(data, fit, weight):
    residual = (data - fit) * weight
    return (residual**2 / residual.size).sum() ** 0.5


def weighted_nrmse(data, fit, weight):
    """This is a local calcuation of normalized root means squared error 
    that uses one weighting paradigm.  
    """
    rmse = weighted_rmse(data, fit, weight)
    std = weighted_std(data, weight)
    return rmse, std, rmse/std