"""Model used int the Tsodyks and Markram 1997 paper.
"""

import numpy as np
import matplotlib.pyplot as plt

n_of_spikes = 10.
spike_freq = 20.
spike_interval = 1 / spike_freq
spike_times = .01 + np.arange(n_of_spikes) * spike_interval

# really just need t-t_AP at every increment
dt = 1e-3
t_vec = np.arange(0, spike_times[-1]+.01, dt)


def convert_time_to_deltat(spike_times, t_vec):
    """Convert time vector to time since spike"""
    # initalize time_since_last_spike vector to be infinity up to the first spike and 0 at first spike
    no_spike_yet=np.where(t_vec < spike_times[0])[0]  # find all indicies before first spike
    time_since_last_spike = np.arange(1,len(no_spike_yet)+1) * np.inf # infinite time before first spike
    
    first_spike_index = len(no_spike_yet)
    time_since_last_spike = np.append(time_since_last_spike, t_vec[first_spike_index] - spike_times[0])  # at spike time_since_last_spike is 0
    
    next_spike_index = 1
    for t in t_vec[first_spike_index + 1:]:  #start for loop at first spike where you left off above.
        if next_spike_index < len(spike_times):  #necessary so that you don't go out of bounds on spike_time index on last spike
            if (t < spike_times[next_spike_index]) and not np.isclose(t, spike_times[next_spike_index]): # isclose is needed for precision error here
                time_since_last_spike = np.append(time_since_last_spike, t - spike_times[next_spike_index - 1]) #continue to take the time since last spike
            else:
                time_since_last_spike = np.append(time_since_last_spike, t - spike_times[next_spike_index]) #take the time from the current spike
                next_spike_index += 1 #update spike number
        else:
            time_since_last_spike = np.append(time_since_last_spike, t - spike_times[next_spike_index - 1]) # always subtract from last spike because we know there are no more

    return time_since_last_spike


time_since_last_spike = convert_time_to_deltat(spike_times, t_vec)

# plot to confirm results
# plt.plot(t_vec, time_since_last_spike)
# plt.plot(spike_times, np.ones(len(spike_times))*0,'|r', ms=50)
# plt.show()

def tsodyks():

    # parameters
    tau_recover = 0.8   # time constant for recovery
    tau_inact = 3.e-3   # time constant of inactivation
    U_SE = 0.67         # utilization of synaptic efficacy
    
    # Kinetic equations for the fraction of resources in each state
    # R: recovered
    # E: effective
    # I: inactive
    
    # Initial conditions:not recorded in output vector
    R = 0
    E = 1
    I = 0
    t = 0 
    
    dRdt = I / tau_recover - U_SE * R * (t_diff)
    dEdt = -E / tau_inact  + U_SE * R * (t_diff)
    
    R_list = R + dRdt
    E_list = E + dEdt
    I_List = 1 - R - E
    
    
    for t_diff in time_since_last_spike[1:]: # starting at one because vector initialized above 
        # Initialize the vectors
        dRdt = I / tau_recover - U_SE * R * (t_diff)
        dEdt = -E / tau_inact  + U_SE * R * (t_diff)
        
        R_list.append(R_list[-1] + dRdt)
        E_list.append(E_list[-1] + dEdt)
        I_list.append(1 - R_list[-1] - E_list[-1])
        
        EPSC_st = A_SE / (f * tau_r)
                
    return R_list, E_list, I_list 


# Net post synaptic current is proportional to the fraction of resources in the effective state, E.  
# Convert current to potential

tau_mem = 50.e-3

v = 1 - exp(-t / tau_mem) # voltage relative to rest.








