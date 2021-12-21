'''
By Mikkel Sinkjaer 
Date: 12/21/2021
'''
# %% 
import numpy as np
import matplotlib.pyplot as plt
from estimate_shift_func import est_shift, adjust_signal

# Generate original signal
t = np.arange(0,10000)
s = np.sin(0.03*t) + np.sin(0.0007*t**1.4)

# Make shift from random walk
np.random.seed(seed=0)  # set seed
shift_true = np.ones(len(t))*100 # Global shift
# Local shift
random_shift_variable = np.random.normal(0.0005,.01,len(t))
shift_i = 0
for i in t:
    shift_i +=random_shift_variable[i]
    shift_true[i] += shift_i
shift_true = shift_true.astype(int)

# Plot the generated shift
plt.plot(shift_true)
plt.title('True shift')
plt.show()

# Make shifted signal 
s_shifted = adjust_signal(s,-shift_true)

# Plot the signals before shift is estimated
fig, ax = plt.subplots(figsize=(15,5))
ax.plot(s, linewidth=1, label = 'Original signal', color= 'tab:orange')
ax.plot(s_shifted-3, linewidth=1, label = 'Shifted signal', color= 'tab:blue')
major_ticks = np.arange(0, len(s), 500000)
ax.set_xticks(major_ticks)
ax.grid(which='major', alpha=1)
ax.set_title('Singals   error = {}'.format(round(np.nanmean(abs(s-s_shifted)),4)))
ax.legend()
ax.grid(False, axis = 'y')
# %% Estimate the shift
fs = 1
int(fs*100),# set parameters for the syncronization
opt = { 'global window s1':int(fs*200),
        'global window s2': int(fs*100),
        'global step': int(fs*500),
        'inner segment':  int(fs*20),
        'outer segment': int(fs*20),
        'local window': int(fs*10), 
        'local shift search': int(fs*5),
        # 'regression treshold': int(4)
}

shift, obs_corr, shift_global = est_shift(s,s_shifted, opt,  plot_path='', label='', n_jobs = 4, regression=False)

# %% estimated vs true shift 
shift_compare = adjust_signal(shift,-shift)  # afjust with negative to compare
fig, ax = plt.subplots(figsize=(15,5))
ax.plot(shift_true, linewidth=1, label = 'True shift', color= 'tab:orange')
ax.plot(shift_compare, linewidth=1, label = 'Estimated shift', color= 'tab:blue')
major_ticks = np.arange(0, len(s), 5000)
ax.set_xticks(major_ticks)
ax.grid(which='major', alpha=1)
ax.set_title('Shift')
ax.legend()
ax.grid(False, axis = 'y')

# %% Signal True vs shifted
s_shifted = adjust_signal(s_shifted, shift) # adjust the shifted signal to the original

# Plot the original and adjusted signal
fig, ax = plt.subplots(figsize=(15,5))
ax.plot(s, linewidth=1, label = 'Original signal', color= 'tab:orange')
ax.plot(s_shifted-3, linewidth=1, label = 'Adjusted signal', color= 'tab:blue')
major_ticks = np.arange(0, len(s), 500000)
ax.set_xticks(major_ticks)
ax.grid(which='major', alpha=1)
ax.set_title('Singals   error = {}'.format(round(np.nanmean(abs(s-s_shifted)),4)))
ax.legend()
ax.grid(False, axis = 'y')



# %%

# %%
