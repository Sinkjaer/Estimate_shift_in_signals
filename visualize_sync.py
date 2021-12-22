'''
By Mikkel Sinkjaer 
Date: 12/21/2021
'''
# %%  Load the syncronized 
import numpy as np
import pandas as pd
import json
from dataapi import data_collection
from scipy import signal
from estimate_shift_func import adjust_signal
import matplotlib.pyplot as plt

side = 'R'
subject_name = 'S01'
night = 'PSG_6'
subq_record = night+'_part_0'
data_path = '/Volumes/T7/UNEEQ/hdf5/' # set path to file 
fs = 207.0310546581987 # sample rate of subq
 
with open(data_path+'shift/shift_'+subject_name+'_'+subq_record+'.json') as json_file:
    shift = json.load(json_file)

dset_path = r''+data_path+'ults_subq_psg_nights_'+subject_name+'.h5'
with data_collection.File(dset_path, 'r') as data_file:
    subject = data_file['/{}'.format(subject_name)]
    rec = list(subject.keys())
    rec = subject[subq_record]
    subq = rec.get_interval()['EEG'].iloc[shift['subset used']['sample_offset_subq']:shift['subset used']['number_of_samples_subq']+shift['subset used']['sample_offset_subq']]

dset_path = r''+data_path+'ults_psg_'+subject_name+'.h5'
with data_collection.File(dset_path, 'r') as data_file:
    subject = data_file['/{}/{}'.format(subject_name,night)]
    psg = subject.get_interval()['PSG'].iloc[shift['subset used']['sample_offset_psg']:shift['subset used']['number_of_samples_psg']+shift['subset used']['sample_offset_psg']]


shift_obs = shift['shift']
corr_obs = shift['corr']


# %%
fig, ax = plt.subplots(figsize=(20,7))
ax.plot(shift_obs, linewidth=1, label = 'shift', color= 'tab:orange')
ax.set_ylabel('shift')
ax.set_ylabel('time')
ax.tick_params(axis='y', labelcolor = 'tab:orange')
major_ticks = np.arange(0, len(shift_obs), 207*60*60)
minor_ticks = np.arange(0, len(shift_obs), 207*60*60)
ax.set_xticks(major_ticks)
ax.grid(which='major', alpha=1)
ax.grid(False, axis = 'y')

ax2 = ax.twinx()
ax2.plot(corr_obs, linewidth=1, label = 'subq', color = 'tab:blue')
ax2.tick_params(axis='y', labelcolor = 'tab:blue')
ax2.set_ylabel('cross corr')
ax2.set_ylim(0,1)
major_ticks = np.arange(0, 1, 0.2)
ax2.set_yticks(major_ticks)
ax2.grid(which='major', alpha=1)

ax.set_title('Shift both')
plt.show()
# %% Shift subq
psg_col = psg.columns
psg = signal.resample(psg, num=subq.shape[0])
psg = pd.DataFrame(psg, columns=psg_col, index=subq.index)

subq_adjusted = adjust_signal(subq,np.array(shift['shift']))
idx = subq_adjusted.dropna().index.values
subq_adjused = subq_adjusted.loc[idx]
subq = subq.loc[idx]
psg = psg.loc[idx]

# %% Plot the shifted data
s1 = psg.F4.values - psg.M2.values
s2 = subq.D.values - subq.P.values
s2_final  = subq_adjusted.D.values - subq_adjusted.P.values
idx = np.arange(3000,len(s1)-3000)
# plot
draw = np.sort(np.random.choice(idx,5))
for i in draw:
    s1_i = (s1[i:i+207*30]-s1[i:i+207*30].mean())/s1[i:i+207*30].std()
    s2_final_i = (s2_final[i:i+207*30]-s2[i:i+207*30].mean())/s2[i:i+207*30].std()
    fig, ax = plt.subplots(figsize=(20,7))
    major_ticks = np.arange(0, len(s1_i), 2070)
    minor_ticks = np.arange(0, len(s1_i), 207)
    ax.plot(s1_i, linewidth=1, label = 'PSG', alpha = 0.5)
    ax.plot(s2_final_i-4, linewidth=1, label = 'subq', color = 'tab:orange', alpha = 0.5)

    ax.legend(loc = 'right')
    ax.axes.get_yaxis().set_visible(False)
    ax.set_title('Syncronized signals, idx = {}'.format(i))
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.5)
    ax.grid(which='major', alpha=1)
    plt.show()
 # %% plot distribution of data
C = psg.F4.values - psg.M2.values
DP = subq.D.values - subq.P.values
MSE_naive = np.mean((C-DP)**2)

_ = plt.hist(C, bins=100)
plt.title('C')
plt.show()

_ = plt.hist(DP, bins=100)
plt.title('DP')
plt.show()
plt.plot(C)
plt.title('DP')
plt.show()
plt.plot(DP)
plt.title('DP')
plt.show()

# %%
