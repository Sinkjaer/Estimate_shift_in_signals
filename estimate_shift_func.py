'''
By Mikkel Sinkjaer 
Date: 12/21/2021
'''

from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import RANSACRegressor, LinearRegression
from tqdm import tqdm
import random
from sklearn.neighbors import KNeighborsRegressor

def adjust_signal(s2,shift_obs):
    '''
    Adjust each sample accoding to estimated shift

    Parameters
    ----------
    s2: pandas DataFrame or numpy array of size [samples, channels]
    shift_obs: numpy array of same length as s2

    Returns
    ---------
    s2_new: s2 shifted
    '''
    s2_new = s2.copy()*np.nan
    head = shift_obs.min()
    if head > 0:
        head = 0
    head = int(head)

    tail = shift_obs.max()
    if tail < 0:
        tail = 0    
    tail = int(tail)
    idx = np.arange(0-head,len(shift_obs)-tail)
    if type(s2) == pd.core.frame.DataFrame:
        s2_new.iloc[idx]=s2.iloc[(idx+shift_obs[idx]).astype(int)]
    else:
        s2_new[idx]=s2[(idx+shift_obs[idx]).astype(int)]
    return  s2_new

def find_shift( s1, s2, window_size=int(200*30), step_size=200*10, shift_search=None, window_s1_add=200*60*1):
    '''Estimate time shift by rolling window. s2 is adjusted to s1.

    Parameters
    ----------
    s1 : array, signal to syncronize too
    s2 : array, signal to syncronize
    window_size : int, size of the window to sync, [t: t+window_size]
    step_size : int, The size of the step the window will be move
    shift_search : int, +- shift interval to search in
    window_s1_add : int, include additional elements add to the s1 window

    Returns
    ---------
    shift : int , The estimated shift of s2 
    np.nanmean(rss_max): float, mean cross correlation for optimal shift
    '''
    # init
    if shift_search == None: # If the user does not define the interval to search in 
        shift_search = int(window_s1_add + int(window_size/2))
    rss = [] # Optimal cross correlation.
    rss_max = []
    t_start = int(shift_search)
    t_end = int(t_start + window_size)
    L = pd.Series(np.arange(-shift_search, shift_search+1)) # The interval to search in

    # Estimate the shift for subsets of the signals
    for i in range(t_end, s1.shape[0]-shift_search+1, step_size):
        s1_i = s1.iloc[t_start-window_s1_add:t_end+window_s1_add]
        s2_i = s1_i*np.nan
        s2_i.iloc[window_s1_add:window_size +
                window_s1_add] = s2.iloc[t_start:t_end]

        # Estimate the cross corelation for different shift values
        rs_1 = L.apply(lambda x: s1_i.corr(s2_i.shift(x))) #estimate the cross corelation for the instal at time i
        rss +=[int(np.floor(len(rs_1)/2)-np.argmax(rs_1))] # Find the shift maximizer
        rss_max += [np.max(rs_1)]

        t_start = t_start + step_size
        t_end = t_end + step_size

    shift = int(np.median(rss))

    return shift, np.mean(rss_max)

def est_segment(s1, s2, i, segment__inner_size, segment__outer_size, local_window, local_shift_search):
    '''
    Estimate shift for a segment. 
    '''
    shift ,rss_max = find_shift(s1[i-segment__outer_size:i+segment__inner_size+segment__outer_size],
                                s2[i-segment__outer_size:i+segment__inner_size+segment__outer_size],
                       window_size=local_window, step_size=int(local_window/2), shift_search=local_shift_search, window_s1_add=0)
    return shift, rss_max

def est_shift(s1, s2, opt, plot_path='', label='', n_jobs = -1, regression= True):
    '''Estimate shift of s2 relative s1.

    Parameters
    ----------
    s1 : array, signal to syncronize too
    s2 : array, signal to syncronize
    opt: dict, options for sync
    plot_path: str, path to where plot is saved
    label: str, name of the plot
    n_jobs: int, number of processes to use. Default is all available
    regression, Boolean, Reather to make regression on end point and large changes in estimated shift.
                        If False endpoint will be estimated from the closest known point.

    Returns
    ---------
    shift_obs: numpy array, estimated shift of each sample in s2
    obs_corr: numpy array, cross correlation for the shift of each sample
    shift_global: int, estimated global shift
    '''
    # init
    s1 = pd.Series(s1)
    s2 = pd.Series(s2)
    global_window = opt['global window']
    global_s1_add = opt['global window s1 additional']
    global_step = opt['global step']
    segment__inner_size = opt['inner segment']
    segment__outer_size = opt['outer segment']  # Size of the window us the estimate to shift
    local_window = opt['local window']
    local_shift_search = opt['local shift search']

    # Calculate the global shift
    shift_global,_ = find_shift(s1, s2,
                          window_size=global_window, step_size=global_step, shift_search=None, window_s1_add=global_s1_add)
    print('Global shift: ',shift_global)
    s2 = s2.shift(int(-shift_global))

    # Estimate shift for segments of the signals to get local estimates
    shift_segment = Parallel(n_jobs=n_jobs)(delayed(est_segment)(s1, s2, i, segment__inner_size, segment__outer_size, local_window, local_shift_search)
                            for i in tqdm(range(segment__outer_size, len(s2)-segment__inner_size-segment__outer_size, segment__inner_size)))
    shift_segment = np.array(shift_segment)

    # Get shift on sample level
    shift_obs = np.ones(s2.shape[0])*np.nan
    obs_corr = np.ones(s2.shape[0])*np.nan
    for counter, i in enumerate(range(segment__outer_size, len(s2)-segment__inner_size-segment__outer_size, segment__inner_size)):
        shift_obs[segment__outer_size+counter*segment__inner_size:segment__inner_size *
            (counter+1)+segment__outer_size] = shift_segment[counter,0]+ shift_global
        obs_corr[segment__outer_size+counter*segment__inner_size:segment__inner_size *
            (counter+1)+segment__outer_size] = shift_segment[counter,1]

    # Estimate endpoint
    if regression == True:
        treshold = opt['regression treshold']
        shift_obs_reg = shift_obs.copy()

        x = pd.DataFrame(shift_obs).dropna()
        idx = random.sample(list(x.index.values),10000) # Use only subset for estimation of regression
        reg = LinearRegression().fit(np.array(idx).reshape(-1,1),x.loc[idx].values.reshape(-1, 1))
        reg_shift = reg.predict(np.arange(0,len(shift_obs)).reshape(-1, 1)).round(0).astype(int).squeeze()

        # Find lag points which deviates from the trend 
        shift_obs_reg[(~np.isnan(shift_obs)) & (abs(reg_shift-shift_obs)>treshold)] = np.nan # point > treshold away from regression line
        shift_obs_reg[np.isnan(shift_obs_reg)] = reg_shift[np.isnan(shift_obs_reg)] # update lag_obs

        # Plot the estimated shift
        shift_plot_data = pd.DataFrame({'shift':shift_obs, 'cross corr':obs_corr,'regression shift':shift_obs_reg})
        shift_plot_data['cross corr']= shift_plot_data['cross corr'].replace(np.nan, 0)
        fig, ax = plt.subplots(figsize=(20,7))
        lns1 = ax.plot(shift_plot_data['shift'].values, linewidth=1, label = 'shift', color= 'tab:orange')
        lns2 = ax.plot(shift_plot_data['regression shift'].values, linewidth=1, label = 'shift regression', color= 'tab:purple')
        ax.set_ylabel('Shift')
        ax.set_xlabel('Time')
        major_ticks = np.arange(0, len(shift_plot_data['shift'].values), 207*60*60)
        ax.set_xticks(major_ticks)
        ax.grid(which='major', alpha=1)
        ax.grid(False, axis = 'y')
        ax2 = ax.twinx()
        lns3 = ax2.plot(shift_plot_data['cross corr'].values, linewidth=1, label = 's2', color = 'tab:blue')
        ax2.set_ylabel('Cross corr')
        ax2.set_ylim(0,1)
        major_ticks = np.arange(0, 1, 0.2)
        ax2.set_yticks(major_ticks)
        ax2.grid(which='major', alpha=1)
        ax.set_title('Shift for {}'. format(label))
        lns = lns1+lns2+lns3
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0)
        fig.savefig(plot_path+label+'_shift.png')

        obs_corr = shift_plot_data['cross corr'].values
        shift_obs = shift_plot_data['regression shift'].values # Use the regression line in the final estimate
    else:
        # Estimate shift in of endpoint by closes estimated shift 
        shift_plot_data = pd.DataFrame({'shift':shift_obs, 'cross corr':obs_corr})
        x_head = shift_plot_data.dropna().index.values[0]
        y_head = shift_plot_data.dropna()['shift'].values[0]
        shift_plot_data.loc[0:x_head-1,'shift'] = y_head

        x_tail = shift_plot_data.dropna().index.values[-1]
        y_tail = shift_plot_data.dropna()['shift'].values[-1]
        shift_plot_data.loc[x_tail+1:,'shift'] = y_tail

        shift_plot_data['cross corr']= shift_plot_data['cross corr'].replace(np.nan, 0)
        fig, ax = plt.subplots(figsize=(20,7))
        lns1 = ax.plot(shift_plot_data['shift'].values, linewidth=1, label = 'shift', color= 'tab:orange')
        ax.set_ylabel('Shift')
        ax.set_xlabel('Time')
        major_ticks = np.arange(0, len(shift_plot_data['shift'].values), 207*60*60)
        ax.set_xticks(major_ticks)
        ax.grid(which='major', alpha=1)
        ax.grid(False, axis = 'y')
        ax2 = ax.twinx()
        lns2 = ax2.plot(shift_plot_data['cross corr'].values, linewidth=1, label = 'cross cor', color = 'tab:blue')
        ax2.set_ylabel('cross corr')
        ax2.set_ylim(0,1)
        major_ticks = np.arange(0, 1, 0.2)
        ax2.set_yticks(major_ticks)
        ax2.grid(which='major', alpha=1)
        ax.set_title('Shift for {}'. format(label))
        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0)
        fig.savefig(plot_path+label+'_shift.png')

        obs_corr = shift_plot_data['cross corr'].values
        shift_obs = shift_plot_data['shift'].values


    return shift_obs, obs_corr, shift_global
