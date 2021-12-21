'''
Use the estimated shift file to adjust the original data. 

By Mikkel Sinkjaer 
Date: 12/21/2021

'''

# %% init
import numpy as np
import pandas as pd
from dataapi import data_collection
from scipy import signal
from estimate_shift_func import est_shift
from datetime import datetime
import json 

hpc = False

subject_name = 'S01'
if hpc == True:
    jobs = 32
    data_path = '/work3/s164548/hdf5/'
    plot_path = '/work3/s164548/sync_plots/'
else:
    jobs = 4
    plot_path = '/Volumes/T7/UNEEQ/synchronizationeeg/syncronized subjects/'
    data_path = '/Volumes/T7/UNEEQ/hdf5/'
psg_subjects = pd.DataFrame({'subject': ['S01', 'S02', 'S04', 'S05', 'S06', 'S08', 'S09', 'S10',
                                         'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19',
                                          'S20', 'S21', 'S22', 'S23', 'S24', 'S25'],
                             'side': ['R', 'R', 'R', 'L','L','R','L','R','L','L','L','L','R','R','R','R','R','L','L','L','R','R']}).set_index('subject')
side = psg_subjects.loc[subject_name, 'side']  # What side subQ is placed

def main(data_path, plot_path, subject_name, side, n_jobs=4):
    min_duration_sec = int(60*10)
    fs = 207.0310546581987 # sample rate of subq
    # Load real data
    # Find number of nights 
    dset_path = r''+data_path+'ults_psg_'+subject_name+'.h5'
    with data_collection.File(dset_path, 'r') as data_file:
        nights = list(data_file['/{}'.format(subject_name)].keys())
    for night in nights:
        #Find number of subq records for the same night
        dset_path = r''+data_path+'ults_subq_psg_nights_'+subject_name+'.h5'
        with data_collection.File(dset_path, 'r') as data_file:
            subject = list(data_file['/{}'.format(subject_name)].keys())
        subq_records = [s for s in subject if night in s]

        # Loop over the each subq record
        for subq_record in subq_records:
            print('Night {}     Record{}'.format(night,subq_record))
            dset_path = r''+data_path+'ults_subq_psg_nights_'+subject_name+'.h5'
            with data_collection.File(dset_path, 'r') as data_file:
                subject = data_file['/{}'.format(subject_name)]
                rec = list(subject.keys())
                rec = subject[subq_record]
                subq = rec.get_interval()
            dset_path = r''+data_path+'ults_psg_'+subject_name+'.h5'
            with data_collection.File(dset_path, 'r') as data_file:
                subject = data_file['/{}/{}'.format(subject_name,night)]
                psg = subject.get_interval()
                if psg['PSG'].empty == True:
                    print('PSG is empty')
                    continue
            subq = subq['EEG']
            start_time = subq.index[0]
            end_time = subq.index[-1]
            sample_off_psg = np.sum((psg['PSG'].index < start_time))
            psg = psg['PSG'][(psg['PSG'].index >= start_time)
                            & (psg['PSG'].index <= end_time)]
            if (psg.empty == True) or (psg.index[-1] - psg.index[0] <= pd.Timedelta(hours=min_duration_sec/60/60)):
                print('To small interval')
                continue

        
            # Trime series
            start_time = psg.index[0]
            end_time = psg.index[-1]
            sample_offset_subq = np.sum(subq.index < start_time)
            subq = subq[(subq.index >= start_time) & (subq.index <= end_time)]
            start_time = subq.index[0]
            end_time = subq.index[-1]
            sample_off_psg += np.sum(psg.index < start_time) 
            psg = psg[(psg.index >= start_time) & (psg.index <= end_time)]
            number_of_samples_subq = len(subq)
            number_of_samples_psg = len(psg) 
            # time used    
            start_time = subq.index[0]
            end_time = subq.index[-1]

            # Output duration
            if (psg.empty == True) or (subq.index[-1] - subq.index[0] <= pd.Timedelta(hours=min_duration_sec/60/60)):
                print('To small interval')
                continue
            else:
                print('Duration: {}'.format(subq.index[-1] - subq.index[0]))


            # resampling subQ to the sampling rate of psg
            psg_col = psg.columns
            psg = signal.resample(psg, num=subq.shape[0])
            psg = pd.DataFrame(psg, columns=psg_col)
            if side == 'R':
                subq = subq.D-subq.P
                psg = psg.F4-psg.M2
            else:
                subq = subq.D-subq.P
                psg = psg.F3-psg.M1
            # set parameters for the syncronization
            opt = { 'global window':int(fs*20),
                    'global window s1 additional': int(fs*60),
                    'global step': int(fs*60*20),
                    'inner segment':  int(fs*60*1),
                    'outer segment': int(fs*60*4),
                    'local window': int(fs*20), 
                    'local shift search': int(fs*5),
                    # 'regression treshold': int(4)
            }
            
            shift, obs_corr, shift_global = est_shift(psg.values,subq.values, opt,  plot_path=plot_path, label=subject_name+'_'+subq_record, n_jobs = n_jobs, regression=False)
            # safe the estimated time shift af json file
            dset_path = r''+data_path+'shift/shift_'+subject_name+'_'+subq_record+'.json'
            shift_save  = {'shift':list(map(int, list(shift))),'corr':list(map(float, list(obs_corr))),'start time':str(start_time), 'end time':str(end_time), 'shift_global':shift_global,
                            'subset used':{'sample_offset_psg': int(sample_off_psg), 'sample_offset_subq':int(sample_offset_subq),'number_of_samples_psg':int(number_of_samples_psg),'number_of_samples_subq':int(number_of_samples_subq)}}
            with open(dset_path, 'w') as outfile:
                json.dump(shift_save, outfile)

            print('\n')



# %% Syncronize for a subject

main(data_path, plot_path, subject_name, side, jobs)


# %%

# %%
