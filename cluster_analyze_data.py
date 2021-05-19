import os 

from pathlib import Path
import scipy.ndimage as ndimage
import numpy as np
from suite2p import default_ops as s2p_default_ops
from suite2p import classification
import shutil
import time
import datetime

from utils import utils_imaging, utils_plot#,utils_pipeline,utils_imaging

import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import tifffile
#from skimage.measure import label
#
%matplotlib qt
#%% read digested behavior data and fluorescence traces, align, find cell candidates
setup = 'DOM3-MMIMS'
subject = 'BCI_10'
session = '2021-05-19'



suite2p_dir = '/groups/svoboda/svobodalab/users/rozmar/BCI_suite2p/{}/{}/{}/_concatenated_movie'.format(setup,subject,session)
bpod_exported = os.path.join(suite2p_dir,'{}-bpod_zaber.npy'.format(session))

behavior_dict = np.load(bpod_exported,allow_pickle = True).tolist()
ops = np.load(os.path.join(suite2p_dir,'ops.npy'),allow_pickle = True).tolist()
#create mean images tiff
# =============================================================================
# imgs = np.load(os.path.join(suite2p_dir,'meanImg.npy'))
# imgs = np.asarray(imgs,dtype = np.int32)
# tifffile.imsave(os.path.join(suite2p_dir,'meanimages.tiff'),imgs)
# =============================================================================

stat = np.load(os.path.join(suite2p_dir,'stat.npy'),allow_pickle = True).tolist()
iscell = np.load(os.path.join(suite2p_dir,'iscell.npy'))
stat = np.asarray(stat)[iscell[:,0]==1].tolist()

try:
    dF = np.load(os.path.join(suite2p_dir,'dF.npy'))[iscell[:,0]==1,:]
    dFF = np.load(os.path.join(suite2p_dir,'dFF.npy'))[iscell[:,0]==1,:]
except:
    print('calculating dff')
    utils_imaging.export_dff(suite2p_dir)#,raw_imaging_dir=raw_imaging_dir,revert_background_subtraction = True)
    dFF = np.load(os.path.join(suite2p_dir,'dFF.npy'))[iscell[:,0]==1,:]
    dF = np.load(os.path.join(suite2p_dir,'dF.npy'))[iscell[:,0]==1,:]
cell_indices = np.where(iscell[:,0]==1)[0]   
fs = ops['fs']
with open(os.path.join(suite2p_dir,'filelist.json')) as f:
    filelist_dict = json.load(f)


#%
motor_steps_mask = np.zeros(dFF.shape[1])
frame_times = np.zeros(dFF.shape[1])*np.nan
gocue_mask = np.zeros(dFF.shape[1])
lick_mask = np.zeros(dFF.shape[1])
reward_mask = np.zeros(dFF.shape[1])
unreward_mask = np.zeros(dFF.shape[1])
threshold_crossing_masks = np.zeros(dFF.shape[1])
task_mask = np.zeros(dFF.shape[1])
reward_consumtion_mask =  np.zeros(dFF.shape[1])
baseline_length = np.nan
trial_number_mask =  np.zeros(dFF.shape[1])*np.nan
prev_frames_so_far = 0
conditioned_neuron_name_list = []
for i,filename in enumerate(behavior_dict['scanimage_file_names']): # generate behavior related vectors
    if filename[0] not in filelist_dict['file_name_list']:
        continue
    movie_idx = np.where(np.asarray(filelist_dict['file_name_list'])==filename[0])[0][0]
    if movie_idx == 0 :
        frames_so_far = 0
    else:
        frames_so_far = np.sum(np.asarray(filelist_dict['frame_num_list'])[:movie_idx])
    frame_num_in_trial = np.asarray(filelist_dict['frame_num_list'])[movie_idx]  
    
    frame_times_now = np.arange(frame_num_in_trial)/fs+behavior_dict['scanimage_first_frame_offset'][i]+(behavior_dict['trial_start_times'][i]-behavior_dict['trial_start_times'][0]).total_seconds()
    frame_times[frames_so_far:frames_so_far+frame_num_in_trial] = frame_times_now 
    go_cue_idx = frames_so_far + int(behavior_dict['go_cue_times'][i][0]*fs)
    gocue_mask[go_cue_idx] = 1
    try:
        threshold_crossing_idx = frames_so_far + int(behavior_dict['threshold_crossing_times'][i][0]*fs)
        threshold_crossing_masks[threshold_crossing_idx] = 1
        task_mask[go_cue_idx:threshold_crossing_idx] = 1
    except:
        task_mask[go_cue_idx:frames_so_far+frame_num_in_trial] = 1
        pass # no threshold crossing
    
    lick_times = np.concatenate([behavior_dict['lick_L'][i],behavior_dict['lick_R'][i]])
    for lick_time in lick_times:
        lick_idx = frames_so_far+int(lick_time*fs)
        lick_mask[lick_idx] = 1
    
    step_times = behavior_dict['zaber_move_forward'][i]
    for step_time in step_times:
        step_idx = frames_so_far+int(step_time*fs)
        try:
            motor_steps_mask[step_idx] += 1
        except:
            pass
    
    
    if len(behavior_dict['reward_L'][i])>0:
        side = 'L'
    elif len(behavior_dict['reward_R'][i])>0:
        side = 'R'
    else:
        side = 'none'
    
    if not side == 'none':
        reward_idx = frames_so_far + int(behavior_dict['reward_{}'.format(side)][i][0]*fs)
        valve_time = behavior_dict['var_ValveOpenTime_{}'.format(side)][i]
        
        lick_times = np.concatenate([behavior_dict['lick_R'][i],behavior_dict['lick_L'][i]])
        last_lick_idx = frames_so_far+int(np.max(lick_times)*fs)
        if valve_time >0:
            reward_mask[reward_idx] = 1
            #reward_consumtion_mask[reward_idx:last_lick_idx]= 1
            reward_consumtion_mask[reward_idx:frames_so_far+frame_num_in_trial]= 1
        else:
            unreward_mask[reward_idx] = 1
            
        task_mask[go_cue_idx:reward_idx] = 1 # everything is task mask before reward
 
    if behavior_dict['var_BaselineZaberForwardStepFrequency'][i]==0 and np.isnan(baseline_length): # this one works if there is an open loop training before the task
        baseline_length = frames_so_far
# =============================================================================
#         if not any(task_mask[:frames_so_far] ==1):
#             task_mask[:frames_so_far] = 1
# =============================================================================
    trial_number_mask[frames_so_far:frames_so_far+frame_num_in_trial] = i

#%% SNR of cells
dFF_scaled = list()
max_SNR = list()
for dff in dFF:
    dff = ndimage.gaussian_filter(dff,3)
    #break
#dff = dFF[cond_s2p_idx,:]
    window = 100
    step=int(window/2)
    starts = np.arange(0,len(dff)-window,step)
    stds = list()
    for start in starts:
       stds.append(np.std(dff[start:start+window]))
    stds_roll = utils_plot.rollingfun(stds,100,'min')
    stds_roll = utils_plot.rollingfun(stds_roll,500,'mean')
    
    dff_scaled = np.copy(dff)
    noise_level = np.ones(len(dff)+1)
    for start,std in zip(starts,stds_roll):
        dff_scaled[start:start+window]=dff[start:start+window]/std
        noise_level[start:start+window]=std
    dff_scaled[start:]=dff[start:]/std
    noise_level[start:]=std
    max_SNR.append(np.max(dff_scaled))
    dFF_scaled.append(dff_scaled)
    #%%
max_SNR = list()
for dff in dFF_scaled:
    max_SNR.append(np.percentile(dff,99))
    #max_SNR.append(np.max(dff))
plt.figure()
plt.hist(max_SNR,50)
needed = np.asarray(max_SNR)>20
dFF = np.asarray(dFF[needed,:])
cell_indices = cell_indices[needed]
#%%

#%% correlations of all neurons
from scipy import stats
def event_based_trace_selection(trace,indices,trial_number_mask,step_back,step_forward):
    event_num = len(indices)
    traces = np.zeros([event_num,step_back+step_forward])*np.nan
    for i,idx in enumerate(indices):
        try:
            y = trace[idx-step_back:idx+step_forward]
            trialnum = trial_number_mask[idx-step_back:idx+step_forward]
            idx_needed = trialnum == trialnum[step_back]
            y = y[idx_needed]
            try:
                traces[i,idx_needed] = y
            except:
                traces[i,:] = y
        except:
            pass
    return traces

def export_trace_modulations(dFF,indices,step_back,step_forward,dozscore=False):
    maxvals = list()
    minvals = list()
    extrema = list()
    for dff in dFF:
        if dozscore:
            dff = stats.zscore(dff)
        reward_traces = event_based_trace_selection(dff,indices,trial_number_mask,step_back,step_forward)
        mean_reward_trace = np.nanmean(reward_traces,0)
        mean_reward_trace =mean_reward_trace -np.nanmean(mean_reward_trace[:step_back])
        maxval = np.max(mean_reward_trace)
        minval = np.min(mean_reward_trace)
        extremum = [maxval,minval][np.argmax(np.abs([maxval,minval]))]
        maxvals.append(maxval)
        minvals.append(minval)
        extrema.append(extremum)
        
    out_dict = {'minimum_values':minvals,
                'maximum_values':maxvals,
                'extrema':extrema,
                'event_indices':indices,
                'trial_number_mask':trial_number_mask}
    return out_dict

step_back_s = .5
step_forward_s = 3
step_back = int(step_back_s*fs)
step_forward = int(step_forward_s*fs)
motor_step_exclusion_window = 5 #frames
lick_exclusion_window = 5
dozscore = True


indices = np.where(reward_mask>0)[0]
modulation_dict_reward = export_trace_modulations(dFF,indices,step_back,step_forward,dozscore=dozscore)

motorstep_indices = np.where(motor_steps_mask>0)[0]
motorstep_indices_diff = np.concatenate([[3600],np.diff(motorstep_indices)])
motorstep_indices = motorstep_indices[motorstep_indices_diff>motor_step_exclusion_window]
modulation_dict_motor_step = export_trace_modulations(dFF,motorstep_indices,step_back,step_forward,dozscore=dozscore)   

indices = np.where(gocue_mask>0)[0]
modulation_dict_gocue = export_trace_modulations(dFF,indices,step_back,step_forward,dozscore=dozscore)
 
lick_indices = np.where(lick_mask>0)[0]
lick_indices_diff = np.concatenate([[3600],np.diff(lick_indices)])
lick_indices = lick_indices[lick_indices_diff>lick_exclusion_window]
modulation_dict_lick = export_trace_modulations(dFF,lick_indices,step_back,step_forward,dozscore=dozscore)

neuron_modulation = {'motor_step':modulation_dict_motor_step,
                     'gocue':modulation_dict_gocue,
                     'reward':modulation_dict_reward,
                     'lick':modulation_dict_lick}
neuron_modulation['motor_step']['color'] = 'black'
neuron_modulation['gocue']['color'] = 'green'
neuron_modulation['reward']['color'] = 'red'
neuron_modulation['lick']['color'] = 'blue'
#%% plot neuron modulations
positive_neuron_n = 10
negative_neuron_n = 10
step_back_s = 2
step_forward_s = 5
step_back = int(step_back_s*fs)
step_forward = int(step_forward_s*fs)
transient_time = np.arange(-step_back,step_forward)/fs
modulation_dict = neuron_modulation
keynum = len(modulation_dict.keys())
fig_modulation = plt.figure(figsize = [15,10])
spec2 = gridspec.GridSpec(ncols=keynum, nrows=2, figure=fig_modulation)
ax_modulation_dict= {}
for i,key in enumerate(modulation_dict.keys()):
    ax_modulation_dict['ax_{}_meantraces'.format(key)] = fig_modulation.add_subplot(spec2[0, i])
    ax_modulation_dict['ax_{}_meantraces'.format(key)].set_title(key)
    order = np.argsort(modulation_dict[key]['extrema'])[::-1]
    previous_value = 0
    for idx in order[:positive_neuron_n]:
        ys = event_based_trace_selection(dFF[idx,:],
                                         modulation_dict[key]['event_indices'],
                                         modulation_dict[key]['trial_number_mask'],
                                         step_back,step_forward)
        y = np.nanmean(ys,0)
        previous_value -= np.nanmax(y)
        ax_modulation_dict['ax_{}_meantraces'.format(key)].plot(transient_time,y+previous_value,'g-')
        
        ax_modulation_dict['ax_{}_meantraces'.format(key)].text(-1*step_back/fs,previous_value,cell_indices[idx])
    previous_value -= 1
    
    for idx in order[-negative_neuron_n:]:
        ys = event_based_trace_selection(dFF[idx,:],
                                         modulation_dict[key]['event_indices'],
                                         modulation_dict[key]['trial_number_mask'],
                                         step_back,step_forward)
        y = np.nanmean(ys,0)
        previous_value -= np.nanmax(y)
        ax_modulation_dict['ax_{}_meantraces'.format(key)].plot(transient_time,y+previous_value,'r-')
        ax_modulation_dict['ax_{}_meantraces'.format(key)].text(-1*step_back/fs,previous_value,cell_indices[idx])
    #break
    ax_modulation_dict['ax_{}_meantraces'.format(key)].set_xlim([-1*step_back_s, step_forward_s])
