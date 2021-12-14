import os 

from pathlib import Path

import numpy as np
from suite2p import default_ops as s2p_default_ops
from suite2p import classification
import shutil
import time
import datetime

from utils import utils_imaging,utils_pipeline,utils_imaging, utils_plot
import datajoint as dj
from pipeline import pipeline_tools,lab,experiment, imaging
import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import tifffile
from skimage.measure import label
#
%matplotlib qt
#%% read digested behavior data and fluorescence traces, align, find cell candidates
save_basedir = '/home/rozmar/Data/Calcium_imaging/BCI_export'
save_data = False

setup = 'KayvonScope'
subject = 'BCI_03'
session = '051121'#

# =============================================================================
# 
# setup = 'DOM3-MMIMS'
# subject = 'BCI_10'
# session = '2021-05-29'
# =============================================================================

setup = 'DOM3-MMIMS'
subject = 'BCI_11'
session = '2021-10-18'
# =============================================================================
# raw_imaging_dir = '/home/rozmar/Data/Calcium_imaging/raw/DOM3-MMIMS/BCI_07/2021-02-15'
# suite2p_dir = '/home/rozmar/Data/Calcium_imaging/suite2p/DOM3-MMIMS/BCI_07/2021-02-15'
# bpod_exported = '/home/rozmar/Data/Behavior/BCI_exported/DOM3-MMIMS/BCI_07/2021-02-15-bpod_zaber.npy'
# =============================================================================

raw_imaging_dir = '/home/rozmar/Data/Calcium_imaging/raw/{}/{}/{}'.format(setup,subject,session)
suite2p_dir = '/home/rozmar/Data/Calcium_imaging/suite2p/{}/{}/{}'.format(setup,subject,session)
bpod_exported = '/home/rozmar/Data/Behavior/BCI_exported/{}/{}/{}-bpod_zaber.npy'.format(setup,subject,session)

behavior_dict = np.load(bpod_exported,allow_pickle = True).tolist()
ops = np.load(os.path.join(suite2p_dir,'ops.npy'),allow_pickle = True).tolist()
#create mean images tiff
files_now = os.listdir(suite2p_dir)
if 'meanimages.tiff' not in files_now:
    imgs = np.load(os.path.join(suite2p_dir,'meanImg.npy'))
    imgs = np.asarray(imgs,dtype = np.int32)
    tifffile.imsave(os.path.join(suite2p_dir,'meanimages.tiff'),imgs)

stat = np.load(os.path.join(suite2p_dir,'stat.npy'),allow_pickle = True).tolist()
iscell = np.load(os.path.join(suite2p_dir,'iscell.npy'))
stat = np.asarray(stat)[iscell[:,0]==1].tolist()

try:
    dF = np.load(os.path.join(suite2p_dir,'dF.npy'))[iscell[:,0]==1,:]
    dFF = np.load(os.path.join(suite2p_dir,'dFF.npy'))[iscell[:,0]==1,:]
    F = np.load(os.path.join(suite2p_dir,'F.npy'))[iscell[:,0]==1,:]
    Fneu = np.load(os.path.join(suite2p_dir,'Fneu.npy'))[iscell[:,0]==1,:]
except:
    print('calculating dff')
    utils_imaging.export_dff(suite2p_dir,raw_imaging_dir=raw_imaging_dir,revert_background_subtraction = True)
    dFF = np.load(os.path.join(suite2p_dir,'dFF.npy'))[iscell[:,0]==1,:]
    dF = np.load(os.path.join(suite2p_dir,'dF.npy'))[iscell[:,0]==1,:]
    F = np.load(os.path.join(suite2p_dir,'F.npy'))[iscell[:,0]==1,:]
    Fneu = np.load(os.path.join(suite2p_dir,'Fneu.npy'))[iscell[:,0]==1,:]
fs = ops['fs']
with open(os.path.join(suite2p_dir,'filelist.json')) as f:
    filelist_dict = json.load(f)


#%
motor_steps_mask = np.zeros(dFF.shape[1])
frame_times = np.zeros(dFF.shape[1])*np.nan
gocue_mask = np.zeros(dFF.shape[1])
lick_mask = np.zeros(dFF.shape[1])
zpos_mask = np.zeros(dFF.shape[1])
reward_mask = np.zeros(dFF.shape[1])
unreward_mask = np.zeros(dFF.shape[1])
threshold_crossing_masks = np.zeros(dFF.shape[1])
task_mask = np.zeros(dFF.shape[1])
reward_consumtion_mask =  np.zeros(dFF.shape[1])
baseline_length = np.nan
trial_number_mask =  np.zeros(dFF.shape[1])*np.nan
prev_frames_so_far = 0
conditioned_neuron_name_list = []
try:
    zcorr_argmax = np.argmax(ops['zcorr'],1)
except:
    zcorr_argmax  = np.zeros(len(ops['filelist']))
for i,filename in enumerate(behavior_dict['scanimage_file_names']): # generate behavior related vectors
    if filename[0] not in filelist_dict['file_name_list']:
        continue
    movie_idx = np.where(np.asarray(filelist_dict['file_name_list'])==filename[0])[0][0]
    if movie_idx == 0 :
        frames_so_far = 0
    else:
        frames_so_far = np.sum(np.asarray(filelist_dict['frame_num_list'])[:movie_idx])
    frame_num_in_trial = np.asarray(filelist_dict['frame_num_list'])[movie_idx]  
    
    zpos_mask[frames_so_far:frames_so_far+frame_num_in_trial] = zcorr_argmax[movie_idx]
    
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
    
#%
for i,filename in enumerate(behavior_dict['scanimage_file_names']):    #find names of conditioned neurons
    if len(behavior_dict['scanimage_roi_outputChannelsRoiNames'][i]) >0:
        for roi_fcn_i, roi_fnc_name in enumerate(behavior_dict['scanimage_roi_outputChannelsNames'][i]):
            bpod_analog_idx = np.nan
            if 'analog' in roi_fnc_name:
                bpod_analog_idx = roi_fcn_i
                break
        try:
            conditioned_neuron_name = (behavior_dict['scanimage_roi_outputChannelsRoiNames'][i][bpod_analog_idx])[0]
        except:
            conditioned_neuron_name = (behavior_dict['scanimage_roi_outputChannelsRoiNames'][i][bpod_analog_idx])
        if len(conditioned_neuron_name) == 0:
            conditioned_neuron_name = ''
    else:
        conditioned_neuron_name  =''
    conditioned_neuron_name_list.append(conditioned_neuron_name)
        
            
 #%    
if baseline_length ==0: # this one tries to guess the end of the baseline based on filenames
    for i,filename in enumerate(behavior_dict['scanimage_file_names']):
        if 'baseline' in filename[0]:
            try:
                movie_idx = np.where(np.asarray(filelist_dict['file_name_list'])==filename[0])[0][0]
                baseline_length = np.sum(np.asarray(filelist_dict['frame_num_list'])[:movie_idx])
            except:
                pass
        
conditioned_neuron_names = np.unique(np.asarray(conditioned_neuron_name_list)[np.asarray(conditioned_neuron_name_list)!=''])       
if len(conditioned_neuron_names)>1:
    conditioned_neuron_name = np.asarray(conditioned_neuron_name_list)[::-1][np.argmax(np.asarray(conditioned_neuron_name_list)[::-1]!='')]
    print('MULTIPLE CONDITIONED NEURONS! going with the last one: {}'.format(conditioned_neuron_name))
else:
    try:
        conditioned_neuron_name = conditioned_neuron_names[0]
        trial_num = np.argmax(np.asarray(conditioned_neuron_name_list) ==conditioned_neuron_name)
        #trial_num = len(conditioned_neuron_name_list)-1-np.argmax(np.asarray(conditioned_neuron_name_list)[::-1] ==conditioned_neuron_name)
        filename = behavior_dict['scanimage_file_names'][trial_num][0]
        metadata = utils_imaging.extract_scanimage_metadata(os.path.join(raw_imaging_dir,filename))
        rois = metadata['metadata']['json']['RoiGroups']['integrationRoiGroup']['rois']   
        roinames_list = list() 
        for roi in rois:
            roinames_list.append(roi['name'])
        roi_idx = np.where(np.asarray(roinames_list)==conditioned_neuron_name)[0][0]+1

    except:
        trial_num = len(behavior_dict['scanimage_file_names'])
        for filename in behavior_dict['scanimage_file_names'][::-1]:
            trial_num -= 1
            if type(filename) == np.ndarray:
                filename = filename[0]
                if 'open' not in filename:
                    break
            
            
        file_base = filename[:filename.rfind('_')]
        if 'vs' in file_base:
            multi_neuron_conditioning = True
            roi_idx = [int(re.findall(r'\d+', file_base)[-2]),int(re.findall(r'\d+', file_base)[-1])]
            roi_sign = [1,-1]
        else:
            multi_neuron_conditioning = False
            roi_idx = [int(re.findall(r'\d+', file_base)[-1])]
            roi_sign = [1]
        
#%       

# =============================================================================
# dFF = np.load(os.path.join(suite2p_dir,'dFF.npy'))[iscell[:,0]==1,:]
# # =============================================================================
# # dFF = np.load(os.path.join(suite2p_dir,'spks.npy'))[iscell[:,0]==1,:]
# # dFF = dFF/np.max(dFF,1)[:,np.newaxis]
# # =============================================================================
# #dFF = np.load(os.path.join(suite2p_dir,'F.npy'))[iscell[:,0]==1,:]
# #dFF = np.load(os.path.join(suite2p_dir,'Fneu.npy'))[iscell[:,0]==1,:]
# =============================================================================

if save_data:
    # have to find conditioned neuron index
    filename = behavior_dict['scanimage_file_names'][trial_num][0]
    metadata = utils_imaging.extract_scanimage_metadata(os.path.join(raw_imaging_dir,filename))
    fovdeg = list()
    for s in metadata['metadata']['hRoiManager']['imagingFovDeg'].strip('[]').split(' '): fovdeg.extend(s.split(';'))
    fovdeg = np.asarray(fovdeg,float)
    fovdeg = [np.min(fovdeg),np.max(fovdeg)]
    rois = metadata['metadata']['json']['RoiGroups']['integrationRoiGroup']['rois']    
    if type(rois) == dict:
        rois = [rois]    
    centerXY_list = list()
    roinames_list = list()
    Lx = float(metadata['metadata']['hRoiManager']['pixelsPerLine'])
    Ly = float(metadata['metadata']['hRoiManager']['linesPerFrame'])
    ax_meanimage.imshow(ops['meanImg'])#,cmap = 'gray')
    cond_s2p_idx = list()
    for roi in rois:
        try:
            centerXY_list.append((roi['scanfields']['centerXY']-fovdeg[0])/np.diff(fovdeg))
        except:
            print('multiple scanfields for {}'.format(roi['name']))
            centerXY_list.append((roi['scanfields'][0]['centerXY']-fovdeg[0])/np.diff(fovdeg))
        ax_meanimage.plot(centerXY_list[-1][0]*Lx,centerXY_list[-1][1]*Ly,'ko')
        roinames_list.append(roi['name'])
    for roi_idx_now in roi_idx:
        conditioned_coordinates = [centerXY_list[roi_idx_now-1][0]*Lx,centerXY_list[roi_idx_now-1][1]*Ly]
        med_list = list()
        dist_list = list()
        for cell_stat in stat:
            ax_meanimage.plot(cell_stat['med'][1],cell_stat['med'][0],'kx')
            dist = np.sqrt((centerXY_list[roi_idx_now-1][0]*Lx-cell_stat['med'][1])**2+(centerXY_list[roi_idx_now-1][1]*Lx-cell_stat['med'][0])**2)
            dist_list.append(dist)
            med_list.append(cell_stat['med'])
            #break
        cond_s2p_idx.append(np.argmin(dist_list))
    
    cond_s2p_idx = np.asarray(cond_s2p_idx)
    save_dir = os.path.join(save_basedir,setup,subject,session)
    Path(save_dir).mkdir(parents = True,exist_ok = True)
    #%
    miss = list()
    ignore = list()
    reward_volume = list()
    for i,(reward_R,reward_L,trial_hit,threshold_crossing_time) in enumerate(zip(behavior_dict['reward_R'],behavior_dict['reward_L'],behavior_dict['trial_hit'],behavior_dict['threshold_crossing_times'])):
        if trial_hit:
            miss.append(False)
            ignore.append(False)
            if len(reward_R)>0:
                reward_volume.append(behavior_dict['var_ValveOpenTime_R'][i])
            elif len(reward_L)>0:
                reward_volume.append(behavior_dict['var_ValveOpenTime_L'][i])
            else:
                print('WTF hit with no reward???')
        else:
            reward_volume.append(0)
            if len(threshold_crossing_time)>0:
                ignore.append(True)
                miss.append(False)
            else:
                ignore.append(False)
                miss.append(True)
    reward_volume = np.asarray(reward_volume)
            
        #%     
            
            
        
    metadata_masks = {'motor_steps':motor_steps_mask,
                      'trial_start':gocue_mask,
                      'reward':reward_mask,
                      'lick':lick_mask,
                      'threshold_crossing':threshold_crossing_masks,
                      'trial_number':trial_number_mask,
                      'conditioned_neuron_index':np.arange(iscell.shape[0])[iscell[:,0]==1][cond_s2p_idx],
                      'frame_rate':ops['fs'],
                      'trial_lickport_step_size':behavior_dict['zaber_trigger_step_size']/1000,
                      'trial_lickport_travel_distance':behavior_dict['zaber_limit_far']-behavior_dict['zaber_reward_zone'],
                      'trial_start_times':behavior_dict['trial_start_times'],
                      'trial_hit':behavior_dict['trial_hit'],
                      'trial_miss':np.asarray(miss),
                      'trial_ignore':np.asarray(ignore),
                      'trial_reward_size':reward_volume,
                      }
    np.save(os.path.join(save_dir,'behavior.npy'),metadata_masks)
    files_to_copy = ['F.npy','Fneu.npy','iscell.npy','ops.npy','spks.npy','stat.npy']
    print('copying files')
    for file in files_to_copy:
        shutil.copy(os.path.join(suite2p_dir,file),os.path.join(save_dir,file))
        #%
    try:
        session_date = datetime.datetime.strptime(session,'%m%d%y').date()
    except:
        try:
            session_date = datetime.datetime.strptime(session,'%Y-%m-%d').date()
        except:
            print('cannot understand date for session dir: {}'.format(session))
            session_date = 'lol'
            #%
    subject_id = (lab.WaterRestriction()&'water_restriction_number = "{}"'.format(subject.replace('_',''))).fetch1('subject_id')
    sessions,session_dates = (experiment.Session()&'subject_id = {}'.format(subject_id)).fetch('session','session_date')
    session = sessions[np.where(session_dates == session_date)[0][0]]
    session_key_wr = {'wr_id':subject.replace('_',''), 'session':session}
    moving_window = 10
    behav_fig = utils_plot.plot_behavior_session(session_key_wr,moving_window)
    behav_fig.savefig(os.path.join(save_dir,'session_behavior.png'))
#%%
#dFF = F#np.mean(F,0).shape
window_size_seconds = 120
moving_window_size = int(fs*window_size_seconds) # moving window
#%
#mask_to_use = np.ones(len(task_mask))#task_mask#task_mask#reward_consumtion_mask#np.ones(len(task_mask))#
mask_to_use_name = 'everything'#'task'#
if mask_to_use_name  == 'task':
    mask_to_use = task_mask#task_mask#task_mask#reward_consumtion_mask#np.ones(len(task_mask))#
elif mask_to_use_name  == 'reward':    
    mask_to_use = reward_consumtion_mask#task_mask#task_mask#reward_consumtion_mask#np.ones(len(task_mask))#
elif mask_to_use_name  == 'everything':    
    mask_to_use = np.ones(len(task_mask))#task_mask#task_mask#reward_consumtion_mask#np.ones(len(task_mask))#
else:
    mask_to_use = None
subtract_baseline_slope = True

fig_meanimage = plt.figure()
spec2 = gridspec.GridSpec(ncols=5, nrows=3, figure=fig_meanimage)
ax_conditioned_neuron =fig_meanimage.add_subplot(spec2[0, :])
ax_meanimage =fig_meanimage.add_subplot(spec2[1, 0])
ax_rois = fig_meanimage.add_subplot(spec2[2, 0],sharex = ax_meanimage,sharey = ax_meanimage)
ax_cum= fig_meanimage.add_subplot(spec2[1, 1])
ax_cum_normalized = fig_meanimage.add_subplot(spec2[2, 1])
ax2 = ax_cum_normalized.twinx()
ax_samples_high = fig_meanimage.add_subplot(spec2[1,2])
ax_samples_low = fig_meanimage.add_subplot(spec2[2, 2])

ax_average_activity = fig_meanimage.add_subplot(spec2[1, 3])
ax_average_activity_normalized = fig_meanimage.add_subplot(spec2[2, 3])

ax_mean_activity_modulation_samples = fig_meanimage.add_subplot(spec2[1, 4])
ax_mean_activity_modulation_to_baseline_samples = fig_meanimage.add_subplot(spec2[2,4])

ax_conditioned_neuron.set_title(raw_imaging_dir)
#%
ax_cum.set_title('cumulative activity during {}'.format(mask_to_use_name))
ax_cum_normalized.set_title('normalized cumulative activity during {}'.format(mask_to_use_name))

ax_samples_high.set_title('activity of increasing cells')
ax_samples_low.set_title('activity of decreasing cells')

ax_average_activity.set_title('mean activity in {} seconds during {}'.format(window_size_seconds,mask_to_use_name))
ax_average_activity_normalized.set_title('normalized mean activity in {} seconds during {}'.format(window_size_seconds,mask_to_use_name))

#%



filename = behavior_dict['scanimage_file_names'][trial_num][0]
#file_base = filename[:filename.rfind('_')]
#roi_idx = int(re.findall(r'\d+', file_base)[-1])
metadata = utils_imaging.extract_scanimage_metadata(os.path.join(raw_imaging_dir,filename))

fovdeg = list()
for s in metadata['metadata']['hRoiManager']['imagingFovDeg'].strip('[]').split(' '): fovdeg.extend(s.split(';'))
fovdeg = np.asarray(fovdeg,float)
fovdeg = [np.min(fovdeg),np.max(fovdeg)]
rois = metadata['metadata']['json']['RoiGroups']['integrationRoiGroup']['rois']    
if type(rois) == dict:
    rois = [rois]    
centerXY_list = list()
roinames_list = list()
Lx = float(metadata['metadata']['hRoiManager']['pixelsPerLine'])
Ly = float(metadata['metadata']['hRoiManager']['linesPerFrame'])
ax_meanimage.imshow(ops['meanImg'])#,cmap = 'gray')
for roi in rois:
    try:
        centerXY_list.append((roi['scanfields']['centerXY']-fovdeg[0])/np.diff(fovdeg))
    except:
        print('multiple scanfields for {}'.format(roi['name']))
        centerXY_list.append((roi['scanfields'][0]['centerXY']-fovdeg[0])/np.diff(fovdeg))
    ax_meanimage.plot(centerXY_list[-1][0]*Lx,centerXY_list[-1][1]*Ly,'ko')
    roinames_list.append(roi['name'])
    #centerXY_morphed.append(np.asarray(roi['scanfields']['affine']) @ np.concatenate([roi['scanfields']['centerXY'],[0]]))
    #break
cond_s2p_idx = list()
for roi_idx_now,sign in zip(roi_idx,roi_sign):    
    if sign>0:
        color = 'green'
    else:
        color = 'red'
    ax_meanimage.plot(centerXY_list[roi_idx_now-1][0]*Lx,centerXY_list[roi_idx_now-1][1]*Ly,'o',color = color)
    #% find conditioned neuron coordinates
    conditioned_coordinates = [centerXY_list[roi_idx_now-1][0]*Lx,centerXY_list[roi_idx_now-1][1]*Ly]
    med_list = list()
    dist_list = list()
    for cell_stat in stat:
        ax_meanimage.plot(cell_stat['med'][1],cell_stat['med'][0],'kx')
        dist = np.sqrt((centerXY_list[roi_idx_now-1][0]*Lx-cell_stat['med'][1])**2+(centerXY_list[roi_idx_now-1][1]*Lx-cell_stat['med'][0])**2)
        dist_list.append(dist)
        med_list.append(cell_stat['med'])
        #break
    cond_s2p_idx.append(np.argmin(dist_list))
offset = 0
for cond_s2p_idx_now,color_now in zip(cond_s2p_idx,['green','red']):
    ax_meanimage.plot(med_list[cond_s2p_idx_now][1],med_list[cond_s2p_idx_now][0],'rx')
    ax_meanimage.set_title('conditioned ROI: {}'.format(cond_s2p_idx))
#%
#baseline_length = 5000


    ax_conditioned_neuron.plot(dFF[cond_s2p_idx_now,:]+offset,'-',color = color_now,alpha = .5)
    #offset += np.max(dFF[cond_s2p_idx_now,:]+offset)
    maxval = np.max(dFF[cond_s2p_idx_now,:])
    minval = np.min(dFF[cond_s2p_idx_now,:])
scale = (maxval-minval)*.1


zposition_norm = zpos_mask - np.min(zpos_mask)
zposition_norm = zposition_norm/np.max(zposition_norm)

ax_conditioned_neuron.plot(lick_mask*scale-scale ,'b-',label = 'lick')#,color = 'purple')
ax_conditioned_neuron.plot(reward_mask *scale-scale*2,'r-',label = 'reward')
ax_conditioned_neuron.plot(motor_steps_mask/np.max(motor_steps_mask)*scale-scale*3,'k-',label = 'motor step')
ax_conditioned_neuron.plot(gocue_mask*scale-scale*2,'g-',label = 'go cue')#%% reward correlation of the conditioned neuron
ax_conditioned_neuron.plot(zposition_norm*scale-scale*4,'c-',label = 'z position: range {} steps'.format(np.max(zpos_mask)-np.min(zpos_mask)))#%% reward correlation of the conditioned neuron
ax_conditioned_neuron.legend()


# =============================================================================
# task_label = label(task_mask)
# for i in np.arange(1,np.max(task_label)):
#     idx = np.where(task_label==i)[0]
#     ax_conditioned_neuron.plot(idx,dFF[cond_s2p_idx,idx],'g-')
# reward_consumption_label = label(reward_consumtion_mask)    
# for i in np.arange(1,np.max(reward_consumption_label)):
#     idx = np.where(reward_consumption_label==i)[0]
#     ax_conditioned_neuron.plot(idx,dFF[cond_s2p_idx,idx],'r-')
# =============================================================================
ax_conditioned_neuron.plot(np.where(unreward_mask ==1)[0],np.zeros_like(np.where(unreward_mask ==1)[0])-1,'ko')
#%
final_values = list()
mean_activity_modulations = list()
mean_activity_modulations_to_baseline = list()
baseline_length_real = int(sum(mask_to_use[:baseline_length]))
if baseline_length_real == 0:
    print('baseline was too low, using 1000 frames')
    baseline_length_real = 1000
#baseline_length_real = 25000
x_axis = np.arange(dFF.shape[1])
for i,dff in enumerate(dFF): # plotting cumulative and average activity

    dff = dff[mask_to_use==1]
    if subtract_baseline_slope:
        baseline_rate = np.cumsum(dff)[baseline_length_real]/baseline_length_real
    else:
        baseline_rate = 0
    vector = np.cumsum(dff)-np.arange(len(dff))*baseline_rate
    ax_cum_normalized.plot(x_axis[mask_to_use==1],vector,'k-',alpha = .3)
    ax_cum.plot(x_axis[mask_to_use==1],np.cumsum(dff),'k-',alpha = .3)
    final_values.append(vector[-1])
    
    dff_mean = np.convolve(dff,np.ones(moving_window_size)/moving_window_size,mode = 'same')#utils_plot.rollingfun(dff,moving_window_size)
    ax_average_activity.plot(x_axis[mask_to_use==1],dff_mean,'-',alpha = .3)
    ax_average_activity_normalized.semilogy(x_axis[mask_to_use==1],dff_mean/np.mean(dff_mean[:baseline_length_real]),'-',alpha = .3)
    maxval = np.max(dff_mean[500:])
    maxidx = np.argmax(dff_mean[500:])+500
    minval = np.min(dff_mean[:maxidx])
    minidx = np.argmin(dff_mean[:maxidx])
    ax_average_activity.plot(x_axis[mask_to_use==1][minidx],dff_mean[minidx],'kv',alpha = .5)
    ax_average_activity.plot(x_axis[mask_to_use==1][maxidx],dff_mean[maxidx],'k^',alpha = .5)
    mean_activity_modulations.append(maxval-minval)
    mean_activity_modulations_to_baseline.append(maxval-np.mean(dff_mean[:baseline_length_real]))

for cond_s2p_idx_now,sign in zip(cond_s2p_idx,roi_sign):
    if sign>0:
        color = 'green'
    else:
        color = 'red'
    dff = dFF[cond_s2p_idx_now,:]
    dff = dff[mask_to_use==1]
    if subtract_baseline_slope:
        baseline_rate = np.cumsum(dff)[baseline_length_real]/baseline_length_real    
    else:
        baseline_rate =0
    ax_cum_normalized.plot(x_axis[mask_to_use==1],np.cumsum(dff)-np.arange(len(dff))*baseline_rate,'-',color = color)#-np.arange(dFF.shape[1])/dFF.shape[1]*np.sum(dFF[cond_s2p_idx,:]))
    ax_cum.plot(x_axis[mask_to_use==1],np.cumsum(dff),'-',color = color)
    
    dff_mean = np.convolve(dff,np.ones(moving_window_size)/moving_window_size,mode = 'same')#utils_plot.rollingfun(dff,moving_window_size)
    ax_average_activity.plot(x_axis[mask_to_use==1],dff_mean,'-',linewidth = 4,color = color)
    ax_average_activity_normalized.semilogy(x_axis[mask_to_use==1],dff_mean/np.mean(dff_mean[:baseline_length_real]),'-',linewidth = 4,color = color)
#% sample traces based on cumulative activity
order = np.argsort(final_values)[::-1]
prev_max = 0
for idx in order[:30]:
    if idx in cond_s2p_idx:
        if np.asarray(roi_sign)[cond_s2p_idx==idx]>0:
            color = 'green'
        else:
            color = 'red'
    else:
        color = 'black'
    prev_max -= np.nanmax(dFF[idx,mask_to_use ==1])
    ax_samples_high.plot(x_axis[mask_to_use==1],dFF[idx,mask_to_use==1]+prev_max,'-',alpha = 1,color = color)
    
for idx in order[-30:]:
    if idx in cond_s2p_idx:
        if np.asarray(roi_sign)[cond_s2p_idx==idx]>0:
            color = 'green'
        else:
            color = 'red'
    else:
        color = 'black'
    prev_max -= np.nanmax(dFF[idx,mask_to_use ==1])
    ax_samples_low.plot(x_axis[mask_to_use==1],dFF[idx,mask_to_use==1]+prev_max,'-',alpha = 1,color = color)

 # maximum modulations   
order = np.argsort(mean_activity_modulations)[::-1]
prev_max = 0
for idx in order[:30]:
    if idx in cond_s2p_idx:
        if np.asarray(roi_sign)[cond_s2p_idx==idx]>0:
            color = 'green'
        else:
            color = 'red'
    else:
        color = 'black'
    prev_max -= np.nanmax(dFF[idx,mask_to_use ==1])
    ax_mean_activity_modulation_samples.plot(x_axis[mask_to_use==1],dFF[idx,mask_to_use==1]+prev_max,'-',alpha = 1,color = color)
order = np.argsort(mean_activity_modulations_to_baseline)[::-1]
prev_max = 0
for idx in order[:30]:
    if idx in cond_s2p_idx:
        if np.asarray(roi_sign)[cond_s2p_idx==idx]>0:
            color = 'green'
        else:
            color = 'red'
    else:
        color = 'black'
    prev_max -= np.nanmax(dFF[idx,mask_to_use ==1])
    ax_mean_activity_modulation_to_baseline_samples.plot(x_axis[mask_to_use==1],dFF[idx,mask_to_use==1]+prev_max,'-',alpha = 1,color = color)
  
    
  
#%

#ax_rois.imshow(ops['meanImg'],cmap = 'gray')
mask = np.zeros_like(ops['meanImg'])
for final_value, roi_stat in zip(final_values,stat):
    mask[roi_stat['ypix'],roi_stat['xpix']] = final_value
    #break
maskimg = ax_rois.imshow(mask)#,alpha = .5)    #cmap = 'hot',
maskimg.set_clim(np.percentile(final_values,5), np.percentile(final_values,95))
fig_meanimage.colorbar(maskimg, ax=ax_rois)#, location='right')#, anchor=(0, 0.3), shrink=0.7)
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
negative_neuron_n = 5
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
    ax_modulation_dict['ax_{}_meantraces'] = fig_modulation.add_subplot(spec2[0, i])
    ax_modulation_dict['ax_{}_meantraces'].set_title(key)
    order = np.argsort(modulation_dict[key]['extrema'])[::-1]
    previous_value = 0
    for idx in order[:positive_neuron_n]:
        ys = event_based_trace_selection(dFF[idx,:],
                                         modulation_dict[key]['event_indices'],
                                         modulation_dict[key]['trial_number_mask'],
                                         step_back,step_forward)
        y = np.nanmean(ys,0)
        previous_value -= np.nanmax(y)
        ax_modulation_dict['ax_{}_meantraces'].plot(transient_time,y+previous_value,'g-')
        
    previous_value -= 1
    
    for idx in order[-negative_neuron_n:]:
        ys = event_based_trace_selection(dFF[idx,:],
                                         modulation_dict[key]['event_indices'],
                                         modulation_dict[key]['trial_number_mask'],
                                         step_back,step_forward)
        y = np.nanmean(ys,0)
        previous_value -= np.nanmax(y)
        ax_modulation_dict['ax_{}_meantraces'].plot(transient_time,y+previous_value,'r-')
#%%
# =============================================================================
# im = plt.imshow(dFF, interpolation='nearest', aspect='auto',cmap = 'magma')
# im.set_clim([-1,5])        
# =============================================================================
        
    #break
#%% rastermap
from rastermap import Rastermap
#traces = (dFF/np.max(dFF,1)[:,np.newaxis])
traces  = stats.zscore(dFF,axis = 1)
#traces = dFF
model = Rastermap(n_components=1, n_X=200, nPC=200, init='pca')
model.fit(traces)
#%
order =model.isort
#order = np.argsort(modulation_dict['reward']['extrema'])
fig_rastermap = plt.figure()
spec = gridspec.GridSpec(ncols = 9, nrows=5, figure=fig_rastermap)

ax_image = fig_rastermap.add_subplot(spec[:-1, :-1])
ax_behavior = fig_rastermap.add_subplot(spec[-1, :-1],sharex = ax_image)
ax_modulation = fig_rastermap.add_subplot(spec[:-1, -1],sharey = ax_image)
image_rastermap = traces[order, :]
#image_rastermap  = scipy.ndimage.gaussian_filter1d(image_rastermap, 1, axis=0)
im = ax_image.imshow(image_rastermap, interpolation='nearest', aspect='auto',cmap = 'magma')
im.set_clim([-1,4])
for i,key in enumerate(modulation_dict.keys()):
    ax_modulation.plot(np.asarray(modulation_dict[key]['extrema'])[order],np.arange(len(modulation_dict[key]['extrema'])),'-',color = neuron_modulation[key]['color'],alpha = .3)
for cond_s2p_idx_now,sign in zip(cond_s2p_idx,roi_sign):
    if sign>0: 
        color = 'green'
    else:
        color = 'red'
    ax_modulation.plot(0,np.where(order==cond_s2p_idx_now)[0][0],'o',color= color)       #

ax_behavior.plot(motor_steps_mask/np.max(motor_steps_mask),'k-',label = 'lickport movement')
ax_behavior.plot(reward_mask + 1,'r-')
ax_behavior.plot(gocue_mask - 1,'g-')#%% reward correlation of the conditioned neuron
ax_behavior.plot(lick_mask +2,'b-')#,color = 'purple')
ax_behavior.set_yticks([-.5,.5,1.5,2.5])
ax_behavior.set_yticklabels(['Go cue','Lickport steps','Reward','Lick'])
#%% reward correlation of the conditioned neuron
#%
                 #dff = dFF[cond_s2p_idx,:]
dff = dFF[cond_s2p_idx[0],:]
step_back = 50
step_forward = 800
motor_step_exclusion_window = 5 #frames
lick_exclusion_window = 5
reward_bin_size = 30






fig_trial_traces = plt.figure(figsize = [15,15])
spec2 = gridspec.GridSpec(ncols=4, nrows=6, figure=fig_trial_traces)
ax_gocue = fig_trial_traces.add_subplot(spec2[0, 0])
ax_gocue_std = fig_trial_traces.add_subplot(spec2[1, 0])
ax_gocue_mean = fig_trial_traces.add_subplot(spec2[2:, 0])
ax_reward = fig_trial_traces.add_subplot(spec2[0, 1])
ax_reward_std = fig_trial_traces.add_subplot(spec2[1, 1])
ax_reward_mean = fig_trial_traces.add_subplot(spec2[2:, 1])
ax_step = fig_trial_traces.add_subplot(spec2[0, 2])
ax_step_std = fig_trial_traces.add_subplot(spec2[1, 2])
ax_step_mean = fig_trial_traces.add_subplot(spec2[2:, 2])

ax_lick = fig_trial_traces.add_subplot(spec2[0, 3])
ax_lick_std = fig_trial_traces.add_subplot(spec2[1, 3])
ax_lick_mean = fig_trial_traces.add_subplot(spec2[2:, 3])

ax_gocue.set_title('Go cue')
ax_reward.set_title('Reward')
ax_step.set_title('Motor step')
ax_lick.set_title('Lick')


reward_traces = event_based_trace_selection(dff,np.where(reward_mask>0)[0],trial_number_mask,step_back,step_forward)
x = np.arange(-step_back,step_forward)/fs
for i,y in enumerate(reward_traces):
    color = plt.cm.inferno(i/reward_traces.shape[0])
    ax_reward.plot(x,y,alpha = .1,color = color)
ax_reward_std.plot(x,np.nanmean(reward_traces,0),'k-',linewidth = 3,alpha = 1)
ax_reward_std.fill_between(x,np.nanmean(reward_traces,0)-np.nanstd(reward_traces,0),np.nanmean(reward_traces,0)+np.nanstd(reward_traces,0),color = 'black',alpha = .5)


gocue_traces = event_based_trace_selection(dff,np.where(gocue_mask>0)[0],trial_number_mask,step_back,step_forward)
for i,y in enumerate(gocue_traces):
    color = plt.cm.inferno(i/gocue_traces.shape[0])
    ax_gocue.plot(x,y,alpha = .1,color = color)
ax_gocue_std.plot(x,np.nanmean(gocue_traces,0),'k-',linewidth = 3,alpha = 1)
ax_gocue_std.fill_between(x,np.nanmean(gocue_traces,0)-np.nanstd(gocue_traces,0),np.nanmean(gocue_traces,0)+np.nanstd(gocue_traces,0),color = 'black',alpha = .5)
#%
motorstep_indices = np.where(motor_steps_mask>0)[0]
motorstep_indices_diff = np.concatenate([[3600],np.diff(motorstep_indices)])
motorstep_indices = motorstep_indices[motorstep_indices_diff>motor_step_exclusion_window]
motorstep_traces = event_based_trace_selection(dff,motorstep_indices,trial_number_mask,step_back,step_forward)
for i,y in enumerate(motorstep_traces):
    color = plt.cm.inferno(i/motorstep_traces.shape[0])
    ax_step.plot(x,y,alpha = .1,color = color)
ax_step_std.plot(x,np.nanmean(motorstep_traces,0),'k-',linewidth = 3,alpha = 1)
ax_step_std.fill_between(x,np.nanmean(motorstep_traces,0)-np.nanstd(motorstep_traces,0),np.nanmean(motorstep_traces,0)+np.nanstd(motorstep_traces,0),color = 'black',alpha = .5)

lick_indices = np.where(lick_mask>0)[0]
lick_indices_diff = np.concatenate([[3600],np.diff(lick_indices)])
lick_indices = lick_indices[lick_indices_diff>lick_exclusion_window]
lick_traces = event_based_trace_selection(dff,lick_indices,trial_number_mask,step_back,step_forward)
for i,y in enumerate(lick_traces):
    color = plt.cm.inferno(i/lick_traces.shape[0])
    ax_lick.plot(x,y,alpha = .1,color = color)
ax_lick_std.plot(x,np.nanmean(lick_traces,0),'k-',linewidth = 3,alpha = 1)
ax_lick_std.fill_between(x,np.nanmean(lick_traces,0)-np.nanstd(lick_traces,0),np.nanmean(lick_traces,0)+np.nanstd(lick_traces,0),color = 'black',alpha = .5)

#%
x = np.arange(-step_back,step_forward)/fs
bin_num = len(np.arange(reward_traces.shape[0]/reward_bin_size+1))
prev_value = 0
for i in np.arange(reward_traces.shape[0]/reward_bin_size+1):
    y= np.nanmean(reward_traces[int(i*reward_bin_size):int((i+1)*reward_bin_size)],0) 
    color = plt.cm.inferno(i/bin_num)
    ax_reward_mean.plot(x,y+ prev_value,alpha = .8,color = color)##-y[step_back]
    prev_value += np.nanmax(y)
ax_reward_mean.vlines(0,ax_reward_mean.get_ylim()[0],ax_reward_mean.get_ylim()[1],'r')

x = np.arange(-step_back,step_forward)/fs
bin_num = len(np.arange(gocue_traces.shape[0]/reward_bin_size+1))
prev_value = 0
for i in np.arange(gocue_traces.shape[0]/reward_bin_size+1):
    y= np.nanmean(gocue_traces[int(i*reward_bin_size):int((i+1)*reward_bin_size)],0) 
    color = plt.cm.inferno(i/bin_num)
    ax_gocue_mean.plot(x,y+ prev_value,alpha = .8,color = color)##-y[step_back]
    prev_value += np.nanmax(y)
ax_gocue_mean.vlines(0,ax_gocue_mean.get_ylim()[0],ax_gocue_mean.get_ylim()[1],'r')

x = np.arange(-step_back,step_forward)/fs
bin_num = len(np.arange(motorstep_traces.shape[0]/reward_bin_size+1))
prev_value = 0
for i in np.arange(motorstep_traces.shape[0]/reward_bin_size+1):
    y= np.nanmean(motorstep_traces[int(i*reward_bin_size):int((i+1)*reward_bin_size)],0) 
    color = plt.cm.inferno(i/bin_num)
    ax_step_mean.plot(x,y+ prev_value,alpha = .8,color = color)##-y[step_back]
    prev_value += np.nanmax(y)
ax_step_mean.vlines(0,ax_step_mean.get_ylim()[0],ax_step_mean.get_ylim()[1],'r')

bin_num = len(np.arange(lick_traces.shape[0]/reward_bin_size+1))
prev_value = 0
for i in np.arange(lick_traces.shape[0]/reward_bin_size+1):
    y= np.nanmean(lick_traces[int(i*reward_bin_size):int((i+1)*reward_bin_size)],0) 
    color = plt.cm.inferno(i/bin_num)
    ax_lick_mean.plot(x,y+ prev_value,alpha = .8,color = color)##-y[step_back]
    prev_value += np.nanmax(y)
ax_lick_mean.vlines(0,ax_lick_mean.get_ylim()[0],ax_lick_mean.get_ylim()[1],'r')
#%% 


#%% SNR of cells
dFF_scaled = list()
max_SNR = list()
for dff in dFF:
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
needed = np.asarray(max_SNR)>10
#%% save data
metadata_masks = {'motor_steps':motor_steps_mask,
                  'trial_start':gocue_mask,
                  'reward':reward_mask,
                  'lick':lick_mask,
                  'trial_number':trial_number_mask,
                  'conditioned_neuron_index':np.arange(iscell.shape[0])[iscell[:,0]==1][cond_s2p_idx],
                  }
np.save(os.path.join(suite2p_dir,'behavior.npy'),metadata_masks)
#%%
np.save(os.path.join(suite2p_dir,'motor_steps.npy'),motor_steps_mask)
#%%
import numpy as np
import os
import matplotlib.pyplot as plt
%matplotlib inline
source_dir = '/home/rozmar/Network/rozsam_google_drive/Data_to_share/BCI/BCI_07/042121'
dFF = np.load(os.path.join(source_dir,'dFF.npy'))
iscell = np.load(os.path.join(source_dir,'iscell.npy'))
behavior = np.load(os.path.join(source_dir,'behavior.npy'),allow_pickle = True).tolist()
ops = np.load(os.path.join(source_dir,'ops.npy'),allow_pickle = True).tolist()
stat = np.load(os.path.join(source_dir,'stat.npy'),allow_pickle = True).tolist()
#%%
dff_conditioned_neuron = dFF[behavior['conditioned_neuron_index'],:]
stat_conditioned_neuron = np.asarray(stat)[behavior['conditioned_neuron_index']]
dFF_cells = dFF[iscell[:,0]==1,:]
stat_cells = np.asarray(stat)[iscell[:,0]==1]
sampling_rate = ops['fs']
meanImg = ops['meanImg']
#%% plot IV
import h5py as h5
from acq4_utils import configfile
def read_h5f_metadata(metadata_h5):
    keys_0 = metadata_h5.keys()
    metadata = None
    for key_0 in keys_0:
        if metadata == None:
            if key_0 == '0':
                metadata = list()
            else:
                metadata = dict()
        if type(metadata_h5[key_0]) == h5._hl.dataset.Dataset:
            datanow = metadata_h5[key_0][()]
        else:
            datanow = read_h5f_metadata(metadata_h5[key_0])
        if type(metadata) == list:
            metadata.append(datanow)
        else:
            metadata[key_0] = datanow
    if len(keys_0) == 0:
        keys_0 = metadata_h5.attrs.keys()
        metadata= dict()
        for key_0 in keys_0:
            if key_0[0]!='_':
                metadata[key_0] = metadata_h5.attrs.get(key_0)
    return metadata


ephisdata_cell = []
sweepstarttimes = []
series_dir =  Path('/home/rozmar/Network/DOM3-MMIMS-2p/video/acq4/2021.10.28_000/cell_002/IVCC_000')
#series_dir = Path('/home/rozmar/Data/acq4/Voltage_rig_1P/2020.07.29_000/cell_000/CCIV_001')#

sweeps = configfile.readConfigFile(series_dir.joinpath('.index'))
if 'Clamp1.ma' in sweeps.keys():
    protocoltype = 'single sweep'
    sweepkeys = ['']
else:
    protocoltype = 'multiple sweeps'
    sweepkeys = sweeps.keys()
for sweep in sweepkeys:
    if sweep != '.' and '.txt' not in sweep and '.ma' not in sweep:
        sweep_dir = series_dir.joinpath(sweep)    
        sweepinfo = configfile.readConfigFile(sweep_dir.joinpath('.index'))
        if sweep=='':
            sweep='0'
        for file in sweepinfo.keys():
            if '.ma' in file:
                
                ephysfile = h5.File(sweep_dir.joinpath(file), "r")
                data = ephysfile['data'][()]
                metadata_h5 = ephysfile['info']
                metadata = read_h5f_metadata(metadata_h5)
                daqchannels = list(metadata[2]['DAQ'].keys())
                sweepstarttime = datetime.datetime.fromtimestamp(metadata[2]['DAQ'][daqchannels[0]]['startTime'])
                #relativetime = (sweepstarttime-cellstarttime).total_seconds()
                if len(ephisdata_cell) > 0 and ephisdata_cell[-1]['sweepstarttime'] == sweepstarttime:
                    ephisdata = ephisdata_cell.pop()   
                else:
                    ephisdata = dict()
                if 'primary' in daqchannels: # ephys data
                    ephisdata['V']=data[1]
                    ephisdata['stim']=data[0]
                    ephisdata['data']=data
                    ephisdata['metadata']=metadata
                    ephisdata['time']=metadata[1]['values']
                    #ephisdata['relativetime']= relativetime
                    ephisdata['sweepstarttime']= sweepstarttime
                    #ephisdata['series']= series
                    ephisdata['sweep']= sweep
                    sweepstarttimes.append(sweepstarttime)
                else:# other daq stuff
                    #%
                    for idx,channel in enumerate(metadata[0]['cols']):    
                        channelname = channel['name'].decode()
                        if channelname[0] == 'u':
                            channelname = channelname[2:-1]
                            if channelname in ['OrcaFlashExposure','Temperature','LED525','FrameCommand','NextFileTrigger']:
                                ephisdata[channelname] = data[idx]
                                #print('{} added'.format(channelname))
                            else:
                                print('waiting in the other daq')
                                timer.sleep(1000)
                ephisdata_cell.append(ephisdata)
#% 24.7g
sweeps = [0,len(ephisdata_cell)-2]
fig = plt.figure()
ax_primary = fig.add_subplot(2,1,1)
ax_stim = fig.add_subplot(2,1,2,sharex = ax_primary)
for sweep in sweeps:
    sr = 1/np.mean(np.diff(ephisdata_cell[sweep]['time']))
    v =ephisdata_cell[sweep]['data'][1,:]*10**3
    #v = utils_ephys.lpFilter(v,2000,1,sr)
    ax_primary.plot(ephisdata_cell[sweep]['time']*1000,v)

    ax_stim.plot(ephisdata_cell[sweep]['time']*1000,ephisdata_cell[sweep]['data'][2,:]*10**12,linewidth = 4)

#ax_primary.set_xlabel('time (ms)')
ax_primary.set_ylabel('Vm (mV)')
ax_stim.set_xlabel('time (ms)')
ax_stim.set_ylabel('Stimulus (pA)')
#%% EPHYS AND BEHAVIOR
from utils import utils_ephys

wsfile = '/home/rozmar/Data/Wavesurfer/DOM3-MMIMS/BCI_13/2021-10-24/cell3_0001.h5'
#wsfile = '/home/rozmar/Network/DOM3-MMIMS-2p/imaging/wavesurfer/BCI_16/2021-10-26/cell0_0001.h5'
wsfile = '/home/rozmar/Data/Wavesurfer/DOM3-MMIMS/BCI_14/2021-10-28/cell2_0001.h5'
wsdata = utils_ephys.load_wavesurfer_file(wsfile)
recording_mode = 'whole_cell' #'cell_attached'

#%%
trace = wsdata[0]['AI-ephys-primary']
sample_rate = wsdata[0]['sampling_rate']
#AP_dict = utils_ephys.findAPs_cell_attached(trace, sample_rate,recording_mode = 'current clamp', SN_min = 5,method = 'diff')
#%% detect intracell APs
import scipy.ndimage as ndimage
# =============================================================================
# trace = wsdata[0]['AI-ephys-primary'][8263118:8263118+1000000]/1000
# sr = wsdata[0]['sampling_rate']
# =============================================================================
def detect_ap_intracell(trace,sr):
    si = 1/sr
    msstep = int(.0005/si)
    sigma = .00005
    trace_f = ndimage.gaussian_filter(trace,sigma/si)
    d_trace_f = np.diff(trace_f)/si
    peaks = d_trace_f > 40
    # =============================================================================
    # sp_starts,sp_ends =(SquarePulse()&key).fetch('square_pulse_start_idx','square_pulse_end_idx') 
    # squarepulses = np.concatenate([sp_starts,sp_ends])
    # for sqidx in squarepulses:
    #     peaks[sqidx-msstep:sqidx+msstep] = False
    # =============================================================================
    peaks = ndimage.morphology.binary_dilation(peaks,np.ones(int(round(.002/si))))
    spikemaxidxes = list()
    counter = 0
    while np.any(peaks):
        if counter ==0:
            print('starting ap detection')
        
        counter +=1
        spikestart = np.argmax(peaks)
        spikeend = np.argmin(peaks[spikestart:])+spikestart
        if spikestart == spikeend:
            if sum(peaks[spikestart:]) == len(peaks[spikestart:]):
                spikeend = len(trace)
        try:
            sipeidx = np.argmax(trace[spikestart:spikeend])+spikestart
        except:
            print(key)
            sipeidx = np.argmax(trace[spikestart:spikeend])+spikestart
        spikemaxidxes.append(sipeidx)
        peaks[spikestart:spikeend] = False
    return np.asarray(spikemaxidxes)


def characterize_AP_intracell(ap_indices,trace,sr):
    counter = 0
    #%
    sigma = .00003 # seconds for filering
    step_time = .0001 # seconds
    threshold_value = 5 # mV/ms
    baseline_length = .01 #s back from threshold
    keylist = list()
    if len(ap_indices)>0:
        si = 1/sr
        step_size = int(np.round(step_time/si))
        ms5_step = int(np.round(.005/si))
        trace_f = ndimage.gaussian_filter(trace,sigma/si)
        d_trace_f = np.diff(trace_f)/si
        tracelength = len(trace)
        baseline_step = int(np.round(baseline_length/si))
        #%
        
        for ap_max_index in ap_indices:
            ap_now = {}
            counter += 1
            dvmax_index = ap_max_index
            while dvmax_index>step_size*2 and trace_f[dvmax_index]>0:
                dvmax_index -= step_size
            while dvmax_index>step_size*2 and dvmax_index < tracelength-step_size and  np.max(d_trace_f[dvmax_index-step_size:dvmax_index])>np.max(d_trace_f[dvmax_index:dvmax_index+step_size]):
                dvmax_index -= step_size
            if dvmax_index < tracelength -1:
                dvmax_index = dvmax_index + np.argmax(d_trace_f[dvmax_index:dvmax_index+step_size])
            else:
                dvmax_index = tracelength-2
                
            
            dvmin_index = ap_max_index
            #%
            while dvmin_index < tracelength-step_size and  (trace_f[dvmin_index]>0 or np.min(d_trace_f[np.max([dvmin_index-step_size,0]):dvmin_index])>np.min(d_trace_f[dvmin_index:dvmin_index+step_size])):
                dvmin_index += step_size
                #%
            dvmin_index -= step_size
            dvmin_index = dvmin_index + np.argmin(d_trace_f[dvmin_index:dvmin_index+step_size])
            
            thresh_index = dvmax_index
            while thresh_index>step_size*2 and (np.min(d_trace_f[thresh_index-step_size:thresh_index])>threshold_value):
                thresh_index -= step_size
            thresh_index = thresh_index - np.argmax((d_trace_f[np.max([0,thresh_index-step_size]):thresh_index] < threshold_value)[::-1])
            ap_threshold = trace_f[thresh_index]
            ap_amplitude = trace_f[ap_max_index]-ap_threshold
            hw_step_back = np.argmax(trace_f[ap_max_index:np.max([ap_max_index-ms5_step,0]):-1]<ap_threshold+ap_amplitude/2)
            hw_step_forward = np.argmax(trace_f[ap_max_index:ap_max_index+ms5_step]<ap_threshold+ap_amplitude/2)
            ap_halfwidth = (hw_step_back+hw_step_forward)*si
            
            
            
            if ap_amplitude > .01 and ap_halfwidth>.0001:
                ap_now['ap_real'] = 1
            else:
                ap_now['ap_real'] = 0
            ap_now['ap_threshold'] = ap_threshold
            ap_now['ap_threshold_index'] = thresh_index
            if thresh_index>10:
                ap_now['ap_baseline_value'] = np.mean(trace_f[np.max([thresh_index - baseline_step,0]) : thresh_index])*1000
            else:
                ap_now['ap_baseline_value'] = ap_threshold
            ap_now['ap_halfwidth'] =  ap_halfwidth
            ap_now['ap_amplitude'] =  ap_amplitude
            ap_now['ap_dv_max'] = d_trace_f[dvmax_index]
            ap_now['ap_dv_max_voltage'] = trace_f[dvmax_index]
            ap_now['ap_dv_min'] =  d_trace_f[dvmin_index]
            ap_now['ap_dv_min_voltage'] = trace_f[dvmin_index]
            ap_now['ap_max_index'] = ap_max_index
            keylist.append(ap_now)
    return keylist

def detect_square_pulse(stim,sr):
    #%
    key = {'square_pulse_num':[],
           'square_pulse_start_idx':[],
           'square_pulse_end_idx':[],
           'square_pulse_start_time':[],
           'square_pulse_length':[],
           'square_pulse_amplitude':[]}
    dstim = np.diff(stim)
    dstim[np.abs(dstim)<5] = 0
    square_pulse_num = -1
    while any(dstim!=0):
        #break
        square_pulse_num += 1
        stimstart = np.argmax(dstim!=0)
        stimstart_ = stimstart+np.argmax(dstim[stimstart:]==0)
        amplitude = sum(dstim[stimstart:stimstart_])
        dstim[stimstart:stimstart_] = 0
        if amplitude>0:
            stimend = np.argmax(dstim<0)
        else:
            stimend = np.argmax(dstim>0)
        stimend_ = stimend+np.argmax(dstim[stimend:]==0)
        
        stimstart += 1
        stimend += 1
        key['square_pulse_num'].append(square_pulse_num)
        key['square_pulse_start_idx'].append(stimstart)
        key['square_pulse_end_idx'].append(stimend)
        key['square_pulse_start_time'].append(stimstart/sr)
        key['square_pulse_length'] .append((stimend-stimstart)/sr)
        key['square_pulse_amplitude'].append(amplitude)
# =============================================================================
#         if key['square_pulse_length'][-1]<=1:
#             dstim[stimend:stimend_] = 0
# =============================================================================
            
        #%
    for k in key.keys():
        key[k] = np.asarray(key[k])
    return key

def calculate_series_resistance(trace,sr,square_pulse_dict):
    time_back = .0002
    time_capacitance = .0001
    time_forward = .0002
    step_back = int(np.round(time_back*sr))
    step_capacitance = int(np.round(time_capacitance*sr))
    step_forward = int(np.round(time_forward*sr))
    
    
    square_pulse_dict['RS'] = []
    for start_idx,stimamplitude in zip(square_pulse_dict['square_pulse_start_idx'],square_pulse_dict['square_pulse_amplitude']):
        if np.abs(stimamplitude)>=40:
            v0_start = np.mean(trace[start_idx-step_back:start_idx])
            vrs_start = np.mean(trace[start_idx+step_capacitance:start_idx+step_capacitance+step_forward])
    
            dv_start = vrs_start-v0_start
            RS_start = dv_start/stimamplitude 
    
            #RS = np.round(RS_start/1000000,2)
            RS = np.round(RS_start*1000,2)
        else:
            RS = np.nan
        square_pulse_dict['RS'].append(RS)   
    return square_pulse_dict

def calculate_input_resistance(trace,sr,square_pulse_dict): # RS still contaminates this measurement
    time_back = .1 #seconds
    step_back = int(np.round(time_back*sr))
    square_pulse_dict['Rin'] = []
    square_pulse_dict['Rin_v_std'] = []
    for start_idx,end_idx,stim_length,stim_amplitude in zip(square_pulse_dict['square_pulse_start_idx'],square_pulse_dict['square_pulse_end_idx'],square_pulse_dict['square_pulse_length'],square_pulse_dict['square_pulse_amplitude']):
        if stim_amplitude<-50 and stim_length>time_back+.05: # only negative pulses
            v0 = np.mean(trace[start_idx-step_back:start_idx])
            v0_std = np.std(trace[start_idx-step_back:start_idx])
            vrin = np.mean(trace[end_idx-step_back:end_idx])
            rin_v_std = np.std(trace[end_idx-step_back:end_idx])
            dv = vrin-v0
            Rin = dv/stim_amplitude 
    
            #RS = np.round(RS_start/1000000,2)
            Rin = np.round(Rin*1000,2)
            Rin_std = [v0_std,rin_v_std]
        else:
            Rin = np.nan
            Rin_std = [np.nan,np.nan]
        square_pulse_dict['Rin'].append(Rin)   
        square_pulse_dict['Rin_v_std'].append(Rin_std)
    return square_pulse_dict


#%% the whole recording in one go
downsample = 1
fig = plt.figure()
time = np.arange(len(wsdata[0]['AI-ephys-primary']))/wsdata[0]['sampling_rate']
ax_v = fig.add_subplot(2,1,1)
ax_i = fig.add_subplot(2,1,2,sharex = ax_v)
#x_recmode = fig.add_subplot(3,1,3,sharex = ax_v)

ax_v.plot(time[::downsample],utils_ephys.lpFilter(wsdata[0]['AI-ephys-primary'],2000,1,wsdata[0]['sampling_rate'])[::downsample],'k-')
ax_i.plot(time[::downsample],wsdata[0]['AI-ephys-secondary'][::downsample],'k-')
ax_i.set_xlabel('time from recording start (s)')
ax_i.set_ylabel('injected current (pA)')
ax_v.set_ylabel('Vm (mV)')
#ax_recmode.plot(time[::10],wsdata[0]['DI-EphysRecordingMode'][::10],'k-')
#%% whole cell plot
fig = plt.figure()
ax_ephys = fig.add_subplot(121)
ax_roi_voltage = fig.add_subplot(122,sharex = ax_ephys,sharey = ax_ephys)

offset = 0
offset_roi_voltage = 0
offset_snr = 0
time_back = 5 #seconds
time_forward = 15
next_trial_starts = np.concatenate([wsdata[0]['trial_start_indices'][1:],[wsdata[0]['trial_start_indices'][-1]+50000]])#[len(wsdata[0]['AI-ephys-primary'])-1]])
next_trial_starts = wsdata[0]['trial_start_indices']
sr = wsdata[0]['sampling_rate']
square_pulse_time_back = 0.1
square_pulse_time_forward = 0.2
square_pulse_steps = [int(square_pulse_time_back*sr),int(square_pulse_time_forward*sr)]
v_list = []
trial_stats = {'firing_rate' : [],
               'RS':[],
               'Rin':[],
               'V0':[],
               'I0':[],
               'AP_threshold':[],
               'AP_amplitude':[],
               'AP_halfwidth':[]}

for i,(trialstart,trialend) in enumerate(zip(wsdata[0]['trial_start_indices'],next_trial_starts)): #wsdata[0]['trial_end_indices']
    square_pulse_list = []
    square_pulse_amplitude_list = []
    step_back = int(time_back * wsdata[0]['sampling_rate'])
    step_forward = int(time_forward * wsdata[0]['sampling_rate'])
    trialend = trialstart+step_forward
    #aps_now = np.where((AP_dict['peak_idx']>trialstart-step_back)&(AP_dict['peak_idx']<trialend))[0]
    
# =============================================================================
#     if any(AP_dict['peak_snr_v'][aps_now]<8):#i>255:
#         break
# =============================================================================
    
    trace_now = wsdata[0]['AI-ephys-primary'][trialstart-step_back:trialend]#'AI-ROI voltage'#'AI-ephys-primary'
    trace_now = utils_ephys.lpFilter(trace_now,2000,1,sr)
    
    stim = wsdata[0]['AI-ephys-secondary'][trialstart-step_back:trialend]#'AI-ROI voltage'#'AI-ephys-primary'
    ap_indices = detect_ap_intracell(trace_now/1000,sr)
    ap_dict = characterize_AP_intracell(ap_indices,trace_now,sr)
    
    square_pulse_dict = detect_square_pulse(stim,sr)
    square_pulse_dict=calculate_series_resistance(trace_now,sr,square_pulse_dict)
    square_pulse_dict=calculate_input_resistance(trace_now,sr,square_pulse_dict)
    trace_noap = trace_now.copy()
    ap_step_back = int(sr*.001)
    ap_step_forward = int(sr*.01)
    for ap_idx in ap_indices:
        trace_noap[np.max([0,ap_idx-ap_step_back]):ap_idx+ap_step_forward] = trace_noap[np.max([0,ap_idx-ap_step_back])]
    v_list.append(trace_noap)
    #%
    for square_i in square_pulse_dict['square_pulse_num']:
        if square_pulse_dict['square_pulse_amplitude'][square_i]<-50:
            sq_idx = square_pulse_dict['square_pulse_start_idx'][square_i]
            square_pulse_list.append(trace_now[sq_idx-square_pulse_steps[0]:sq_idx+square_pulse_steps[1]])
            square_pulse_amplitude_list.append(square_pulse_dict['square_pulse_amplitude'][square_i])
            #break
    #break
    
    #%
    time = np.arange(len(trace_now))/wsdata[0]['sampling_rate']-step_back/wsdata[0]['sampling_rate']
    ax_ephys.plot(time,trace_now+offset,'k-')
# =============================================================================
#     if len(ap_indices)>0:
#         ax_ephys.plot(time[ap_indices],(trace_now+offset)[ap_indices],'ro')
# =============================================================================
    
    if i%10 == 0:
        ax_ephys.text(-1*time_back-1,offset+np.median(trace_now),str(i))
    roi_voltage_now = wsdata[0]['AI-ROI voltage'][trialstart-step_back:trialend]#'AI-ROI voltage'#'AI-ephys-primary'
    ax_roi_voltage.plot(time,roi_voltage_now/3.3*(np.max(trace_now)-np.min(trace_now))+offset+np.median(trace_now),'g-')
    offset = offset-(np.abs(np.nanmin(trace_now)) + np.abs(np.nanmax(trace_now)))
    offset_roi_voltage = -1+offset_roi_voltage-(np.abs(np.nanmin(roi_voltage_now)) + np.abs(np.nanmax(roi_voltage_now)))
    
    ampl_list = []
    thresh_list = []
    hw_list = []
    for ap_now in ap_dict:
        ampl_list.append(ap_now['ap_amplitude'])
        thresh_list.append(ap_now['ap_threshold'])
        hw_list.append(ap_now['ap_halfwidth'])
        
    trial_stats['firing_rate'].append(len(ap_indices)/(len(trace_now)/sr))
    trial_stats['RS'].append(np.nanmedian(square_pulse_dict['RS']))
    trial_stats['Rin'].append(np.nanmedian(square_pulse_dict['Rin'])-np.nanmedian(square_pulse_dict['RS']))
    trial_stats['V0'].append(np.median(trace_now))
    trial_stats['I0'].append(np.median(stim))
    trial_stats['AP_threshold'].append(np.median(thresh_list))
    trial_stats['AP_amplitude'].append(np.median(ampl_list))
    trial_stats['AP_halfwidth'].append(np.median(hw_list))
   # break
# =============================================================================
#     if i >100:
#         break
# =============================================================================
ax_roi_voltage.set_xlabel('time from GO cue')    
ax_ephys.set_xlabel('time from GO cue')    
ax_ephys.set_title('Whole Cell')
ax_roi_voltage.set_title('Generated motor command voltage')
    #break

#%% Whole cell II
mean_win = 2
fig = plt.figure()
ax1 = fig.add_subplot(711)
ax2 = fig.add_subplot(712,sharex = ax1)
ax3 = fig.add_subplot(713,sharex = ax1)
ax4 = fig.add_subplot(714,sharex = ax1)
ax5 = fig.add_subplot(715,sharex = ax1)
ax6 = fig.add_subplot(716,sharex = ax1)
ax7 = fig.add_subplot(717,sharex = ax1)

ax1.plot(utils_plot.rollingfun(trial_stats['firing_rate'], window = mean_win, func = 'mean'),'r-',label = 'firing rate')
ax2.plot(utils_plot.rollingfun(trial_stats['RS'], window = mean_win, func = 'mean'),'b-',label = 'RS after bridge')
ax2.plot(utils_plot.rollingfun(trial_stats['Rin'], window = mean_win, func = 'mean'),'g-',label ='Rin')
ax3.plot(utils_plot.rollingfun(trial_stats['V0'], window = mean_win, func = 'mean'),'k-',label ='V0')
ax4.plot(utils_plot.rollingfun(trial_stats['I0'], window = mean_win, func = 'mean'),'k-',label ='I0')

ax5.plot(utils_plot.rollingfun(trial_stats['AP_threshold'], window = mean_win, func = 'mean'),'k-',label ='AP threshold')
ax6.plot(utils_plot.rollingfun(trial_stats['AP_amplitude'], window = mean_win, func = 'mean'),'k-',label ='AP amplitude')
ax7.plot(utils_plot.rollingfun(trial_stats['AP_halfwidth'], window = mean_win, func = 'mean')*1000,'k-',label ='AP halfwidth')

ax1.set_ylabel('Hz')
ax2.set_ylabel('MOhm')
ax3.set_ylabel('mV')
ax4.set_ylabel('pA')
ax5.set_ylabel('mV')
ax6.set_ylabel('mV')
ax7.set_ylabel('ms')
ax7.set_xlabel('trial number')
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend()
ax6.legend()
ax7.legend()

#%%
from matplotlib import cm

group_size = 10
last_trial = 80
offset_step = 5
offset = 0
fig = plt.figure()
ax_v0 = fig.add_subplot(111)
v_array = np.asarray(v_list)
cm_subsection = np.linspace(0, 1, int(np.ceil(last_trial/group_size))) 
for i,color_i in zip(range(group_size,len(v_list)+group_size,group_size),cm_subsection):
    v_now =np.mean(v_array[i-group_size:i],0)
    v_now = utils_ephys.lpFilter(v_now,100,1,sr)
    color = cm.hot(color_i)
    ax_v0.plot(time,v_now-offset,color = color,alpha = .8,label = 'trial {}-{}'.format(i-group_size,i))
    offset+=offset_step
    if i>last_trial:
        break
ax_v0.legend()
    #break

#%% Cell attached plot

fig = plt.figure()
ax_ephys = fig.add_subplot(121)
ax_roi_voltage = fig.add_subplot(122,sharex = ax_ephys,sharey = ax_ephys)

offset = 0

offset_snr = 0
time_back = 5 #seconds
next_trial_starts = np.concatenate([wsdata[0]['trial_start_indices'][1:],[len(wsdata[0]['AI-ephys-primary'])-1]])
sr = wsdata[0]['sampling_rate']

for i,(trialstart,trialend) in enumerate(zip(wsdata[0]['trial_start_indices'],next_trial_starts)): #wsdata[0]['trial_end_indices']
    step_back = int(time_back * wsdata[0]['sampling_rate'])
    #aps_now = np.where((AP_dict['peak_idx']>trialstart-step_back)&(AP_dict['peak_idx']<trialend))[0]
    
# =============================================================================
#     if any(AP_dict['peak_snr_v'][aps_now]<8):#i>255:
#         break
# =============================================================================
    
    trace_now = wsdata[0]['AI-ephys-primary'][trialstart-step_back:trialend]#'AI-ROI voltage'#'AI-ephys-primary'
    AP_dict = utils_ephys.findAPs_cell_attached(trace_now, sr,recording_mode = 'current clamp', SN_min = 5,method = 'diff')
    ap_indices = AP_dict['peak_idx']

    time = np.arange(len(trace_now))/wsdata[0]['sampling_rate']-step_back/wsdata[0]['sampling_rate']
    ax_ephys.plot(time,trace_now+offset,'k-')
    
    ax_ephys.plot(time[ap_indices],(trace_now+offset)[ap_indices],'ro')
    
    
    roi_voltage_now = wsdata[0]['AI-ROI voltage'][trialstart-step_back:trialend]#'AI-ROI voltage'#'AI-ephys-primary'
    ax_roi_voltage.plot(time,roi_voltage_now/3.3*(np.max(trace_now)-np.min(trace_now))+offset+np.median(trace_now),'g-')
    offset = offset-(np.abs(np.nanmin(trace_now)) + np.abs(np.nanmax(trace_now)))
    

# =============================================================================
#     if i >100:
#         break
# =============================================================================
ax_roi_voltage.set_xlabel('time from GO cue')    
ax_ephys.set_xlabel('time from GO cue')    
ax_ephys.set_title('cell attached')
ax_roi_voltage.set_title('Generated motor command voltage')
    #break