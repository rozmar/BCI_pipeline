import datajoint as dj
from pipeline import pipeline_tools,lab,experiment,videography,imaging
from pipeline.ingest import datapipeline_metadata
from utils import utils_pybpod, utils_ephys, utils_imaging,utils_pipeline, utils_plot

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import datetime
import numpy as np
import time as timer
import os
from scipy.io import loadmat
%matplotlib qt

#%% show example video
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
class MplColorHelper:

  def __init__(self, cmap_name, start_val, stop_val):
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

  def get_rgb(self, val):
    return (np.asarray(np.asarray(self.scalarMap.to_rgba(val))*255,int)[np.asarray([2,1,0])]).tolist()


#%
output_dir = '/home/rozmar/Data/temp'
session_key_wr = {'wr_id':'BCI14', 'session':2}
show_tracking = False
subject_id = (lab.WaterRestriction()&'water_restriction_number = "{}"'.format(session_key_wr['wr_id'])).fetch1('subject_id')
session_key = {'subject_id':subject_id,
               'session':session_key_wr['session']}
trial_key = {'subject_id':subject_id,
               'session':session_key_wr['session'],
               'trial':55}

dlc_key_lickport = {'camera_position':'side',
                        'bodypart':'lickport'}
dlc_key_tongue = {'camera_position':'side',
                        'bodypart':'tongue_tip'}#tongue_tip
steps_back = 160
go_time = float((experiment.TrialEvent()&trial_key&'trial_event_type = "go"').fetch1('trial_event_time'))

camera_position = 'bottom'
folder,file,frame_times = (videography.BehaviorVideo()&trial_key&'camera_position = "{}"'.format(camera_position)).fetch1('video_folder','vide_file_name','video_frame_times')
frame_times = frame_times-go_time
bodyparts = np.unique((videography.DLCTracking()&trial_key&'camera_position = "{}"'.format(camera_position)).fetch('bodypart'))
dlc_dict = {}
for bodypart in bodyparts:
    dlc_key_lickport['bodypart'] = bodypart
    lickport_x,lickport_y,lickport_p = (videography.DLCTracking()&dlc_key_lickport&trial_key&'camera_position = "{}"'.format(camera_position)).fetch1('x','y','p')
    lickport_x = lickport_x.copy()
    lickport_y = lickport_y.copy()
    to_nan= lickport_p<.99
    lickport_x[to_nan] = np.nan
    lickport_y[to_nan] = np.nan
    dlc_dict[bodypart] = {'x':lickport_x,
                            'y':lickport_y}

videofile = os.path.join(folder,file)
cap = cv2.VideoCapture(videofile)

out = cv2.VideoWriter(os.path.join(output_dir,'outpy.avi'),cv2.VideoWriter_fourcc('M','J','P','G'), 31, (640,290))


COL = MplColorHelper('viridis', 0, len(bodyparts))
frame_count = 0
#movie = []
frame_nums = []
frame = True
while not frame is None:
    
    ret, frame = cap.read()
    if frame_count%5 ==0:
        frame_nums.append(frame_count)
        #movie.append(frame[:,:,0])
        time = '{:.2f} s'.format(frame_times[frame_count])
        frame = cv2.putText(frame, time, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        if show_tracking:
            for i,bodypart in enumerate(dlc_dict.keys()):
                lickport_x = dlc_dict[bodypart]['x']
                lickport_y = dlc_dict[bodypart]['y']
                
                try:
                    frame = cv2.circle(img = frame,center = (int(lickport_x[frame_count]),int(lickport_y[frame_count])),radius = 5,color = COL.get_rgb(i),thickness=2)
                    if frame_count>0:
                        hist_x = lickport_x[np.max([0,frame_count-steps_back]):frame_count:5]
                        hist_y = lickport_y[np.max([0,frame_count-steps_back]):frame_count:5]
                        for x_start,x_end,y_start,y_end in zip(hist_x[:-1],hist_x[1:],hist_y[:-1],hist_y[1:]):
                            if not np.isnan(x_start ):
                                frame = cv2.line(frame, (int(x_start),int(y_start)), (int(x_end),int(y_end)), COL.get_rgb(i), 1)
              
                
                except ValueError:
                    pass
                #break
       # break
        out.write(frame)
    if frame_count%100 ==0:
        print(frame_count)
    frame_count+=1
    #break
    #%
cap.release()
out.release()

#%% FrameAutoEncoder sample video
import pandas as pd
base_dir = '/home/rozmar/Data/FrameAutoEncoderEmbeddings/sample'
df = pd.read_csv(os.path.join(base_dir,'data.csv'))
movie_idx = 3
xin = np.load(os.path.join(base_dir,'xIn.npy'))
xout = np.load(os.path.join(base_dir,'xOut.npy'))
#%
movie = xin[movie_idx,:,0,:,:]
#%
out = cv2.VideoWriter(os.path.join(output_dir,'FrameAutoEncoder_in.avi'),cv2.VideoWriter_fourcc('M','J','P','G'), 15, (xin.shape[4],xin.shape[3]),False)
frame_count = 1484-800#int(movie.shape[0]/2)

for frame in movie:
    frame = np.asarray(frame*255,np.uint8)
    time = '{:.2f} s'.format(frame_times[frame_count])
    frame = cv2.putText(frame, time, (10,20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255))
    frame_count+=10
    out.write(frame)
out.release()

movie = xout[movie_idx,:,0,:,:]
#%
out = cv2.VideoWriter(os.path.join(output_dir,'FrameAutoEncoder_out.avi'),cv2.VideoWriter_fourcc('M','J','P','G'), 15, (xin.shape[4],xin.shape[3]),False)
frame_count = 1484-800#int(movie.shape[0]/2)

for frame in movie:
    frame[frame<0] = 0
    frame[frame>1] = 1
    frame = np.asarray(frame*255,np.uint8)
    
    time = '{:.2f} s'.format(frame_times[frame_count])
    frame = cv2.putText(frame, time, (10,20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255))
    frame_count+=10
    out.write(frame)
out.release()
#%%
#pd.l


#%% compare autoencoder with DLC -UNFINISHED

dlc_key_lickport = {'camera_position':'side',
                        'bodypart':'lickport'}
dlc_key_tongue = {'camera_position':'side',
                        'bodypart':'jaw'}#tongue_tip
session_key_wr = {'wr_id':'BCI14', 'session':2}
subject_id = (lab.WaterRestriction()&'water_restriction_number = "{}"'.format(session_key_wr['wr_id'])).fetch1('subject_id')
session_key = {'subject_id':subject_id,
               'session':session_key_wr['session']}
conditioned_rois = imaging.ROI()*imaging.ConditionedROI()&session_key
frame_times = (imaging.FOVFrameTimes()&session_key).fetch1('frame_times')
frame_rate = (imaging.FOV()&session_key).fetch1('fov_frame_rate')
for roi_cond in conditioned_rois:
    dff = (imaging.ROITrace()&roi_cond).fetch1('roi_dff')
    break # only first ROI now
#%
for trial in experiment.SessionTrial()&session_key:
    if trial['trial']<55:
        continue
    embedding_dim, vector = (videography.EmbeddingVector()&session_key&'trial = {}'.format(trial['trial'])).fetch('embedding_dimension','embedding_vector')
    vector = np.stack(vector)
    lickport_position = (videography.DLCTracking()&dlc_key_lickport&session_key&'trial = {}'.format(trial['trial'])).fetch1('x')
    tongue_position,p = (videography.DLCTracking()&dlc_key_tongue&session_key&'trial = {}'.format(trial['trial'])).fetch1('y','p')
    tongue_position = tongue_position.copy()
    tongue_position[p<.99]=np.nan#np.median(tongue_position)
    
    go_time = float((experiment.TrialEvent()&dlc_key_lickport&session_key&'trial = {}'.format(trial['trial'])&'trial_event_type = "go"').fetch1('trial_event_time'))
    movie_frame_times = (videography.BehaviorVideo()&dlc_key_lickport&session_key&'trial = {}'.format(trial['trial'])).fetch1('video_frame_times')
    movie_frame_times = movie_frame_times-go_time
    
    break

#%
fig = plt.figure()
ax_embedding = fig.add_subplot(2,1,1)
ax_lickport = fig.add_subplot(2,1,2,sharex = ax_embedding)
maxval = 0
for vector_now in vector[:3,:]:
    difi = maxval-np.min(vector_now)
    ax_embedding.plot(movie_frame_times,vector_now+difi)
    maxval = np.max(vector_now+difi)
    #break
ax_lickport.plot(movie_frame_times,(lickport_position-np.min(lickport_position))/(np.max(lickport_position)-np.min(lickport_position))*-1,label = 'lickport-dlc')
ax_lickport.plot(movie_frame_times,(tongue_position-np.nanmin(tongue_position))/(np.nanmax(tongue_position)-np.nanmin(tongue_position)),label = 'jaw-dlc')
ax_lickport.legend()
ax_lickport.set_xlabel('time relative to GO cue')
ax_lickport.set_ylabel('Part tracking')
ax_embedding.set_ylabel('Embeddings')

#%% video analysis copy movies  and metadata to dm11
import random
from pathlib import Path
import json
import shutil
remote_base_dir = '/home/rozmar/Network/dm11/svobodalab/users/rozmar/BCI_videos/raw/'
#remote_base_dir_name = '/groups/svoboda/svobodalab/users/rozmar/BCI_videos/raw/'
session_key_wr = {'wr_id':'BCI14', 'session':10}#sessions 2 and 10 so far
#session_key_wr = {'wr_id':'BCI15', 'session':10}#sessions 5 and 10 so far
camera = 'side'
movie_number_to_use = 5000
first_frame_offset = .058 # the start of the first frame from trial start time

subject_id = (lab.WaterRestriction()&'water_restriction_number = "{}"'.format(session_key_wr['wr_id'])).fetch1('subject_id')
session = experiment.Session()&'subject_id = {}'.format(subject_id)&'session = {}'.format(session_key_wr['session'])

session_date = str(session.fetch1('session_date'))
wr_id = (lab.WaterRestriction()&session).fetch1('water_restriction_number')
wr_id  = wr_id[:3]+'_'+wr_id[3:] # stupid thing, directories are not accurate..
bpod_file_names = np.unique((experiment.TrialMetaData().BpodMetaData()&session).fetch('bpod_file_name'))
setup = session.fetch1('rig')

if setup == 'DOM3-2p-Resonant-MMIMS':
    setup = 'DOM3-MMIMS'
else:
    setup = 'Bergamo-2P'

trials = list(experiment.TrialMetaData().BpodMetaData()*experiment.TrialMetaData().VideoMetaData()&session)
random.shuffle(trials)
target_raw_video_dir = os.path.join(remote_base_dir,'{}_{}'.format(wr_id,session_date),'video')
target_raw_metadata_dir = os.path.join(remote_base_dir,'{}_{}'.format(wr_id,session_date),'metadata')
Path(target_raw_video_dir).mkdir(parents=True, exist_ok=True)
Path(target_raw_metadata_dir).mkdir(parents=True, exist_ok=True)
prev_camera_config_file = ''
for trial in trials[:movie_number_to_use]:
    config_file =  os.path.join('/home/rozmar/Data/Behavior_videos/',setup,wr_id,camera,trial['bpod_file_name'],'camera_config.json')
    if config_file != prev_camera_config_file: #get exposition time
        try:
            with open(config_file) as f:
                camera_config = json.load(f)
            exposition_time = camera_config['camera']['properties']['shutter']['absoluteValue']/1000
            prev_camera_config_file = config_file
        except:
            exposition_time = .001 # educated guess
    
    video_dir = os.path.join('/home/rozmar/Data/Behavior_videos/',setup,wr_id,camera,trial['bpod_file_name'],'trial_{0:03d}'.format(trial['bpod_trial_num']))
    video_files = os.listdir(video_dir)
    if len(video_files) != 2:
        print('too many/too few files, skipping')
        continue
    video_name = ''
    exposition_times = ''
    for video_file in video_files:
        if video_file.endswith('txt'):
            from numpy import loadtxt
            exposition_times = loadtxt(os.path.join(video_dir,video_file), comments="#", delimiter=",", unpack=False)
            exposition_times += first_frame_offset + exposition_time/2
        elif video_file.endswith('avi'):
            video_name = video_file
        else:
            print('unknown file format {}'.format(video_ffile))
    if len(video_name)>0 and len(exposition_times)>0:
        try:
            go_cue_time = float((experiment.TrialEvent()&trial & 'trial_event_type = "go"').fetch1('trial_event_time'))
            try:
                threshold_cross_time = float((experiment.TrialEvent()&trial & 'trial_event_type = "threshold cross"').fetch1('trial_event_time'))
            except:
                threshold_cross_time = np.nan
            try   :
                reward_time = float((experiment.TrialEvent()&trial & 'trial_event_type = "reward"').fetch1('trial_event_time'))
            except:
                reward_time = np.nan
        except:
            print('no go cue in trial')
            continue
        
        go_cue_frame = int(np.argmin(np.abs(exposition_times-go_cue_time)))
        if not np.isnan(threshold_cross_time):
            threshold_frame = int(np.argmin(np.abs(exposition_times-threshold_cross_time)))
        else:
            threshold_frame  = np.nan
        if not np.isnan(reward_time):
            reward_frame = int(np.argmin(np.abs(exposition_times-reward_time)))
        else:
            reward_frame  = np.nan    
        trial['behavior_video_name'] = os.path.join(video_dir,video_name)
        trial['go_cue_frame'] = go_cue_frame
        trial['threshold_frame'] = threshold_frame
        trial['reward_frame'] = reward_frame
        trial['wr_id'] = session_key_wr['wr_id']
        trial['frame_number'] = len(exposition_times)
        
        fname = 'trial_{0:04d}'.format(trial['trial'])
        print(fname)
        with open(os.path.join(target_raw_metadata_dir,fname+'.json'), "w") as json_file_out:
            json.dump(trial, json_file_out, indent=4, sort_keys=True)
        shutil.copyfile(trial['behavior_video_name'],os.path.join(target_raw_video_dir,fname+'.avi'))

#%% FrameAutoEncoder training progress
import re
import json
from pathlib import Path
log_dir = '/home/rozmar/Network/dm11/svoboda$/rozsam/Scripts/Python/FrameAutoEncoder/log/train_log'
_training_names = np.sort(os.listdir(log_dir))
training_names = list()
for training_name in _training_names:
    if '.json' not in training_name:
        training_names.append(training_name)
    else:
        with open(os.path.join(log_dir,training_name)) as f:
            training_meta = json.load(f)
#training_name = '9-BCI_14_2021-06-12--2021-08-07'
#training_name = '17-BCI_14_2021-06-12_gpu_threshold--2021-08-10'
#training_name = '18-BCI_14_and_BCI_15--2021-08-12'
#training_name = '22-BCI_14_and_BCI_15--2021-08-13'
#training_name = '25-BCI_14_and_BCI_15_per_subject--2021-08-13'
#training_name = '26-BCI_14_two_sessions_merged--2021-08-13'
fig = plt.figure()
    
for training_i,training_name in enumerate(training_names):
    training_id = re.findall(r'\d+', training_name)[0]
    if training_i==0:
        ax1 = fig.add_subplot(4,4,training_i+1)
        ax2 = ax1.twinx()
        ax_first = ax1
        ax2_first =ax2
    else:
        ax1 = fig.add_subplot(4,4,training_i+1,sharex = ax_first,sharey = ax_first)
        ax2 = ax1.twinx()
        ax2.get_shared_y_axes().join(ax2, ax2_first)
    training_dir = os.path.join(log_dir,training_name)
    training_file = os.path.join(training_dir,'cmd_log.txt')
    training_file_mtime =  datetime.datetime.fromtimestamp(Path(training_file).stat().st_mtime)
    iteration_num_list = []
    validation_error_list = []
        #%
    with open(training_file, 'r') as file1:
        losslist = list()
        while True:
           
            line = file1.readline()
            if not line:
                break
            if 'iteration' in line:
                iteration_num_list.append(int(re.findall(r'\d+', line)[-1]))
            elif line.startswith('val') and '[' in line:
                validation_error_list.append(float(line.strip('val').strip('[]').strip('\n').split('__')[0]))
                #break
            if len(line)>200:
                linelist = line.split(',')
                #%
                for entry in linelist:
                    entry = entry.strip('[]').split('__')
                    if len(entry)==4:
                        entry = np.asarray(entry,float)
                        losslist.append(entry[0])
                    #break
    
    #ax2=ax1.twinx()
    ax1.plot(losslist,'k-',label = 'training error')
    ax1.plot(iteration_num_list,validation_error_list,'ro-',label = 'validation error')
    ax1.set_yscale('log')
    #ax2.set_yscale('log')
    if (datetime.datetime.now()-training_file_mtime).total_seconds()<60:
        titlecolor = 'red'
    else:
        titlecolor = 'black'
    try:
        ax1.set_title('{}  \n  {}'.format(training_name,training_meta[training_id]),color = titlecolor)
    except:
        ax1.set_title(training_name,color = titlecolor)
    
    ax1.legend()
    checkpoints_dir = os.path.join(training_dir,'checkpoints')
    checkpoint_iter = list()
    checkpoint_time = list()
    checkpoint_files = os.listdir(checkpoints_dir)
    for checkpoint_file in checkpoint_files:
        checkpoint_iter.append(int(re.findall(r'\d+', checkpoint_file)[0]))
        fname = Path(os.path.join(checkpoints_dir,checkpoint_file))
        mtime = datetime.datetime.fromtimestamp(fname.stat().st_mtime)
        checkpoint_time.append(mtime)
    checkpoint_order = np.argsort(checkpoint_iter)
    checkpoint_iter = np.asarray(checkpoint_iter)[checkpoint_order]
    checkpoint_time = np.asarray(checkpoint_time)[checkpoint_order]
    checkpoint_diff = list()
    for dif_now,iternum in zip(np.diff(checkpoint_time),np.diff(checkpoint_iter)):
        checkpoint_diff.append(iternum/dif_now.total_seconds())
        
    ax2.plot(np.nanmean([checkpoint_iter[:-1],checkpoint_iter[1:]],0),checkpoint_diff,'b-^')
    ax2.set_ylabel('training speed (iter/sec)')
    ax2.yaxis.label.set_color('blue')
    ax2.plot(len(losslist)/2,len(losslist)/(training_file_mtime-checkpoint_time[0]).total_seconds(),'bo')
    ax2.set_yscale('log')
    #break
#ax2.legend()
#%% FrameAutoEncoder - check embedding
img_dir = remote_base_dir = '/home/rozmar/Network/dm11/svobodalab/users/rozmar/BCI_videos/jpg/'
embedding_dir = remote_base_dir = '/home/rozmar/Network/dm11/svobodalab/users/rozmar/BCI_videos/embedding/'
sessions = os.listdir(embedding_dir)
for session in sessions:
# =============================================================================
#     if 'BCI_14' in session:
#         continue
# =============================================================================
    embedding_session_dir = os.path.join(embedding_dir,session)
    trials = np.sort(os.listdir(embedding_session_dir))
    for trial in trials[::-1]:
        mu = np.load(os.path.join(embedding_session_dir,trial))
        break
    break
plt.plot(mu[:,:5])