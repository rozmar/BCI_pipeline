import datajoint as dj
from pipeline import lab,experiment,videography
import json
import numpy as np
import os
import cv2 as cv
import pandas as pd
import re
import sklearn.metrics

first_frame_offset = .058 # the start of the first frame from trial start time
cameras = ['side','bottom']
video_base_dir = '/home/rozmar/Data/Behavior_videos/'
deeplabcut_base_dir = '/home/rozmar/Data/DLC_output'
frameautoencoder_base_dir = '/home/rozmar/Data/FrameAutoEncoderEmbeddings/'
#%%
def populate_behaviorvideo():
    sessions = experiment.Session()
    for session in sessions:
        if len(videography.BehaviorVideo()&session)>0:
            continue # session already uploaded
        trials = experiment.TrialMetaData().BpodMetaData()*experiment.TrialMetaData().VideoMetaData()&session
        prev_camera_config_file = ''
        BehaviorVideo_list = list()
        for trial_i,trial in enumerate(trials):
            if type(trial['behavior_video_name']) == str:
                if trial['behavior_video_name'] == 'no behavior video':
                    continue
            #session_date = str((experiment.Session&trial).fetch1('session_date'))
            wr_id = (lab.WaterRestriction()&trial).fetch1('water_restriction_number')
            wr_id  = wr_id[:3]+'_'+wr_id[3:] # stupid thing, directories are not accurate..
            setup = (experiment.Session&trial).fetch1('rig')
            
            if setup == 'DOM3-2p-Resonant-MMIMS':
                setup = 'DOM3-MMIMS'
            else:
                setup = 'Bergamo-2P'
            for camera in cameras:
                config_file =  os.path.join(video_base_dir,setup,wr_id,camera,trial['bpod_file_name'],'camera_config.json')
                if config_file != prev_camera_config_file: #get exposition time
                    try:
                        #%
                        with open(config_file) as f:
                            camera_config = json.load(f)
                            #%
                        exposition_time = camera_config['camera']['properties']['shutter']['absoluteValue']/1000
                        prev_camera_config_file = config_file
                    except:
                        print('camera config file not found, aborting')
                        continue
                    
                
                
        #% this part will fill the main videography tables - finish it!
            #for i in [1]:
                video_dir = os.path.join(video_base_dir,setup,wr_id,camera,trial['bpod_file_name'],'trial_{0:03d}'.format(trial['bpod_trial_num']))
                video_files = os.listdir(video_dir)
                if len(video_files) != 2:
                    print('too many/too few files, skipping')
                    #continue
                video_name = ''
                exposition_times = ''
                for video_file in video_files:
                    if video_file.endswith('txt'):
                        from numpy import loadtxt
                        exposition_times = loadtxt(os.path.join(video_dir,video_file), comments="#", delimiter=",", unpack=False)
                        exposition_times += first_frame_offset + exposition_time/2
                    elif video_file.endswith('avi'):
                        video_name = video_file
                        video = cv.VideoCapture(os.path.join(video_dir,video_file))
                        frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
                        video.release()
                        
                    else:
                        print('unknown file format {}'.format(video_file))
                        
                if len(video_name)<0 or len(exposition_times)<0:
                    continue
                if len(exposition_times) != frame_count:
                    print('exposition time mismatch')
                    continue
                if exposition_times[-1]>90:
                    print('too long video - {} s- skipping'.format(exposition_times[-1]))
                    continue
                BehaviorVideo_key = {'subject_id':trial['subject_id'],
                                     'session':trial['session'],
                                     'trial':trial['trial'],
                                     'camera_position':camera,
                                     'video_frame_count':frame_count,
                                     'video_frame_rate':1/np.mean(np.diff(exposition_times)),
                                     'video_frame_times':exposition_times,
                                     'video_exposition_time':exposition_time*1000,
                                     'video_folder':video_dir,
                                     'vide_file_name':video_name}
                BehaviorVideo_list.append(BehaviorVideo_key)
        if len(BehaviorVideo_list)>0:
            with dj.conn().transaction: #inserting one movie
                print('uploading videograpy of session {}'.format(session)) #movie['movie_name']
                videography.BehaviorVideo().insert(BehaviorVideo_list,allow_direct_insert=True)    
#%%

def populate_dlctracking():
    #%%
    sessions = experiment.Session()
    for session in sessions:
        if len(videography.BehaviorVideo()&session)==0:
            continue # no video for session
        if len(videography.DLCTracking()&session)>0:
            continue # session already uploaded
        DLCTracking_list = list()
        camera_positions = np.unique((videography.BehaviorVideo()&session).fetch('camera_position'))
        #print('starting {}'.format(session))
        for camera_position in camera_positions:
            behaviorvideos = videography.BehaviorVideo()&session&'camera_position = "{}"'.format(camera_position)
            for behaviorvideo in behaviorvideos:
                #check if the tracking exists
                dlcfolder = os.path.join(deeplabcut_base_dir,behaviorvideo['video_folder'][len(video_base_dir):])
                if not os.path.exists(dlcfolder):
                    break
                #%
                dlcdata = pd.read_csv(os.path.join(dlcfolder,'{}csv'.format(behaviorvideo['vide_file_name'][:-3])),index_col = 0, header = None)
                bodyparts = np.unique(dlcdata.loc['bodyparts'])
                dlc_dict = {}
                for bodypart in bodyparts:
                    dlc_dict[bodypart] = dict()
                for colname in dlcdata.keys():
                    dlc_dict[dlcdata[colname]['bodyparts']][dlcdata[colname]['coords']] = np.asarray(dlcdata[colname][3:].values,float)
                for bodypart in dlc_dict.keys():
                    bodypart_name= re.sub('-', '_', bodypart)
                    bodypart_name= re.sub('W', 'whisker', bodypart)
                    dlctracking_dict = {'subject_id':behaviorvideo['subject_id'],
                                        'session':behaviorvideo['session'],
                                        'trial':behaviorvideo['trial'],
                                        'camera_position':behaviorvideo['camera_position'],
                                        'bodypart':bodypart_name,
                                        'x':dlc_dict[bodypart]['x'],
                                        'y':dlc_dict[bodypart]['y'],
                                        'p':dlc_dict[bodypart]['likelihood']}
                    DLCTracking_list.append(dlctracking_dict)
                dj.conn().ping()
            if len(DLCTracking_list)>0:
                with dj.conn().transaction: #inserting one movie
                    print('uploading DLC tracking of session {}'.format(session)) #movie['movie_name']
                    videography.DLCTracking().insert(DLCTracking_list,allow_direct_insert=True)    
                    
 #%%                   
def populate_embedding_vectors():
    #%%
    
    
    sessions = experiment.Session()
    for session in sessions:
        if len(videography.BehaviorVideo()&session)==0:
            continue # no video for session
        if len(videography.EmbeddingVector()&session)>0:
            continue # session already uploaded
        wr_id = (lab.WaterRestriction()&session).fetch1('water_restriction_number')
        wr_id  = wr_id[:3]+'_'+wr_id[3:] 
        session_date = str(session['session_date'])
        embedding_session_dir = os.path.join(frameautoencoder_base_dir,'{}_{}'.format(wr_id,session_date))
        if not os.path.exists(embedding_session_dir):
            continue # embedding not exported yet
        Embedding_list = list()
        #break
        print('starting {}'.format(session))
        for trial in experiment.SessionTrial()&session:
            filename = os.path.join(embedding_session_dir,'trial_{0:04d}.npy'.format(trial['trial']))
            vectors = np.load(filename)
            for embedding_dimension in range(vectors.shape[1]):
                embedding_dict = {'subject_id':trial['subject_id'],
                                  'session':trial['session'],
                                  'trial':trial['trial'],
                                  'embedding_dimension':embedding_dimension,
                                  'embedding_vector':vectors[:,embedding_dimension]}
                Embedding_list.append(embedding_dict)
            dj.conn().ping()
        if len(Embedding_list)>0:
            with dj.conn().transaction: #inserting one movie
                print('uploading embedding of session {}'.format(session)) #movie['movie_name']
                videography.EmbeddingVector().insert(Embedding_list,allow_direct_insert=True) 
#%% jaw tracking - can detect all movements separately if needed
# =============================================================================
# 
# jaw_key = {'camera_position':'side',
#               'bodypart':'jaw'}
# min_dlc_p = .99
# min_lick_bout_len = 10 # 0.1 s
# #%
# 
# for session in experiment.Session():
#     if len(videography.DLCTracking()&session)==0:
#         continue
# # =============================================================================
# #     if len(videography.DLCLickBout()&session)>0:
# #         continue
# # =============================================================================
#     
#     #%
#     x,y,p = (videography.DLCTracking()&session&jaw_key).fetch('x','y','p')
#     x = np.concatenate(x)
#     y = np.concatenate(y)
#     p = np.concatenate(p)
#     x[p<min_dlc_p] = np.nan
#     y[p<min_dlc_p] = np.nan
#     break
# #%%
#     origin_x = np.nanmedian(x)
#     origin_y = np.nanmedian(y)
#     distance = np.sqrt((x-origin_x)**2+(y-origin_y)**2)
#     distance_filt = utils_plot.rollingfun(distance,10)
# 
# =============================================================================
#%%
def populate_dlc_lick_bouts():
    #%%
    tongue_key = {'camera_position':'side',
                  'bodypart':'tongue_tip'}

    min_dlc_p = .99
    min_lick_bout_len = 10 # 0.033 s
    #%
    
    for session in experiment.Session():
        if len(videography.DLCTracking()&session)==0:
            continue
        if len(videography.DLCLickBout()&session)>0:
            continue
        
        #%
        x,y,p = (videography.DLCTracking()&session&tongue_key).fetch('x','y','p')
        x = np.concatenate(x)
        y = np.concatenate(y)
        p = np.concatenate(p)
        x[p<min_dlc_p] = np.nan
        y[p<min_dlc_p] = np.nan
        session_start_x = np.percentile(x[np.isnan(x)==False],1)
        session_start_y = np.percentile(y[np.isnan(y)==False],1)
        #%
        lick_bout_list = []
        lick_contact_list = []
    
        for trial in experiment.SessionTrial()&session:
            if len(videography.BehaviorVideo()&trial) == 0:
                print('no video for {}'.format(trial))
                continue
            frame_times = (videography.BehaviorVideo()&trial&tongue_key).fetch1('video_frame_times')
            action_event_ids, action_event_times = (experiment.ActionEvent()&trial).fetch('action_event_id','action_event_time')
            action_event_times = np.asarray(action_event_times,float)
            frame_interval = np.median(np.diff(frame_times))
            lickboutdict_base = trial.copy()
            lickboutdict_base.pop('trial_start_time')
            lickboutdict_base.pop('trial_end_time')
            if len(videography.DLCTracking()&trial)==0:
                continue
            x,y,p = (videography.DLCTracking()&trial&tongue_key).fetch1('x','y','p')
            #x_jaw,y_jaw,p_jaw = (videography.DLCTracking()&trial&jaw_key).fetch1('x','y','p')
            #%
            x = x.copy()
            y = y.copy()
            #%
            x[p<min_dlc_p] = np.nan
            y[p<min_dlc_p] = np.nan
            #%
            template = np.diff(np.asarray((np.concatenate([[False],np.isnan(x) == False,[False]])),int))
            starts = np.where(template == 1)[0]
            ends = np.where(template == -1)[0]
            lick_bout_id = -1
            for start,end in zip(starts,ends):
                lickboutdict = lickboutdict_base.copy()
                if end-start<min_lick_bout_len:
                    continue
                lick_bout_id += 1
                dist = sklearn.metrics.pairwise_distances(np.asarray([[session_start_x]*(end-start),[session_start_y]*(end-start)]).T,np.asarray([x[start:end],y[start:end]]).T)
                lick_trajectory = dist[0,:]
                peak_idx = np.argmax(lick_trajectory)
                lickboutdict['lick_bout_id'] = lick_bout_id
                lickboutdict['lick_bout_start_time']=frame_times[start]
                lickboutdict['lick_bout_start_frame']= start
                lickboutdict['lick_bout_end_frame']= end
                lickboutdict['lick_bout_peak_time'] =  frame_times[start+peak_idx]
                lickboutdict['lick_bout_peak_frame']=  start+peak_idx
                lickboutdict['lick_bout_amplitude'] =  lick_trajectory[peak_idx]
                lickboutdict['lick_bout_amplitude_x']= np.max(x[start:end]-session_start_x)
                lickboutdict['lick_bout_amplitude_y']= np.max(y[start:end]-session_start_y)
                lickboutdict['lick_bout_half_width'] = np.sum(lick_trajectory>lick_trajectory[peak_idx]/2)*frame_interval
                lickboutdict['lick_bout_rise_time']  = frame_interval*peak_idx
                lick_bout_list.append(lickboutdict)
                
                lick_identity = (action_event_times>lickboutdict['lick_bout_start_time']) & (action_event_times<=frame_times[lickboutdict['lick_bout_end_frame']-1])
                if any(lick_identity):
                    contact_dict =lickboutdict_base.copy()
                    contact_dict['lick_bout_id'] = lick_bout_id
                    contact_dict['action_event_id'] = action_event_ids[lick_identity][0]
                    contact_dict['contact_frame_number'] = np.argmin(np.abs(frame_times-action_event_times[lick_identity][0]))
                    lick_contact_list.append(contact_dict)
            dj.conn().ping()
                
                #%
        if len(lick_bout_list)>0:
            with dj.conn().transaction: #inserting one movie
                print('uploading lick bouts of session {}'.format(session)) #movie['movie_name']
                videography.DLCLickBout().insert(lick_bout_list,allow_direct_insert=True) 
                videography.DLCLickBoutContact().insert(lick_contact_list,allow_direct_insert=True) 
 #%%   
def populate_event_frame_ids():
    #%%
    for session in experiment.Session():
        for camera in videography.BehaviorCamera()&session:
            if len(videography.TrialEventFrame()&session)>0:
                continue
            if len(videography.DLCTracking()&session) == 0:
                continue
            trial_event_list = list()
            action_event_list = list()

            for trial in experiment.SessionTrial()&session:
                try:
                    frame_times = (videography.BehaviorVideo()&trial&camera).fetch1('video_frame_times')
                except:
                    print('no video')
                    continue
                trial_dict_base = trial.copy()
                trial_dict_base.pop('trial_start_time')
                trial_dict_base.pop('trial_end_time')
                trial_dict_base['camera_position'] = camera['camera_position']
               
                for action_event in experiment.ActionEvent()&trial:
                    action_event_dict = trial_dict_base.copy()
                    action_event_dict['action_event_id'] = action_event['action_event_id']
                    time_now = float(action_event['action_event_time'])
                    frame_num = np.argmin(np.abs(frame_times-time_now))
                    dt = frame_times[frame_num]-time_now
                    action_event_dict['frame_num'] = frame_num
                    action_event_dict['dt'] = dt
                    action_event_list.append(action_event_dict)
                for trial_event in experiment.TrialEvent()&trial:
                    trial_event_dict = trial_dict_base.copy()
                    trial_event_dict['trial_event_id'] = trial_event['trial_event_id']
                    time_now = float(trial_event['trial_event_time'])
                    frame_num = np.argmin(np.abs(frame_times-time_now))
                    dt = frame_times[frame_num]-time_now
                    trial_event_dict['frame_num'] = frame_num
                    trial_event_dict['dt'] = dt
                    trial_event_list.append(trial_event_dict)
            with dj.conn().transaction: #inserting one FOV
                print('uploading videography frame indices to datajoint -  subject-session-camera: {}-{}-{}'.format(session['subject_id'],session['session'],camera['camera_position'])) #movie['movie_name']
                videography.ActionEventFrame().insert(action_event_list, allow_direct_insert=True)
                videography.TrialEventFrame().insert(trial_event_list, allow_direct_insert=True)