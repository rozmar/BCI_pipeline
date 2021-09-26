from utils import utils_pybpod, utils_ephys, utils_imaging
import numpy as np
import datetime
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
try:
    import datajoint as dj
    from pipeline import pipeline_tools,lab,experiment, videography, imaging
    from pipeline.ingest import datapipeline_metadata
except:
    print('could not import datajoint')
from matplotlib import cm

import matplotlib

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}
matplotlib.rc('font', **font)
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Tahoma']
#%
def gaussFilter(sig,sRate,sigma = .00005):
    si = 1/sRate
    #sigma = .00005
    sig_f = ndimage.gaussian_filter(sig,sigma/si)
    return sig_f

def rollingfun(y, window = 10, func = 'mean'):
    """
    rollingfun
        rolling average, min, max or std
    
    @input:
        y = array, window, function (mean,min,max,std)
    """
    if len(y)<=window:
        if func =='mean':
            out = np.ones(len(y))*np.nanmean(y)
        elif func == 'min':
            out = np.ones(len(y))*np.nanmin(y)
        elif func == 'max':
            out = np.ones(len(y))*np.nanmaxn(y)
        elif func == 'std':
            out = np.ones(len(y))*np.nanstd(y)
        elif func == 'median':
            out = np.ones(len(y))*np.nanmedian(y)
        else:
            print('undefinied funcion in rollinfun')
    else:
        y = np.concatenate([y[window::-1],y,y[:-1*window:-1]])
        ys = list()
        for idx in range(window):    
            ys.append(np.roll(y,idx-round(window/2)))
        if func =='mean':
            out = np.nanmean(ys,0)[window:-window]
        elif func == 'min':
            out = np.nanmin(ys,0)[window:-window]
        elif func == 'max':
            out = np.nanmax(ys,0)[window:-window]
        elif func == 'std':
            out = np.nanstd(ys,0)[window:-window]
        elif func == 'median':
            out = np.nanmedian(ys,0)[window:-window]
        else:
            print('undefinied funcion in rollinfun')
    return out

def plot_behavior_session(session_key_wr = {'wr_id':'BCI07', 'session':2},moving_window = 10):
    #%
    wr_names = lab.WaterRestriction().fetch('water_restriction_number')
    if session_key_wr['wr_id'] not in wr_names:
        print('mouse ID not in database, use one of these: {}'.format(wr_names))
        return
    subject_id = (lab.WaterRestriction()&'water_restriction_number = "{}"'.format(session_key_wr['wr_id'])).fetch1('subject_id')
    #%
    sessions,session_dates = (experiment.Session()&'subject_id = {}'.format(subject_id)).fetch('session','session_date')
    
    if session_key_wr['session'] not in sessions:
        texttodisplay = dict()
        for session,sessiondate in zip(sessions,session_dates):
            texttodisplay['session {}'.format(session)] =  str(sessiondate)
        print('session not in database, use one of these for {}:'.format(session_key_wr['wr_id']))
        print(texttodisplay)
        return texttodisplay
    #%
    session_key = {'subject_id':subject_id,
                   'session':session_key_wr['session']}
    session_date, session_user = (experiment.Session()&session_key).fetch1('session_date','username')
    #%
     # trial number
    
    valve_time_trials, valve_times = (experiment.LickPortSetting.OpenDuration()&experiment.TrialLickportChoice()&session_key).fetch('trial','open_duration')
    
    gocue_trial, go_cue = (experiment.TrialEvent()&session_key&'trial_event_type = "go"').fetch('trial','trial_event_time')
    thresholdcross_trial, threshold_cross = (experiment.TrialEvent()&session_key&'trial_event_type = "threshold cross"').fetch('trial','trial_event_time')
    reward_trial, reward_time = (experiment.TrialEvent()&session_key&'trial_event_type = "reward"').fetch('trial','trial_event_time')
    time_to_hit = np.zeros(len(gocue_trial))*np.nan
    for trial,threshold_t in zip(thresholdcross_trial,threshold_cross):
        idx = gocue_trial==trial
        time_to_hit[idx] = threshold_t-go_cue[idx]
    
    time_to_collect_reward = np.zeros(len(gocue_trial))*np.nan
    for trial,reward_t in zip(reward_trial,reward_time):
        idx_th = thresholdcross_trial==trial
        idx = gocue_trial==trial
        time_to_collect_reward[idx] = reward_t-threshold_cross[idx_th]
        
    trial_num,outcome,lickport_auto_step_freq,task_protocol,lickport_step_size,trial_start_time,trial_end_time,lickport_maximum_speed,lickport_limit_far = (experiment.SessionTrial()*experiment.LickPortSetting()*experiment.BehaviorTrial()&session_key).fetch('trial','outcome','lickport_auto_step_freq','task_protocol','lickport_step_size','trial_start_time','trial_end_time','lickport_maximum_speed','lickport_limit_far')
    lickport_step_size = np.asarray(lickport_step_size,float)
    trial_lengths = np.asarray(trial_end_time-trial_start_time,float)
    trial_num_bci,threshold_low,threshold_high,bci_conditioned_neuron_sign,bci_conditioned_neuron_idx = (experiment.BCISettings().ConditionedNeuron()&session_key).fetch('trial','bci_threshold_low','bci_threshold_high','bci_conditioned_neuron_sign','bci_conditioned_neuron_idx')
    hits = outcome == 'hit'
    hit_rate = np.zeros(len(outcome))*np.nan
    
    trial_average_speed = np.abs(np.asarray(lickport_limit_far,float))/time_to_hit
    trial_max_speed = np.zeros(len(outcome))*np.nan
    for trial_i, step_size in zip(trial_num,lickport_step_size):
        step_ts = (experiment.TrialEvent()&session_key&'trial_event_type = "lickport step"'&'trial = {}'.format(trial_i)).fetch('trial_event_time')
        dts = np.diff(np.asarray(step_ts,float))
        if len(dts)>0:
            trial_max_speed[trial_i] =step_size/np.min(dts) 
        
        
    
    rewardsperminute = np.zeros(len(outcome))*np.nan
    trial_length_moving = np.zeros(len(outcome))*np.nan
    timetohit_moving = np.zeros(len(outcome))*np.nan
    timetocollectreward_moving = np.zeros(len(outcome))*np.nan
    max_speed_moving = np.zeros(len(outcome))*np.nan
    trial_average_speed_moving = np.zeros(len(outcome))*np.nan
    task_change_idx = np.where(np.abs(np.diff(task_protocol))>0)[0]
    task_change_idx = np.concatenate([[0],task_change_idx,[len(task_protocol)]])
    for idx in np.arange(len(task_change_idx)-1):
        hit_rate[task_change_idx[idx]:task_change_idx[idx+1]] = rollingfun(hits[task_change_idx[idx]:task_change_idx[idx+1]],moving_window)
        rewardsperminute[task_change_idx[idx]:task_change_idx[idx+1]] = rollingfun((hits/trial_lengths*60)[task_change_idx[idx]:task_change_idx[idx+1]],moving_window)
        trial_length_moving[task_change_idx[idx]:task_change_idx[idx+1]] = rollingfun(trial_lengths[task_change_idx[idx]:task_change_idx[idx+1]],moving_window)
        timetohit_moving[task_change_idx[idx]:task_change_idx[idx+1]] = rollingfun(time_to_hit[task_change_idx[idx]:task_change_idx[idx+1]],moving_window)
        timetocollectreward_moving[task_change_idx[idx]:task_change_idx[idx+1]] = rollingfun(time_to_collect_reward[task_change_idx[idx]:task_change_idx[idx+1]],moving_window)
        trial_average_speed_moving[task_change_idx[idx]:task_change_idx[idx+1]] = rollingfun(trial_average_speed[task_change_idx[idx]:task_change_idx[idx+1]],moving_window)
        max_speed_moving[task_change_idx[idx]:task_change_idx[idx+1]] = rollingfun(trial_max_speed[task_change_idx[idx]:task_change_idx[idx+1]],moving_window)
    
    
    #%
    fig_behavior = plt.figure(figsize = [8,15])
    ax_movement_speed = fig_behavior.add_subplot(412)
    ax_distance = ax_movement_speed.twinx()
    ax_distance.set_ylabel('Distance to travel (mm)')
    #ax_movement_speed.set_ylabel(r'\textcolor{red}{Max }'+r'\textcolor{green}{Mean }'+r'\textcolor{blue}{Baseline }'+ r'{\color{red}speed} (mm/s)')
    ax_movement_speed.set_ylabel('Max/Mean/Baseline speed (mm/s)')
    #ax_movement_speed.tick_params(axis='y', colors='blue')
    #ax_movement_speed.yaxis.label.set_color('blue')
    
    
    ax_rewardrate = fig_behavior.add_subplot(411,sharex = ax_movement_speed)
    ax_rewardrate.set_ylabel('Reward rate over {} trials'.format(moving_window))
    ax_rewardsperminute = ax_rewardrate.twinx()
    ax_rewardsperminute.set_ylabel('Reward/ minute over {} trials'.format(moving_window))
    ax_rewardsperminute.tick_params(axis='y', colors='green')
    ax_rewardsperminute.yaxis.label.set_color('green')
    
    ax_timing = fig_behavior.add_subplot(413,sharex = ax_movement_speed)
    ax_time_to_lick = ax_timing.twinx()
    
    ax_valve = fig_behavior.add_subplot(414,sharex = ax_movement_speed)
    ax_threshold = ax_valve.twinx()
    ax_threshold.set_ylabel('BCI thresholds')
    ax_timing.semilogy(trial_num,trial_lengths,'rx')
    ax_timing.semilogy(trial_num,trial_length_moving,'r-')
    ax_timing.semilogy(trial_num,time_to_hit,'g.')
    ax_timing.semilogy(trial_num,timetohit_moving,'g-')
    ax_time_to_lick.plot(trial_num,time_to_collect_reward,'b^')
    ax_time_to_lick.plot(trial_num,timetocollectreward_moving,'b-')
    
    
    for conditioned_neuron_idx_now in np.unique(bci_conditioned_neuron_idx):
        idx_now = bci_conditioned_neuron_idx == conditioned_neuron_idx_now
        if np.mean(bci_conditioned_neuron_sign[idx_now])>0:
            cellcolor = 'green'
        else:
            cellcolor = 'red'
        ax_threshold.plot(trial_num_bci[idx_now],threshold_low[idx_now],'-',color = cellcolor)
        ax_threshold.plot(trial_num_bci[idx_now],threshold_high[idx_now],'-.',color = cellcolor)
    
    ax_movement_speed.plot(trial_num,np.asarray(lickport_auto_step_freq,float)*np.asarray(lickport_step_size,float),'b-',label = 'baseline speed')
    ax_movement_speed.plot(trial_num,np.asarray(lickport_maximum_speed,float),'r-',label = 'max speed')
    ax_distance.plot(trial_num,np.abs(lickport_limit_far),'k-')
    ax_movement_speed.plot(trial_num,trial_average_speed,'g.',label = 'average speed')
    ax_movement_speed.plot(trial_num,trial_average_speed_moving,'g-')
    ax_movement_speed.plot(trial_num,trial_max_speed,'rx',label = 'maximum speed')
    ax_movement_speed.plot(trial_num,max_speed_moving,'r--')
    
    ax_rewardrate.plot(trial_num,hit_rate,'k-')
    ax_rewardsperminute.plot(trial_num,rewardsperminute,'g--')
    outcomes = np.ones(len(outcome))*np.nan
    outcomes[outcome=='hit']=1
    ax_rewardrate.plot(trial_num,outcomes,'g|')
    outcomes = np.ones(len(outcome))*np.nan
    outcomes[outcome=='miss']=0
    ax_rewardrate.plot(trial_num,outcomes,'r|')
    outcomes = np.ones(len(outcome))*np.nan
    outcomes[outcome=='ignore']=0
    ax_rewardrate.plot(trial_num,outcomes,'rx')
    
    ax_valve.plot(valve_time_trials, valve_times,'ko')
    ax_valve.set_ylabel('Valve time (s)')
    
    ax_timing.set_ylabel('Length of trial and time to hit (s)')
    ax_rewardrate.set_title('{} - {} (session {}) - {}'.format(session_key_wr['wr_id'],session_date,session_key_wr['session'],session_user))
    ax_time_to_lick.set_ylabel('Time to lick from threshold crossing (s)')
    #%
    return fig_behavior

def plot_motor_precision():
    csvfile = '/home/rozmar/Data/Behavior/Behavior_rigs/DOM3-MMIMS/BCI/experiments/BCI/setups/DOM3/sessions/20210323-092017/20210323-092017.csv'
    ephysfile = '/home/rozmar/Data/Wavesurfer/DOM3-MMIMS/test_subject/2021-03-23/test_0001.h5'
    
    csv_data = utils_pybpod.load_and_parse_a_csv_file(csvfile)
    behavior_dict = utils_pybpod.minethedata(csv_data,extract_variables = True)
    
    #% Load zaber stuff
    subject_name = csv_data['subject'][0]
    setup_name = csv_data['setup'][0]
    zaber_dict = utils_pybpod.generate_zaber_info_for_pybpod_dict(behavior_dict,subject_name,setup_name,zaber_folder_root = '/home/rozmar/Data/Behavior/BCI_Zaber_data')
    
    
    #% Load wavesurfer
    
    ephysdata = utils_ephys.load_wavesurfer_file(ephysfile)[0]
    
    
    #%
    sampling_rate = ephysdata['sampling_rate']
    trial_start_idxs = np.where(np.diff(ephysdata['DI-TrialStart'])==1)[0]
    trial_end_idxs = np.where(np.diff(ephysdata['DI-TrialStart'])==-1)[0]
    trial_start_times_ws = trial_start_idxs/ephysdata['sampling_rate']#+ephysdata['sweep_start_timestamp']
    trial_starttimedeltas = np.asarray([datetime.timedelta(seconds = starttime) for starttime in trial_start_times_ws])
    trial_start_timestamps_ws = ephysdata['sweep_start_timestamp'] + trial_starttimedeltas
    bitcode_trial_nums = utils_ephys.decode_bitcode(ephysdata)
    #%decode bitcode
    
        #break
    #%
    channel_to_extract = 'AI-ROI voltage'
    ys = list()
    for trial_start_idx,trial_end_idx in zip(trial_start_idxs,trial_end_idxs):
        y = ephysdata[channel_to_extract][trial_start_idx:trial_end_idx]
        ys.append(y)
            
       
    

    
    trial_num = 11
    
    s =zaber_dict['trigger_step_size'][trial_num]/1000
    v = zaber_dict['speed'][trial_num]
    a = zaber_dict['acceleration'][trial_num]
    zaber_step_time = utils_pybpod.calculate_step_time(s,v,a)
    zaber_step_step_number = int(zaber_step_time*sampling_rate)
    zaber_reward_zone_start = zaber_dict['reward_zone'][trial_num]
    analog_y = ys[trial_num]
    analog_x = np.arange(len(analog_y))/sampling_rate
    zaber_move_forward = behavior_dict['zaber_move_forward'][trial_num]
    zaber_step_size = s*int('{}1'.format(zaber_dict['direction'][trial_num]))
    zaber_position = np.zeros(len(analog_y))
    for zaber_move_forward_now in zaber_move_forward:
        zaber_move_idx = np.argmax(analog_x>=zaber_move_forward_now)
        zaber_position[zaber_move_idx:zaber_move_idx+zaber_step_step_number] += zaber_step_size/zaber_step_step_number
    zaber_position = np.cumsum(zaber_position)+zaber_dict['limit_far'][trial_num]
    
    threshold_crossing_time = behavior_dict['threshold_crossing_times'][trial_num]
    
    threshold_crossing_idx = np.argmax(analog_x>threshold_crossing_time)
    calculated_position_at_threshold_crossing_time =  zaber_position[np.argmax(analog_x>threshold_crossing_time)]    
    calculated_threshold_crossing_time = np.argmax(zaber_position>zaber_reward_zone_start)/sampling_rate
    
    
    fig = plt.figure(figsize = [12,15])
    ax_analog = fig.add_subplot(2,1,1)
    #ax_ttl = fig.add_subplot(2,1,2,sharex = ax_analog)
    ax_location = fig.add_subplot(2,1,2,sharex = ax_analog)
    #ax_speed = fig.add_subplot(4,1,4,sharex = ax_analog)
    ax_speed = ax_analog.twinx()#fig.add_subplot(4,1,4,sharex = ax_analog)
       
    
    ax_analog.plot(analog_x,analog_y,'k-',label = 'analog scanimage input')
    ax_analog.plot(zaber_move_forward,np.zeros(len(zaber_move_forward)),'k|')
    
    ax_analog.set_ylabel('Signal from scanimage (V)')
    
    ax_location.plot(analog_x,zaber_position,'k-')
    
    #ax_location.plot(threshold_crossing_time,calculated_position_at_threshold_crossing_time,'ro',ms = 12)
    #ax_location.plot(calculated_threshold_crossing_time,zaber_reward_zone_start,'go',ms = 12)
    ax_location.vlines(threshold_crossing_time,0,np.max(zaber_position),'r',linestyles='dashed')
    ax_location.vlines(calculated_threshold_crossing_time,0,np.max(zaber_position),'g',linestyles='dashed')
    ax_location.hlines(zaber_dict['limit_close'],0,np.max(analog_x),'r',linestyles='dashed')
    ax_location.hlines(zaber_dict['limit_far'],0,np.max(analog_x),'r',linestyles='dashed')
    ax_location.set_ylabel('Lickport position (mm)')
    ax_location.set_xlabel('Time (s)')
    
    downsample_x = 10
    x_down = analog_x[::downsample_x]
    position_down = zaber_position[::downsample_x]
    
    speed_down = np.diff(position_down)/np.diff(x_down)[0]
    speed_filt = gaussFilter(speed_down,sampling_rate/downsample_x,sigma = .02)
    
    ax_speed.plot(x_down[:-1]+np.diff(x_down)[0]/2,speed_filt,'r-',label = 'calculated speed')
    ax_speed.set_ylabel('Lickport speed (mm/s)')
    ax_speed.yaxis.label.set_color('red')
#%
def plot_behavior_session_stats():
    #%
    fig_sessions = plt.figure(figsize = [10,15])
    ax_trialnum = fig_sessions.add_subplot(311)
    ax_valvetime =  fig_sessions.add_subplot(313)
    ax_dates = fig_sessions.add_subplot(312)
    subject_ids,wr_names = lab.WaterRestriction().fetch('subject_id','water_restriction_number')
    order = np.argsort(wr_names)
    subject_ids = subject_ids[order]
    wr_names = wr_names[order]
    
    
    cm_subsection = np.linspace(0, 1, len(wr_names)) 
    
    for  i,(subject_id,wr_name,color_i) in enumerate(zip(subject_ids,wr_names,cm_subsection)):
        color = cm.inferno(color_i)
        sessions = experiment.Session()&'subject_id = {}'.format(subject_id)
        session_nums,session_dates = sessions.fetch('session','session_date')
        surgery_dates = (lab.Surgery()&'subject_id = {}'.format(subject_id)).fetch('start_time')
        trial_nums = list()
        trial_nums_cl = list()
        trial_nums_ol = list()
        valve_times_mean = list()
        valve_times_std = list()
        
        for session_num in session_nums:
            session = sessions&'session = {}'.format(session_num)
            trial_nums.append(len(experiment.SessionTrial()&session))
            trial_nums_cl.append(len(experiment.BehaviorTrial()&session & 'task = "BCI CL"'))
            trial_nums_ol.append(len(experiment.BehaviorTrial()&session & 'task = "BCI OL"'))
            open_duration = np.asarray((experiment.LickPortSetting.OpenDuration()*experiment.TrialLickportChoice()&session).fetch('open_duration'),float)
            valve_times_mean.append(np.nanmean(open_duration))
            valve_times_std.append(np.nanstd(open_duration))
        ax_trialnum.plot(session_nums,trial_nums_cl,'-',label = wr_name,color=color, linewidth = 4)
        ax_trialnum.plot(session_nums,trial_nums_ol,'--',color=color, linewidth = 2)
        ax_valvetime.plot(valve_times_mean,trial_nums,'o',color = color)
        ax_dates.plot(session_dates,np.ones(len(session_dates))*i,'o',color=color)
        ax_dates.plot(surgery_dates,np.ones(len(surgery_dates))*i,'kx',ms = 8)#,color=color
        #break
    
    ax_trialnum.set_xlabel('Session number')
    ax_trialnum.set_ylabel('Trial num in session')
    
    ax_valvetime.set_xlabel('Mean valve time (s)')
    ax_valvetime.set_ylabel('Trial num in session')
    
    ax_dates.set_xlabel('Session date')
    ax_dates.set_yticks(np.arange(len(wr_names)))
    ax_dates.set_yticklabels(wr_names)
    ax_dates.invert_yaxis()
    ax_trialnum.legend()
    
def plot_lickport_tracking(session_key_wr = {'wr_id':'BCI14', 'session':2},
                           example_trial_num = 101,
                           min_probability = .99):
    # correlate DLC lickport tracking with zaber steps
    #wr_names = lab.WaterRestriction().fetch('water_restriction_number')
    subject_id = (lab.WaterRestriction()&'water_restriction_number = "{}"'.format(session_key_wr['wr_id'])).fetch1('subject_id')
    session_key = {'subject_id':subject_id,
                   'session':session_key_wr['session']}
    
    dlc_key = {'camera_position':'side',
               'bodypart':'lickport'}
    
    fig = plt.figure()
    ax_dlc = fig.add_subplot(211)
    ax_step = ax_dlc.twinx()
    ax_all_points = fig.add_subplot(212)
    si = .001
    #min_probability = .99
    #example_trial_num = 101
    for trial in experiment.SessionTrial()&session_key:
        try:
            dlc_exp_times = (videography.BehaviorVideo()&trial&dlc_key).fetch1('video_frame_times')
            dlc_lickport_position = (videography.DLCTracking&trial&dlc_key).fetch1('x')*-1
            dlc_lickport_prob = (videography.DLCTracking&trial&dlc_key).fetch1('p')
            dlc_lickport_position[dlc_lickport_prob<min_probability] = np.nan
            
            lickport_step_times = np.asarray((experiment.TrialEvent()&trial&'trial_event_type = "lickport step"').fetch('trial_event_time'),float)
            licport_step_size = float((experiment.LickPortSetting()&trial).fetch1('lickport_step_size'))
            lickport_step_duration = float((experiment.LickPortSetting()&trial).fetch1('lickport_step_time'))
            #lickport_distance = float((experiment.LickPortSetting()&trial).fetch1('lickport_limit_far'))*-1
            go_cue = float((experiment.TrialEvent()&trial&'trial_event_type = "go"').fetch1('trial_event_time'))
            reward = float((experiment.TrialEvent()&trial&'trial_event_type = "reward"').fetch1('trial_event_time'))
            zaber_step_step_number = int(lickport_step_duration/si)
            lickport_position_t = np.arange(np.min(dlc_exp_times),np.max(dlc_exp_times),si)
            lickport_position = np.zeros(len(lickport_position_t))
            
            for zaber_move_forward_now in lickport_step_times:
                zaber_move_idx = np.argmax(lickport_position_t>=zaber_move_forward_now)
                lickport_position[zaber_move_idx:zaber_move_idx+zaber_step_step_number] += licport_step_size/zaber_step_step_number
            lickport_position = np.cumsum(lickport_position)
            #lickport_position[:np.argmax(lickport_position_t>go_cue)] = lickport_distance
            
            dlc_needed_frames = (dlc_exp_times>go_cue) & (dlc_exp_times< reward)
            dlc_vals = dlc_lickport_position[dlc_needed_frames]
            zaber_locations = list()
            for dlc_frame_time in dlc_exp_times[dlc_needed_frames]:
                zaber_locations.append(lickport_position[np.argmax(lickport_position_t>dlc_frame_time)])
            ax_all_points.plot(dlc_vals,zaber_locations,'ko',alpha = .1)
            if trial['trial']==example_trial_num:
                
                ax_dlc.plot(dlc_exp_times,dlc_lickport_position,'k-', label = 'DLC lickport position')
                ax_step.plot(lickport_position_t,lickport_position,'r', label = 'zaber motor position')
        except:
            pass
        
    #    break
    #%
    session_date = (experiment.Session()&session_key).fetch1('session_date')
    ax_dlc.set_title('{} - session {} - {}'.format(session_key_wr['wr_id'],session_key_wr['session'],session_date))
    ax_dlc.set_xlabel('time from trial start (s)')
    ax_dlc.set_ylabel('Lickport tip location (pixel)')
    ax_step.set_ylabel('Motor location (mm)')
    ax_all_points.set_ylabel('Motor location (mm)')
    ax_all_points.set_xlabel('Lickport tip location (pixel)')

def plot_session_and_licks(session_key_wr):
    subject_id = (lab.WaterRestriction()&'water_restriction_number = "{}"'.format(session_key_wr['wr_id'])).fetch1('subject_id')
    session_key = {'subject_id':subject_id,
                   'session':session_key_wr['session']}
    dlc_min_probability = .99

    moving_window = 30 # trials
    dff_averaging_win = 180
    lick_bin_window = 180 #s
    lick_bin_step = 10 #s
    
    dlc_key_lickport = {'camera_position':'side',
                        'bodypart':'lickport'}
    
    fig = plt.figure(figsize = [10,15])
    ax_dff = fig.add_subplot(4,1,2)
    ax_dff.set_ylabel('Activity of conditioned neuron (dF/F)')
    ax_dff_mean = ax_dff.twinx()
    ax_rewardrate = fig.add_subplot(4,1,1,sharex = ax_dff)
    ax_rewardrate.set_ylabel('Reward rate over {} trials'.format(moving_window))
    ax_rewardsperminute = ax_rewardrate.twinx()
    ax_rewardsperminute.set_ylabel('Reward/ minute over {} trials'.format(moving_window))
    ax_rewardsperminute.tick_params(axis='y', colors='green')
    ax_rewardsperminute.yaxis.label.set_color('green')
    
    
    ax_tongue = fig.add_subplot(4,1,3,sharex = ax_dff)
    ax_licknum = ax_tongue.twinx()
    ax_trial_details = fig.add_subplot(4,1,4,sharex = ax_dff)
    
    trial_num,outcome,lickport_auto_step_freq,task_protocol,lickport_step_size,trial_start_time,trial_end_time,lickport_maximum_speed,lickport_limit_far = (experiment.SessionTrial()*experiment.LickPortSetting()*experiment.BehaviorTrial()&session_key).fetch('trial','outcome','lickport_auto_step_freq','task_protocol','lickport_step_size','trial_start_time','trial_end_time','lickport_maximum_speed','lickport_limit_far')
    trial_lengths = np.asarray(trial_end_time-trial_start_time,float)
    hits = outcome == 'hit'
    hit_rate = np.zeros(len(outcome))*np.nan
    
    rewardsperminute = np.zeros(len(outcome))*np.nan
    task_change_idx = np.where(np.abs(np.diff(task_protocol))>0)[0]
    task_change_idx = np.concatenate([[0],task_change_idx,[len(task_protocol)]])
    for idx in np.arange(len(task_change_idx)-1):
        hit_rate[task_change_idx[idx]:task_change_idx[idx+1]] = rollingfun(hits[task_change_idx[idx]:task_change_idx[idx+1]],moving_window)
        rewardsperminute[task_change_idx[idx]:task_change_idx[idx+1]] = rollingfun((hits/trial_lengths*60)[task_change_idx[idx]:task_change_idx[idx+1]],moving_window)
    
    ax_rewardrate.plot(trial_end_time,hit_rate,'k-')
    ax_rewardsperminute.plot(trial_end_time,rewardsperminute,'g--')
    outcomes = np.ones(len(outcome))*np.nan
    outcomes[outcome=='hit']=1
    ax_rewardrate.plot(trial_end_time,outcomes,'g|')
    outcomes = np.ones(len(outcome))*np.nan
    outcomes[outcome=='miss']=0
    ax_rewardrate.plot(trial_end_time,outcomes,'r|')
    outcomes = np.ones(len(outcome))*np.nan
    outcomes[outcome=='ignore']=0
    ax_rewardrate.plot(trial_end_time,outcomes,'rx')
    
    conditioned_rois = imaging.ROI()*imaging.ConditionedROI()&session_key
    frame_times = (imaging.FOVFrameTimes()&session_key).fetch1('frame_times')
    for roi_cond in conditioned_rois:
        dff = (imaging.ROITrace()&roi_cond).fetch1('roi_dff')
        dff_down = rollingfun(dff,30)[::30]
        frame_times_down = frame_times[::30]
        dff_mean = rollingfun(dff_down,dff_averaging_win)
        if roi_cond['cond_roi_multiplier']>0:
            color = 'green'
        elif roi_cond['cond_roi_multiplier']<0:
            color = 'red'
            #continue
        else:
            color= 'black'
        ax_dff.plot(frame_times,dff,color = color, alpha = .4)
        ax_dff_mean.plot(frame_times_down,dff_mean,'--',color = color, alpha = 1)
    ax_dff_mean.set_ylabel('mean activity over {} s'.format(dff_averaging_win))
    #% lick amplitudes - contact vs non-contact
    
    #%
    lick_times = list()
    lick_amplitudes = list()
    lick_times_contact = list()
    lick_amplitudes_contact = list()
    lick_times_noncontact = list()
    lick_amplitudes_noncontact = list()
    
    for trial in experiment.SessionTrial()&session_key:
# =============================================================================
#         try:
#             reward_time = float((experiment.TrialEvent()&trial&'trial_event_type = "reward"').fetch1('trial_event_time'))
#         except:
#             reward_time  = 100
# =============================================================================
        lick_bout_start_times, lick_bout_amplitudes = (videography.DLCLickBout()&trial&'lick_bout_amplitude<150').fetch('lick_bout_start_time','lick_bout_amplitude')
    # =============================================================================
    #     needed = lick_bout_start_times<reward_time
    #     lick_bout_start_times = lick_bout_start_times[needed]
    #     lick_bout_amplitudes = lick_bout_amplitudes[needed]
    # =============================================================================
        lick_times.extend(lick_bout_start_times+float(trial['trial_start_time']))
        lick_amplitudes.extend(lick_bout_amplitudes)
        
        lick_bout_start_times, lick_bout_amplitudes = (videography.DLCLickBoutContact()*videography.DLCLickBout()&trial&'lick_bout_amplitude<150').fetch('lick_bout_start_time','lick_bout_amplitude')
    # =============================================================================
    #     needed = lick_bout_start_times<reward_time
    #     lick_bout_start_times = lick_bout_start_times[needed]
    #     lick_bout_amplitudes = lick_bout_amplitudes[needed]
    # =============================================================================
        lick_times_contact.extend(lick_bout_start_times+float(trial['trial_start_time']))
        lick_amplitudes_contact.extend(lick_bout_amplitudes)
        #break
        #%
    for lick_time,lick_amplitude in zip(lick_times,lick_amplitudes):
        if lick_time not in lick_times_contact:
            lick_times_noncontact.append(lick_time)
            lick_amplitudes_noncontact.append(lick_amplitude)
    #%
    lick_times = np.asarray(lick_times)
    lick_amplitudes = np.asarray(lick_amplitudes)
    lick_times_contact = np.asarray(lick_times_contact)
    lick_amplitudes_contact = np.asarray(lick_amplitudes_contact)
    lick_times_noncontact = np.asarray(lick_times_noncontact)
    lick_amplitudes_noncontact = np.asarray(lick_amplitudes_noncontact)
    
    bin_centers = np.arange(np.min(lick_times),np.max(lick_times),lick_bin_step)
    lick_bin_mean_contact = list()
    lick_bin_mean_noncontact = list()
    lick_num_contact = list()
    lick_num_noncontact = list()
    for bin_center in bin_centers:
        idxes = (lick_times_contact>bin_center-lick_bin_window/2) & (lick_times_contact<bin_center+lick_bin_window/2)
        lick_bin_mean_contact.append(np.median(lick_amplitudes_contact[idxes]))   
        idxes_2 = (lick_times_noncontact>bin_center-lick_bin_window/2) & (lick_times_noncontact<bin_center+lick_bin_window/2)
        lick_bin_mean_noncontact.append(np.median(lick_amplitudes_noncontact[idxes_2]))   
        lick_num_contact.append(sum(idxes))#sum(idxes)+
        lick_num_noncontact.append(sum(idxes_2))#sum(idxes)+
        #%
    ax_tongue.plot(lick_times_noncontact,lick_amplitudes_noncontact,'ko', alpha = .4,label = 'non-contact licks') 
    ax_tongue.plot(bin_centers,lick_bin_mean_noncontact,'w-',linewidth = 6)
    ax_tongue.plot(bin_centers,lick_bin_mean_noncontact,'k-',linewidth = 4)#,label = 'average over {}s'.format(lick_bin_window))
    ax_tongue.plot(lick_times_contact,lick_amplitudes_contact,'ro', alpha = .4,label = 'contact licks') 
    ax_tongue.plot(bin_centers,lick_bin_mean_contact,'w-',linewidth = 6)
    ax_tongue.plot(bin_centers,lick_bin_mean_contact,'r-',linewidth = 4)#,label = 'average over {}s'.format(lick_bin_window))
    ax_tongue.legend()
    ax_tongue.set_ylabel('Lick distance (pixels)')
    #%
    ax_licknum.plot(bin_centers,(np.asarray(lick_num_contact)+np.asarray(lick_num_noncontact))/lick_bin_window,'b-', linewidth = 4)
    ax_licknum.set_ylabel('lick rate (Hz)')
    ax_licknum.yaxis.label.set_color('blue')
    
    ylim = ax_tongue.get_ylim()
    ax_tongue.set_ylim([ylim[0]-np.diff(ylim)/3,ylim[1]])
    ylim = ax_licknum.get_ylim()
    ax_licknum.set_ylim([ylim[0],ylim[1]+2*np.diff(ylim)/3])
    #%
    lickport_x,lickport_p,frame_times,trial_start_time = (experiment.SessionTrial()*videography.BehaviorVideo()*videography.DLCTracking()&session_key&dlc_key_lickport).fetch('x','p','video_frame_times','trial_start_time')
    lickport_x = np.concatenate(lickport_x)
    lickport_p = np.concatenate(lickport_p)
    lickport_x[lickport_p<dlc_min_probability] = np.nan
    lickport_time = []
    lickport_x = lickport_x*-1
    for t,st in zip(frame_times,trial_start_time):
        lickport_time.append(t+float(st))
    lickport_time  = np.concatenate(lickport_time)
    needed = np.isnan(lickport_x)==False
    lickport_x = lickport_x[needed]
    lickport_time = lickport_time[needed]
    lickport_x = lickport_x - np.percentile(lickport_x,1)
    lickport_x  = lickport_x/np.percentile(lickport_x,99)
    lickport_x_filt = rollingfun(lickport_x,5)
    ax_trial_details.plot(lickport_time,lickport_x_filt,'k-')
    ax_trial_details.set_ylim([-0.1,1.1])
    ax_trial_details.set_ylabel('Lickport position')
    #%
    go_cue_times, trial_start_times = (experiment.SessionTrial()*experiment.TrialEvent()&session_key&'trial_event_type = "go"').fetch('trial_event_time','trial_start_time')
    go_cue_times = np.asarray(trial_start_times+go_cue_times,float)
    reward_times,trial_start_times = (experiment.SessionTrial()*experiment.TrialEvent()&session_key&'trial_event_type = "reward"').fetch('trial_event_time','trial_start_time')
    reward_times = np.asarray(trial_start_times+reward_times,float)
    ax_trial_details.vlines(go_cue_times,0,1,color = 'green')
    ax_trial_details.plot(go_cue_times,np.zeros(len(go_cue_times))+.5,'>',color = 'green')
    ax_trial_details.vlines(reward_times,0,1,color = 'red')
    ax_trial_details.plot(reward_times,np.zeros(len(reward_times))+.5,'<',color = 'red')
    # =============================================================================
    # #% paw movement
    # x,y,p,frame_times,trial_start_times = (experiment.SessionTrial()*videography.BehaviorVideo()*videography.DLCTracking()&session_key&dlc_key_hand).fetch('x','y','p','video_frame_times','trial_start_time')
    # paw_time = []
    # for t,st in zip(frame_times,trial_start_time):
    #     paw_time.append(t+float(st))
    # x=np.concatenate(x)
    # y =np.concatenate(y)
    # p =np.concatenate(p)
    # paw_time =np.concatenate(paw_time)
    # x[p<dlc_min_probability] = np.nan
    # y[p<dlc_min_probability] = np.nan
    # x=x[::10]
    # y=y[::10]
    # paw_time=paw_time[::10]
    # stdx = rollingfun(x,30,'std')
    # stdy = rollingfun(y,30,'std')
    # std_paw = np.nanmean([stdx,stdy],0)
    # std_paw = std_paw/np.nanmax(std_paw)
    # ax_licknum.plot(paw_time,std_paw,color = 'brown')
    # =============================================================================
    
def plot_go_reward_heatmap_dff(session_key_wr):
    #%%
    subject_id = (lab.WaterRestriction()&'water_restriction_number = "{}"'.format(session_key_wr['wr_id'])).fetch1('subject_id')
    session_key = {'subject_id':subject_id,
                   'session':session_key_wr['session']}
    conditioned_rois = imaging.ROI()*imaging.ConditionedROI()&session_key
    
    
    fig = plt.figure()
    subplot_idx = 0
    for roi_cond in conditioned_rois:
        dff = (imaging.ROITrace()&roi_cond).fetch1('roi_dff')
        subplot_idx += 1
        ax_go_matrix = fig.add_subplot(len(conditioned_rois),2,subplot_idx)
        subplot_idx += 1
        ax_reward_matrix = fig.add_subplot(len(conditioned_rois),2,subplot_idx)
        
        ax_go_matrix.set_title('GO cue neuron {}, multiplier = {}'.format(roi_cond['roi_number'],roi_cond['cond_roi_multiplier']))
        ax_reward_matrix.set_title('Reward')
        
        frame_rate = (imaging.FOV()&session_key).fetch1('fov_frame_rate')
        
        time_back_go = 10 #s
        time_forward_go = 20 #s
        time_back_reward = 20 #s
        time_forward_reward = 10 #s
        
        step_back_go = int(time_back_go*frame_rate)
        step_forward_go = int(time_forward_go*frame_rate)
        
        step_back_reward = int(time_back_reward*frame_rate)
        step_forward_reward = int(time_forward_reward*frame_rate)
        
        
        #frame_times = (imaging.FOVFrameTimes()&session_key).fetch1('frame_times')
        
            #break # only first ROI now
        
        time_vector_go = np.arange(-1*step_back_go,step_forward_go,1)/frame_rate
        go_matrix = np.zeros([len(experiment.SessionTrial()&session_key),len(time_vector_go)])#*np.nan
        time_vector_reward = np.arange(-1*step_back_reward,step_forward_reward,1)/frame_rate
        reward_matrix = np.zeros([len(experiment.SessionTrial()&session_key),len(time_vector_reward)])#*np.nan
        
        start_trials, start_frames = (imaging.TrialStartFrame()&session_key).fetch('trial','frame_num')
        end_trials, end_frames = (imaging.TrialEndFrame()&session_key).fetch('trial','frame_num')
        go_trials, go_frames = (experiment.TrialEvent()*imaging.TrialEventFrame()&'trial_event_type = "go"'&session_key).fetch('trial','frame_num')
        reward_trials, reward_frames = (experiment.TrialEvent()*imaging.TrialEventFrame()&'trial_event_type = "reward"'&session_key).fetch('trial','frame_num')
        
        start_times = []
        end_times = []
        reward_times = []
        for trial,go_frame in zip(go_trials,go_frames):
            try:
                go_matrix[trial,:] = dff[go_frame-step_back_go:go_frame+step_forward_go]
            except:
                pass
            if trial in start_trials:
                start_times.append((start_frames[start_trials==trial][0]-go_frame)/frame_rate)
            if trial in end_trials:
                end_times.append((end_frames[end_trials==trial][0]-go_frame)/frame_rate)
            if trial in reward_trials:
                reward_times.append((reward_frames[reward_trials==trial][0]-go_frame)/frame_rate)
                
        ax_go_matrix.imshow(go_matrix, aspect='auto', extent=[-1*time_back_go,time_forward_go,go_matrix.shape[0],0],cmap = 'magma')
        ax_go_matrix.plot(start_times,start_trials,'w|',linewidth = 4)
        ax_go_matrix.plot(end_times,end_trials,'w|',linewidth = 4)
        ax_go_matrix.plot(reward_times,reward_trials,'ro')
        ax_go_matrix.set_xlim([time_vector_go[0],time_vector_go[-1]])
        ax_go_matrix.set_xlabel('time relative to go cue (s)')
        ax_go_matrix.set_ylabel('trial #')
        
        start_times = []
        start_trials_ = []
        end_times = []
        end_trials_ = []
        go_times = []
        go_trials_ = []
        for trial,reward_frame in zip(reward_trials,reward_frames):
            try:
                reward_matrix[trial,:] = dff[reward_frame-step_back_reward:reward_frame+step_forward_reward]
            except:
                
                pass
            if trial in start_trials:
                start_times.append((start_frames[start_trials==trial][0]-reward_frame)/frame_rate)
                start_trials_.append(np.where(start_trials==trial)[0][0])
            if trial in end_trials:
                end_times.append((end_frames[end_trials==trial][0]-reward_frame)/frame_rate)
                end_trials_.append(np.where(end_trials==trial)[0][0])
            if trial in go_trials:
                go_times.append((go_frames[go_trials==trial][0]-reward_frame)/frame_rate)
                go_trials_.append(np.where(go_trials==trial)[0][0])
                
        ax_reward_matrix.imshow(reward_matrix, aspect='auto', extent=[-1*time_back_reward,time_forward_reward,reward_matrix.shape[0],0],cmap = 'magma')
        ax_reward_matrix.plot(start_times,start_trials_,'w|',linewidth = 4)
        ax_reward_matrix.plot(end_times,end_trials_,'w|',linewidth = 4)
        ax_reward_matrix.plot(go_times,go_trials_,'go')
        ax_reward_matrix.set_xlim([time_vector_reward[0],time_vector_reward[-1]])
        ax_reward_matrix.set_xlabel('time relative to reward (s)')
        ax_reward_matrix.set_ylabel('trial #')   
#%%
def plot_session_dlc_heatmap(session_key_wr = {'wr_id':'BCI14', 'session':2},bodypart = 'tongue_tip',video_axis = 'y'):
    subject_id = (lab.WaterRestriction()&'water_restriction_number = "{}"'.format(session_key_wr['wr_id'])).fetch1('subject_id')
    session_key = {'subject_id':subject_id,
                   'session':session_key_wr['session']}
    
    dlc_key = {'camera_position':'side',
               'bodypart':bodypart}#jaw#tongue_tip
    #video_axis = 'y'
    
    
    fig = plt.figure()
    ax_go_matrix = fig.add_subplot(1,2,1)
    ax_reward_matrix = fig.add_subplot(1,2,2)
    
    ax_go_matrix.set_title('GO cue')
    ax_reward_matrix.set_title('Reward')
    
    frame_rates = (videography.BehaviorVideo()*videography.DLCTracking()&session_key&dlc_key).fetch('video_frame_rate')
    frame_rate = np.median(frame_rates)
        
    time_back_go = 5 #s
    time_forward_go = 20 #s
    time_back_reward = 20 #s
    time_forward_reward = 5 #s
    dlc_min_probability = .99
    
    step_back_go = int(time_back_go*frame_rate)
    step_forward_go = int(time_forward_go*frame_rate)
    
    step_back_reward = int(time_back_reward*frame_rate)
    step_forward_reward = int(time_forward_reward*frame_rate)
    
    dlc_trial_num,movement_vector,p = (videography.DLCTracking()&session_key&dlc_key).fetch('trial',video_axis,'p')
    dlc_baseline_value = np.percentile(np.concatenate(movement_vector)[np.concatenate(p)>=dlc_min_probability],5)
    dlc_edges = np.percentile(np.concatenate(movement_vector)[np.concatenate(p)>=dlc_min_probability],[1,99])
    
    time_vector_go = np.arange(-1*step_back_go,step_forward_go,1)/frame_rate
    go_matrix = np.zeros([len(experiment.SessionTrial()&session_key),len(time_vector_go)])+dlc_baseline_value#*np.nan
    time_vector_reward = np.arange(-1*step_back_reward,step_forward_reward,1)/frame_rate
    reward_matrix = np.zeros([len(experiment.SessionTrial()&session_key),len(time_vector_reward)])+dlc_baseline_value#*np.nan
    
    
    go_trials, go_frames = (experiment.TrialEvent()*videography.TrialEventFrame()&'trial_event_type = "go"'&session_key).fetch('trial','frame_num')
    reward_trials, reward_frames = (experiment.TrialEvent()*videography.TrialEventFrame()&'trial_event_type = "reward"'&session_key).fetch('trial','frame_num')
    
    
    reward_times = []
    for go_trial,go_frame in zip(go_trials,go_frames):
        trial_idx = np.argmax(go_trial == dlc_trial_num)
        vector_now =movement_vector[trial_idx].copy()
        p_now = p[trial_idx]
        vector_now[p_now<dlc_min_probability] = dlc_baseline_value
        
        if go_frame>=step_back_go:
            nan_beginning = 1
        else:
            nan_beginning = step_back_go-go_frame
        if len(vector_now)-go_frame>=step_forward_go:
            nan_end = 0
        else:
            nan_end = step_forward_go-(len(vector_now)-go_frame)
        go_matrix[go_trial,nan_beginning-1:-nan_end-1] = vector_now[np.max([go_frame+1 - step_back_go,0]):step_forward_go+go_frame]
        
        if go_trial in reward_trials:
            reward_times.append((reward_frames[reward_trials==go_trial][0]-go_frame)/frame_rate)
    
    go_times = []
    go_trials_ = list()
    for reward_trial,reward_frame in zip(reward_trials,reward_frames):
        trial_idx = np.argmax(reward_trial == dlc_trial_num)
        vector_now =movement_vector[trial_idx].copy()
        p_now = p[trial_idx]
        vector_now[p_now<dlc_min_probability] = dlc_baseline_value
        
        if reward_frame>=step_back_reward:
            nan_beginning = 1
        else:
            nan_beginning = step_back_reward-reward_frame
        if len(vector_now)-reward_frame>=step_forward_reward:
            nan_end = 0
        else:
            nan_end = step_forward_reward-(len(vector_now)-reward_frame)
        reward_matrix[reward_trial,nan_beginning-1:-nan_end-1] = vector_now[np.max([reward_frame+1 - step_back_reward,0]):step_forward_reward+reward_frame]
        
        if reward_trial in go_trials:
            go_times.append((go_frames[go_trials==reward_trial][0]-reward_frame)/frame_rate)
            go_trials_.append(reward_trial)
        #break
    #%
    im_go = ax_go_matrix.imshow(go_matrix, aspect='auto', extent=[-1*time_back_go,time_forward_go,go_matrix.shape[0],0],cmap = 'magma')#magma
    ax_go_matrix.plot(reward_times,reward_trials,'ro')
    ax_go_matrix.set_xlim([time_vector_go[0],time_vector_go[-1]])
    ax_go_matrix.set_xlabel('time relative to go cue (s)')
    ax_go_matrix.set_ylabel('trial #')
    im_go.set_clim(dlc_edges[0],dlc_edges[1])
    
    
    im_reward = ax_reward_matrix.imshow(reward_matrix, aspect='auto', extent=[-1*time_back_reward,time_forward_reward,reward_matrix.shape[0],0],cmap = 'magma')#magma
    ax_reward_matrix.plot(go_times,go_trials_,'go')
    ax_reward_matrix.set_xlim([time_vector_reward[0],time_vector_reward[-1]])
    ax_reward_matrix.set_xlabel('time relative to reward (s)')
    ax_reward_matrix.set_ylabel('trial #')
    im_reward.set_clim(dlc_edges[0],dlc_edges[1])
    
    
    
def plot_lick_triggered_dff_over_session(session_key_wr):
    #%%
    max_baseline_dff = 1
    dff_baseline = .5
    lick_amplitude_percentiles = [1,25,50,75,99]
    lick_time_percentiles = np.arange(0,101,10)#[0,25,50,75,100]
    #session_key_wr = {'wr_id':'BCI14', 'session':2}
    
    
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax_overlay = fig.add_subplot(2,2,2)
    ax_high_transients = fig.add_subplot(2,2,3)
    ax_overlay.set_title('average dF/F by lick distance')
    ax_overlay.set_xlabel('Time from lick onset (s)')
    ax_overlay.set_ylabel('dF/F')
    ax_high_transients.set_ylabel('Q4 lick#')
    ax_high_transients.set_xlabel('Time from lick onset (s)')
    ax_high_transients_overlay = fig.add_subplot(2,2,4)
    ax_high_transients_overlay.set_xlabel('Time from lick onset (s)')
    ax_high_transients_overlay.set_ylabel('dF/F')
    
    subject_id = (lab.WaterRestriction()&'water_restriction_number = "{}"'.format(session_key_wr['wr_id'])).fetch1('subject_id')
    session_key = {'subject_id':subject_id,
                   'session':session_key_wr['session']}
    conditioned_rois = imaging.ROI()*imaging.ConditionedROI()&session_key
    frame_times = (imaging.FOVFrameTimes()&session_key).fetch1('frame_times')
    frame_rate = (imaging.FOV()&session_key).fetch1('fov_frame_rate')
    for roi_cond in conditioned_rois:
        dff = (imaging.ROITrace()&roi_cond).fetch1('roi_dff')
        #break # only first ROI now
    time_back = 1 #s
    time_forward = 3 #s
    step_back = int(time_back*frame_rate)
    step_forward = int(time_forward*frame_rate)
    baseline_step = int(dff_baseline*frame_rate)
    time_vector= np.arange(-1*step_back,step_forward,1)/frame_rate
    lick_trial_times,lick_times,lick_amplitudes = (experiment.SessionTrial()*videography.DLCLickBout()&session_key).fetch('trial_start_time','lick_bout_start_time','lick_bout_amplitude')
    lick_matrix = np.zeros([len(lick_amplitudes),len(time_vector)])
    lick_time_list = []
    for i,(lick_trial_time,lick_time,lick_amplitude) in enumerate(zip(lick_trial_times,lick_times,lick_amplitudes)):
        lick_time = float(lick_trial_time)+lick_time
        lick_idx = np.argmin(np.abs(frame_times-lick_time))
        lick_matrix[i,:] = dff[lick_idx-step_back:lick_idx+step_forward]
        lick_time_list.append(lick_time)
        #break
    lick_time_list = np.asarray(lick_time_list)
    baseline_vals = np.mean(lick_matrix[:,step_back-baseline_step:step_back],1)
    lick_matrix_now = lick_matrix[baseline_vals<=max_baseline_dff,:]
    #ax1.imshow(lick_matrix_now, aspect='auto', extent=[-1*time_back,time_forward,lick_matrix_now.shape[0],0],cmap = 'magma')
    ax1.hist(lick_amplitudes[(lick_amplitudes>=np.percentile(lick_amplitudes,1))&(lick_amplitudes<=np.percentile(lick_amplitudes,99))],100,color = 'black')
    ax1.set_ylabel('# of licks')
    ax1.set_xlabel('Lick distance (pixels)')
    ax1.vlines(np.percentile(lick_amplitudes,lick_amplitude_percentiles),ax1.get_ylim()[0],ax1.get_ylim()[1],color = 'red',linewidth = 4)
    for i,(pctl_low,pctl_high) in enumerate(zip(lick_amplitude_percentiles[:-1],lick_amplitude_percentiles[1:])):
        needed = (baseline_vals<=max_baseline_dff) & (lick_amplitudes<=np.percentile(lick_amplitudes,pctl_high)) & (lick_amplitudes>=np.percentile(lick_amplitudes,pctl_low))
        lick_matrix_now = lick_matrix[needed,:]
        #ax_now =fig.add_subplot(3,3,i+2)
        #ax_now.imshow(lick_matrix_now, aspect='auto', extent=[-1*time_back,time_forward,lick_matrix_now.shape[0],0],cmap = 'magma')
        ax_overlay.plot(time_vector,np.mean(lick_matrix_now,0),label = '{}% - {}%'.format(pctl_low,pctl_high),linewidth = 4)
    ax_overlay.legend()
    
    #ax_high_transients.imshow(lick_matrix_now, aspect='auto', extent=[-1*time_back,time_forward,lick_matrix_now.shape[0],0],cmap = 'magma')
    ax_high_transients.hist(lick_time_list[needed],50,color= 'black')
    ax_high_transients.vlines(np.percentile(lick_time_list[needed],lick_time_percentiles),ax_high_transients.get_ylim()[0],ax_high_transients.get_ylim()[1],color = 'red',linewidth = 4)
    lick_times_now = lick_time_list[needed]
    baseline_vals_now = baseline_vals[needed]
    for i,(pctl_low,pctl_high) in enumerate(zip(lick_time_percentiles[:-1],lick_time_percentiles[1:])):
        needed_ = (baseline_vals_now<=max_baseline_dff) & (lick_times_now<=np.percentile(lick_times_now,pctl_high)) & (lick_times_now>=np.percentile(lick_times_now,pctl_low))
        lick_matrix_now_ = lick_matrix_now[needed_,:]
        #ax_now =fig.add_subplot(3,3,i+2)
        #ax_now.imshow(lick_matrix_now, aspect='auto', extent=[-1*time_back,time_forward,lick_matrix_now.shape[0],0],cmap = 'magma')
        ax_high_transients_overlay.plot(time_vector,np.mean(lick_matrix_now_,0),label = '{}% - {}%'.format(pctl_low,pctl_high),linewidth = 4)
        #break
    ax_high_transients_overlay.legend()