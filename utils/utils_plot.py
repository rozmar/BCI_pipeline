from utils import utils_pybpod, utils_ephys, utils_imaging, utils_plot
import numpy as np
import datetime
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import datajoint as dj
from pipeline import pipeline_tools,lab,experiment
from pipeline.ingest import datapipeline_metadata
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
    time_to_hit = np.zeros(len(gocue_trial))*np.nan
    for trial,threshold_t in zip(thresholdcross_trial,threshold_cross):
        idx = gocue_trial==trial
        time_to_hit[idx] = threshold_t-go_cue[idx]
    
    trial_num,outcome,lickport_auto_step_freq,task_protocol,lickport_step_size,trial_start_time,trial_end_time,lickport_maximum_speed,lickport_limit_far = (experiment.SessionTrial()*experiment.LickPortSetting()*experiment.BehaviorTrial()&session_key).fetch('trial','outcome','lickport_auto_step_freq','task_protocol','lickport_step_size','trial_start_time','trial_end_time','lickport_maximum_speed','lickport_limit_far')
    lickport_step_size = np.asarray(lickport_step_size,float)
    trial_lengths = np.asarray(trial_end_time-trial_start_time,float)
    trial_num_bci,threshold_low,threshold_high = (experiment.BCISettings()&session_key).fetch('trial','bci_threshold_low','bci_threshold_high')
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
    max_speed_moving = np.zeros(len(outcome))*np.nan
    trial_average_speed_moving = np.zeros(len(outcome))*np.nan
    task_change_idx = np.where(np.abs(np.diff(task_protocol))>0)[0]
    task_change_idx = np.concatenate([[0],task_change_idx,[len(task_protocol)]])
    for idx in np.arange(len(task_change_idx)-1):
        hit_rate[task_change_idx[idx]:task_change_idx[idx+1]] = utils_plot.rollingfun(hits[task_change_idx[idx]:task_change_idx[idx+1]],moving_window)
        rewardsperminute[task_change_idx[idx]:task_change_idx[idx+1]] = utils_plot.rollingfun((hits/trial_lengths*60)[task_change_idx[idx]:task_change_idx[idx+1]],moving_window)
        trial_length_moving[task_change_idx[idx]:task_change_idx[idx+1]] = utils_plot.rollingfun(trial_lengths[task_change_idx[idx]:task_change_idx[idx+1]],moving_window)
        timetohit_moving[task_change_idx[idx]:task_change_idx[idx+1]] = utils_plot.rollingfun(time_to_hit[task_change_idx[idx]:task_change_idx[idx+1]],moving_window)
        trial_average_speed_moving[task_change_idx[idx]:task_change_idx[idx+1]] = utils_plot.rollingfun(trial_average_speed[task_change_idx[idx]:task_change_idx[idx+1]],moving_window)
        max_speed_moving[task_change_idx[idx]:task_change_idx[idx+1]] = utils_plot.rollingfun(trial_max_speed[task_change_idx[idx]:task_change_idx[idx+1]],moving_window)
    
    
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
    
    ax_valve = fig_behavior.add_subplot(414,sharex = ax_movement_speed)
    ax_threshold = ax_valve.twinx()
    ax_threshold.set_ylabel('BCI thresholds')
    ax_timing.semilogy(trial_num,trial_lengths,'rx')
    ax_timing.semilogy(trial_num,trial_length_moving,'r-')
    ax_timing.semilogy(trial_num,time_to_hit,'g.')
    ax_timing.semilogy(trial_num,timetohit_moving,'g-')
    
    
    ax_threshold.plot(trial_num_bci,threshold_low,'r-.')
    ax_threshold.plot(trial_num_bci,threshold_high,'g-.')
    
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