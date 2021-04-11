import datajoint as dj
from pipeline import lab,experiment
import datetime
import numpy as np
import time as timer
import os
from scipy.io import loadmat
#utils_pipeline.export_pybpod_files(overwrite = True)
#%% load available data and location
def list_exported_bpod_data(behavior_export_basedir = '/home/rozmar/Data/Behavior/BCI_exported'):
    behavior_file_ending = '-bpod_zaber.npy'
    
    session_dict = {}
    
    setups = os.listdir(behavior_export_basedir)
    for setup in setups:
        if '.' in setup:
            continue
        setup_dir = os.path.join(behavior_export_basedir,setup)
        subjects = np.sort(os.listdir(setup_dir))
        for subject in subjects:
            if '.' in subject:
                continue
            subject_dir = os.path.join(setup_dir,subject)
            sessions = np.sort(os.listdir(subject_dir))
            for file_name in sessions:
                if behavior_file_ending not in file_name:
                    continue
                session = file_name[:-len(behavior_file_ending)]
                try:
                    session_date = datetime.datetime.strptime(session,'%m%d%y')
                except:
                    try:
                        session_date = datetime.datetime.strptime(session,'%Y-%m-%d')
                    except:
                        print('cannot understand date for session dir: {}'.format(session))
                        continue
                if subject not in session_dict.keys():
                    session_dict[subject]={'setup':list(),
                                           'date':list(),
                                           'filename':list(),
                                           'dir_name':list()}
                session_dict[subject]['setup'].append(setup)
                session_dict[subject]['date'].append(session_date)
                session_dict[subject]['filename'].append(file_name)
                session_dict[subject]['dir_name'].append(session)
    for subject in session_dict.keys():
        order = np.argsort(session_dict[subject]['date'])
        for key in session_dict[subject].keys():
            session_dict[subject][key] = np.asarray(session_dict[subject][key])[order]
    return session_dict


#%%
def populate_behavior():
    #%%
    behavior_export_basedir = '/home/rozmar/Data/Behavior/BCI_exported'
    session_dict = list_exported_bpod_data(behavior_export_basedir = '/home/rozmar/Data/Behavior/BCI_exported')
    for subject in session_dict.keys():
        subject_id = (lab.WaterRestriction()&'water_restriction_number = "{}"'.format(subject.replace('_',''))).fetch1('subject_id')
        for session_num,(setup,session_date,filename,dir_name) in enumerate(zip(session_dict[subject]['setup'], session_dict[subject]['date'], session_dict[subject]['filename'], session_dict[subject]['dir_name'])):
            print(dir_name)
            behavfile = os.path.join(behavior_export_basedir,setup,subject,filename)
            raw_imaging_dir = os.path.join(dj.config['locations.imagingdata_raw'],setup,subject,dir_name)
            raw_imaging_files = os.listdir(raw_imaging_dir)
            bpoddata = np.load(behavfile,allow_pickle = True).tolist()
            try:
                user = bpoddata['experimenter_name'][0]
            except:
                print('no experimenter info found, assuming daiek')
                user = 'daiek'
            
            
            if setup.lower() == 'kayvonscope':
                rig = 'Bergamo-2p-Resonant'
            elif  'dom3' in setup.lower():
                rig = 'DOM3-2p-Resonant-MMIMS'
            else:
                print('unknown rig: {}'.format(rig))
            session_key = {'subject_id':subject_id,
                           'session':session_num,
                           'session_date':session_date.date(),
                           'session_time':bpoddata['trial_start_times'][0].time(),
                           'username':user,
                           'rig':rig}
            if len(experiment.Session()&'subject_id = {}'.format(subject_id)&'session_date = "{}"'.format(session_date.date()))>0:
                print('already uploaded')
                continue
            trial_key_list = list()
            behaviortrial_key_list = list()
            bci_key_list = list()
            lickport_key_list = list()
            lickport_duration_key_list = list()
            trial_event_list = list()
            action_event_list = list()
            trial_choice_list = list()
            for trial_num,trial_num_bpod in enumerate(bpoddata['trial_num']):
                trial_event_id = 0
                action_event_id = 0
                trial_key = {'subject_id':subject_id,
                             'session':session_num,
                             'trial':trial_num,
                             'trial_start_time':(bpoddata['trial_start_times'][trial_num]-bpoddata['trial_start_times'][0]).total_seconds(),
                             'trial_end_time': (bpoddata['trial_end_times'][trial_num]-bpoddata['trial_start_times'][0]).total_seconds()}
                direction = float('{}1'.format(bpoddata['zaber_direction'][trial_num]))
                lickport_key = {'subject_id':subject_id,
                                'session':session_num,
                                'trial':trial_num,
                                'lickport_limit_far':(bpoddata['zaber_limit_far'][trial_num]-bpoddata['zaber_reward_zone'][trial_num])*direction,
                                'lickport_limit_close':(bpoddata['zaber_limit_close'][trial_num]-bpoddata['zaber_reward_zone'][trial_num])*direction,
                                'lickport_threshold':0,
                                'lickport_step_size':bpoddata['zaber_trigger_step_size'][trial_num]/1000,
                                'lickport_step_time':bpoddata['zaber_trigger_step_time'][trial_num],
                                'lickport_auto_step_freq':bpoddata['var_BaselineZaberForwardStepFrequency'][trial_num],
                                'lickport_maximum_speed':bpoddata['zaber_max_speed'][trial_num]}
                if bpoddata['var_BaselineZaberForwardStepFrequency'][trial_num]>0:
                    task = 'BCI OL'
                    task_protocol = 0
                else:
                    task = 'BCI CL'
                    task_protocol = 10
                if not type(bpoddata['scanimage_file_names'][trial_num]) == str or '.tif' in bpoddata['scanimage_file_names'][trial_num]:
                    if not type(bpoddata['scanimage_file_names'][trial_num]) == str:
                        tiffname = bpoddata['scanimage_file_names'][trial_num][0]
                    else:
                        tiffname = bpoddata['scanimage_file_names'][trial_num]
                    try:
                        fileidx = int(tiffname[tiffname.rfind('_')+1:tiffname.rfind('.')])
                        thresholdfilename = '{}_threshold_{}.mat'.format(tiffname[:tiffname.rfind('_')],fileidx)
                    except:
                        print('weird tiff file name: {}'.format(tiffname))
                        if ').' in tiffname:
                            fileidx = int(tiffname[tiffname.rfind('_')+1:tiffname.rfind(' ')])
                            addendum = tiffname[tiffname.rfind(' '):tiffname.rfind('.')]
                            thresholdfilename = '{}_threshold_{}{}.mat'.format(tiffname[:tiffname.rfind('_')],fileidx,addendum)
                    if thresholdfilename not in raw_imaging_files: #if there is no movie file, I assume open loop configuration
                        task = 'BCI OL'
                        task_protocol = 0
                    else:
                        task = 'BCI CL'
                        task_protocol = 10
                        
                        threshold_mat = loadmat(os.path.join(raw_imaging_dir,thresholdfilename))
                        bci_key = {'subject_id':subject_id,
                                   'session':session_num,
                                   'trial':trial_num,
                                   'bci_conditioned_neuron_idx':threshold_mat['selected_rois'][0],
                                   'bci_threshold_low':threshold_mat['BCI_threshold'][0][0],
                                   'bci_threshold_high':threshold_mat['BCI_threshold'][0][1],
                                   'bci_minimum_voltage_out':0,
                                   'bci_movement_punishment_t':0,
                                   'bci_movement_punishment_pix':0,}
                        if len(threshold_mat.keys())>5: # marton added extra fields - this is not a hard criterion, might introduce bugs later
                            if 'enable_motion_punishment' not in threshold_mat.keys():
                                threshold_mat['enable_motion_punishment'] = [[1]]
                                print('enable motion punishment added manually')
                            if 'movement_punishment_time' not in threshold_mat.keys():
                                threshold_mat['movement_punishment_time'] = [[1]]
                                print('motion punishment time added manually')
                            if threshold_mat['enable_motion_punishment'][0][0] ==1:
                                bci_key['bci_movement_punishment_t'] = threshold_mat['movement_punishment_time'][0][0]
                                bci_key['bci_movement_punishment_pix'] = threshold_mat['max_pixel_movement'][0][0]
                            bci_key['bci_minimum_voltage_out'] = threshold_mat['minimum_voltage_out'][0][0]
                            
                        if bci_key['bci_minimum_voltage_out']>0 and bci_key['bci_threshold_low']>5000: #this is a way for open loop configuration
                            task = 'BCI OL'
                            task_protocol = 0
                        lickport_key['lickport_auto_step_freq'] += (bci_key['bci_minimum_voltage_out']/3.3)*bpoddata['zaber_max_speed'][trial_num]
                        bci_key_list.append(bci_key)  #we add this only if there is a scanimage file
                
                if len(bpoddata['reward_L'][trial_num])>0:
                    lickportchoice_key = {'subject_id':subject_id,
                                          'session':session_num,
                                          'trial':trial_num,
                                          'lick_port':'left'}
                    trial_choice_list.append(lickportchoice_key)
                elif len(bpoddata['reward_R'][trial_num])>0:
                    lickportchoice_key = {'subject_id':subject_id,
                                      'session':session_num,
                                      'trial':trial_num,
                                      'lick_port':'right'}
                    trial_choice_list.append(lickportchoice_key)
                    
                
                if bpoddata['trial_hit'][trial_num]:
                    outcome = 'hit'
                elif len(bpoddata['threshold_crossing_times'][trial_num]) == 0:
                    outcome = 'miss'
                else:
                    outcome = 'ignore'
                
                if len(bpoddata['autowater_R'][trial_num])>0 or len(bpoddata['autowater_L'][trial_num])>0:
                    autowater = True
                else:
                    autowater = False
                
                
                
                behaviortrial_key = {'subject_id':subject_id,
                                     'session':session_num,
                                     'trial':trial_num,
                                     'task':task,
                                     'task_protocol':task_protocol,
                                     'trial_instruction':'none',#to be defined in later experiments
                                     'outcome':outcome,
                                     'auto_water':autowater}
                
                
                # add go cue, threshold crossing, lickport steps, and licks
                trial_event_id+=1
                trial_event_key = {'subject_id':subject_id,
                                   'session':session_num,
                                   'trial':trial_num,
                                   'trial_event_id':trial_event_id,
                                   'trial_event_type':'go',
                                   'trial_event_time':bpoddata['go_cue_times'][trial_num][0],
                                   'trial_event_duration':0}
                trial_event_list.append(trial_event_key)
                if len(bpoddata['threshold_crossing_times'][trial_num])>0:
                    trial_event_id+=1
                    trial_event_key = {'subject_id':subject_id,
                                       'session':session_num,
                                       'trial':trial_num,
                                       'trial_event_id':trial_event_id,
                                       'trial_event_type':'threshold cross',
                                       'trial_event_time':bpoddata['threshold_crossing_times'][trial_num][0],
                                       'trial_event_duration':0}
                    trial_event_list.append(trial_event_key)
                
                
                for step_t_now in bpoddata['zaber_move_forward'][trial_num]:
                    trial_event_id+=1
                    trial_event_key = {'subject_id':subject_id,
                                       'session':session_num,
                                       'trial':trial_num,
                                       'trial_event_id':trial_event_id,
                                       'trial_event_type':'lickport step',
                                       'trial_event_time':step_t_now,
                                       'trial_event_duration':bpoddata['zaber_trigger_step_time'][trial_num]}
                    trial_event_list.append(trial_event_key)
                for reward_time in np.concatenate([bpoddata['reward_R'][trial_num],bpoddata['reward_L'][trial_num]]):
                    trial_event_id+=1
                    trial_event_key = {'subject_id':subject_id,
                                       'session':session_num,
                                       'trial':trial_num,
                                       'trial_event_id':trial_event_id,
                                       'trial_event_type':'reward',
                                       'trial_event_time':reward_time,
                                       'trial_event_duration':0}
                    trial_event_list.append(trial_event_key)
                
                for action_event_time,action_event_type in zip(np.concatenate([bpoddata['lick_L'][trial_num],bpoddata['lick_R'][trial_num]]),np.concatenate([len(bpoddata['lick_L'][trial_num])*['left lick'],len(bpoddata['lick_R'][trial_num])*['right lick']])):
                    action_event_id += 1
                    action_event_key = {'subject_id':subject_id,
                                       'session':session_num,
                                       'trial':trial_num,
                                       'action_event_id':action_event_id,
                                       'action_event_type':action_event_type,
                                       'action_event_time':action_event_time}
                    action_event_list.append(action_event_key)
                
                lickport_openduration_key = {'subject_id':subject_id,
                                             'session':session_num,
                                             'trial':trial_num,           
                                             'lick_port':'left',
                                             'open_duration':bpoddata['var_ValveOpenTime_L'][trial_num]}
                lickport_duration_key_list.append(lickport_openduration_key)
                lickport_openduration_key = {'subject_id':subject_id,
                                             'session':session_num,
                                             'trial':trial_num,           
                                             'lick_port':'right',
                                             'open_duration':bpoddata['var_ValveOpenTime_R'][trial_num]}
                lickport_duration_key_list.append(lickport_openduration_key)
                trial_key_list.append(trial_key)
                behaviortrial_key_list.append(behaviortrial_key)
                lickport_key_list.append(lickport_key)
                #break
                
            with dj.conn().transaction: #inserting one movie
                print('uploading session {}'.format(session_key)) #movie['movie_name']
    # =============================================================================
    #             try:
    # =============================================================================
                experiment.Session().insert1(session_key,allow_direct_insert=True)
                experiment.SessionTrial().insert(trial_key_list,allow_direct_insert=True)
                experiment.BehaviorTrial().insert(behaviortrial_key_list,allow_direct_insert=True)
                
                experiment.LickPortSetting().insert(lickport_key_list,allow_direct_insert=True)
                experiment.LickPortSetting.OpenDuration().insert(lickport_duration_key_list,allow_direct_insert=True)
                experiment.BCISettings().insert(bci_key_list,allow_direct_insert=True)
                experiment.ActionEvent().insert(action_event_list,allow_direct_insert=True)
                experiment.TrialEvent().insert(trial_event_list,allow_direct_insert=True)
                experiment.TrialLickportChoice().insert(trial_choice_list,allow_direct_insert=True)
                
                dj.conn().ping()
    # =============================================================================
    #             except dj.errors.DuplicateError:
    #                 print('already uploaded')
    # =============================================================================
                
    # =============================================================================
    #         break
    #     break
    # =============================================================================
        