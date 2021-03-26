from utils import utils_pybpod, utils_imaging
from scipy.io import savemat
from pathlib import Path
import os
from statistics import mode
import numpy as np
import time as timer
import datetime
import datajoint as dj
from pybpodgui_api.models.project import Project

def find_pybpod_sessions(subject_names_list,date_now,projects):
    #%
    if type(subject_names_list) != list:
        subject_names_list = [subject_names_list]
    sessions_now = list()
    session_start_times_now = list()
    experimentnames_now = list()
    for proj in projects: #
        exps = proj.experiments
        for exp in exps:
            stps = exp.setups
            for stp in stps:
                #sessions = stp.sessions
                for session in stp.sessions:
                    for subject_now in subject_names_list:
                        if session.subjects and session.subjects[0].find(subject_now) >-1 and session.name.startswith(date_now):
                            sessions_now.append(session)
                            session_start_times_now.append(session.started)
                            experimentnames_now.append(exp.name)
    order = np.argsort(session_start_times_now)
    outdict = {'sessions':np.asarray(sessions_now)[order],
               'session_start_times':np.asarray(session_start_times_now)[order],
               'experiment_names':np.asarray(experimentnames_now)[order]}
    #%
    return outdict



def extract_files_from_dir(basedir):
    #%
    files = os.listdir(basedir)
    exts = list()
    basenames = list()
    fileindexs = list()
    dirs = list()
    for file in files:
        if Path(os.path.join(basedir,file)).is_dir():
            exts.append('')
            basenames.append('')
            fileindexs.append(np.nan)
            dirs.append(file)
        else:
            if '_' in file:# and ('cell' in file.lower() or 'stim' in file.lower()):
                basenames.append(file[:-1*file[::-1].find('_')-1])
                try:
                    fileindexs.append(int(file[-1*file[::-1].find('_'):file.find('.')]))
                except:
                    print('weird file index: {}'.format(file))
                    fileindexs.append(-1)
            else:
                basenames.append(file[:file.find('.')])
                fileindexs.append(-1)
            exts.append(file[file.find('.'):])
    tokeep = np.asarray(exts) != ''
    files = np.asarray(files)[tokeep]
    exts = np.asarray(exts)[tokeep]
    basenames = np.asarray(basenames)[tokeep]
    fileindexs = np.asarray(fileindexs)[tokeep]
    out = {'dir':basedir,
           'filenames':files,
           'exts':exts,
           'basenames':basenames,
           'fileindices':fileindexs,
           'dirs':dirs
           }
    #%
    return out

#%% this script will export behavior and pair it to imaging, then save it in a neat directory structure
def export_pybpod_files(overwrite=False,behavior_export_basedir = '/home/rozmar/Data/Behavior/BCI_exported'):
#overwrite = False
    calcium_imaging_raw_basedir = dj.config['locations.imagingdata_raw']
    raw_behavior_dirs =  dj.config['locations.behavior_dirs_raw'] 
    
    projects = list()
    for projectdir in raw_behavior_dirs:
        projects.append(Project())
        projects[-1].load(projectdir)
    setups = os.listdir(calcium_imaging_raw_basedir)
    for setup in setups[::-1]:
        calcium_imaging_raw_setup_dir = os.path.join(calcium_imaging_raw_basedir,setup)
        subjects = np.sort(os.listdir(calcium_imaging_raw_setup_dir))
        for subject in subjects:
            if 'bci' in subject.lower():
                calcium_imaging_raw_subject_dir = os.path.join(calcium_imaging_raw_setup_dir,subject)
                subject_wr_name = 'BCI{}'.format(subject[subject.find('_')+1:])
                sessions = np.sort(os.listdir(calcium_imaging_raw_subject_dir))
                for session in sessions:
                    try:
                        session_date = datetime.datetime.strptime(session,'%m%d%y')
                    except:
                        try:
                            session_date = datetime.datetime.strptime(session,'%Y-%m-%d')
                        except:
                            print('cannot understand date for session dir: {}'.format(session))
                            continue
                        
                    bpod_export_dir = os.path.join(behavior_export_basedir,setup,subject)
                    Path(bpod_export_dir).mkdir(parents=True, exist_ok=True)
                    bpod_export_file = '{}-bpod_zaber.npy'.format(session)
                    if not overwrite and os.path.exists(os.path.join(bpod_export_dir,bpod_export_file)):
                        print('session already exported: {}'.format(os.path.join(bpod_export_dir,bpod_export_file)))
                        continue
                    bpod_session_dict = find_pybpod_sessions([subject_wr_name,subject],session_date.strftime('%Y%m%d'),projects)
                    #%
                    behavior_dict_list = list()
                    sessionfile_start_times = list()
                    for sessionfile in bpod_session_dict['sessions']:
                        csvfile = sessionfile.filepath
                        csv_data = utils_pybpod.load_and_parse_a_csv_file(csvfile)
                        behavior_dict = utils_pybpod.minethedata(csv_data,extract_variables = True)
                        subject_name = csv_data['subject'][0]
                        setup_name = csv_data['setup'][0]
                        zaber_dict = utils_pybpod.generate_zaber_info_for_pybpod_dict(behavior_dict,subject_name,setup_name,zaber_folder_root = '/home/rozmar/Data/Behavior/BCI_Zaber_data')
                        zaber_step_times = list()
                        for v,a,s in zip(zaber_dict['speed'],zaber_dict['acceleration'],zaber_dict['trigger_step_size']):
                            zaber_step_times.append(utils_pybpod.calculate_step_time(s/1000,v,a))
                        zaber_dict['trigger_step_time'] =np.asarray(zaber_step_times)
                        for key in zaber_dict.keys():
                            behavior_dict['zaber_{}'.format(key)] = zaber_dict[key]
                        trialnum = len(behavior_dict['trial_num'])
                        behavior_dict['bpod_file_names'] = np.asarray([sessionfile.filepath.split('/')[-1]]*trialnum)
                        if trialnum>0:
                            behavior_dict_list.append(behavior_dict)
                            sessionfile_start_times.append(behavior_dict['trial_start_times'][0])
                        #%
                    if  len(behavior_dict_list)>1:
                        #%
                        order = np.argsort(sessionfile_start_times)
                        behavior_dict_list = np.asarray(behavior_dict_list)[order]
                        behavior_dict = {}
                        for key in behavior_dict_list[0].keys():
                            keylist = list()
                            for behavior_dict_now in behavior_dict_list:
                                for element in behavior_dict_now[key]:
                                    keylist.append(element)
                            behavior_dict[key] = np.asarray(keylist)
                        #%
    # =============================================================================
    #                     print('multiple bpod files, handle me')
    #                     timer.sleep(10000)
    # =============================================================================
                    #%
                    calcium_imaging_raw_session_dir = os.path.join(calcium_imaging_raw_subject_dir,session)
                    files = extract_files_from_dir(calcium_imaging_raw_session_dir)
                    tiffiles = files['exts']=='.tif'
                    uniquebasenames = np.unique(files['basenames'][tiffiles])
                    filenames_all = list()
                    frame_timestamps_all = list()
                    nextfile_timestamps_all = list()
                    acqtrigger_timestamps_all = list()
                    trigger_arrived_timestamps_all = list()
                    for basename in uniquebasenames:
                        file_idxs_now = (files['exts']=='.tif') & (files['basenames']==basename)
                        filenames = files['filenames'][file_idxs_now]
                        fileindices = files['fileindices'][file_idxs_now]
                        order = np.argsort(fileindices)
                        filenames = filenames[order]
                        for filename in filenames:
                            try:
                                metadata = utils_imaging.extract_scanimage_metadata(os.path.join(calcium_imaging_raw_session_dir,filename))
                            except:
                                print('tiff file read error: {}'.format(os.path.join(calcium_imaging_raw_session_dir,filename)))
                                continue
                                
                            #%
                            movie_start_time = metadata['movie_start_time']
                            
                            if float(metadata['description_first_frame']['frameTimestamps_sec'])<0:
                                valence=-1
                            else:
                                valence = 1
                            frame_timestamp = movie_start_time+valence*datetime.timedelta(seconds = np.abs(float(metadata['description_first_frame']['frameTimestamps_sec'])))
                            
                            if float(metadata['description_first_frame']['acqTriggerTimestamps_sec'])<0:
                                valence=-1
                            else:
                                valence = 1
                            acqtrigger_timestamp = movie_start_time+valence*datetime.timedelta(seconds = np.abs(float(metadata['description_first_frame']['acqTriggerTimestamps_sec'])))
                            if float(metadata['description_first_frame']['nextFileMarkerTimestamps_sec'])<0:
                                valence=-1
                            else:
                                valence = 1
                            nextfile_timestamp = movie_start_time+valence*datetime.timedelta(seconds = np.abs(float(metadata['description_first_frame']['nextFileMarkerTimestamps_sec'])))
                            if float(metadata['description_first_frame']['nextFileMarkerTimestamps_sec']) == -1:
                                trigger_arrived_timestamp =acqtrigger_timestamp
                            else:
                                trigger_arrived_timestamp = nextfile_timestamp
                                
                            filenames_all.append(filename)
                            frame_timestamps_all.append(frame_timestamp)
                            nextfile_timestamps_all.append(nextfile_timestamp)
                            acqtrigger_timestamps_all.append(acqtrigger_timestamp)
                            
                            trignextstopenable = 'true' in metadata['metadata']['hScan2D']['trigNextStopEnable'].lower()
                            if metadata['metadata']['extTrigEnable'] == '0':
                                trigger_arrived_timestamps_all.append(np.nan)
                                print('not triggered')
                            else:
                                trigger_arrived_timestamps_all.append(trigger_arrived_timestamp)
                                #print('triggered')
                            
                    #%
                    filenames_all= np.asarray(filenames_all)
                    frame_timestamps_all= np.asarray(frame_timestamps_all)
                    nextfile_timestamps_all= np.asarray(nextfile_timestamps_all)
                    acqtrigger_timestamps_all= np.asarray(acqtrigger_timestamps_all)
                    trigger_arrived_timestamps_all = np.asarray(trigger_arrived_timestamps_all)
                    file_order = np.argsort(frame_timestamps_all)
                    
                    filenames_all = filenames_all[file_order]
                    frame_timestamps_all = frame_timestamps_all[file_order]
                    nextfile_timestamps_all = nextfile_timestamps_all[file_order]
                    acqtrigger_timestamps_all = acqtrigger_timestamps_all[file_order]
                    trigger_arrived_timestamps_all = trigger_arrived_timestamps_all[file_order]
                    
                    istriggered = list()
                    for stamp in trigger_arrived_timestamps_all: istriggered.append(type(stamp)==datetime.datetime)
                    
                    
                    bpod_trial_start_times = behavior_dict['trial_start_times']
                    #%
                    dist_list = list()
                    for i,t_now in enumerate(bpod_trial_start_times):
                        for t_next in trigger_arrived_timestamps_all[istriggered]:
                            dt = (t_next-t_now).total_seconds()
                            if dt<600:
                                dist_list.append(dt)
                    dist_list = np.asarray(dist_list)
                    center_sec = mode(np.asarray(dist_list,int))
                    dist_list = dist_list[(dist_list>center_sec-1) &  (dist_list<center_sec+1)]
                    time_offset = np.median(dist_list)
                    print('time_offset: {} s'.format(time_offset))
                    #%
                    bpod_trial_file_names = list()
                    for trial_start_time,trial_end_time in zip(behavior_dict['trial_start_times'],behavior_dict['trial_end_times']):
                        trial_start_time = trial_start_time +datetime.timedelta(seconds = time_offset-.5) #gets a 0.5 second extra
                        trial_end_time = trial_end_time +datetime.timedelta(seconds = time_offset)
                        movie_idxes = (trigger_arrived_timestamps_all[istriggered]>trial_start_time) & (trigger_arrived_timestamps_all[istriggered]<trial_end_time)
                        if any(movie_idxes):
                            if sum(movie_idxes) == 1:
                                moviename = np.asarray(filenames_all[istriggered][movie_idxes])
                            else:
                                moviename = np.asarray(filenames_all[istriggered][movie_idxes])
                        else:
                            moviename = 'no movie for this trial'
                        bpod_trial_file_names.append(moviename)
                        #%
                    behavior_dict['scanimage_file_names'] = bpod_trial_file_names
                    bpod_export_dir = os.path.join(behavior_export_basedir,setup,subject)
                    Path(bpod_export_dir).mkdir(parents=True, exist_ok=True)
                    bpod_export_file = '{}-bpod_zaber.npy'.format(session)
                    np.save(os.path.join(bpod_export_dir,bpod_export_file),behavior_dict)
                    bpod_export_file = '{}-bpod_zaber.mat'.format(session)
                    behavior_dict['trial_start_times'] = np.asarray(behavior_dict['trial_start_times'],str)
                    behavior_dict['trial_end_times'] = np.asarray(behavior_dict['trial_end_times'],str)
                    savemat(os.path.join(bpod_export_dir,bpod_export_file),behavior_dict)
                    print('{}/{} saved'.format(subject,session))

 