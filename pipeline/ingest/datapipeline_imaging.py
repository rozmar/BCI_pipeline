import os 

from pathlib import Path

import numpy as np

import shutil

import datetime

from utils import utils_imaging,utils_pipeline, utils_plot
import datajoint as dj
from pipeline import pipeline_tools,lab,experiment,imaging
import re

import json
import tifffile
from skimage.measure import label
#
#import matplotlib.pyplot as plt
#%matplotlib qt

#%%
def populate_session(setup,subject,session):
    
    #%%        
    #% vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv set setup, subject, session
    setup = 'DOM3-MMIMS'
    subject = 'BCI_14'
    session = '2021-06-25'
    #% ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ set setup, subject, session
    
    raw_imaging_dir = '/home/rozmar/Data/Calcium_imaging/raw/{}/{}/{}'.format(setup,subject,session)
    suite2p_dir = '/home/rozmar/Data/Calcium_imaging/suite2p/{}/{}/{}'.format(setup,subject,session)
    bpod_exported = '/home/rozmar/Data/Behavior/BCI_exported/{}/{}/{}-bpod_zaber.npy'.format(setup,subject,session)
    
    behavior_dict = np.load(bpod_exported,allow_pickle = True).tolist()
    ops = np.load(os.path.join(suite2p_dir,'ops.npy'),allow_pickle = True).tolist()
    #create mean images tiff
    files_now = os.listdir(suite2p_dir)
    
    stat = np.load(os.path.join(suite2p_dir,'stat.npy'),allow_pickle = True).tolist()
    iscell = np.load(os.path.join(suite2p_dir,'iscell.npy'))
    cell_indices_s2p = np.where(iscell[:,0]==1)[0]
    stat = np.asarray(stat)[iscell[:,0]==1].tolist()
    
    
    print('calculating dff')
    utils_imaging.export_dff(suite2p_dir,raw_imaging_dir=raw_imaging_dir,revert_background_subtraction = True)
    dFF = np.load(os.path.join(suite2p_dir,'dFF.npy'))[iscell[:,0]==1,:]
    dF = np.load(os.path.join(suite2p_dir,'dF.npy'))[iscell[:,0]==1,:]
    F = np.load(os.path.join(suite2p_dir,'F.npy'))[iscell[:,0]==1,:]
    Fcorr = np.load(os.path.join(suite2p_dir,'Fcorr.npy'))[iscell[:,0]==1,:]
    Fneu = np.load(os.path.join(suite2p_dir,'Fneu.npy'))[iscell[:,0]==1,:]
    F_background = np.load(os.path.join(suite2p_dir,'F_background_values.npy'))
    F_background_correction = np.load(os.path.join(suite2p_dir,'F_background_correction.npy'))
    F += F_background_correction
    Fneu += F_background_correction
    fs = ops['fs']
    with open(os.path.join(suite2p_dir,'filelist.json')) as f:
        filelist_dict = json.load(f)
    with open(os.path.join(suite2p_dir,'s2p_params.json')) as f:
        s2p_params = json.load(f)  
    #% vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv extract metadata
    basename_prev = ''
    movie_metadata_dict = {}
    for file_name,frame_num in zip(filelist_dict['file_name_list'],filelist_dict['frame_num_list']):
        basename = file_name[:-1*file_name[::-1].find('_')-1]
        if basename != basename_prev:
            scanimage_metadata = utils_imaging.extract_scanimage_metadata(os.path.join(raw_imaging_dir,file_name))
            
            try:
                xyzstring = scanimage_metadata['metadata']['hMotors']['samplePosition']
            except: # old file format
                xyzstring = scanimage_metadata['metadata']['hMotors']['motorPosition']
            xyz = np.asarray(xyzstring.strip(']').strip('[').split(' '),float)  
            print(xyz)
            try:
                mask = np.asarray(scanimage_metadata['metadata']['hScan2D']['mask'].strip(']').strip('[').split(';'),int)
            except: # old file format
                mask = None
            try:
                scanmode = scanimage_metadata['metadata']['hScan2D']['scanMode'].strip("'")
            except: # old file format
                scanmode = scanimage_metadata['metadata']['hScan2D']['scannerType'].strip("'").lower()
            #%
            fovum = list()
            for s in scanimage_metadata['metadata']['hRoiManager']['imagingFovUm'].strip('[]').split(' '): fovum.extend(s.split(';'))
            fovum = np.asarray(fovum,float)
            fovum = [np.min(fovum),np.max(fovum)]
            pixel_size = np.diff(fovum)[0]/int(scanimage_metadata['metadata']['hRoiManager']['pixelsPerLine'])
            #%
            dict_now = {'x_size' : int(scanimage_metadata['metadata']['hRoiManager']['pixelsPerLine']),
                        'y_size' : int(scanimage_metadata['metadata']['hRoiManager']['linesPerFrame']),
                        'frame_rate': float(scanimage_metadata['metadata']['hRoiManager']['scanVolumeRate']),
                        'pixel_size': pixel_size,
                        'hbeam_power': np.asarray((scanimage_metadata['metadata']['hBeams']['powers']).strip('[]').split(' '),float)[0],
                        'hmotors_sample_x':xyz[0],
                        'hmotors_sample_y':xyz[1],
                        'hmotors_sample_z':xyz[2],
                        'hroimanager_lineperiod':float(scanimage_metadata['metadata']['hRoiManager']['linePeriod']),
                        'hroimanager_scanframeperiod':float(scanimage_metadata['metadata']['hRoiManager']['scanFramePeriod']),
                        'hroimanager_scanzoomfactor':float(scanimage_metadata['metadata']['hRoiManager']['scanZoomFactor']),
                        'hscan2d_fillfractionspatial':float(scanimage_metadata['metadata']['hScan2D']['fillFractionSpatial']),
                        'hscan2d_fillfractiontemporal':float(scanimage_metadata['metadata']['hScan2D']['fillFractionTemporal']),
                        'hscan2d_mask':mask,
                        'hscan2d_scanmode':scanmode,
                        'flyback_time':float(scanimage_metadata['metadata']['hRoiManager']['scanFramePeriod'])-float(scanimage_metadata['metadata']['hRoiManager']['linePeriod'])*int(scanimage_metadata['metadata']['hRoiManager']['linesPerFrame'])}
    
            savechannels = np.asarray(scanimage_metadata['metadata']['hChannels']['channelSave'].strip(']').strip('[').split(';'),int)
            savechannels_idx = savechannels -1
            offset = np.asarray(scanimage_metadata['metadata']['hChannels']['channelOffset'].strip(']').strip('[').split(' '),int)
            subtractoffset = np.asarray(scanimage_metadata['metadata']['hChannels']['channelSubtractOffset'].strip(']').strip('[').split(' '),str)=='true'
            color = np.asarray(scanimage_metadata['metadata']['hChannels']['channelMergeColor'].strip("'}").strip("{'").split("' '"),str)
            name = np.asarray(scanimage_metadata['metadata']['hChannels']['channelName'].strip("'}").strip("{'").split("' '"),str)
           
            offset = offset[savechannels_idx]
            subtractoffset = np.asarray(subtractoffset[savechannels_idx],int)
            color = color[savechannels_idx]
            name = name[savechannels_idx]
            offset = offset[savechannels_idx]
            #%
            MovieChannels_list = []
            for channel_number,channel_color,channel_offset,channel_subtract_offset,channel_name in zip(savechannels,
                                                                                                        color,
                                                                                                        offset,
                                                                                                        subtractoffset,
                                                                                                        name):
                moviechannelsdata = {'channel_number':channel_number,
                                     'channel_color':channel_color,
                                    'channel_offset':channel_offset,
                                    'channel_subtract_offset':channel_subtract_offset,
                                    'channel_name':name}
                MovieChannels_list.append(moviechannelsdata)
            dict_now['channels'] = MovieChannels_list
            
            movie_metadata_dict[basename] = dict_now
            basename_prev = basename
    
    for key in dict_now.keys(): # we assume that acquisition parameters are constant during the experiment, let's check it here
        value_list = list()
        for basename in movie_metadata_dict.keys():
            value_list.append(movie_metadata_dict[basename][key])
        if type(movie_metadata_dict[basename][key])==dict or type(movie_metadata_dict[basename][key])==np.ndarray or type(movie_metadata_dict[basename][key])==list or 'motor' in key:
            continue
        if len(np.unique(value_list))>1:
            print('potential error: multiple values for {}: {}'.format(key,np.unique(value_list)))
       #%
    #% extract Z locations
    if 'zcorr' in ops.keys(): # get all needed Z correction metadata here
        scanimage_metadata = utils_imaging.extract_scanimage_metadata(os.path.join(raw_imaging_dir,s2p_params['z_stack_name']))
        zs = np.asarray(scanimage_metadata['metadata']['hStackManager']['zs'].strip('[]').split(' '),float) # in microns
        zcorr_argmax = np.argmax(ops['zcorr'],1)
        zcorr_mask = np.zeros(F.shape[1])*np.nan
        zpos_mask = np.zeros(F.shape[1])*np.nan
        if len(zcorr_argmax)<len(ops['xoff']): # there is only 
            zcorr_type = 'trialwise'
            framenum_start = 0
            for framenum,zcorr in zip(filelist_dict['frame_num_list'],zcorr_argmax):
                zcorr_mask[framenum_start:framenum_start+framenum]  = zcorr
                framenum_start+=framenum
    
        else:
            zcorr_type = 'running'
            zcorr_mask = zcorr_argmax
        zpos_mask = zs[np.asarray(zcorr_mask,int)] # now it's in microns
    else:
        zcorr_mask = None
        zpos_mask = None
    
    #% ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ extract metadata      
    
    
                  
    #% vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv extract frame_times
    frame_center_offset = dict_now['y_size']*dict_now['hroimanager_lineperiod']/2 # we assume that frame rate and flyback time is constant across trials
    frame_times = np.zeros(dFF.shape[1])*np.nan
    trial_number_mask =  np.zeros(dFF.shape[1])*np.nan
    prev_frames_so_far = 0
    conditioned_neuron_name_list = []
    for i,filename in enumerate(behavior_dict['scanimage_file_names']): # generate behavior related vectors
        if filename[0] not in filelist_dict['file_name_list']:
            continue # image file not present now..
            
        movie_idx = np.where(np.asarray(filelist_dict['file_name_list'])==filename[0])[0][0]
        if movie_idx == 0 :
            frames_so_far = 0
        else:
            frames_so_far = np.sum(np.asarray(filelist_dict['frame_num_list'])[:movie_idx])
        frame_num_in_trial = np.asarray(filelist_dict['frame_num_list'])[movie_idx]  
        frame_times_now = np.arange(frame_num_in_trial)/fs+behavior_dict['scanimage_first_frame_offset'][i]+(behavior_dict['trial_start_times'][i]-behavior_dict['trial_start_times'][0]).total_seconds()
        frame_times_now  += frame_center_offset
        frame_times[frames_so_far:frames_so_far+frame_num_in_trial] = frame_times_now 
        trial_number_mask[frames_so_far:frames_so_far+frame_num_in_trial] = i
    
    # rest of the movies without behavior
    
    for i in range(len(behavior_dict['residual_tiff_files']['next_trial_index'])):
        if behavior_dict['residual_tiff_files']['scanimage_file_names'][i] not in filelist_dict['file_name_list']:
            continue
        movie_idx =np.argmax(np.asarray(filelist_dict['file_name_list'])==behavior_dict['residual_tiff_files']['scanimage_file_names'][i])
        movie_frame_num = np.asarray(filelist_dict['frame_num_list'])[movie_idx]
        movie_frame_start_idx = np.sum(np.asarray(filelist_dict['frame_num_list'])[:movie_idx])
        if not np.isnan(behavior_dict['residual_tiff_files']['next_trial_index'][i]):
            movie_frame_start_time = (behavior_dict['trial_start_times'][behavior_dict['residual_tiff_files']['next_trial_index'][i]]-behavior_dict['trial_start_times'][0]).total_seconds()-behavior_dict['residual_tiff_files']['time_to_next_trial_start'][i]
            frame_times_now = np.arange(movie_frame_num)/fs + movie_frame_start_time
            frame_times_now  += frame_center_offset
            frame_times[movie_frame_start_idx:movie_frame_start_idx+movie_frame_num] = frame_times_now 
            #break
        elif not np.isnan(behavior_dict['residual_tiff_files']['previous_trial_index'][i]):
            movie_frame_start_time = (behavior_dict['trial_start_times'][behavior_dict['residual_tiff_files']['previous_trial_index'][i]]-behavior_dict['trial_start_times'][0]).total_seconds()+behavior_dict['residual_tiff_files']['time_from_previous_trial_start'][i]
            frame_times_now = np.arange(movie_frame_num)/fs + movie_frame_start_time
            frame_times_now  += frame_center_offset
            frame_times[movie_frame_start_idx:movie_frame_start_idx+movie_frame_num] = frame_times_now 
        
    #% ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ extract frame_times          
    #% vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv extract conditioned neuron information
    # this method gives a single conditioned neuron / conditioned neuron pair for the whole session
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
    
        except: # try to read out conditioned neuron identity from file names
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
                
                
    # have to find conditioned neuron index in suite2p
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
    
    cond_s2p_idx = list()
    for roi in rois:
        try:
            centerXY_list.append((roi['scanfields']['centerXY']-fovdeg[0])/np.diff(fovdeg))
        except:
            print('multiple scanfields for {}'.format(roi['name']))
            centerXY_list.append((roi['scanfields'][0]['centerXY']-fovdeg[0])/np.diff(fovdeg))
        roinames_list.append(roi['name'])
    for roi_idx_now in roi_idx:
        conditioned_coordinates = [centerXY_list[roi_idx_now-1][0]*Lx,centerXY_list[roi_idx_now-1][1]*Ly]
        med_list = list()
        dist_list = list()
        for cell_stat in stat:
    
            dist = np.sqrt((centerXY_list[roi_idx_now-1][0]*Lx-cell_stat['med'][1])**2+(centerXY_list[roi_idx_now-1][1]*Lx-cell_stat['med'][0])**2)
            dist_list.append(dist)
            med_list.append(cell_stat['med'])
            #break
        cond_s2p_idx.append(np.argmin(dist_list))
    
    cond_s2p_idx = np.asarray(cond_s2p_idx)
    
    conditioned_neuron_dict = {'scanimage_roi_idx':roi_idx,
                               'suite2p_roi_idx':cond_s2p_idx,
                               'multi_neuron_conditioning':multi_neuron_conditioning,
                               'roi_sign':roi_sign}
    #% ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ extract conditioned neuron information        
    #% create dictioneries for schemas and upload
    try:
        session_date = datetime.datetime.strptime(session,'%m%d%y').date()
    except:
        try:
            session_date = datetime.datetime.strptime(session,'%Y-%m-%d').date()
        except:
            print('cannot understand date for session dir: {}'.format(session))
            session_date = 'lol'
    subject_id = (lab.WaterRestriction()&'water_restriction_number = "{}"'.format(subject.replace('_',''))).fetch1('subject_id')
    sessions,session_dates = (experiment.Session()&'subject_id = {}'.format(subject_id)).fetch('session','session_date')
    session_num = sessions[np.where(session_dates == session_date)[0][0]]
    session_key_wr = {'wr_id':subject.replace('_',''), 'session':session}
    
    fov_dict = {'subject_id':subject_id,
                'session':session_num,
                'fov_number':0,
                'fov_x_size':movie_metadata_dict[list(movie_metadata_dict.keys())[0]]['x_size'],
                'fov_y_size':movie_metadata_dict[list(movie_metadata_dict.keys())[0]]['y_size'],
                'fov_frame_rate':fs,
                'fov_frame_num':F.shape[1],
                'fov_pixel_size':movie_metadata_dict[list(movie_metadata_dict.keys())[0]]['pixel_size']}
    
    fov_metadata_dict = {'subject_id':subject_id,
                         'session':session_num,
                         'fov_number':0,
                         'fov_directory':suite2p_dir,
                         'fov_movie_names':filelist_dict['file_name_list'],
                         'fov_movie_frame_nums':filelist_dict['frame_num_list'],
                         'fov_zoomfactor':movie_metadata_dict[list(movie_metadata_dict.keys())[0]]['hroimanager_scanzoomfactor'],
                         'fov_beam_power':movie_metadata_dict[list(movie_metadata_dict.keys())[0]]['hbeam_power'],
                         'fov_flyback_time':movie_metadata_dict[list(movie_metadata_dict.keys())[0]]['flyback_time'],
                         'fov_lineperiod':movie_metadata_dict[list(movie_metadata_dict.keys())[0]]['hroimanager_lineperiod'],
                         'fov_scanframeperiod':movie_metadata_dict[list(movie_metadata_dict.keys())[0]]['hroimanager_scanframeperiod']}
    
    channels_dict_list = []
    for channel_dict in movie_metadata_dict[list(movie_metadata_dict.keys())[0]]['channels']:
        channels_dict = {'subject_id':subject_id,
                         'session':session_num,
                         'fov_number':0,
                         'channel_number':channel_dict['channel_number'],
                         'channel_color':channel_dict['channel_color'],
                         'channel_offset':int(np.mean(F_background_correction))*-1,
                         'channel_subtract_offset':1
                         }
        if channel_dict['channel_number'] ==ops['functional_chan']:
            channels_dict['channel_description'] = 'functional GCaMP signal'
        else:
            channels_dict['channel_description'] = 'structural channel'
        channels_dict_list.append(channels_dict)
        
    # we have mean image only for the functional channel for now
    fovmeanimage_dict ={'subject_id':subject_id,
                        'session':session_num,
                        'fov_number':0,
                        'channel_number':ops['functional_chan'], 
                        'fov_mean_image':ops['meanImg'],
                        'fov_mean_image_enhanced':ops['meanImgE'],
                        'fov_max_projection':None}
    
    fovframetimes_dict={'subject_id':subject_id,
                        'session':session_num,
                        'fov_number':0,
                        'frame_times':frame_times}
    
    fov_motion_corr_dict_base={'subject_id':subject_id,
                               'session':session_num,
                               'fov_number':0}
    fov_motion_correction_list = []
    motion_corr_descriptions = ['rigid registration','nonrigid registration']
    for motion_correction_id,motion_corr_description in enumerate(motion_corr_descriptions):
        motion_correction = fov_motion_corr_dict_base.copy()
        motion_correction['motion_correction_id'] = motion_correction_id
        motion_correction['motion_corr_description'] = motion_corr_description
        if 'nonrigid' in motion_corr_description:
            motion_correction['motion_corr_x_block']=ops['xblock']
            motion_correction['motion_corr_y_block']=ops['yblock']
            motion_correction['motion_corr_x_offset']=ops['xoff1']
            motion_correction['motion_corr_y_offset'] =ops['yoff1']
        else:
            motion_correction['motion_corr_x_block']=None
            motion_correction['motion_corr_y_block']=None
            motion_correction['motion_corr_x_offset']= ops['xoff']
            motion_correction['motion_corr_y_offset'] =ops['yoff'] 
        fov_motion_correction_list.append(motion_correction)
    
    
    roi_dict_base= {'subject_id':subject_id,
                    'session':session_num,
                    'fov_number':0}
    
    conditioned_neuron_list  = list()
    roi_list = list()
    roi_neuropil_list = list()
    roi_trace_list = list()
    roi_neuropil_trace_list = list()
    
    for cell_index in range(len(stat)):   
        dj.conn().ping()
    
        xpix = stat[cell_index]['xpix']
        ypix = stat[cell_index]['ypix']
        roi = roi_dict_base.copy()
        roi['roi_number'] = cell_index
        roi_trace = roi.copy()
        roi_neuropil = roi.copy()
        roi_neuropil['neuropil_number'] = 0
        roi_neuropil_trace = roi_neuropil.copy()
        roi['roi_centroid_x'] = stat[cell_index]['med'][1]
        roi['roi_centroid_y'] = stat[cell_index]['med'][0]
        roi['roi_xpix'] = xpix
        roi['roi_ypix'] = ypix
        roi['roi_weights'] = stat[cell_index]['lam']
        roi['roi_pixel_num'] = len(roi['roi_weights'])
        roi['roi_aspect_ratio'] = stat[cell_index]['aspect_ratio']
        roi['roi_compact'] = stat[cell_index]['compact']
        roi['roi_radius'] = stat[cell_index]['radius']
        roi['roi_skew'] =  stat[cell_index]['skew']
        roi['roi_s2p_idx'] = cell_indices_s2p[cell_index]
        roi_list.append(roi)
        
        if cell_index in conditioned_neuron_dict['suite2p_roi_idx']:
            idx_ = np.where(np.asarray(conditioned_neuron_dict['suite2p_roi_idx']==cell_index))[0][0]
            conditioned_neuron_dict_now = roi_dict_base.copy()
            conditioned_neuron_dict_now['roi_number'] = cell_index
            
            conditioned_neuron_dict_now['cond_roi_index_scanimage'] = np.asarray(conditioned_neuron_dict['scanimage_roi_idx'])[idx_]
            conditioned_neuron_dict_now['cond_roi_multiplier'] = np.asarray(conditioned_neuron_dict['roi_sign'])[idx_]
            conditioned_neuron_dict_now['multi_neuron_conditioning'] = conditioned_neuron_dict['multi_neuron_conditioning']
            conditioned_neuron_list.append(conditioned_neuron_dict_now)
            
        for channel in channels_dict_list:
            roi_trace['channel_number'] = channel['channel_number']
            if channel['channel_number'] == ops['functional_chan']:
                roi_trace_green = roi_trace.copy()
                roi_trace_green['roi_f_raw'] = F[cell_index,:]
                roi_trace_green['roi_f_corr'] = Fcorr[cell_index,:]
                roi_trace_green['roi_dff'] = dFF[cell_index,:]
                roi_trace_green['roi_f_mean'] = np.nanmean(F[cell_index,:])
                roi_trace_green['roi_f_median'] = np.nanmedian(F[cell_index,:])
                roi_trace_green['roi_f_min'] = np.nanmin(F[cell_index,:])
                roi_trace_green['roi_f_max'] = np.nanmax(F[cell_index,:])
                roi_trace_list.append(roi_trace_green)
            else: #unfinished
                roi_trace_red = roi_trace.copy()
                roi_trace_red['roi_f'] = F_chan2[cell_index,:]
                roi_trace_red['roi_f_mean'] = np.nanmean(F_chan2[cell_index,:])
                roi_trace_red['roi_f_median'] = np.nanmedian(F_chan2[cell_index,:])
                roi_trace_red['roi_f_min'] = np.nanmin(F_chan2[cell_index,:])
                roi_trace_red['roi_f_max'] = np.nanmax(F_chan2[cell_index,:])
                roi_trace_list.append(roi_trace_red)
                
                    #%
                        
        neuropil_number = 0
        roi_neuropil_now = roi_neuropil.copy()
        roi_neuropil_now['neuropil_number'] = neuropil_number
        roi_neuropil_now['neuropil_ypix'],roi_neuropil_now['neuropil_xpix'] = np.unravel_index(stat[cell_index]['neuropil_mask'],[ops['Ly'],ops['Lx']])
        roi_neuropil_now['neuropil_pixel_num'] =  len(roi_neuropil_now['neuropil_xpix'])
        roi_neuropil_list.append(roi_neuropil_now)
        for channel in channels_dict_list:
            roi_neuropil_trace['channel_number'] = channel['channel_number']
            roi_neuropil_trace['neuropil_number'] = neuropil_number
            if channel['channel_number'] == ops['functional_chan']:
                roi_neuropil_trace_green = roi_neuropil_trace.copy()
                roi_neuropil_trace_green['neuropil_f'] = Fneu[cell_index,:]
                roi_neuropil_trace_green['neuropil_f_mean'] =np.nanmean(roi_neuropil_trace_green['neuropil_f'])
                roi_neuropil_trace_green['neuropil_f_median']=np.nanmedian(roi_neuropil_trace_green['neuropil_f'])
                roi_neuropil_trace_green['neuropil_f_min']=np.nanmin(roi_neuropil_trace_green['neuropil_f'])
                roi_neuropil_trace_green['neuropil_f_max']=np.nanmax(roi_neuropil_trace_green['neuropil_f'])
                
                roi_neuropil_trace_list.append(roi_neuropil_trace_green)
            else:
                roi_neuropil_trace_red = roi_neuropil_trace.copy()
               
                roi_neuropil_trace_red['neuropil_f'] = Fneu_chan2[cell_index,:]
               
                roi_neuropil_trace_red['neuropil_f_mean'] =np.nanmean(roi_neuropil_trace_red['neuropil_f'])
                roi_neuropil_trace_red['neuropil_f_median']=np.nanmedian(roi_neuropil_trace_red['neuropil_f'])
                roi_neuropil_trace_red['neuropil_f_min']=np.nanmin(roi_neuropil_trace_red['neuropil_f'])
                roi_neuropil_trace_red['neuropil_f_max']=np.nanmax(roi_neuropil_trace_red['neuropil_f'])
    
                roi_neuropil_trace_list.append(roi_neuropil_trace_red)
    
    with dj.conn().transaction: #inserting one FOV
        print('uploading ROIs to datajoint -  subject-session: {}-{}'.format(subject_id,session)) #movie['movie_name']
        imaging.FOV().insert1(fov_dict, allow_direct_insert=True)
        imaging.FOVMetaData().insert1(fov_metadata_dict, allow_direct_insert=True)
        imaging.FOVChannel().insert(channels_dict_list, allow_direct_insert=True)
        imaging.FOVMeanImage().insert1(fovmeanimage_dict, allow_direct_insert=True)
        imaging.FOVFrameTimes().insert1(fovframetimes_dict, allow_direct_insert=True)
        imaging.MotionCorrection().insert(fov_motion_correction_list, allow_direct_insert=True)   
        imaging.ROI().insert(roi_list, allow_direct_insert=True)
        dj.conn().ping()
        imaging.ROINeuropil().insert(roi_neuropil_list, allow_direct_insert=True)
        dj.conn().ping()
        imaging.ROITrace().insert(roi_trace_list, allow_direct_insert=True)
        dj.conn().ping()
        imaging.ROINeuropilTrace().insert(roi_neuropil_trace_list, allow_direct_insert=True)
        imaging.ConditionedROI.insert(conditioned_neuron_list, allow_direct_insert=True)


#%% populate trial frames
def populate_trial_frame_ids():
    #%%
    for fov in imaging.FOV():
        if len(imaging.TrialStartFrame()&fov)>0:
            continue
        frame_times = (imaging.FOVFrameTimes()&fov).fetch1('frame_times')
        trial_event_list = list()
        action_event_list = list()
        trial_start_list = list()
        trial_end_list = list()
        for trial in experiment.SessionTrial()&fov:
            trial_dict_base = trial.copy()
            trial_dict_base.pop('trial_start_time')
            trial_dict_base.pop('trial_end_time')
            trial_dict_base['fov_number'] = fov['fov_number']
            trial_dict = trial_dict_base.copy()
            trial_start_idx = np.argmin(np.abs(frame_times-float(trial['trial_start_time'])))
            trial_start_dt = frame_times[trial_start_idx]-float(trial['trial_start_time'])
            trial_dict['frame_num'] = trial_start_idx
            trial_dict['dt'] = trial_start_dt
            trial_start_list.append(trial_dict)
            trial_end_dict = trial_dict_base.copy()
            trial_end_idx = np.argmin(np.abs(frame_times-float(trial['trial_end_time'])))
            trial_end_dt = frame_times[trial_start_idx]-float(trial['trial_end_time'])
            trial_end_dict['frame_num'] = trial_end_idx
            trial_end_dict['dt'] = trial_end_dt
            trial_end_list.append(trial_end_dict)
            
            for action_event in experiment.ActionEvent()&trial:
                action_event_dict = trial_dict_base.copy()
                action_event_dict['action_event_id'] = action_event['action_event_id']
                time_now = float(trial['trial_start_time'])+float(action_event['action_event_time'])
                frame_num = np.argmin(np.abs(frame_times-time_now))
                dt = frame_times[frame_num]-time_now
                action_event_dict['frame_num'] = frame_num
                action_event_dict['dt'] = dt
                action_event_list.append(action_event_dict)
            for trial_event in experiment.TrialEvent()&trial:
                trial_event_dict = trial_dict_base.copy()
                trial_event_dict['trial_event_id'] = trial_event['trial_event_id']
                time_now = float(trial['trial_start_time'])+float(trial_event['trial_event_time'])
                frame_num = np.argmin(np.abs(frame_times-time_now))
                dt = frame_times[frame_num]-time_now
                trial_event_dict['frame_num'] = frame_num
                trial_event_dict['dt'] = dt
                trial_event_list.append(trial_event_dict)
        with dj.conn().transaction: #inserting one FOV
            print('uploading ROIs to datajoint -  subject-session-fov: {}-{}-{}'.format(fov['subject_id'],fov['session'],fov['fov_number'])) #movie['movie_name']
            imaging.TrialStartFrame().insert(trial_start_list, allow_direct_insert=True)
            imaging.TrialEndFrame().insert(trial_end_list, allow_direct_insert=True)
            imaging.ActionEventFrame().insert(action_event_list, allow_direct_insert=True)
            imaging.TrialEventFrame().insert(trial_event_list, allow_direct_insert=True)
        
            