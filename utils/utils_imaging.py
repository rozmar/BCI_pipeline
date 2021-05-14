from ScanImageTiffReader import ScanImageTiffReader
import json
import numpy as np
import datetime
import os
import time
from os import path
from scipy.ndimage import filters

from suite2p import default_ops as s2p_default_ops
from suite2p import run_s2p, io,registration, run_plane
#%%
def extract_scanimage_metadata(file): # this function is also in utils_io
    #%
    image = ScanImageTiffReader(file)
    metadata_raw = image.metadata()
    description_first_image = image.description(0)
    description_first_image_dict = dict(item.split(' = ') for item in description_first_image.rstrip(r'\n ').rstrip('\n').split('\n'))
    metadata_str = metadata_raw.split('\n\n')[0]
    metadata_json = metadata_raw.split('\n\n')[1]
    metadata_dict = dict(item.split('=') for item in metadata_str.split('\n') if 'SI.' in item)
    metadata = {k.strip().replace('SI.','') : v.strip() for k, v in metadata_dict.items()}
    for k in list(metadata.keys()):
        if '.' in k:
            ks = k.split('.')
            # TODO just recursively create dict from .-containing values
            if k.count('.') == 1:
                if not ks[0] in metadata.keys():
                    metadata[ks[0]] = {}
                metadata[ks[0]][ks[1]] = metadata[k]
            elif k.count('.') == 2:
                if not ks[0] in metadata.keys():
                    metadata[ks[0]] = {}
                    metadata[ks[0]][ks[1]] = {}
                elif not ks[1] in metadata[ks[0]].keys():
                    metadata[ks[0]][ks[1]] = {}
                metadata[ks[0]][ks[1]] = metadata[k]
            elif k.count('.') > 2:
                print('skipped metadata key ' + k + ' to minimize recursion in dict')
            metadata.pop(k)
    metadata['json'] = json.loads(metadata_json)
    frame_rate = metadata['hRoiManager']['scanVolumeRate']
    try:
        z_collection = metadata['hFastZ']['userZs']
        num_planes = len(z_collection)
    except: # new scanimage version
        if metadata['hFastZ']['enable'] == 'true':
            print('multiple planes not handled in metadata collection.. HANDLE ME!!!')
            #time.sleep(1000)
            num_planes = 1
        else:
            num_planes = 1
    
    roi_metadata = metadata['json']['RoiGroups']['imagingRoiGroup']['rois']
    
    
    if type(roi_metadata) == dict:
        roi_metadata = [roi_metadata]
    num_rois = len(roi_metadata)
    roi = {}
    w_px = []
    h_px = []
    cXY = []
    szXY = []
    for r in range(num_rois):
        roi[r] = {}
        roi[r]['w_px'] = roi_metadata[r]['scanfields']['pixelResolutionXY'][0]
        w_px.append(roi[r]['w_px'])
        roi[r]['h_px'] = roi_metadata[r]['scanfields']['pixelResolutionXY'][1]
        h_px.append(roi[r]['h_px'])
        roi[r]['center'] = roi_metadata[r]['scanfields']['centerXY']
        cXY.append(roi[r]['center'])
        roi[r]['size'] = roi_metadata[r]['scanfields']['sizeXY']
        szXY.append(roi[r]['size'])
        #print('{} {} {}'.format(roi[r]['w_px'], roi[r]['h_px'], roi[r]['size']))
    
    w_px = np.asarray(w_px)
    h_px = np.asarray(h_px)
    szXY = np.asarray(szXY)
    cXY = np.asarray(cXY)
    cXY = cXY - szXY / 2
    cXY = cXY - np.amin(cXY, axis=0)
    mu = np.median(np.transpose(np.asarray([w_px, h_px])) / szXY, axis=0)
    imin = cXY * mu
    
    n_rows_sum = np.sum(h_px)
    n_flyback = (image.shape()[1] - n_rows_sum) / np.max([1, num_rois - 1])
    
    irow = np.insert(np.cumsum(np.transpose(h_px) + n_flyback), 0, 0)
    irow = np.delete(irow, -1)
    irow = np.vstack((irow, irow + np.transpose(h_px)))
    
    data = {}
    data['fs'] = frame_rate
    data['nplanes'] = num_planes
    data['nrois'] = num_rois #or irow.shape[1]?
    if data['nrois'] == 1:
        data['mesoscan'] = 0
    else:
        data['mesoscan'] = 1
    
    if data['mesoscan']:
        #data['nrois'] = num_rois #or irow.shape[1]?
        data['dx'] = []
        data['dy'] = []
        data['lines'] = []
        for i in range(num_rois):
            data['dx'] = np.hstack((data['dx'], imin[i,1]))
            data['dy'] = np.hstack((data['dy'], imin[i,0]))
            data['lines'] = list(range(irow[0,i].astype('int32'), irow[1,i].astype('int32') - 1)) ### TODO NOT QUITE RIGHT YET
        data['dx'] = data['dx'].astype('int32')
        data['dy'] = data['dy'].astype('int32')
        print(data['dx'])
        print(data['dy'])
        print(data['lines'])
            #data['lines']{i} = 
            #data.dx(i) = int32(imin(i,2));
            #data.dy(i) = int32(imin(i,1));
            #data.lines{i} = irow(1,i):(irow(2,i)-1)
    movie_start_time = datetime.datetime.strptime(description_first_image_dict['epoch'].rstrip(']').lstrip('['), '%Y %m %d %H %M %S.%f')
    out = {'metadata':metadata,
           'roidata':data,
           'roi_metadata':roi_metadata,
           'frame_rate':frame_rate,
           'num_planes':num_planes,
           'shape':image.shape(),
           'description_first_frame':description_first_image_dict,
           'movie_start_time': movie_start_time}
    #%%
    return out

def register_trial(target_movie_directory,file):
    #%%
    with open(os.path.join(target_movie_directory,'s2p_params.json'), "r") as read_file:
        s2p_params = json.load(read_file)
    dir_now = os.path.join(target_movie_directory,file[:-4])
    tiff_now = os.path.join(target_movie_directory,file[:-4],file)
    reg_json_file = os.path.join(target_movie_directory,file[:-4],'reg_progress.json')
    if 'reg_progress.json' in os.listdir(dir_now):
        with open(reg_json_file, "r") as read_file:
            reg_dict = json.load(read_file)
    else:
        reg_dict = {'registration_started':False}
        
# =============================================================================
#     if reg_dict['registration_started']:
#         return
# =============================================================================
    reg_dict = {'registration_started':True,
                'registration_started_time':str(time.time()),
                'registration_finished':False}
    with open(reg_json_file, "w") as data_file:
        json.dump(reg_dict, data_file, indent=2)
    metadata = extract_scanimage_metadata(tiff_now)
    pixelsize = metadata['roi_metadata'][0]['scanfields']['sizeXY']
    movie_dims = metadata['roi_metadata'][0]['scanfields']['pixelResolutionXY']
    zoomfactor = float(metadata['metadata']['hRoiManager']['scanZoomFactor'])
    
    XFOV = 1500*np.exp(-zoomfactor/11.5)+88 # HOTFIX for 16x objective
    pixelsize_real =  XFOV/movie_dims[0]
    print('pixel size changed from {} to {} '.format(pixelsize,pixelsize_real))
    pixelsize = pixelsize_real
    
    FOV = np.min(pixelsize)*np.asarray(movie_dims)
    ops = s2p_default_ops()#run_s2p.default_ops()

    ops['reg_tif'] = False # save registered movie as tif files
    #ops['num_workers'] = s2p_params['num_workers']
    ops['delete_bin'] = 0 
    ops['keep_movie_raw'] = 0
    ops['save_path0'] = dir_now
    ops['fs'] = float(metadata['frame_rate'])
    if '[' in metadata['metadata']['hChannels']['channelSave']:
        ops['nchannels'] = 2
    else:
        ops['nchannels'] = 1
    ops['tau'] = 1
    ops['maxregshift'] =  s2p_params['max_reg_shift']/np.max(FOV)
    ops['nimg_init'] = 500
    ops['nonrigid'] = True
    ops['maxregshiftNR'] = int(s2p_params['max_reg_shift_NR']/np.min(pixelsize)) # this one is in pixels...
    block_size_optimal = np.round((s2p_params['block_size']/np.min(pixelsize)))
    potential_bases = np.asarray([2**np.floor(np.log(block_size_optimal)/np.log(2)),2**np.ceil(np.log(block_size_optimal)/np.log(2)),3**np.floor(np.log(block_size_optimal)/np.log(3)),3**np.ceil(np.log(block_size_optimal)/np.log(3))])
    block_size = int(potential_bases[np.argmin(np.abs(potential_bases-block_size_optimal))])
    ops['block_size'] = np.ones(2,int)*block_size
    ops['smooth_sigma'] = s2p_params['smooth_sigma']/np.min(pixelsize_real)#pixelsize_real #ops['diameter']/10 #
    #ops['smooth_sigma_time'] = s2p_params['smooth_sigma_time']*float(metadata['frame_rate']) # ops['tau']*ops['fs']#
    ops['data_path'] = target_movie_directory
    ops['tiff_list'] = [tiff_now]
    ops['batch_size'] = 250
    ops['do_registration'] = 1
    ops['roidetect'] = False
    meanimage_dict = np.load(os.path.join(target_movie_directory,'mean_image.npy'),allow_pickle = True).tolist()
    refImg = meanimage_dict['refImg']
    ops['refImg'] = refImg
    ops['force_refImg'] = True
    print('regstering {}'.format(tiff_now))
    ops['do_regmetrics'] = False
    #%
    ops = run_s2p(ops)
    
    #%%
    try:
        #%
        file = s2p_params['z_stack_name']
#%
        zstack_tiff = os.path.join(target_movie_directory,file[:-4],file)
        reader=ScanImageTiffReader(zstack_tiff)
        stack=reader.data()
        if stack.shape[1]/ops['Lx'] == 2:
            stack = stack[:,::2,::2]
        elif stack.shape[1]/ops['Lx'] == 4:
            stack = stack[:,::4,::4]
        elif stack.shape[1]/ops['Lx'] == 8:
            stack = stack[:,::8,::8]
        #%
        #ops_orig, zcorr = registration.zalign.compute_zpos(stack, ops)
        ops_orig, zcorr = registration.zalign.compute_zpos_single_frame(stack, ops['meanImg'][np.newaxis,:,:], ops)
        np.save(ops['ops_path'], ops_orig)
        #%%
        #reader.close()
        #%
    except:
        pass # no z-stack
  #%%
    with open(reg_json_file, "r") as read_file:
        reg_dict = json.load(read_file)
    reg_dict['registration_finished'] = True
    reg_dict['registration_finished_time'] = str(time.time())
    with open(reg_json_file, "w") as data_file:
        json.dump(reg_dict, data_file, indent=2)
       #%% 
def generate_mean_image_from_trials(target_movie_directory,trial_num_to_use):
    #%%
    with open(os.path.join(target_movie_directory,'s2p_params.json'), "r") as read_file:
        s2p_params = json.load(read_file)
    reference_movie_dir = os.path.join(target_movie_directory,'_reference_image')
    file_dict = np.load(os.path.join(target_movie_directory,'copy_data.npy'),allow_pickle = True).tolist()
    file_now = file_dict['copied_files'][0]
    metadata = extract_scanimage_metadata(os.path.join(target_movie_directory,file_now[:-4],file_now))
    pixelsize = metadata['roi_metadata'][0]['scanfields']['sizeXY']
    movie_dims = metadata['roi_metadata'][0]['scanfields']['pixelResolutionXY']
    zoomfactor = float(metadata['metadata']['hRoiManager']['scanZoomFactor'])
    
    XFOV = 1500*np.exp(-zoomfactor/11.5)+88 # HOTFIX for 16x objective
    pixelsize_real =  XFOV/movie_dims[0]
    print('pixel size changed from {} to {} '.format(pixelsize,pixelsize_real))
    pixelsize = pixelsize_real
    
    FOV = np.min(pixelsize)*np.asarray(movie_dims)
    ops = s2p_default_ops()#run_s2p.default_ops()
    
    ops['reg_tif'] = False # save registered movie as tif files
    #ops['num_workers'] = s2p_params['num_workers']
    ops['delete_bin'] = 0 
    ops['keep_movie_raw'] = 0
    ops['fs'] = float(metadata['frame_rate'])
    if '[' in metadata['metadata']['hChannels']['channelSave']:
        ops['nchannels'] = 2
    else:
        ops['nchannels'] = 1
    ops['tau'] = 1
    ops['maxregshift'] =  s2p_params['max_reg_shift']/np.max(FOV)
    ops['nimg_init'] = 500
    ops['nonrigid'] = True
    ops['maxregshiftNR'] = int(s2p_params['max_reg_shift_NR']/np.min(pixelsize)) # this one is in pixels...
    block_size_optimal = np.round((s2p_params['block_size']/np.min(pixelsize)))
    potential_bases = np.asarray([2**np.floor(np.log(block_size_optimal)/np.log(2)),2**np.ceil(np.log(block_size_optimal)/np.log(2)),3**np.floor(np.log(block_size_optimal)/np.log(3)),3**np.ceil(np.log(block_size_optimal)/np.log(3))])
    block_size = int(potential_bases[np.argmin(np.abs(potential_bases-block_size_optimal))])
    ops['block_size'] = np.ones(2,int)*block_size
    ops['smooth_sigma'] = s2p_params['smooth_sigma']/np.min(pixelsize_real)#pixelsize_real #ops['diameter']/10 #
    #ops['smooth_sigma_time'] = s2p_params['smooth_sigma_time']*float(metadata['frame_rate']) # ops['tau']*ops['fs']#
    ops['data_path'] = reference_movie_dir
    ops['batch_size'] = 250
    ops['do_registration'] = 0
    ops['roidetect'] = False
    ops['do_bidiphase'] = True
    #%
    tiff_list = list()
    filename_list = list()
    for file_now in file_dict['copied_files']:
        tiff_list.append(os.path.join(target_movie_directory,file_now[:-4],file_now))
        filename_list.append(file_now)
        if len(tiff_list)>=trial_num_to_use:
            break
    ops['tiff_list'] = tiff_list
    ops['save_path0'] = reference_movie_dir
    #%
    run_s2p(ops)
    refImg = None
    raw = True
    ops = np.load(os.path.join(target_movie_directory,'_reference_image','suite2p/plane0/ops.npy'),allow_pickle = True).tolist()
    if ops['frames_include'] != -1:
        ops['nframes'] = min((ops['nframes'], ops['frames_include']))
    else:
        nbytes = path.getsize(ops['raw_file'] if ops.get('keep_movie_raw') and path.exists(ops['raw_file']) else ops['reg_file'])
        ops['nframes'] = int(nbytes / (2 * ops['Ly'] * ops['Lx'])) # this equation is only true with int16 :)
    # get binary file paths
    raw = raw and ops.get('keep_movie_raw') and 'raw_file' in ops and path.isfile(ops['raw_file'])
    reg_file_align = ops['reg_file'] if ops['nchannels'] < 2 or ops['functional_chan'] == ops['align_by_chan'] else ops['reg_file_chan2']
    raw_file_align = ops.get('raw_file') if ops['nchannels'] < 2 or ops['functional_chan'] == ops['align_by_chan'] else ops.get('raw_file_chan2')
    raw_file_align = raw_file_align if raw and ops.get('keep_movie_raw') and 'raw_file' in ops and path.isfile(ops['raw_file']) else []
    
    ### ----- compute and use bidiphase shift -------------- ###
    if refImg is None or (ops['do_bidiphase'] and ops['bidiphase'] == 0):
        # grab frames
        with io.BinaryFile(Lx=ops['Lx'], Ly=ops['Ly'], read_filename=raw_file_align if raw else reg_file_align) as f:
            frames = f[np.linspace(0, ops['nframes'], 1 + np.minimum(ops['nimg_init'], ops['nframes']), dtype=int)[:-1]]    
    
    if refImg is not None:
        print('NOTE: user reference frame given')
    else:
        t0 = time.time()
        refImg = registration.register.compute_reference(ops, frames)
        print('Reference frame, %0.2f sec.'%(time.time()-t0))
    ops['refImg'] = refImg
    meanimage_dict = {'refImg':refImg,
                      'movies_used':filename_list}
    np.save(os.path.join(target_movie_directory,'mean_image.npy'),meanimage_dict)    
    
    reference_movie_json = os.path.join(target_movie_directory,'_reference_image','refimage_progress.json')
    with open(reference_movie_json, "r") as read_file:
        refimage_dict = json.load(read_file)
    refimage_dict['ref_image_finished'] = True
    refimage_dict['ref_image_finished_time'] = str(time.time())
    with open(reference_movie_json, "w") as data_file:
        json.dump(refimage_dict, data_file, indent=2)
    #%%
def find_ROIs(full_movie_dir):
    #%%
    ops_path = os.path.join(full_movie_dir,'ops.npy')
    
    ops = np.load(ops_path,allow_pickle = True).tolist()
    #%
    keys = list(ops.keys())
    for key in keys:
        if key.endswith('_list') and 'Img' in key:
            ops[key[:-5]]=ops[key]
            #print(key)
        elif key =='fs_list':
            ops[key[:-5]]=np.median(ops[key])
            print('there were multiple frame rates: {} , using: {}'.format(np.unique(ops[key]),np.median(ops[key])))
        elif key.endswith('_list'):
            ops[key[:-5]]=ops[key]
        if key.endswith('_list'):
            ops.pop(key, None)
    
    concatenated_movie_filelist_json = os.path.join(full_movie_dir,'filelist.json')
    with open(concatenated_movie_filelist_json, "r") as read_file:
        filelist_dict = json.load(read_file)
            
    roifind_progress_dict = {'roifind_started':True,
                             'roifind_finished':False,
                             'roifind_start_time':str(time.time()),
                             'roifind_source_movies':list(filelist_dict['file_name_list'])}
    #print(roifind_progress_dict)
    roifindjson_file = os.path.join(full_movie_dir,'roifind_progress.json')
    with open(roifindjson_file, "w") as write_file:
        json.dump(roifind_progress_dict, write_file,indent=2)
    
    
            #%
    if 'BCI_10' in full_movie_dir: #GCaMP8s
        ops['tau'] = .25
    ops['do_registration'] = 0
    ops['save_path'] = full_movie_dir
    ops['allow_overlap'] = False
    ops['nframes'] = np.sum(ops['nframes'])
    ops['save_folder'] = ''
    ops['save_path0'] = full_movie_dir
    ops['fast_disk'] = full_movie_dir
    ops['reg_file'] = os.path.join(full_movie_dir,'data.bin')
    ops['roidetect'] = True
    ops['ops_path'] = full_movie_dir
    ops['xrange'] = [np.max(ops['xrange'][::2]),np.min(ops['xrange'][1::2])]
    ops['yrange'] = [np.max(ops['yrange'][::2]),np.min(ops['yrange'][1::2])]
    ops['save_mat']=1
    if type(ops['fs']) == list:
        ops['fs'] = ops['fs'][-1]
    if type(ops['bidi_corrected']) == list or type(ops['bidi_corrected']) == np.ndarray:
        ops['bidi_corrected'] = ops['bidi_corrected'][-1]
    #%% #np.save(os.path.join(full_movie_dir,'ops.npy'),ops)
    run_plane(ops)
    roifind_progress_dict['roifind_finished'] = True
    roifind_progress_dict['roifind_finish_time']=str(time.time())
    with open(roifindjson_file, "w") as write_file:
        json.dump(roifind_progress_dict, write_file,indent=2)
    #%%
def registration_metrics(full_movie_dir):
    #%%
    ops_path = os.path.join(full_movie_dir,'ops.npy')
    
    ops = np.load(ops_path,allow_pickle = True).tolist()
    #%
    keys = list(ops.keys())
    for key in keys:
        if key.endswith('_list') and 'Img' in key:
            ops[key[:-5]]=ops[key]
            #print(key)
        elif key.endswith('_list'):
            ops[key[:-5]]=ops[key]
# =============================================================================
#         if key.endswith('_list'):
#             ops.pop(key, None)
# =============================================================================
            #%
    ops['do_registration'] = 0
    ops['save_path'] = full_movie_dir
    ops['allow_overlap'] = True
    ops['nframes'] = np.sum(ops['nframes'])
    ops['save_folder'] = ''
    ops['save_path0'] = full_movie_dir
    ops['fast_disk'] = full_movie_dir
    ops['reg_file'] = os.path.join(full_movie_dir,'data.bin')
    ops['roidetect'] = True
    ops['ops_path'] = full_movie_dir
    ops['xrange'] = [np.max(ops['xrange'][::2]),np.min(ops['xrange'][1::2])]
    ops['yrange'] = [np.max(ops['yrange'][::2]),np.min(ops['yrange'][1::2])]
    t0 = time.time()
    ops = registration.get_pc_metrics(ops)
    #print('Registration metrics, %0.2f sec.' % time.time()-t0)
    if 'fs_list' in ops.keys():
        ops['fs'] = np.median(ops['fs_list'])
    np.save(os.path.join(ops['save_path'], 'ops.npy'), ops)
    
def export_dff(suite2p_dir,raw_imaging_dir=None,revert_background_subtraction = False):
    #%%
    
    if revert_background_subtraction:
        with open(os.path.join(suite2p_dir,'filelist.json')) as f:
            filelist_dict = json.load(f)
        background_to_subtract = []
        basename_prev = ''
        for file_name,frame_num in zip(filelist_dict['file_name_list'],filelist_dict['frame_num_list']):
            basename = file_name[:-1*file_name[::-1].find('_')-1]
            if basename != basename_prev:
                metadata = extract_scanimage_metadata(os.path.join(raw_imaging_dir,file_name))
                offsets = np.asarray(metadata['metadata']['hScan2D']['channelOffsets'].strip('[]').split(' '),int)
                subtract_offset = np.asarray(metadata['metadata']['hScan2D']['channelsSubtractOffsets'].strip('[]').split(' '))=='true'
                if  not subtract_offset[0]:
                    offset_value = 0
                else:
                    offset_value = offsets[0]
                basename_prev = basename
                #print(file_name)  
            background_to_subtract.append(np.ones(frame_num)*offset_value)
        background_to_subtract = np.concatenate(background_to_subtract)
           # break
        
    
    #%%

    #%
# =============================================================================
#     suite2p_dir = '/home/rozmar/Data/Calcium_imaging/suite2p/DOM3-MMIMS/BCI_07/2021-02-15'
#     suite2p_dir = '/home/rozmar/Data/Calcium_imaging/suite2p/KayvonScope/BCI_03/121420'
#     suite2p_dir = '/home/rozmar/Data/Calcium_imaging/suite2p/KayvonScope/BCI_07/042121'
# =============================================================================
    
    F = np.load(os.path.join(suite2p_dir,'F.npy'))+background_to_subtract -np.min(background_to_subtract)
    Fneu = np.load(os.path.join(suite2p_dir,'Fneu.npy')) + background_to_subtract -np.min(background_to_subtract)
    #iscell = np.load(os.path.join(suite2p_dir,'iscell.npy'))
    ops = np.load(os.path.join(suite2p_dir,'ops.npy'),allow_pickle = True).tolist()
    fs = ops['fs']
    sig_baseline = 10 
    win_baseline = int(60*fs)
    #noncell = iscell[:,0] ==0
    Fcorr= F-0.7*Fneu
    to_correct = np.min(Fcorr,1)<1
    Fcorr[to_correct,:] = Fcorr[to_correct,:]-np.min(Fcorr,1)[to_correct,np.newaxis]+1 # we don't allow anything to be below 0
    #Fcorr[noncell] = 0
    #%
    Flow = filters.gaussian_filter(Fcorr,    [0., sig_baseline])
    Flow = filters.minimum_filter1d(Flow,    win_baseline)
    Flow = filters.maximum_filter1d(Flow,    win_baseline)
    #%
    dF = Fcorr-Flow
    dFF = (Fcorr-Flow)/Flow
    #Fcorr[noncell] = 0
    #%
    np.save(os.path.join(suite2p_dir,'Fcorr.npy'),Fcorr )
    np.save(os.path.join(suite2p_dir,'dFF.npy'),dFF )
    np.save(os.path.join(suite2p_dir,'dF.npy'),dF )
    
    
