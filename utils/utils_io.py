import os
import shutil
from pathlib import Path
import numpy as np
from ScanImageTiffReader import ScanImageTiffReader
import json
import datetime

def extract_scanimage_metadata(file): # this is a duplicate function - also in utils_imaging
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

def copy_tiff_files_in_order(source_movie_directory,target_movie_directory):
    Path(target_movie_directory).mkdir(parents = True,exist_ok = True)
    dirs_in_target_dir = os.listdir(target_movie_directory)

    files_dict = extract_files_from_dir(source_movie_directory)
    file_idxs = (files_dict['exts']=='.tif')
    fnames = files_dict['filenames'][file_idxs]
    file_indices = files_dict['fileindices'][file_idxs]
    basenames = files_dict['basenames'][file_idxs]
    order  = np.argsort(file_indices)
    fnames = fnames[order]
    basenames = basenames[order]
    file_indices = file_indices[order]
    uniquebasenames = np.unique(basenames)
    for basename in uniquebasenames:
        if 'stack' in basename:
            needed = basenames != basename
            basenames = basenames[needed]
            fnames = fnames[needed]
            file_indices = file_indices[needed]
    uniquebasenames = np.unique(basenames)
    start_times = list()
    for basename in uniquebasenames:
        fname = fnames[np.where(basename==basenames)[0][0]]
        metadata = extract_scanimage_metadata(os.path.join(files_dict['dir'],fname))
        start_times.append(metadata['movie_start_time'])
    order = np.argsort(start_times)  
    fnames_new = list()
    for idx in order:
        fnames_new.append(fnames[basenames==uniquebasenames[idx]])
    fnames = np.concatenate(fnames_new)
    
    #%
    try:
        file_dict = np.load(os.path.join(target_movie_directory,'copy_data.npy'),allow_pickle = True).tolist()
    except:
        file_dict = {'copied_files':list()}
        
    for fname in fnames:
        if fname[:-4] not in dirs_in_target_dir: 
            try:
                metadata = extract_scanimage_metadata(os.path.join(files_dict['dir'],fname))
            except:
                print('could not read metadata for {} - probably still in the making'.format(fname))
                break
            target_dir = os.path.join(target_movie_directory,fname[:-4])
            Path(target_dir).mkdir(parents = True,exist_ok = True)
            sourcefile = os.path.join(source_movie_directory,fname)
            destfile = os.path.join(os.path.join(target_movie_directory,fname[:-4]),fname)
            shutil.copyfile(sourcefile,destfile+'_tmp')
            os.rename(destfile+'_tmp',destfile)
            file_dict['copied_files'].append(fname)
            np.save(os.path.join(target_movie_directory,'copy_data.npy'),file_dict)
            #break
