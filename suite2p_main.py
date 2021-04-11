#%% import libraries
from utils import utils_imaging, utils_pipeline
from pathlib import Path
import os
from os import path
import numpy as np
import shutil
import time
import json

from suite2p import default_ops as s2p_default_ops
from suite2p import run_s2p, io,registration

#%% define session and suite2p working dir
source_movie_directory_base = '/home/rozmar/Data/Calcium_imaging/raw/'
target_movie_directory_base = '/home/rozmar/Data/temp/suite2p/'
source_movie_directory = '/home/rozmar/Data/Calcium_imaging/raw/DOM3-MMIMS/BCI_07/2021-02-15/'
target_movie_directory = os.path.join(target_movie_directory_base,source_movie_directory[len(source_movie_directory_base):])
s2p_params = {'max_reg_shift':50, # microns
              'max_reg_shift_NR': 20, # microns
              'block_size': 200, # microns
              'smooth_sigma':0.5, # microns
              'smooth_sigma_time':0, #seconds,
              'overwrite': False,
              'num_workers':4} # folder where the suite2p output is saved
sp2_params_file = os.path.join(target_movie_directory,'s2p_params.json')
Path(target_movie_directory).mkdir(parents = True,exist_ok = True)
with open(sp2_params_file, "w") as data_file:
        json.dump(s2p_params, data_file, indent=2)
#%% Check for new .tiff files in a given directory and copy them when they are finished - should be run every few seconds
utils_pipeline.copy_1_tiff_file_in_order(source_movie_directory,target_movie_directory)
#%% obtain shared reference image from zstack - or from multiple trials
trial_num_to_use = 10
utils_imaging.generate_mean_image_from_trials(target_movie_directory,trial_num_to_use)

#%% write to binary and perform motion correction if a new file appears, create mean image

file_dict = np.load(os.path.join(target_movie_directory,'copy_data.npy'),allow_pickle = True).tolist()
for file in file_dict['copied_files']:
    dir_now = os.path.join(target_movie_directory,file[:-4])
    tiff_now = os.path.join(target_movie_directory,file[:-4],file)
    reg_json_file = os.path.join(target_movie_directory,file[:-4],'reg_progress.json')
    if 'reg_progress.json' in os.listdir(dir_now):
        with open(reg_json_file, "r") as read_file:
            reg_dict = json.load(read_file)
    else:
        reg_dict = {'registration_started':False}
        
    if reg_dict['registration_started']:
        continue
    utils_imaging.register_trial(target_movie_directory,file) # this should spawn a worker instead
    
    
#%%

#%% generate concatenated binary file 

#%% run cell detection on concatenated binary file