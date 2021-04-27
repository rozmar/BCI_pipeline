#%% define session and suite2p working dir
# =============================================================================
# source_movie_directory_base = '/home/rozmar/Data/Calcium_imaging/raw/'
# target_movie_directory_base = '/home/rozmar/Data/temp/suite2p/'
# source_movie_directory = '/home/rozmar/Data/Calcium_imaging/raw/DOM3-MMIMS/BCI_07/2021-02-15/'
# =============================================================================

# =============================================================================
# source_movie_directory_base = '/run/user/62266/gvfs/sftp:host=10.102.10.46/home/rozmar/Data/Calcium_imaging/raw/'
# target_movie_directory_base = '/groups/svoboda/home/rozsam/Data/BCI_data/'
# source_movie_directory = '/run/user/62266/gvfs/sftp:host=10.102.10.46/home/rozmar/Data/Calcium_imaging/raw/DOM3-MMIMS/BCI_07/2021-02-15/'
# =============================================================================

# =============================================================================
# #source_movie_directory_base = '/run/user/62266/gvfs/sftp:host=10.102.10.46/home/rozmar/Data/Calcium_imaging/raw/'
# target_movie_directory_base = '/groups/svoboda/home/rozsam/Data/BCI_data/'
# #source_movie_directory = '/run/user/62266/gvfs/sftp:host=10.102.10.46/home/rozmar/Data/Calcium_imaging/raw/DOM3-MMIMS/BCI_07/2021-02-15/'
# =============================================================================

#target_movie_directory_base = '/home/rozmar/Data/temp/suite2p/'
target_movie_directory_base = '/groups/svoboda/home/rozsam/Data/BCI_data/'

# =============================================================================
# setup = 'DOM3-MMIMS'
# subject = 'BCI_07'
# session = '2021-02-15'
# =============================================================================

# =============================================================================
# setup = 'KayvonScope'
# subject = 'BCI_03'
# session = '121420'
# =============================================================================

setup = 'DOM3-MMIMS'
subject = 'BCI_07'
session = '2021-02-17'


# =============================================================================
# setup = 'KayvonScope'
# subject = 'BCI_08'
# session = '012121'
# =============================================================================


# =============================================================================
# setup = 'KayvonScope'
# subject = 'BCI_07'
# session = '042121'
# =============================================================================

s2p_params = {'max_reg_shift':50, # microns
              'max_reg_shift_NR': 20, # microns
              'block_size': 200, # microns
              'smooth_sigma':0.5, # microns
              'smooth_sigma_time':0, #seconds,
              'overwrite': False,
              'num_workers':4} # folder where the suite2p output is saved

on_cluster = True
#% import libraries
if on_cluster:
    
    import json
    from utils import utils_io
    from pathlib import Path
    
else:
    from utils import utils_imaging, utils_io #utils_pipeline,
    from pathlib import Path
    import json
#from threading import Timer
import os
import threading
import multiprocessing
import numpy as np
import shutil
import datetime
class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer     = None
        self.function   = function
        self.interval   = interval
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = threading.Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False

#%
#target_movie_directory = os.path.join(target_movie_directory_base,source_movie_directory[len(source_movie_directory_base):])
target_movie_directory = os.path.join(target_movie_directory_base,setup,subject,session)
sp2_params_file = os.path.join(target_movie_directory,'s2p_params.json')
Path(target_movie_directory).mkdir(parents = True,exist_ok = True)
with open(sp2_params_file, "w") as data_file:
        json.dump(s2p_params, data_file, indent=2)
#% Check for new .tiff files in a given directory and copy them when they are finished - should be run every few seconds
copyfile_json_file = os.path.join(target_movie_directory_base,'copyfile.json')
copyfile_params = {'setup':setup,
                   'subject':subject,
                   'session':session}
with open(copyfile_json_file, "w") as data_file:
        json.dump(copyfile_params, data_file, indent=2)
# =============================================================================
# copy_thread = multiprocessing.Process(target=utils_io.copy_tiff_files_in_loop, args=(source_movie_directory,target_movie_directory))
# copy_thread.start()
# =============================================================================
#%%
#copy_thread = threading.Thread(target=utils_io.copy_tiff_files_in_loop, name="copy tiffs", args=(source_movie_directory,target_movie_directory))
#copy_thread.start()   
#rt = RepeatedTimer(3600, utils_io.copy_tiff_files_in_order, source_movie_directory,target_movie_directory) # it auto-starts, no need of rt.start()
#utils_io.copy_1_tiff_file_in_order(source_movie_directory,target_movie_directory)
#%% obtain shared reference image from zstack - or from multiple trials
trial_num_to_use = 10
# =============================================================================
# cluster_command = ' && '.join(cluster_command_list)
# full_command = 'bsub -n 1 -J meanimage " {} > ~/Scripts/meanimage_out.txt"'.format(cluster_command)
# =============================================================================
#%
if not os.path.exists(os.path.join(target_movie_directory,'mean_image.npy')):
    if on_cluster:
        cluster_command_list = ['eval "$(conda shell.bash hook)"',
                                'conda activate suite2p',
                                'cd ~/Scripts/Python/BCI_pipeline/',
                                'python cluster_helper.py {} "\'{}\'" {}'.format('utils_imaging.generate_mean_image_from_trials',target_movie_directory,trial_num_to_use)]
        with open("/groups/svoboda/home/rozsam/Scripts/runBCI.sh","w") as shfile:
            #shfile.writelines(cluster_command_list) 
            for L in cluster_command_list:
                shfile.writelines(L+'\n') 
        bash_command = "bsub -n 1 -J BCI_job 'sh /groups/svoboda/home/rozsam/Scripts/runBCI.sh > ~/Scripts/BCI_output.txt'"
        os.system(bash_command)
    else:
        utils_imaging.generate_mean_image_from_trials(target_movie_directory,trial_num_to_use)
else:
    print('reference image is already present')

#%% write to binary and perform motion correction if a new file appears, create mean image
import time
num = 0
file_dict = np.load(os.path.join(target_movie_directory,'copy_data.npy'),allow_pickle = True).tolist()

for file in file_dict['copied_files']:
    if not os.path.exists(os.path.join(target_movie_directory,'mean_image.npy')):
        print('no reference image!!')
        break
    
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
    print('starting {}'.format(file))
    if on_cluster: # this part will spawn a worker for each trial
        #%
        cluster_command_list = ['eval "$(conda shell.bash hook)"',
                                'conda activate suite2p',
                                'cd ~/Scripts/Python/BCI_pipeline/',
                                "python cluster_helper.py {} '\"{}\"' '\"{}\"'".format('utils_imaging.register_trial',target_movie_directory,file)]
        cluster_output_file = os.path.join(dir_now,'s2p_registration_output.txt')
        bash_command = r"bsub -n 1 -J BCI_register_{} -o /dev/null '{} > {}'".format(file,' && '.join(cluster_command_list),cluster_output_file)
        os.system(bash_command)
        
    else:
        utils_imaging.register_trial(target_movie_directory,file) 
    

#%% generate concatenated binary file 
concatenated_movie_dir = os.path.join(target_movie_directory,'_concatenated_movie')
Path(concatenated_movie_dir).mkdir(parents = True,exist_ok = True)
#utils_io.concatenate_suite2p_files(target_movie_directory)

if on_cluster: # this part will spawn a worker for each trial
    #%
    cluster_command_list = ['eval "$(conda shell.bash hook)"',
                            'conda activate suite2p',
                            'cd ~/Scripts/Python/BCI_pipeline/',
                            "python cluster_helper.py {} '\"{}\"'".format('utils_io.concatenate_suite2p_files',target_movie_directory)]
    cluster_output_file = os.path.join(os.path.join(target_movie_directory,'_concatenated_movie'),'s2p_concatenation_output.txt')
    bash_command = r"bsub -n 1 -J BCI_concatenate_files '{} > {}'".format(' && '.join(cluster_command_list),cluster_output_file)
    os.system(bash_command)
    
else:
    utils_io.concatenate_suite2p_files(target_movie_directory)



#%% run cell detection on concatenated binary file

# =============================================================================
# concatenated_movie_dir = os.path.join(target_movie_directory,'_concatenated_movie')
# full_movie_dir = os.path.join(target_movie_directory,'full_movie_00')
# Path(full_movie_dir).mkdir(parents = True,exist_ok = True)
# for file in os.listdir(concatenated_movie_dir):
#     shutil.copy(os.path.join(concatenated_movie_dir,file),os.path.join(full_movie_dir,file))
# =============================================================================

concatenated_movie_dir = os.path.join(target_movie_directory,'_concatenated_movie')
full_movie_dir = concatenated_movie_dir
#%
cluster_command_list = ['eval "$(conda shell.bash hook)"',
                        'conda activate suite2p',
                        'cd ~/Scripts/Python/BCI_pipeline/',
                        "python cluster_helper.py {} '\"{}\"'".format('utils_imaging.find_ROIs',full_movie_dir)]
cluster_output_file = os.path.join(full_movie_dir,'s2p_ROI_finding_output.txt')
bash_command = r"bsub -n 2 -J BCI_ROIfind '{} > {}'".format(' && '.join(cluster_command_list),cluster_output_file)
os.system(bash_command) # -o /dev/null

#%% registration metrics
if on_cluster:
    cluster_command_list = ['eval "$(conda shell.bash hook)"',
                            'conda activate suite2p',
                            'cd ~/Scripts/Python/BCI_pipeline/',
                            "python cluster_helper.py {} '\"{}\"'".format('utils_imaging.registration_metrics',full_movie_dir)]
    cluster_output_file = os.path.join(full_movie_dir,'s2p_registration_metrics_output.txt')
    bash_command = r"bsub -n 2 -J BCI_registration_metric '{} > {}'".format(' && '.join(cluster_command_list),cluster_output_file)
    os.system(bash_command) # -o /dev/null
else:
    utils_imaging.registration_metrics(full_movie_dir)    
#%% stop deamons
#copy_thread.kill()

# =============================================================================
# #%%
# file_dict = np.load(os.path.join(target_movie_directory,'copy_data.npy'),allow_pickle = True).tolist()
# for file_idx,file in enumerate(file_dict['copied_files']):
#     
#     dir_now = os.path.join(target_movie_directory,file[:-4])
#     ops = np.load(os.path.join(dir_now,'suite2p','plane0','ops.npy'),allow_pickle = True).tolist()
#     if len(ops['xoff']) != ops['nframes']:
#         print([file,ops['nframes'],len(ops['xoff'])])
# =============================================================================
#%% z-stack analysis
z_stack_dir = '/groups/svoboda/home/rozsam/Data/zstack/940/'
z_stack_fname = os.listdir(z_stack_dir)[0]





