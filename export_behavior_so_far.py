from utils import utils_pipeline
import json
import os
import datetime
import sys
print(sys.argv)
setup = sys.argv[1]
subject_names = [sys.argv[2]]
try: 
    session = sys.argv[3]
except:
    session = datetime.date.today().strftime('%Y-%m-%d')
    
if setup == 'dom3':
    raw_behavior_dirs = [r'W:\Users\labadmin\Documents\Pybpod\BCI'] # TODO refer to the behavior PC
    target_movie_directory_base = r'Z:\users\rozmar\BCI_suite2p\DOM3-MMIMS'
    calcium_imaging_raw_session_dir = os.path.join(r'D:\Marton\scanimage',subject_names[0],session)
    save_dir = os.path.join(target_movie_directory_base,subject_names[0],session,'_concatenated_movie')
utils_pipeline.export_single_pybpod_session(raw_behavior_dirs,subject_names,session,calcium_imaging_raw_session_dir,save_dir)
    
        
        
