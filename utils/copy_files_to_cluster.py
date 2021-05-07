from utils import utils_io
import json
import os
import time
target_movie_directory_base = '/home/rozmar/Network/dm11/svoboda$/rozsam/Data/BCI_data'
#target_movie_directory_base = '/home/rozmar/Data/temp/suite2p/'
source_movie_directory_base = '/home/rozmar/Data/Calcium_imaging/raw'
while True:
    try:
        copyfile_json_file = os.path.join(target_movie_directory_base,'copyfile.json')
        with open(copyfile_json_file, "r") as read_file:
                    copyfile_dict = json.load(read_file)
        target_movie_directory = os.path.join(target_movie_directory_base,copyfile_dict['setup'],copyfile_dict['subject'],copyfile_dict['session'])
        source_movie_directory = os.path.join(source_movie_directory_base,copyfile_dict['setup'],copyfile_dict['subject'],copyfile_dict['session'])
        utils_io.copy_tiff_files_in_order(source_movie_directory,target_movie_directory)
    except:
        print('error reading copy json file')
        time.sleep(5)
        
        
