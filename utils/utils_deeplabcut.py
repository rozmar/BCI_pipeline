#%%
import deeplabcut

#%% initialize 
deeplabcut.create_new_project('BCI_DOM3_side', 
                              'rozmar', 
                              ['/home/rozmar/Data/DLC_training/BCI_01/raw_samples/side'], 
                              working_directory='/home/rozmar/Data/DLC_training/BCI_01/side', 
                              copy_videos=False, 
                              multianimal=False)
configfile = '/home/rozmar/Data/DLC_training/BCI_01/side/BCI_DOM3_side-rozmar-2021-08-15/config.yaml'
#%%
deeplabcut.create_new_project('BCI_DOM3_bottom', 
                              'rozmar', 
                              ['/home/rozmar/Data/DLC_training/BCI_01/raw_samples/bottom'], 
                              working_directory='/home/rozmar/Data/DLC_training/BCI_01/bottom', 
                              copy_videos=False, 
                              multianimal=False)
configfile = '/home/rozmar/Data/DLC_training/BCI_01/bottom/BCI_DOM3_bottom-rozmar-2021-09-05/config.yaml'
#%% extract frames

deeplabcut.extract_frames(configfile,userfeedback = False)
#%% extract csv file from h5 file that DLC generated - than train on cluster
import os
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

base_folder = configfile[:-11]
labeledfolder = os.path.join(base_folder,'labeled-data')
destination_dir = os.path.join(base_folder,'labeled_for_cluster')
Path(destination_dir).mkdir(parents = True,exist_ok = True)
labeledmovies = os.listdir(labeledfolder)
movie_counter = 0
csv_dict = {}
final_frame_name_list = []
original_frame_names = []

for movie in labeledmovies:
    movie_dir = os.path.join(labeledfolder,movie)
    files = np.sort(os.listdir(movie_dir))
    frames = list()
    for file in files:
        if '.csv' in file:
            meta = pd.read_csv(os.path.join(movie_dir,file),index_col = 0, header = None).T
            break
    bodyparts = np.unique(meta['bodyparts'].values)
    for fname in meta.keys():
        if '.png' not in fname:
            continue
        movie_counter+=1
        for bodypart in bodyparts:
            if bodypart not in csv_dict.keys():
                csv_dict[bodypart] = {'X':[],
                                      'Y':[],
                                      'Slice':[]}
            x = round(float(meta.loc[(meta['bodyparts']==bodypart) & (meta['coords']=='x'),fname].values[0]),2)
            y = round(float(meta.loc[(meta['bodyparts']==bodypart) & (meta['coords']=='y'),fname].values[0]),2)
            csv_dict[bodypart]['X'].append(x)
            csv_dict[bodypart]['Y'].append(y)
            csv_dict[bodypart]['Slice'].append(movie_counter)
        original_frame_names.append(fname)
        final_frame_name_list.append('img_{:05d}.png'.format(movie_counter))
       
for bodypart in csv_dict.keys():
    df_out_now = pd.DataFrame(csv_dict[bodypart])
    df_out_now.index = np.arange(1, len(df_out_now)+1)
    df_out_now.to_csv(os.path.join(destination_dir,'{}.csv'.format(bodypart)))
for f_orig,f_dest in zip(original_frame_names,final_frame_name_list):
    shutil.copyfile(os.path.join(base_folder,f_orig),os.path.join(destination_dir,f_dest))
       
    
    #break