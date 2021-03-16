#%%
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import time
import os
import pickle
import shutil
#%%
paths = ['/home/rozmar/Data/Behavior/Behavior_rigs/KayvonScope']
for defpath in paths:
    if os.path.exists(defpath):
        break
#defpath = 'C:\\Users\\labadmin\\Documents\\Pybpod\\Projects'#'/home/rozmar/Network/BehaviorRig/Behavroom-Stacked-2/labadmin/Documents/Pybpod/Projects'

def loaddirstucture(projectdir = Path(defpath),projectnames_needed = None, experimentnames_needed = None,  setupnames_needed=None):
    dirstructure = dict()
    projectnames = list()
    experimentnames = list()
    setupnames = list()
    sessionnames = list()
    subjectnames = list()
    if type(projectdir) != type(Path()):
        projectdir = Path(projectdir)
    for projectname in projectdir.iterdir():
        if projectname.is_dir() and (not projectnames_needed or projectname.name in projectnames_needed):
            dirstructure[projectname.name] = dict()
            projectnames.append(projectname.name)
            
            for subjectname in (projectname / 'subjects').iterdir():
                if subjectname.is_dir() : 
                    subjectnames.append(subjectname.name)            
            
            for experimentname in (projectname / 'experiments').iterdir():
                if experimentname.is_dir() and (not experimentnames_needed or experimentname.name in experimentnames_needed ): 
                    dirstructure[projectname.name][experimentname.name] = dict()
                    experimentnames.append(experimentname.name)
                    
                    for setupname in (experimentname / 'setups').iterdir():
                        if setupname.is_dir() and (not setupnames_needed or setupname.name in setupnames_needed ): 
                            setupnames.append(setupname.name)
                            dirstructure[projectname.name][experimentname.name][setupname.name] = list()
                            
                            for sessionname in (setupname / 'sessions').iterdir():
                                if sessionname.is_dir(): 
                                    sessionnames.append(sessionname.name)
                                    dirstructure[projectname.name][experimentname.name][setupname.name].append(sessionname.name)
    return dirstructure, projectnames, experimentnames, setupnames, sessionnames, subjectnames              

def load_and_parse_a_csv_file(csvfilename,subject_needed = ''):
    df = pd.read_csv(csvfilename,delimiter=';',skiprows = 6)
    df = df[df['TYPE']!='|'] # delete empty rows
    df = df[df['TYPE']!= 'During handling of the above exception, another exception occurred:'] # delete empty rows
    df = df[df['MSG']!= ' '] # delete empty rows
    df = df[df['MSG']!= '|'] # delete empty rows
    df = df.reset_index(drop=True) # resetting indexes after deletion
    try:
        df['PC-TIME']=df['PC-TIME'].apply(lambda x : datetime.strptime(x,'%Y-%m-%d %H:%M:%S.%f')) # converting string time to datetime
    except ValueError: # sometimes pybpod don't write out the whole number...
        badidx = df['PC-TIME'].str.find('.')==-1
        if len(df['PC-TIME'][badidx]) == 1:
            df['PC-TIME'][badidx] = df['PC-TIME'][badidx]+'.000000'
        else:
            df['PC-TIME'][badidx] = [df['PC-TIME'][badidx]+'.000000']
        df['PC-TIME']=df['PC-TIME'].apply(lambda x : datetime.strptime(x,'%Y-%m-%d %H:%M:%S.%f')) # converting string time to datetime
    tempstr = df['+INFO'][df['MSG']=='CREATOR-NAME'].values[0]
    experimenter = tempstr[2:tempstr[2:].find('"')+2] #+2
    tempstr = df['+INFO'][df['MSG']=='SUBJECT-NAME'].values[0]
    subject = tempstr[2:tempstr[2:].find("'")+2] #+2
    if len(subject_needed)>0 and subject.lower() != subject_needed.lower():
        return None
    df['experimenter'] = experimenter
    df['subject'] = subject
    # adding trial numbers in session
    idx = (df[df['TYPE'] == 'TRIAL']).index.to_numpy()
    idx = np.concatenate(([0],idx,[len(df)]),0)
    idxdiff = np.diff(idx)
    Trialnum = np.array([])
    for i,idxnumnow in enumerate(idxdiff): #zip(np.arange(0:len(idxdiff)),idxdiff):#
        Trialnum  = np.concatenate((Trialnum,np.zeros(idxnumnow)+i),0)
    df['Trial_number_in_session'] = Trialnum
    indexes = df[df['MSG'] == 'Trialnumber:'].index + 1 #+2
    if len(indexes)>0:
        if 'Trial_number' not in df.columns:
            df['Trial_number']=np.NaN
        trialnumbers_real = df['MSG'][indexes]
        trialnumbers = df['Trial_number_in_session'][indexes].values
        for trialnumber_real,trialnum in zip(trialnumbers_real,trialnumbers):
            #df['Trial_number'][df['Trial_number_in_session'] == trialnum] = int(blocknumber)
            try:
                df.loc[df['Trial_number_in_session'] == trialnum, 'Trial_number'] = int(trialnumber_real)
            except:
                df.loc[df['Trial_number_in_session'] == trialnum, 'Block_number'] = np.nan
    # saving variables (if any)
    variableidx = (df[df['MSG'] == 'Variables:']).index.to_numpy()
    if len(variableidx)>0:
        d={}
        exec('variables = ' + df['MSG'][variableidx+1].values[0], d)
        for varname in d['variables'].keys():
            if isinstance(d['variables'][varname], (list,tuple)):
                templist = list()
                for idx in range(0,len(df)):
                    templist.append(d['variables'][varname])
                df['var:'+varname]=templist
            else:
                df['var:'+varname] = d['variables'][varname]
    # updating variables
    variableidxs = (df[df['MSG'] == 'Variables updated:']).index.to_numpy()
    for variableidx in variableidxs:
        d={}
        exec('variables = ' + df['MSG'][variableidx+1], d)
        for varname in d['variables'].keys():
            if isinstance(d['variables'][varname], (list,tuple)):
                templist = list()
                idxs = list()
                for idx in range(variableidx,len(df)):
                    idxs.append(idx)
                    templist.append(d['variables'][varname])
                df['var:'+varname][variableidx:]=templist.copy()
            else:
                #df['var:'+varname][variableidx:] = d['variables'][varname]
                df.loc[range(variableidx,len(df)), 'var:'+varname] = d['variables'][varname]

    return df


#%%
def minethedata(data):
    #%%
    Zaber_moves_channel = ['Wire1High','Wire1Low']
    trial_start_idxs = data.loc[data['TYPE'] == 'TRIAL'].index.to_numpy()
    trial_end_idxs = data.loc[data['TYPE'] == 'END-TRIAL'].index.to_numpy()
    if len(trial_start_idxs) > len(trial_end_idxs):
        trial_end_idxs = np.concatenate([trial_end_idxs,[len(data)]])
    
    data_dict = {'go_cue_times':list(),
                 'trial_start_times':list(),
                 'lick_L':list(),
                 'lick_R':list(),
                 'reward_L':list(),
                 'reward_R':list(),
                 'autowater_L':list(),
                 'autowater_R':list(),
                 'zaber_move_forward': list(),
                 'trial_hit':list(),
                 'time_to_hit':list(),
                 'trial_num':list(),
                 
                 }
    
    for trial_num,(trial_start_idx, trial_end_idx) in enumerate(zip(trial_start_idxs,trial_end_idxs)):
        df_trial =  data[trial_start_idx:trial_end_idx]
        trial_start_time = data['PC-TIME'][trial_start_idx]
        go_cue_time = df_trial.loc[(df_trial['MSG'] == 'GoCue') & (df_trial['TYPE'] == 'TRANSITION'),'PC-TIME'].values#[0]#.index.to_numpy()[0]
        if len(go_cue_time) == 0:
            break # no go cue no trial
        lick_left_times = df_trial.loc[data['var:WaterPort_L_ch_in'] == data['+INFO'],'PC-TIME'].values
        lick_right_times = df_trial.loc[data['var:WaterPort_R_ch_in'] == data['+INFO'],'PC-TIME'].values
        reward_left_times = df_trial.loc[(data['MSG'] == 'Reward_L') & (data['TYPE'] == 'TRANSITION'),'PC-TIME'].values
        reward_right_times = df_trial.loc[(data['MSG'] == 'Reward_R') & (data['TYPE'] == 'TRANSITION'),'PC-TIME'].values
        autowater_left_times = df_trial.loc[(data['MSG'] == 'Auto_Water_L') & (data['TYPE'] == 'TRANSITION'),'PC-TIME'].values
        autowater_right_times = df_trial.loc[(data['MSG'] == 'Auto_Water_R') & (data['TYPE'] == 'TRANSITION'),'PC-TIME'].values
        ITI_start_times = df_trial.loc[(data['MSG'] == 'ITI') & (data['TYPE'] == 'TRANSITION'),'PC-TIME'].values
        zaber_motor_movement_times = df_trial.loc[(Zaber_moves_channel[0] == data['+INFO']) | (Zaber_moves_channel[1] == data['+INFO']),'PC-TIME'].values
        
        # convert to seconds from trial start
        zero_time = np.asarray(trial_start_time,dtype = 'datetime64[us]')
        go_cue_time = (np.asarray(np.asarray(go_cue_time,dtype = 'datetime64[us]')-zero_time,float)/1000000)[0]
        lick_left_times = (np.asarray(np.asarray(lick_left_times,dtype = 'datetime64[us]')-zero_time,float)/1000000)
        lick_right_times = (np.asarray(np.asarray(lick_right_times,dtype = 'datetime64[us]')-zero_time,float)/1000000)
        
        reward_left_times = (np.asarray(np.asarray(reward_left_times,dtype = 'datetime64[us]')-zero_time,float)/1000000)
        reward_right_times = (np.asarray(np.asarray(reward_right_times,dtype = 'datetime64[us]')-zero_time,float)/1000000)
        autowater_left_times = (np.asarray(np.asarray(autowater_left_times,dtype = 'datetime64[us]')-zero_time,float)/1000000)
        autowater_right_times = (np.asarray(np.asarray(autowater_right_times,dtype = 'datetime64[us]')-zero_time,float)/1000000)
        ITI_start_times = (np.asarray(np.asarray(ITI_start_times,dtype = 'datetime64[us]')-zero_time,float)/1000000)
        zaber_motor_movement_times = (np.asarray(np.asarray(zaber_motor_movement_times,dtype = 'datetime64[us]')-zero_time,float)/1000000)
        
        
        
        data_dict['trial_num'].append(trial_num)
        data_dict['go_cue_times'].append(go_cue_time)
        data_dict['trial_start_times'].append(trial_start_time)
        data_dict['lick_L'].append(lick_left_times)
        data_dict['lick_R'].append(lick_right_times)
        data_dict['reward_L'].append(reward_left_times)
        data_dict['reward_R'].append(reward_right_times)
        data_dict['autowater_L'].append(autowater_left_times)
        data_dict['autowater_R'].append(autowater_right_times)
        data_dict['zaber_move_forward'].append(zaber_motor_movement_times)
        
        reward_times = np.concatenate([reward_left_times,reward_right_times])
        if len(reward_times)>0:
            data_dict['trial_hit'].append(True)
            data_dict['time_to_hit'].append(reward_times[0]-go_cue_time)
        else:
            data_dict['trial_hit'].append(False)
            data_dict['time_to_hit'].append(np.nan)
       #%% 
    return data_dict
        
  

def generate_pickles_from_csv(projectdir = Path(defpath),
                              projectnames_needed = None, 
                              experimentnames_needed = None,  
                              setupnames_needed=None,
                              load_only_last_day = False):
    #%%
# =============================================================================
#     projectdir = Path(defpath)
#     projectnames_needed = None 
#     experimentnames_needed = None
#     setupnames_needed=None
#     load_only_last_day = False
# =============================================================================
    
    
    dirstructure = dict()
    projectnames = list()
    experimentnames = list()
    setupnames = list()
    sessionnames = list()
    #projectdir= defpath
    if type(projectdir) != type(Path()):
        projectdir = Path(projectdir)
    for projectname in projectdir.iterdir():
        if projectname.is_dir() and (not projectnames_needed or projectname.name in projectnames_needed):
            dirstructure[projectname.name] = dict()
            projectnames.append(projectname.name)
            
            projectdir_export = projectname/'experiments_exported'
            if 'experiments_exported' not in os.listdir(projectname):
                (projectdir_export).mkdir()
                
            for experimentname in (projectname / 'experiments').iterdir():
                if experimentname.is_dir() and (not experimentnames_needed or experimentname.name in experimentnames_needed ): 
                    dirstructure[projectname.name][experimentname.name] = dict()
                    experimentnames.append(experimentname.name)
                    
                    experimentname_export = projectdir_export/experimentname.name
                    if experimentname.name not in os.listdir(projectdir_export):
                        (experimentname_export).mkdir()
                        experimentname_export = experimentname_export/'setups'
                        (experimentname_export).mkdir()
                    else:
                        experimentname_export = experimentname_export/'setups'
                    for setupname in (experimentname / 'setups').iterdir():
                        if setupname.is_dir() and (not setupnames_needed or setupname.name in setupnames_needed ): 
                            setupnames.append(setupname.name)
                            dirstructure[projectname.name][experimentname.name][setupname.name] = list()
                            #%
                            setupname_export = experimentname_export/setupname.name
                            if setupname.name not in os.listdir(experimentname_export):
                                (setupname_export).mkdir()
                                setupname_export = setupname_export/'sessions'
                                (setupname_export).mkdir()
                            else:
                                setupname_export = setupname_export/'sessions'
                            
                            if load_only_last_day:
                                sessionnames_forsort = list()
                                for sessionname in (setupname / 'sessions').iterdir():
                                    if sessionname.is_dir(): 
                                        sessionnames_forsort.append(sessionname.name[:8])#only the date
                                sessionnames_forsort = np.sort(np.unique(sessionnames_forsort))
                                sessiondatestoload = sessionnames_forsort[-5:]
                            
                            for sessionname in (setupname / 'sessions').iterdir():
                                if sessionname.is_dir() and (not load_only_last_day or sessionname.name[:8] in sessiondatestoload): 
                                    sessionnames.append(sessionname.name)
                                    dirstructure[projectname.name][experimentname.name][setupname.name].append(sessionname.name)
                                    if not os.path.exists(setupname_export/ (sessionname.name+'.pkl')):
                                        doit = True
                                    elif os.stat(setupname_export/ (sessionname.name+'.pkl')).st_mtime < os.stat(sessionname/ (sessionname.name+'.csv')).st_mtime:
                                        doit = True
                                    else:
                                        doit = False
                                    if doit and os.path.exists(sessionname/ (sessionname.name+'.csv')):
                                        df = load_and_parse_a_csv_file(sessionname/ (sessionname.name+'.csv'))
                                        
                                        variables = dict()
# =============================================================================
#                                         try:
# =============================================================================
#%

                                        variables = minethedata(df)  
                                        variables['experimenter'] = df['experimenter'][0]
                                        variables['subject'] = df['subject'][0]
                                        
# =============================================================================
#                                         print([len(variables['trial_num']),len(variables['time_to_hit'])])
#                                         if np.diff([len(variables['trial_num']),len(variables['time_to_hit'])])[0] != 0:
#                                             time.sleep(1000)
# =============================================================================
                                        
                                        
                                        #%
# =============================================================================
#                                         except:
#                                             variables = dict()
# =============================================================================
                                        with open(setupname_export/ (sessionname.name+'.tmp'), 'wb') as outfile:
                                            pickle.dump(variables, outfile)
                                        shutil.move(setupname_export/ (sessionname.name+'.tmp'),setupname_export/ (sessionname.name+'.pkl'))

   #%%    %                                
def load_pickles_for_online_analysis(projectdir = Path(defpath),projectnames_needed = None, experimentnames_needed = None,  setupnames_needed=None, subjectnames_needed = None, load_only_last_day = False):
    #%%
# =============================================================================
#     projectdir = Path(defpath)
#     projectnames_needed = None
#     experimentnames_needed = None
#     setupnames_needed=None
#     subjectnames_needed = 'BCI03'
#     load_only_last_day = True   
# =============================================================================
        
    variables_out = dict()
    if type(projectdir) != type(Path()):
        projectdir = Path(projectdir)
    for projectname in projectdir.iterdir():
        if projectname.is_dir() and (not projectnames_needed or projectname.name in projectnames_needed):    
            for experimentname in (projectname / 'experiments_exported').iterdir():
                if experimentname.is_dir() and (not experimentnames_needed or experimentname.name in experimentnames_needed ): 
    
                    for setupname in (experimentname / 'setups').iterdir():
                        if setupname.is_dir() and (not setupnames_needed or setupname.name in setupnames_needed ): 
                            if load_only_last_day:
                                sessionnames= list()
                                for sessionname in os.listdir(setupname / 'sessions'):
                                    sessionnames.append(sessionname[:8])#only the date
                                sessionnames = np.sort(np.unique(sessionnames))
                                sessiondatestoload = sessionnames[-5:]
                                sessions = np.sort(os.listdir(setupname / 'sessions'))
                            for sessionname in sessions:
                                if sessionname[-3:] == 'pkl' and (not load_only_last_day or sessionname[:8] in sessiondatestoload): 
                                    #print('opening '+ sessionname)
                                    with open(setupname / 'sessions'/ sessionname,'rb') as readfile:
                                        variables_new = pickle.load(readfile)
                                    if len(variables_new.keys()) > 0:
# =============================================================================
#                                         print([len(variables_new['trial_num']),len(variables_new['time_to_hit'])])
#                                         if np.diff([len(variables_new['trial_num']),len(variables_new['time_to_hit'])])[0] != 0:
#                                             time.sleep(1000)
# =============================================================================
                                        if  not subjectnames_needed or variables_new['subject'] in subjectnames_needed:
                                            variables_new['file_start_time']=[variables_new['trial_start_times'][0]]
                                            
                                            if len(variables_out.keys()) == 0: # initialize dictionary
                                                variables_out = variables_new
                                                variables_out['experimenter'] = [variables_out['experimenter']]
                                                variables_out['subject'] = [variables_out['subject']]
                                                variables_new['file_start_trialnum'] = [variables_new['trial_num'][0]]
                                            else:
                                                variables_new['trial_num']=list(np.asarray(variables_new['trial_num'])+variables_out['trial_num'][-1]+1)
                                                variables_new['file_start_trialnum'] = [variables_new['trial_num'][0]]
                                                for key in variables_new.keys():
                                                    if type(variables_new[key]) == list:
                                                        variables_out[key].extend(variables_new[key])
                                                    else:
                                                        variables_out[key].append(variables_new[key])
      #%%                                              
    return variables_out
