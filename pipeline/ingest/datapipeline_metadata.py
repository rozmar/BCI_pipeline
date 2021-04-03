import notebook_google.notebook_main as online_notebook
from datetime import datetime, timedelta
import pandas as pd
import json
#import time as timer
import numpy as np
#% connect to server
import datajoint as dj
dj.conn()
#from pipeline import pipeline_tools
from pipeline import lab#, experiment
#%%
def populatemetadata(updatemetadata = False):
    #%% save metadata from google drive if necessairy
    if updatemetadata:
        online_notebook.update_metadata(dj.config['metadata.notebook_name'],dj.config['locations.metadata_surgery_experiment'])
        lastmodify = online_notebook.fetch_lastmodify_time('Lab metadata')
        with open(dj.config['locations.metadata_lab']+'last_modify_time.json') as timedata:
            lastmodify_prev = json.loads(timedata.read())
        if lastmodify != lastmodify_prev:
            print('updating Lab metadata from google drive')
            dj.config['locations.metadata_lab']
            IDs = ['Experimenter','Rig','Virus']
            for ID in IDs:
                df_wr = online_notebook.fetch_lab_metadata(ID)
                if type(df_wr) == pd.DataFrame:
                    df_wr.to_csv(dj.config['locations.metadata_lab']+ID+'.csv') 
        
            with open(dj.config['locations.metadata_lab']+'last_modify_time.json', "w") as write_file:
                json.dump(lastmodify, write_file)
            print('Lab metadata updated')
    
    #%% add users
    df_experimenters = pd.read_csv(dj.config['locations.metadata_lab']+'Experimenter.csv')
    experimenterdata = list()
    for experimenter in df_experimenters.iterrows():
        experimenter = experimenter[1]
        dictnow = {'username':experimenter['username'],'fullname':experimenter['fullname']}
        experimenterdata.append(dictnow)
    print('adding experimenters')
    for experimenternow in experimenterdata:
        try:
            lab.Person().insert1(experimenternow)
        except dj.errors.DuplicateError:
            print('duplicate. experimenter: ',experimenternow['username'], ' already exists')
    
    #%% add rigs
    df_rigs = pd.read_csv(dj.config['locations.metadata_lab']+'Rig.csv')
    rigdata = list()
    for rig in df_rigs.iterrows():
        rig = rig[1]
        dictnow = {'rig':rig['rig'],'room':rig['room'],'rig_description':rig['rig_description']}
        rigdata.append(dictnow)
    print('adding rigs')
    for rignow in rigdata:
        try:
            lab.Rig().insert1(rignow)
        except dj.errors.DuplicateError:
            print('duplicate. rig: ',rignow['rig'], ' already exists')
            
    #%% add viruses
    df_viruses = pd.read_csv(dj.config['locations.metadata_lab']+'Virus.csv')
    virusdata = list()
    serotypedata = list()
    for virus in df_viruses.iterrows():
        virus = virus[1]
        if type(virus['remarks']) != str:
            virus['remarks'] = ''
        dictnow = {'virus_id':virus['virus_id'],
                   'virus_source':virus['virus_source'],
                   'serotype':virus['serotype'],
                   'username':virus['username'],
                   'virus_name':virus['virus_name'],
                   'titer':virus['titer'],
                   'order_date':virus['order_date'],
                   'remarks':virus['remarks']}
        virusdata.append(dictnow)
        dictnow = {'serotype':virus['serotype']}
        serotypedata.append(dictnow)
    print('adding rigs')
    for virusnow,serotypenow in zip(virusdata,serotypedata):
        try:
            lab.Serotype().insert1(serotypenow)
        except dj.errors.DuplicateError:
            print('duplicate serotype: ',serotypenow['serotype'], ' already exists')
        try:
            lab.Virus().insert1(virusnow)
        except dj.errors.DuplicateError:
            print('duplicate virus: ',virusnow['virus_name'], ' already exists')
    #%% populate subjects, surgeries and water restrictions
    print('adding surgeries and stuff')
    df_surgery = pd.read_csv(dj.config['locations.metadata_surgery_experiment']+'Surgeries-BCI.csv')
    #%%
    for item in df_surgery.iterrows():
        
        if item[1]['project'].lower() == dj.config['project'].lower() and (item[1]['status'] == 'training' or item[1]['status'] == 'sacrificed'):
            
            subjectdata = {
                    'subject_id': item[1]['animal#'],
                    'cage_number': item[1]['cage#'],
                    'date_of_birth': item[1]['DOB'],
                    'sex': item[1]['sex'],
                    'username': item[1]['experimenter'],
                    }
            try:
                lab.Subject().insert1(subjectdata)
            except dj.errors.DuplicateError:
                print('duplicate. animal :',item[1]['animal#'], ' already exists')
            surgeryidx = 1
            while 'surgery date ('+str(surgeryidx)+')' in item[1].keys() and item[1]['surgery date ('+str(surgeryidx)+')'] and type(item[1]['surgery date ('+str(surgeryidx)+')']) == str:
                start_time = datetime.strptime(item[1]['surgery date ('+str(surgeryidx)+')']+' '+item[1]['surgery time ('+str(surgeryidx)+')'],'%Y-%m-%d %H:%M')
                end_time = start_time + timedelta(minutes = int(item[1]['surgery length (min) ('+str(surgeryidx)+')']))
                surgerydata = {
                        'surgery_id': surgeryidx,
                        'subject_id':item[1]['animal#'],
                        'username': item[1]['experimenter'],
                        'start_time': start_time,
                        'end_time': end_time,
                        'surgery_description': item[1]['surgery type ('+str(surgeryidx)+')'] + ':-: comments: ' + str(item[1]['surgery comments ('+str(surgeryidx)+')']),
                        }
                try:
                    lab.Surgery().insert1(surgerydata)
                except dj.errors.DuplicateError:
                    print('duplicate. surgery for animal ',item[1]['animal#'], ' already exists: ', start_time)
                #checking craniotomies
                #%
                cranioidx = 1
                while 'craniotomy diameter ('+str(cranioidx)+')' in item[1].keys() and item[1]['craniotomy diameter ('+str(cranioidx)+')'] and (type(item[1]['craniotomy surgery id ('+str(cranioidx)+')']) == int or type(item[1]['craniotomy surgery id ('+str(cranioidx)+')']) == float):
                    if item[1]['craniotomy surgery id ('+str(cranioidx)+')'] == surgeryidx:
                        proceduredata = {
                                'surgery_id': surgeryidx,
                                'subject_id':item[1]['animal#'],
                                'procedure_id':cranioidx,
                                'skull_reference':item[1]['craniotomy reference ('+str(cranioidx)+')'],
                                'ml_location':item[1]['craniotomy lateral ('+str(cranioidx)+')'],
                                'ap_location':item[1]['craniotomy anterior ('+str(cranioidx)+')'],
                                'surgery_procedure_description': 'craniotomy: ' + str(item[1]['craniotomy comments ('+str(cranioidx)+')']),
                                }
                        try:
                            lab.Surgery.Procedure().insert1(proceduredata)
                        except dj.errors.DuplicateError:
                            print('duplicate cranio for animal ',item[1]['animal#'], ' already exists: ', cranioidx)
                    cranioidx += 1
                #% 
                
                virusinjidx = 1
                while 'virus inj surgery id ('+str(virusinjidx)+')' in item[1].keys() and item[1]['virus inj virus id ('+str(virusinjidx)+')'] and item[1]['virus inj surgery id ('+str(virusinjidx)+')']:
                    if item[1]['virus inj surgery id ('+str(virusinjidx)+')'] == surgeryidx:
    # =============================================================================
    #                     print('waiting')
    #                     timer.sleep(1000)
    # =============================================================================
                        if '[' in str(item[1]['virus inj lateral ('+str(virusinjidx)+')']):
                            virus_ml_locations = eval(item[1]['virus inj lateral ('+str(virusinjidx)+')'])
                        else:
                            virus_ml_locations = [int(item[1]['virus inj lateral ('+str(virusinjidx)+')'])]
                        if '[' in str(item[1]['virus inj anterior ('+str(virusinjidx)+')']):
                            virus_ap_locations = eval(item[1]['virus inj anterior ('+str(virusinjidx)+')'])
                        else:
                            virus_ap_locations = [int(item[1]['virus inj anterior ('+str(virusinjidx)+')'])]
                        if '[' in str(item[1]['virus inj ventral ('+str(virusinjidx)+')']):
                            virus_dv_locations = eval(item[1]['virus inj ventral ('+str(virusinjidx)+')'])
                        else:
                            virus_dv_locations = [int(item[1]['virus inj ventral ('+str(virusinjidx)+')'])]
                        if '[' in str(item[1]['virus inj volume (nl) ('+str(virusinjidx)+')']):
                            virus_volumes = eval(item[1]['virus inj volume (nl) ('+str(virusinjidx)+')'])
                        else:
                            virus_volumes = [int(item[1]['virus inj volume (nl) ('+str(virusinjidx)+')'])]
                        if '[' in str(item[1]['virus inj dilution ('+str(virusinjidx)+')']):
                            virus_dilutions = eval(item[1]['virus inj dilution ('+str(virusinjidx)+')'])
                        else:
                            virus_dilutions = np.ones(len(virus_ml_locations))*float(item[1]['virus inj dilution ('+str(virusinjidx)+')'])
                            
                            
                        for virus_ml_location,virus_ap_location,virus_dv_location,virus_volume,virus_dilution in zip(virus_ml_locations,virus_ap_locations,virus_dv_locations,virus_volumes,virus_dilutions):
                            injidx = len(lab.Surgery.VirusInjection() & surgerydata) +1
                            virusinjdata = {
                                    'surgery_id': surgeryidx,
                                    'subject_id':item[1]['animal#'],
                                    'injection_id':injidx,
                                    'virus_id':item[1]['virus inj virus id ('+str(virusinjidx)+')'],
                                    'skull_reference':item[1]['virus inj reference ('+str(virusinjidx)+')'],
                                    'ml_location':virus_ml_location,
                                    'ap_location':virus_ap_location,
                                    'dv_location':virus_dv_location,
                                    'volume':virus_volume,
                                    'dilution':virus_dilution, #item[1]['virus inj dilution ('+str(virusinjidx)+')'],
                                    'description': 'virus injection: ' + str(item[1]['virus inj comments ('+str(virusinjidx)+')']),
                                    }
                            try:
                                lab.Surgery.VirusInjection().insert1(virusinjdata)
                            except dj.errors.DuplicateError:
                                print('duplicate virus injection for animal ',item[1]['animal#'], ' already exists: ', injidx)
                    virusinjidx += 1    
                #%
                
                surgeryidx += 1
                    
                #%
            if item[1]['ID'] and item[1]['wr start date']:
                wrdata = {
                        'subject_id':item[1]['animal#'],
                        'water_restriction_number': item[1]['ID'],
                        'wr_cage_number': item[1]['cage#'],
                        'wr_start_date': item[1]['wr start date'],
                        'wr_start_weight': item[1]['wr start weight'],
                        }
                ##print(wrdata)
                try:
                    lab.WaterRestriction().insert1(wrdata)
                except dj.errors.DuplicateError:
                    print('duplicate. water restriction :',item[1]['animal#'], ' already exists')
                      
