from pywavesurfer import ws
import numpy as np
import time as timer
import datetime
#%
def recover_digital_channels(array_raw,channelnum):
    digitalarrays = list()
    for i in np.arange(channelnum)[::-1]:
        array_now = np.zeros_like(array_raw)
        array_now[array_raw>=2**i] = 1
        array_raw[array_raw>=2**i] -=2**i
        digitalarrays.append(np.asarray(array_now,int))
    digitalarrays = np.concatenate(digitalarrays[::-1])
    return digitalarrays
#%
def load_wavesurfer_file(WS_path): # works only for single sweep files
    #%
    ws_data = ws.loadDataFile(filename=WS_path, format_string='double' )
    #units = np.array(ws_data['header']['AIChannelUnits']).astype(str)
    channelnames_analog = np.array(ws_data['header']['AIChannelNames']).astype(str)
    activechannels_analog = np.concatenate(ws_data['header']['IsAIChannelActive'])==1
    channelnames_analog = channelnames_analog[activechannels_analog]
    channelnames_digital = np.array(ws_data['header']['DIChannelNames']).astype(str)
    activechannels_digital = np.concatenate(ws_data['header']['IsDIChannelActive'])==1
    channelnames_digital = channelnames_digital[activechannels_digital]
    keys = list(ws_data.keys())
    del keys[keys=='header']
# =============================================================================
#     if len(keys)>1:
#         print('MULTIPLE SWEEPS! HANDLE ME!! : {}'.format(WS_path))
#         timer.sleep(10000)
# =============================================================================
    outdict_list = list()
    for key in keys:
        sweep = ws_data[key]
        sRate = ws_data['header']['AcquisitionSampleRate'][0][0]
        timestamp = ws_data['header']['ClockAtRunStart']
        outdict = {'sampling_rate':sRate,
                   'timestamp':timestamp,
                   'sweep_timestamp':sweep['timestamp'][0][0],
                   'sweep_name':key}
        
        for idx,name in enumerate(channelnames_analog):
            outdict['AI-{}'.format(name)] = sweep['analogScans'][idx,:]
        if 'digitalScans' in sweep.keys():
            digitalarrays = recover_digital_channels(sweep['digitalScans'],len(channelnames_digital))
            for idx,name in enumerate(channelnames_digital):
                outdict['DI-{}'.format(name)] = digitalarrays[idx,:]
        seconds = np.floor(outdict['timestamp'][5])
        microseconds = (outdict['timestamp'][5]-seconds)*1000000
        outdict['sweep_start_timestamp'] = datetime.datetime(outdict['timestamp'][0],outdict['timestamp'][1],outdict['timestamp'][2],outdict['timestamp'][3],outdict['timestamp'][4],seconds,microseconds) + datetime.timedelta(seconds = outdict['sweep_timestamp'])
        try:
            outdict = decode_bitcode(outdict)
        except:
            pass
        outdict_list.append(outdict)
        outdict['ws_file_path'] = WS_path
    return outdict_list

def decode_bitcode(ephysdata):
    #%%
    sampling_rate = ephysdata['sampling_rate']
    if 'DI-TrialStart' in ephysdata.keys():
        trial_start_idxs = np.where(np.diff(ephysdata['DI-TrialStart'])==1)[0]
        trial_end_idxs_real = np.where(np.diff(ephysdata['DI-TrialStart'])==-1)[0]
        trial_end_idxs = np.concatenate([trial_start_idxs[1:],[len(ephysdata['DI-TrialStart'])-1]])
    else:
        trial_start_idxs = [0]
        trial_end_idxs = [-1]
    bitcode_trial_nums = list()
    
    for trial_start_idx,trial_end_idx in zip(trial_start_idxs,trial_end_idxs):
        if 'AI-Bitcode' in ephysdata.keys():
            bitcode_channel_now = (ephysdata['AI-Bitcode'][trial_start_idx:trial_end_idx]>3)*1
        else:
            for key in ephysdata.keys():
                if 'AI-' in key:
                    break
            #print('no bitcode channel found in wavesurfer file, falling back to {}'.format(key))
            bitcode_channel_now = (ephysdata[key][trial_start_idx:trial_end_idx]>3)*1
        pulse_start_idx = np.where(np.diff(bitcode_channel_now)==1)[0]
        pulse_end_idx = np.where(np.diff(bitcode_channel_now)==-1)[0]
        pulse_lengths = list()
        for pulse_start,pulse_end in zip(pulse_start_idx,pulse_end_idx):
            pulse_lengths.append((pulse_end-pulse_start)/sampling_rate)
        pulse_lengths = np.asarray(pulse_lengths)
        digits = ''
        if len(pulse_lengths)<20:
            #print('too few pulses in bitcode: {}'.format(len(pulse_lengths)))
            bitcode_trial_nums.append(np.nan)
            continue
        while len(pulse_lengths)>1:
            if len(pulse_lengths[:np.argmax(pulse_lengths[1:]<.01)+2]) == 3:
                digits += '1'
            else:
                digits += '0'
            pulse_lengths = pulse_lengths[np.argmax(pulse_lengths[1:]<.01)+2:]
        if len(digits)<11:
            bitcode_trial_nums.append(np.nan)
        else:
            bitcode_trial_nums.append(int(digits,2))
    
    ephysdata['bitcode_trial_nums'] = bitcode_trial_nums
    ephysdata['trial_start_indices'] = trial_start_idxs
    ephysdata['trial_end_indices'] = trial_end_idxs_real
    #%%
    return ephysdata