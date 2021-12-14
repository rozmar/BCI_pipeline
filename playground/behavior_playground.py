import datajoint as dj
from pipeline import pipeline_tools,lab,experiment,videography,imaging
from pipeline.ingest import datapipeline_metadata
from utils import utils_pybpod, utils_ephys, utils_imaging,utils_pipeline, utils_plot

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import datetime
import numpy as np
import time as timer
import os
from scipy.io import loadmat
%matplotlib qt


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 21}
matplotlib.rc('font', **font)
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Tahoma']
#utils_pipeline.export_pybpod_files(overwrite = True)
#%%
utils_plot.plot_behavior_session_stats()
#%%
session_key_wr = {'wr_id':'BCI14', 'session':10}#2
#session_key_wr = {'wr_id':'BCI15', 'session':10}
#session_key_wr = {'wr_id':'BCI07', 'session':43}
#session_key_wr = {'wr_id':'BCI10', 'session':20}
#session_key_wr = {'wr_id':'BCI06', 'session':5}
moving_window = 30
utils_plot.plot_behavior_session(session_key_wr,moving_window)
#%%
session_key_wr = {'wr_id':'BCI14', 'session':10}
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 11}
matplotlib.rc('font', **font)
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Tahoma']
utils_plot.plot_session_and_licks(session_key_wr)

#%% dff heatmap relative to GO cue and reward
session_key_wr = {'wr_id':'BCI14', 'session':10}
utils_plot.plot_go_reward_heatmap_dff(session_key_wr)

#%% licks relative to go cue and reward
session_key_wr = {'wr_id':'BCI14', 'session':2}
utils_plot.plot_session_dlc_heatmap(session_key_wr= session_key_wr,bodypart = 'tongue_tip',video_axis = 'y')

#%% lick-locked dff activity
utils_plot.plot_lick_triggered_dff_over_session(session_key_wr)
#%%

#%% sound  - noise reduction - quite a dead-end
import IPython
from scipy.io import wavfile
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import librosa


import time
from datetime import timedelta as td


def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)


def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _db_to_amp(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)


def plot_spectrogram(signal, title):
    fig, ax = plt.subplots(figsize=(20, 4))
    cax = ax.matshow(
        signal,
        origin="lower",
        aspect="auto",
        cmap=plt.cm.seismic,
        vmin=-1 * np.max(np.abs(signal)),
        vmax=np.max(np.abs(signal)),
    )
    fig.colorbar(cax)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_statistics_and_filter(
    mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
):
    fig, ax = plt.subplots(ncols=2, figsize=(20, 4))
    plt_mean, = ax[0].plot(mean_freq_noise, label="Mean power of noise")
    plt_std, = ax[0].plot(std_freq_noise, label="Std. power of noise")
    plt_std, = ax[0].plot(noise_thresh, label="Noise threshold (by frequency)")
    ax[0].set_title("Threshold for mask")
    ax[0].legend()
    cax = ax[1].matshow(smoothing_filter, origin="lower")
    fig.colorbar(cax)
    ax[1].set_title("Filter for smoothing Mask")
    plt.show()


def removeNoise(
    audio_clip,
    noise_clip,
    n_grad_freq=2,
    n_grad_time=4,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    n_std_thresh=1.5,
    prop_decrease=1.0,
    verbose=False,
    visual=False,
):
    """Remove noise from audio based upon a clip containing only noise

    Args:
        audio_clip (array): The first parameter.
        noise_clip (array): The second parameter.
        n_grad_freq (int): how many frequency channels to smooth over with the mask.
        n_grad_time (int): how many time channels to smooth over with the mask.
        n_fft (int): number audio of frames between STFT columns.
        win_length (int): Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
        hop_length (int):number audio of frames between STFT columns.
        n_std_thresh (int): how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
        prop_decrease (float): To what extent should you decrease noise (1 = all, 0 = none)
        visual (bool): Whether to plot the steps of the algorithm

    Returns:
        array: The recovered signal with noise subtracted

    """
    if verbose:
        start = time.time()
    # STFT over noise
    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))  # convert to dB
    # Calculate statistics over noise
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
    if verbose:
        print("STFT on noise:", td(seconds=time.time() - start))
        start = time.time()
    # STFT over signal
    if verbose:
        start = time.time()
    sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))
    if verbose:
        print("STFT on signal:", td(seconds=time.time() - start))
        start = time.time()
    # Calculate value to mask dB to
    mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))
    print(noise_thresh, mask_gain_dB)
    # Create a smoothing filter for the mask in time and frequency
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    # calculate the threshold for each frequency/time bin
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T
    # mask if the signal is above the threshold
    sig_mask = sig_stft_db < db_thresh
    if verbose:
        print("Masking:", td(seconds=time.time() - start))
        start = time.time()
    # convolve the mask with a smoothing filter
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease
    if verbose:
        print("Mask convolution:", td(seconds=time.time() - start))
        start = time.time()
    # mask the signal
    sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )  # mask real
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
        1j * sig_imag_masked
    )
    if verbose:
        print("Mask application:", td(seconds=time.time() - start))
        start = time.time()
    # recover the signal
    recovered_signal = _istft(sig_stft_amp, hop_length, win_length)
    recovered_spec = _amp_to_db(
        np.abs(_stft(recovered_signal, n_fft, hop_length, win_length))
    )
    if verbose:
        print("Signal recovery:", td(seconds=time.time() - start))
    if visual:
        plot_spectrogram(noise_stft_db, title="Noise")
    if visual:
        plot_statistics_and_filter(
            mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
        )
    if visual:
        plot_spectrogram(sig_stft_db, title="Signal")
    if visual:
        plot_spectrogram(sig_mask, title="Mask applied")
    if visual:
        plot_spectrogram(sig_stft_db_masked, title="Masked signal")
    if visual:
        plot_spectrogram(recovered_spec, title="Recovered spectrogram")
    return recovered_signal
#%% audio
from utils import utils_ephys

wsfile = '/home/rozmar/Data/Wavesurfer/DOM3-MMIMS/BCI_14/2021-08-03/baseline_0001.h5'
wsdata = utils_ephys.load_wavesurfer_file(wsfile)
wsdata = wsdata[0]

#%%
trial_needed = 2
trial_span = 5
sound = wsdata['AI-Microphone'][wsdata['trial_start_indices'][trial_needed]:wsdata['trial_end_indices'][trial_needed+trial_span]:]
sound = sound/np.max(sound)

#%%
noise_edges_s = [[15,18],[34,38],[53,57],[72,76]]
noise_concat = []
for noise_edges_now in noise_edges_s:
    noise_concat.append(wsdata['AI-Microphone'][int(noise_edges_now[0]*wsdata['sampling_rate']):int(noise_edges_now[1]*wsdata['sampling_rate'])])
noise = np.concatenate(noise_concat)
sound_denoised_now = removeNoise(audio_clip=sound, noise_clip=noise,verbose=True,visual=True)



#%%
import pylab
#Audio(sound,rate = 20000)
plt.figure()
pylab.specgram(sound,Fs = 20000,mode = 'psd',cmap = 'jet',scale = 'dB',scale_by_freq= False,NFFT = 128, noverlap = 100)
plt.figure()
pylab.specgram(sound_denoised_now,Fs = 20000,mode = 'psd',cmap = 'jet',scale = 'dB',scale_by_freq= False,NFFT = 128, noverlap = 100)
#plt.yscale('log')
#%%
from scipy.io.wavfile import write
scaled = np.int16(sound_denoised_now/np.max(np.abs(sound_denoised_now)) * 32767)
write('test_denoised.wav', 20000, scaled)
scaled = np.int16(sound/np.max(np.abs(sound)) * 32767)
write('test.wav', 20000, scaled)
#%% are bias frame times reliable?

session = experiment.Session() &'subject_id = 478612'&'session = 33'
session_date = str(session.fetch1('session_date'))
wr_id = (lab.WaterRestriction()&session).fetch1('water_restriction_number')
wr_id  = wr_id[:3]+'_'+wr_id[3:] # stupid thing, directories are not accurate..
bpod_file_names = np.unique((experiment.TrialMetaData().BpodMetaData()&session).fetch('bpod_file_name'))
setup = session.fetch1('rig')

if setup == 'DOM3-2p-Resonant-MMIMS':
    setup = 'DOM3-MMIMS'
else:
    setup = 'Bergamo-2P'
camera = 'bottom'

# load wavesurfer stuff
ws_dir = os.path.join(dj.config['locations.elphysdata_wavesurfer'],setup,wr_id,session_date)
ws_files = os.listdir(ws_dir)
wsdata_all = list()
for ws_file in ws_files:
    wsdata = utils_ephys.load_wavesurfer_file(os.path.join(ws_dir,ws_file))
    wsdata_all.extend(wsdata)
ws_timestamp_list = list()
for wsdata in wsdata_all:
    ws_timestamp_list.append(datetime.datetime(wsdata['timestamp'][0], wsdata['timestamp'][1], wsdata['timestamp'][2],wsdata['timestamp'][3],wsdata['timestamp'][4],wsdata['timestamp'][5]+wsdata['sweep_timestamp']))
ws_order = np.argsort(ws_timestamp_list)

#%% go trial by trial and find videos
import cv2 as cv
import pylab
ws_trial_end_indices = np.concatenate([wsdata['trial_start_indices'][1:],[-1]])
ws_exposure_name = 'DI-Exposition{}Camera'.format(camera.capitalize())
ws_sweep_idx = 0
wsdata = wsdata_all[ws_sweep_idx]
prev_trial = -1
first_frame_times = list()
for trial in experiment.TrialMetaData().BpodMetaData()*experiment.TrialMetaData().VideoMetaData()&session:
    if trial['bpod_trial_num']<prev_trial:
        ws_sweep_idx += 1
        wsdata = wsdata_all[ws_sweep_idx]
    prev_trial = trial['bpod_trial_num']
    
    if not any(np.asarray(wsdata['bitcode_trial_nums'])==trial['bpod_trial_num']):
        print('wavesurfer trial not found,skipping')
        continue
    
    video_dir = os.path.join('/home/rozmar/Data/Behavior_videos/',setup,wr_id,camera,trial['bpod_file_name'],'trial_{0:03d}'.format(trial['bpod_trial_num']))
    video_files = os.listdir(video_dir)
    if len(video_files)>2:
        print('too many video files, skipping')
        continue
    for video_file in video_files:
        if video_file.endswith('txt'):
            from numpy import loadtxt
            exposition_times = loadtxt(os.path.join(video_dir,video_file), comments="#", delimiter=",", unpack=False)
        elif video_file.endswith('avi'):
            video = cv.VideoCapture(os.path.join(video_dir,video_file))
            frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
            video.release()
        else:
            print('unknown file format {}'.format(video_ffile))
           
    ws_trial_idx = np.where(np.asarray(wsdata['bitcode_trial_nums'])==trial['bpod_trial_num'])[0][0]
    
    frametriggers = wsdata[ws_exposure_name][wsdata['trial_start_indices'][ws_trial_idx]:ws_trial_end_indices[ws_trial_idx]]
    #trialon = wsdata['DI-TrialStart'][wsdata['trial_start_indices'][ws_trial_idx]:ws_trial_end_indices[ws_trial_idx]]
    
    time = np.arange(len(frametriggers))/wsdata['sampling_rate']
    frame_start = time[np.where(np.diff(frametriggers)>.5)[0]]
    frame_end = time[np.where(np.diff(frametriggers)<-.5)[0]]
    
    
    if len(frame_start) == len(frame_end) and len(frame_start)-len(exposition_times) == 2:
        frame_mid = (frame_start+frame_end)/2
        first_frame_time = frame_start[0]
        first_frame_times.append(first_frame_time)
        frame_time_diffs = frame_start[2:]-exposition_times
        plt.plot(exposition_times,frame_time_diffs*1000,'ko',alpha = .1)
        
        
# =============================================================================
#         sound=  wsdata['AI-Microphone'][wsdata['trial_start_indices'][ws_trial_idx]:ws_trial_end_indices[ws_trial_idx]]
#         fig = plt.figure()
#         ax_specgram = fig.add_subplot(4,1,2)
#         pylab.specgram(sound,Fs = 20000,mode = 'psd',cmap = 'jet',scale = 'dB',scale_by_freq= False)
#         ax_frames = fig.add_subplot(4,1,3,sharex = ax_specgram)
#         ax_frames .plot(np.arange(len(frametriggers))/20000,frametriggers)
#         ax_sound = fig.add_subplot(4,1,1,sharex = ax_specgram)
#         ax_sound.plot(np.arange(len(frametriggers))/20000,sound)
#         ax_framenum = fig.add_subplot(4,1,4,sharex = ax_specgram)
#         ax_framenum.plot(frame_start,np.arange(len(frame_start))+1,'ko')
#         break
# =============================================================================
    
    
    
# =============================================================================
#     if prev_trial>10:
#         break
# =============================================================================
#%% show movie frame by frame
cap = cv.VideoCapture(os.path.join(video_dir,video_file))
frame_seq = 9


frame_no = (frame_seq /frame_count)



cap.set(cv.CAP_PROP_POS_FRAMES, frame_seq)
res, frame = cap.read()

# =============================================================================
# cap.set(2,frame_no);
# 
# ret, frame = cap.read()
# =============================================================================
#Set grayscale colorspace for the frame. 
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#Cut the video extension to have the name of the video
my_video_name = video_file.split(".")[0]

#Display the resulting frame
cv.imshow(my_video_name+' frame '+ str(frame_seq),gray)

#Set waitKey 
#cv.waitKey()

#Store this frame to an image
#cv2.imwrite(my_video_name+'_frame_'+str(frame_seq)+'.jpg',gray)

# When everything done, release the capture
cap.release()
#cv2.destroyAllWindows()




#%%  check if lick times are consistent between bpod and wavesurfer
 
from utils import utils_ephys

wsfile = '/home/rozmar/Data/Wavesurfer/DOM3-MMIMS/BCI_14/2021-08-03/baseline_0001.h5'
wsfile = '/home/rozmar/Data/Wavesurfer/DOM3-MMIMS/BCI_14/2021-08-03/Neuron1vs2_0001.h5'
wsdata = utils_ephys.load_wavesurfer_file(wsfile)
wsdata = wsdata[0]
#%
from skimage.measure import label
session = experiment.Session() &'subject_id = 478612'&'session = 33'
licktimes_diff = list()
licktimes_bpod_all = list()
for trialnum,startidx,endidx in zip(wsdata['bitcode_trial_nums'],wsdata['trial_start_indices'],wsdata['trial_end_indices']):
    if np.isnan(trialnum):
        continue
    licktimes_bpod,realtrialnums = (experiment.TrialMetaData().BpodMetaData()*experiment.ActionEvent() &session&'bpod_trial_num = {}'.format(trialnum)).fetch('action_event_time','trial')
    licktimes_bpod = np.asarray(licktimes_bpod,float)
    needed = realtrialnums == np.max(realtrialnums)
    licktimes_bpod = licktimes_bpod[needed]
    lick_mask_ws = label(wsdata['AI-Lick'][startidx:endidx]>3)
    licktimes_ws = list()
    for i in range(1,np.max(lick_mask_ws)+1):
        licktimes_ws.append(np.argmax(lick_mask_ws==i)/wsdata['sampling_rate'])
    if len(licktimes_ws) == len(licktimes_bpod):
        plt.plot(licktimes_ws,licktimes_bpod,'ko')
        licktimes_diff.extend(licktimes_bpod*1000-np.asarray(licktimes_ws)*1000)
        licktimes_bpod_all.extend(licktimes_bpod)
# =============================================================================
#         if any(np.abs(licktimes_bpod*1000-np.asarray(licktimes_ws)*1000)>1.5):
#             break
# =============================================================================
            
    #break
    
plt.figure()
plt.plot(licktimes_bpod_all,licktimes_diff,'ko',alpha = .1)
plt.gca().set_xlabel('Time in trial (s)')
plt.gca().set_ylabel('Bpod-NI error (ms)')
#%%
# =============================================================================
# %matplotlib qt
# import matplotlib
# from matplotlib.backends.backend_pgf import FigureCanvasPgf
# matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
# #matplotlib.use('ps')
# font = {'family' : 'normal',
#         'weight' : 'normal',
#         'size'   : 12}
# matplotlib.rc('font', **font)
# matplotlib.rcParams['font.family'] = 'sans-serif'
# matplotlib.rcParams['font.sans-serif'] = ['Helvetica']
# # =============================================================================
# # pgf_with_latex = {
# #     "text.usetex": True,            # use LaTeX to write all text
# #     "pgf.rcfonts": False,           # Ignore Matplotlibrc
# #     "pgf.preamble": [
# #         r'\usepackage{color}'     # xcolor for colours
# #     ]
# # }
# # matplotlib.rcParams.update(pgf_with_latex)
# # =============================================================================
# #%%
# 
# =============================================================================





