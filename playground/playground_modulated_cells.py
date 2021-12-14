#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 21:31:52 2021

@author: rozmar
"""
#%%

final_values = list()
mean_activity_modulations = list()
mean_activity_modulations_to_baseline = list()
baseline_length_real = int(sum(mask_to_use[:baseline_length]))

x_axis = np.arange(dFF.shape[1])
for dff in dFF: # plotting cumulative and average activity
    dff = dff[mask_to_use==1]
    if subtract_baseline_slope:
        baseline_rate = np.cumsum(dff)[baseline_length_real]/baseline_length_real
    else:
        baseline_rate = 0
    vector = np.cumsum(dff)-np.arange(len(dff))*baseline_rate
    final_values.append(vector[-1])
    dff_mean = np.convolve(dff,np.ones(moving_window_size)/moving_window_size,mode = 'same')#utils_plot.rollingfun(dff,moving_window_size)
    maxval = np.max(dff_mean[500:])
    maxidx = np.argmax(dff_mean[500:])+500
    minval = np.min(dff_mean[:maxidx])
    minidx = np.argmin(dff_mean[:maxidx])
    mean_activity_modulations.append(maxval-minval)
    mean_activity_modulations_to_baseline.append(maxval-np.mean(dff_mean[:baseline_length_real]))
  
  #%%
from utils import utils_plot
fig_fft = plt.figure()
ax_fft = fig_fft.add_subplot(211)
ax_so = fig_fft.add_subplot(212)
fftval = scipy.signal.stft(dff, fs = 30)
ax_fft.imshow(np.abs(fftval[2]*fftval[0][:,np.newaxis]),aspect = 'auto', extent = [fftval[1][0],fftval[1][-1],fftval[0][0],fftval[0][-1]],origin = 'lower')
fft_matrix = np.abs(fftval[2]*fftval[0][:,np.newaxis])
so_vector = np.sum(fft_matrix[fftval[0]<=4,:],0)/np.sum(fft_matrix,0)
ax_so.plot(utils_plot.rollingfun(so_vector,30))
  #%%
  
order = np.argsort(final_values)[::-1]
neuron_idx_now = order[0]
dff  = np.median(Fneu,0)

fig = plt.figure()
ax_conditioned_neuron = fig.add_subplot(111)


ax_conditioned_neuron.plot(dff,'k-')
maxval = np.max(dff)
minval = np.min(dff)
scale = (maxval-minval)*.1


zposition_norm = zpos_mask - np.min(zpos_mask)
zposition_norm = zposition_norm/np.max(zposition_norm)

ax_conditioned_neuron.plot(lick_mask*scale-scale ,'b-',label = 'lick')#,color = 'purple')
ax_conditioned_neuron.plot(reward_mask *scale-scale*2,'r-',label = 'reward')
ax_conditioned_neuron.plot(motor_steps_mask/np.max(motor_steps_mask)*scale-scale*3,'k-',label = 'motor step')
ax_conditioned_neuron.plot(gocue_mask*scale-scale*2,'g-',label = 'go cue')#%% reward correlation of the conditioned neuron
ax_conditioned_neuron.plot(zposition_norm*scale-scale*4,'c-',label = 'z position: range {} steps'.format(np.max(zpos_mask)-np.min(zpos_mask)))#%% reward correlation of the conditioned neuron
ax_conditioned_neuron.legend()


# =============================================================================
# task_label = label(task_mask)
# for i in np.arange(1,np.max(task_label)):
#     idx = np.where(task_label==i)[0]
#     ax_conditioned_neuron.plot(idx,dFF[cond_s2p_idx,idx],'g-')
# reward_consumption_label = label(reward_consumtion_mask)    
# for i in np.arange(1,np.max(reward_consumption_label)):
#     idx = np.where(reward_consumption_label==i)[0]
#     ax_conditioned_neuron.plot(idx,dFF[cond_s2p_idx,idx],'r-')
# =============================================================================
ax_conditioned_neuron.plot(np.where(unreward_mask ==1)[0],np.zeros_like(np.where(unreward_mask ==1)[0])-1,'ko')
#%
#%
#dff = dFF[neuron_idx_now,:]
step_back = 50
step_forward = 350
motor_step_exclusion_window = 5 #frames
lick_exclusion_window = 5
reward_bin_size = 25






fig_trial_traces = plt.figure(figsize = [15,15])
spec2 = gridspec.GridSpec(ncols=4, nrows=6, figure=fig_trial_traces)
ax_gocue = fig_trial_traces.add_subplot(spec2[0, 0])
ax_gocue_std = fig_trial_traces.add_subplot(spec2[1, 0])
ax_gocue_mean = fig_trial_traces.add_subplot(spec2[2:, 0])
ax_reward = fig_trial_traces.add_subplot(spec2[0, 1])
ax_reward_std = fig_trial_traces.add_subplot(spec2[1, 1])
ax_reward_mean = fig_trial_traces.add_subplot(spec2[2:, 1])
ax_step = fig_trial_traces.add_subplot(spec2[0, 2])
ax_step_std = fig_trial_traces.add_subplot(spec2[1, 2])
ax_step_mean = fig_trial_traces.add_subplot(spec2[2:, 2])

ax_lick = fig_trial_traces.add_subplot(spec2[0, 3])
ax_lick_std = fig_trial_traces.add_subplot(spec2[1, 3])
ax_lick_mean = fig_trial_traces.add_subplot(spec2[2:, 3])

ax_gocue.set_title('Go cue')
ax_reward.set_title('Reward')
ax_step.set_title('Motor step')
ax_lick.set_title('Lick')


reward_traces = event_based_trace_selection(dff,np.where(reward_mask>0)[0],trial_number_mask,step_back,step_forward)
x = np.arange(-step_back,step_forward)/fs
for i,y in enumerate(reward_traces):
    color = plt.cm.inferno(i/reward_traces.shape[0])
    ax_reward.plot(x,y,alpha = .1,color = color)
ax_reward_std.plot(x,np.nanmean(reward_traces,0),'k-',linewidth = 3,alpha = 1)
ax_reward_std.fill_between(x,np.nanmean(reward_traces,0)-np.nanstd(reward_traces,0),np.nanmean(reward_traces,0)+np.nanstd(reward_traces,0),color = 'black',alpha = .5)


gocue_traces = event_based_trace_selection(dff,np.where(gocue_mask>0)[0],trial_number_mask,step_back,step_forward)
for i,y in enumerate(gocue_traces):
    color = plt.cm.inferno(i/gocue_traces.shape[0])
    ax_gocue.plot(x,y,alpha = .1,color = color)
ax_gocue_std.plot(x,np.nanmean(gocue_traces,0),'k-',linewidth = 3,alpha = 1)
ax_gocue_std.fill_between(x,np.nanmean(gocue_traces,0)-np.nanstd(gocue_traces,0),np.nanmean(gocue_traces,0)+np.nanstd(gocue_traces,0),color = 'black',alpha = .5)
#%
motorstep_indices = np.where(motor_steps_mask>0)[0]
motorstep_indices_diff = np.concatenate([[3600],np.diff(motorstep_indices)])
motorstep_indices = motorstep_indices[motorstep_indices_diff>motor_step_exclusion_window]
motorstep_traces = event_based_trace_selection(dff,motorstep_indices,trial_number_mask,step_back,step_forward)
for i,y in enumerate(motorstep_traces):
    color = plt.cm.inferno(i/motorstep_traces.shape[0])
    ax_step.plot(x,y,alpha = .1,color = color)
ax_step_std.plot(x,np.nanmean(motorstep_traces,0),'k-',linewidth = 3,alpha = 1)
ax_step_std.fill_between(x,np.nanmean(motorstep_traces,0)-np.nanstd(motorstep_traces,0),np.nanmean(motorstep_traces,0)+np.nanstd(motorstep_traces,0),color = 'black',alpha = .5)

lick_indices = np.where(lick_mask>0)[0]
lick_indices_diff = np.concatenate([[3600],np.diff(lick_indices)])
lick_indices = lick_indices[lick_indices_diff>lick_exclusion_window]
lick_traces = event_based_trace_selection(dff,lick_indices,trial_number_mask,step_back,step_forward)
for i,y in enumerate(lick_traces):
    color = plt.cm.inferno(i/lick_traces.shape[0])
    ax_lick.plot(x,y,alpha = .1,color = color)
ax_lick_std.plot(x,np.nanmean(lick_traces,0),'k-',linewidth = 3,alpha = 1)
ax_lick_std.fill_between(x,np.nanmean(lick_traces,0)-np.nanstd(lick_traces,0),np.nanmean(lick_traces,0)+np.nanstd(lick_traces,0),color = 'black',alpha = .5)

#%
x = np.arange(-step_back,step_forward)/fs
bin_num = len(np.arange(reward_traces.shape[0]/reward_bin_size+1))
prev_value = 0
for i in np.arange(reward_traces.shape[0]/reward_bin_size+1):
    y= np.nanmean(reward_traces[int(i*reward_bin_size):int((i+1)*reward_bin_size)],0) 
    color = plt.cm.inferno(i/bin_num)
    ax_reward_mean.plot(x,y+ prev_value,alpha = .8,color = color)##-y[step_back]
    prev_value += np.nanmax(y)
ax_reward_mean.vlines(0,ax_reward_mean.get_ylim()[0],ax_reward_mean.get_ylim()[1],'r')

x = np.arange(-step_back,step_forward)/fs
bin_num = len(np.arange(gocue_traces.shape[0]/reward_bin_size+1))
prev_value = 0
for i in np.arange(gocue_traces.shape[0]/reward_bin_size+1):
    y= np.nanmean(gocue_traces[int(i*reward_bin_size):int((i+1)*reward_bin_size)],0) 
    color = plt.cm.inferno(i/bin_num)
    ax_gocue_mean.plot(x,y+ prev_value,alpha = .8,color = color)##-y[step_back]
    prev_value += np.nanmax(y)
ax_gocue_mean.vlines(0,ax_gocue_mean.get_ylim()[0],ax_gocue_mean.get_ylim()[1],'r')

x = np.arange(-step_back,step_forward)/fs
bin_num = len(np.arange(motorstep_traces.shape[0]/reward_bin_size+1))
prev_value = 0
for i in np.arange(motorstep_traces.shape[0]/reward_bin_size+1):
    y= np.nanmean(motorstep_traces[int(i*reward_bin_size):int((i+1)*reward_bin_size)],0) 
    color = plt.cm.inferno(i/bin_num)
    ax_step_mean.plot(x,y+ prev_value,alpha = .8,color = color)##-y[step_back]
    prev_value += np.nanmax(y)
ax_step_mean.vlines(0,ax_step_mean.get_ylim()[0],ax_step_mean.get_ylim()[1],'r')

bin_num = len(np.arange(lick_traces.shape[0]/reward_bin_size+1))
prev_value = 0
for i in np.arange(lick_traces.shape[0]/reward_bin_size+1):
    y= np.nanmean(lick_traces[int(i*reward_bin_size):int((i+1)*reward_bin_size)],0) 
    color = plt.cm.inferno(i/bin_num)
    ax_lick_mean.plot(x,y+ prev_value,alpha = .8,color = color)##-y[step_back]
    prev_value += np.nanmax(y)
ax_lick_mean.vlines(0,ax_lick_mean.get_ylim()[0],ax_lick_mean.get_ylim()[1],'r')