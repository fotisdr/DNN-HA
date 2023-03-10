"""
DNN-based hearing-aid strategy for real-time processing: One size fits all

This script can be used to compensate for the elevated hearing thresholds of 
a listener using a DNN-based HA model for real-time sound processing.
By default, the script is used to process an example sentence presented at 70 
dB SPL for a high-frequency sloping audiogram (30-35 dB HL). The parameters in
lines 44-54 can be adjusted according to the desired usage. 

A wavfile is specified as the input (wavfile_input) and is processed based on
the audiogram of a listener (audiogram_input) at the desired intensity level L.
The DNN-HA model was trained to use audiogram inputs defined at 8 frequencies:
[125,250,500,1000,2000,3000,4000,6000,8000] Hz
Noise can also be added to the input stimulus at a desired SNR, while the 
frame_size parameter can be used to process sound in frames.
The save_wav parameter specifies the folder to save the processed stimulus.

Fotios Drakopoulos, UGent, March 2023.
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, model_from_json, load_model
import tensorflow.keras.backend as K

import time
import os, sys
import scipy.signal as sp_sig
import scipy.io as sp_io
import scipy.io.wavfile as sio_wav
from extra_functions import *

import matplotlib.pyplot as plt

#tf.compat.v1.disable_eager_execution() # speeds up execution in Tensorflow v2

# define the DNN-HA models to be evaluated 
parentdir = ''
modeldirs = [
'CNN-HA-12layers',
]

# Define the inputs
audiogram_input = [30,30,30,30,31.6,32.58,33.28,34.28,35] # the audiogram (in dB HL) across the 8 frequencies
wavfile_input = '00131.wav' # wavfile to process - 00131 corresponds to the Flemish sentence 'David draag drie gele boten'
# Input parameters
L = 70 # sound pressure level of the sentence (dB SPL)
SNR = '' # when a number is given, noise is added at the specified SNR - if empty then no noise is added
wavfile_noise = '' # the location of the noise wavfile to be used - if empty then white noise is used
save_wav = 'wavfiles' # path to save the wavfile before and after processing - if empty then the wavfiles are not saved
wav_dBref = 110 # dB reference for the saved wavfiles
# Frame processing
frame_size = 0 # frame size (best defined as a power of two) - use 0 for processing the whole signal, minimum frame is 64 samples for the current model
overlap = 0 # values of [0,1) - 0 adds no overlap, 0.5 corresponds to 50%

# Fixed parameters for the trained DNN-HA architecture
audiogram_frequencies = [125,250,500,1000,2000,3000,4000,6000,8000] # Hz
fs_model = 20e3 # sampling frequency for the DNN-HA model
Nenc = 6
p0 = 2e-5 # Pa reference

# Plotting limits (in ms) - 0 to 2000 for the full sentence
tstart = 490
tend = 590

#%%
speechsignal, fs_speech = wavfile_read(wavfile_input) 
if fs_model != fs_speech :
    print("Resampling to " + str(fs_model) + "Hz")
    stim_full = sp_sig.resample_poly (speechsignal, fs_model, fs_speech)
else :
    stim_full = speechsignal

stim_length = stim_full.size
rem = stim_length % (2**Nenc)
if rem:
    stim_length = stim_length + int(2**Nenc-rem)

stim = np.zeros((1,stim_length))
stim[0,:stim_full.size] = p0 * 10**(L/20) * stim_full / rms(stim_full)

if SNR != '' and SNR != []:
    stim_clean = stim.copy()
    if wavfile_noise: # noise wavfile is provided
        noise_full, fs_noise = wavfile_read(wavfile_noise)
        if fs_model != fs_noise:
            noise_full = sp_sig.resample_poly (noise_full, fs_model, fs_noise)

        noisei = np.random.randint(0,noise_full.size-stim_length)
        noise = noise_full[noisei:noisei+stim_length]
    else: # Gaussian noise
        noise = np.random.normal(size=stim_length)

    speechrms=rms(stim)
    noiserms=rms(noise)
    ratio=(speechrms/noiserms)*np.sqrt(10**(-SNR/10))
    stim[0,:] = stim[0,:] + ratio*noise

stim = np.expand_dims(stim, axis=2)
#print("Stimulus shape to be processed", stim.shape)
print("Unprocessed stimulus level: %.2f dB SPL" % (20*np.log10(rms(stim[0],axis=None))-20*np.log10(p0)))

# Initialize parameters
t_c = np.arange(stim.shape[1]) / fs_model
istart, _ = min(enumerate(1000*t_c), key=lambda x: abs( x [1]- tstart))
iend, _ = min(enumerate(1000*t_c), key=lambda x: abs( x [1]- tend))

legend_labels=[str(int(L))]
stim_labels = ["Unprocessed"]
stim_all = np.zeros((stim.shape[1], len(modeldirs)+1))
stim_all[:,0] = stim[0,:,0]

if save_wav:
    if not os.path.exists(save_wav):
        os.makedirs(save_wav)
    sio_wav.write(save_wav + '/' + wavfile_input[:-4] + '_20k.wav',int(fs_model),stim[0,:,0]*10**(-(wav_dBref+20*np.log10(p0))/20))

# load the compensation models
stimp = np.zeros((len(modeldirs),stim.shape[0],stim.shape[1],stim.shape[2]))
for modeli, modeldir in enumerate(modeldirs):
    dirtitle = modeldir
    
    weights_name = "/Gmodel.h5"
    print ("loading model from " + modeldir)
    json_file = open (modeldir + "/Gmodel.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()

    modelp = model_from_json(loaded_model_json)
    modelp.load_weights(modeldir + weights_name)
    
    if frame_size: # process sound in frames
        stim_cropped = slice_1dsignal(stim[0,:,0], frame_size, 0, stride=1-overlap)
        stim_cropped = np.reshape(stim_cropped, (-1,frame_size,1))
    else:
        stim_cropped = stim
        
    # repeat the audiogram input to match the bottleneck dimension of stim_cropped
    audiogram_rep = np.tile(audiogram_input, (stim_cropped.shape[0],int(stim_cropped.shape[1]/(2**Nenc)),1))
    
    # process sound
    t = time.time()
    stimp_cropped = modelp.predict([stim_cropped,audiogram_rep])
    print('Processing time for %d frames (%d samples): %.4f s' % (stimp_cropped.shape[0],stim_cropped.shape[1],(time.time()-t)))

    if frame_size:
        stimp_cropped = reconstruct_wav(stimp_cropped[:,:,0], stride_factor=1-overlap)
    #print(stimp_cropped.shape)

    stimp[modeli,0] = np.reshape(stimp_cropped[:,:stim.shape[1]], (1,-1,1))

    stim_all[:,1+modeli] = stimp[modeli,0,:,0]
    #stim_labels.append(dirtitle)
    stim_labels.append('DNN-HA processed')
    
    print(dirtitle + ": %.2f dB SPL" % (20*np.log10(rms(stimp[modeli,0],axis=None))-20*np.log10(p0)))

    if save_wav:
        sio_wav.write(save_wav + '/' + wavfile_input[:-4] + '_' + dirtitle + '_20k.wav',int(fs_model),stimp[modeli,0,:,0]*10**(-(wav_dBref+20*np.log10(p0))/20))

# Plot the time-domain signal before and after processing
plt.figure(1)
lineObj = plt.plot(1000*t_c, stim_all,linewidth=0.5)
plt.title('Audio Input')
plt.xlabel('Time [ms]'), plt.ylabel('Sound pressure [Pa]'), plt.xlim((tstart, tend))
plt.legend(lineObj, stim_labels)
plt.grid(linewidth=0.4,linestyle='--')
plt.tight_layout()

# Plot the spectrogram before and after processing
f, axarr = plt.subplots(stim_all.shape[1]+1, sharex=True)
if len(modeldirs):
    axarr[0].plot(1000*t_c, stim_all[:,1],'c',linewidth=0.5,label='DNN-HA processed')
axarr[0].plot(1000*t_c, stim_all[:,0],'k',linewidth=0.5,label='Unprocessed')
axarr[0].legend(frameon=False)
axarr[0].set_title('Audio Input')
#axarr[0].set_xlim((tstart, tend)) # commented for the whole signal
extent = axarr[0].get_window_extent().transformed(f.dpi_scale_trans.inverted())
axarr[0].set_ylabel('Sound pressure [Pa]')

for j in range(stim_all.shape[1]):
    freqs, times, spec = sp_sig.spectrogram(stim_all[:,j], fs_model, nperseg=256)
    if j == 0:
        vmin = np.min(20*np.log10(spec))
        vmax = np.max(20*np.log10(spec))
    cax = axarr[j+1].imshow(np.flip(20*np.log10(spec),axis=0), cmap='turbo', extent=(0,1000*t_c[-1],freqs[0]/1000,freqs[-1]/1000), aspect="auto", vmin=vmin, vmax=vmax) 
    axarr[j+1].set_title('Spectrogram - ' + stim_labels[j])
    axarr[j+1].set_ylabel('Frequency [kHz]')
    #axarr[j+1].set_xlim((tstart, tend))
    #plt.colorbar(cax)
axarr[j+1].set_xlabel('Time [ms]')
f.tight_layout()

f, axarr = plt.subplots(2) #, sharex=True)
axarr[0].semilogx(audiogram_frequencies, audiogram_input, 'kx-')
axarr[0].set_xlim((audiogram_frequencies[0],audiogram_frequencies[-1]))
axarr[0].set_ylim((-10,90))
axarr[0].set_xlabel('Frequency [kHz]')
axarr[0].set_ylabel('Gain loss [dB HL]')
axarr[0].set_xticks(audiogram_frequencies)
axarr[0].set_xticklabels(np.array(audiogram_frequencies)/1000)
axarr[0].invert_yaxis()
axarr[0].grid(linewidth=0.4,linestyle='--')
axarr[0].set_title('Audiogram input')

# Plot the magnitude spectrum before and after processing
# Adopted from https://github.com/jmrplens/PyOctaveBand
try:
    import PyOctaveBand
    for j in range(stim_all.shape[1]):
        spl, freq = PyOctaveBand.octavefilter(stim_all[:,j], fs=fs_model, fraction=3, order=6, limits=[10, 8000], show=0)
        axarr[1].semilogx(freq, spl, linewidth=0.5)  
    axarr[1].set_xlabel('Frequency [Hz]')
    axarr[1].set_ylabel('Magnitude [dB]')
    axarr[1].set_title('Audio input')
    axarr[1].legend(stim_labels)
    axarr[1].grid(linewidth=0.4,linestyle='--')
    axarr[1].set_xlim((10,10000))
except:
    print('Add PyOctaveBand.py to the directory to plot the frequency spectra of the stimuli!')
f.tight_layout()

plt.show()

