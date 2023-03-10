# -*- coding: utf-8 -*-
"""
This script contains all the supplementary functions.

@author: Fotios Drakopoulos, UGent, Mar 2022
"""
from os import path
import scipy.io.wavfile as sp_wavfile
import numpy as np

def slice_1dsignal(signal, window_size, minlength, stride=0.5):
    """ 
    Return windows of the given signal by sweeping in stride fractions
    of window
    Slices that are less than minlength are omitted
    """
    n_samples = signal.shape[0]
    offset = int(window_size * stride)
    num_slices = (n_samples)
    slices = np.array([]).reshape(0, window_size) # initialize empty array
    for beg_i in range(0, n_samples, offset):
        end_i = beg_i + window_size
        if n_samples - beg_i < minlength :
            break
        if end_i <= n_samples :
            slice_ = np.array([signal[beg_i:end_i]])
        else :
            slice_ = np.concatenate((np.array([signal[beg_i:]]), np.zeros((1, end_i - n_samples))), axis=1)
        slices = np.concatenate((slices, slice_), axis=0)
    return slices #.astype('float32')

def reconstruct_wav(wavmat, stride_factor=0.5):
  """
  Reconstructs the audiofile from sliced matrix wavmat
  """
  window_length = wavmat.shape[1]
  window_stride = int(stride_factor * window_length)
  wav_length = (wavmat.shape[0] -1 ) * window_stride + window_length
  wav_recon = np.zeros((1,wav_length))
  #print ("wav recon shape " + str(wav_recon.shape))
  for k in range (wavmat.shape[0]):
    wav_beg = k * window_stride
    wav_end = wav_beg + window_length
    wav_recon[0, wav_beg:wav_end] += wavmat[k, :]

  # now compute the scaling factor for multiple instances
  noverlap = int(np.ceil(1/stride_factor))
  scale_ = (1/float(noverlap)) * np.ones((1, wav_length))
  for s in range(noverlap-1):
    s_beg = s * window_stride
    s_end = s_beg + window_stride
    scale_[0, s_beg:s_end] = 1/ (s+1)
    scale_[0, -s_beg - 1 : -s_end:-1] = 1/ (s+1)

  return wav_recon * scale_
  
def rms (x, axis=None):
    # compute rms of a matrix
    sq = np.mean(np.square(x), axis = axis)
    return np.sqrt(sq)
    
def next_power_of_2(x):
    return 1 if x == 0 else int(2**np.ceil(np.log2(x)))

def wavfile_read(wavfile,fs=[]):
    # read a wavfile and normalize it
    # if fs is given the signal is resampled to the given sampling frequency
    fs_signal, speech = sp_wavfile.read(wavfile)
    if not fs:
        fs=fs_signal

    if speech.dtype != 'float32' and speech.dtype != 'float64':
        if speech.dtype == 'int16':
            nb_bits = 16 # -> 16-bit wav files
        elif speech.dtype == 'int32':
            nb_bits = 32 # -> 32-bit wav files
        max_nb_bit = float(2 ** (nb_bits - 1))
        speech = speech / (max_nb_bit + 1.0) # scale the signal to [-1.0,1.0]

    if fs_signal != fs :
        signalr = sp_sig.resample_poly(speech, fs, fs_signal)
    else:
        signalr = speech

    return signalr, fs

