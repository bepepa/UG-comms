#! /usr/bin/env python3

# File: pulse_shaping.py - pulse_shaping and matched filtering; 
#       

import numpy as np

#
# Various pulse shapes
#
# Note: pulses may be scaled to produce samples of a continuous-time pulse or like a discrete-time pulse.
# For a continuous-time pulse, the integral over the square of the pulse equals 1. For a discrete-time pulse,
# the sum over the square of the samples equals 1

def sine_squared_pulse(fsT, fs=1):
    """synthesize a sine squared pulse
    
    Inputs:
    -------
    fsT: samples per symbol period
    fs: if a sampling rate is specified, then pulses will be scaled like samples of a continuous-time pulse, i.e.,
        such that \int p^2(t) dt = 1

    Returns:
    --------
    pulse of length fsT samples
    """
    nn = np.arange(fsT)
    pp = np.sqrt(8*fs/(3*fsT)) * np.sin(np.pi * nn/fsT)**2

    return pp

def rect_pulse(fsT, fs=1):
    """synthesize a rectangular pulse
    
    Inputs:
    -------
    fsT: samples per symbol period
    fs: if a sampling rate is specified, then pulses will be scaled like samples of a continuous-time pulse, i.e.,
        such that \int p^2(t) dt = 1

    Returns:
    --------
    pulse of length fsT samples
    """
    pp = np.sqrt(fs/(fsT)) * np.ones(fsT)

    return pp

def half_sine_pulse(fsT, fs=1):
    """synthesize a half-sine pulse
    
    Inputs:
    -------
    fsT: samples per symbol period
    fs: if a sampling rate is specified, then pulses will be scaled like samples of a continuous-time pulse, i.e.,
        such that \int p^2(t) dt = 1

    Returns:
    --------
    pulse of length fsT samples
    """
    nn = np.arange(fsT)
    pp = np.sqrt(2*fs/(fsT)) * np.sin(np.pi * nn/fsT)

    return pp


#
# pulse shaping
#
def pulse_shape(syms, pp, fsT):
    """perform pulse shaping for a sequence of symbols"""

    # upsample the symbol sequence
    N_dd = (len(syms)-1) *fsT + 1  # this avoids extra zeros at end
    dd = np.zeros(N_dd, dtype=syms.dtype)
    dd[0::fsT] = syms

    # convolve with pulse
    return np.convolve(dd, pp)

