import numpy as np
from scipy.signal import butter, lfilter, cheby1, filtfilt
from scipy.optimize import curve_fit
import allantools as al
from DigWorker import *
from NumericalTools import *

pi = np.pi

#Analysis  
def find_he_xe_freqs(x_data, y_data, sample_freq):
    '''
    Finds frequency of peak for xenon and helium. Does by finding peak of 
    fft of data where peak is around 14 for xe and 39 for he. Can make better
    by applying a fit around
    the guess area and finding peak of fit.
    
    -Inputs-
    x_data: time data (np.array)
    y_data: signal data (np.array)
    sample_freq: 1/sample_freq in Hz (float)
    
    -Returns-
    he_freq: frequency of He-3 (float)
    xe_freq: frequency of Xe-129 (float)
    freqs: frequencies from FFT (np.array)
    fourier: amplitudes coordinating to freqs in FFT response (np.array)
    '''
    fourier = np.abs(np.fft.fft(y_data)) #amplitudes of FFT response
    x_n = x_data.size
    
    #creates axis of frequencies corresponding to fft peaks
    freqs = np.fft.fftfreq(x_n,1.0/sample_freq) 

    he = [i for i in freqs if 36 < i < 45] #regions to search for he 
    he_loc = np.where(freqs == he[0])[0][0] #index of beginning of he region
    he_loc_end = np.where(freqs == he[-1])[0][0] #index of end of he region

    xe = [i for i in freqs if 12 < i < 24] #same as he
    xe_loc = np.where(freqs == xe[0])[0][0]
    xe_loc_end = np.where(freqs == xe[-1])[0][0]

    he_max = np.amax(fourier[he_loc:he_loc_end])#he_maxes[0]

    xe_max = np.amax(fourier[xe_loc:xe_loc_end])

    he_max_loc = np.where(fourier == he_max)[0][0] #index of peak
    xe_max_loc = np.where(fourier == xe_max)[0][0] #same
    
    he_freq = freqs[he_max_loc] #actual frequency of peak found
    xe_freq = freqs[xe_max_loc] #same
        
    return he_freq, xe_freq, freqs, fourier

#fix data length

def skip_artifacts(data, sample_freq, cutoff, time_steps = 5):
    '''Used to skip any artifacts created from a lowpass filter. Uses
    5 time steps as a standard'''
    skip = int(time_steps / cutoff * sample_freq)
    return data[skip:]
    
# Demodulate

def sin_stat(time, f0, method = 'sin'):
    '''Creates sine and cosine data to be used for manipulating data'''
    if method == 'sin':
        #2 in front to cancel out 1/2 later
        return [2.0 * np.sin(2*pi*f0 * t) for t in time] 
    if method == 'cos':
        return [2.0 * np.cos(2*pi*f0 * t) for t in time]

def demodulate(time, signal, f0, sample_freq, cutoff = 1): #cutoff is in hz
    '''
    first creates list of sine and cosine data to numerically multiply all 
    data by. Then Makes X, Y, which are data times cosine data and sine data
    respectively. Then does a lowpass to only look at signal close to actual 
    frequency of gas. Then finds amplitude and phase of data.
    '''

    C = sin_stat(time, f0, method = 'cos')
    S = sin_stat(time, f0, method = 'sin')
    
    X = [d * c for d,c in zip(signal,C)]
    Y = [d * s for d,s in zip(signal,S)]
    
    X_low = butter_lowpass_filter(X, cutoff, sample_freq)
    Y_low = butter_lowpass_filter(Y, cutoff, sample_freq)
    
    amp = [np.sqrt(x**2 + y**2) for x,y in zip(X_low, Y_low)]
    phase = [np.arctan(-y/x) for y,x in zip(Y_low,X_low)]
    
    phase = arctan_unwrap(phase)
    return (amp,phase)
    
def demodulate_for_both_species(time, signal, sample_freq, he_cutoff = 2, \
                                xe_cutoff = 1):
    '''
    Performs demodulation on for both He-3 and Xe-129. Does this by finding
    their average larmor frequency and uses these in the demodulation code.
    '''
    hef0, xef0, f1,f2 = find_he_xe_freqs(time,signal, sample_freq)
    he_amp, he_phase = demodulate(time, signal,hef0,sample_freq,\
                                  cutoff = he_cutoff )
    xe_amp, xe_phase = demodulate(time, signal, xef0, sample_freq, \
                                  cutoff = xe_cutoff)
    
    lower_cutoff = min([he_cutoff, xe_cutoff])
    he_amp = skip_artifacts(he_amp, sample_freq, lower_cutoff)
    he_phase = skip_artifacts(he_phase, sample_freq, lower_cutoff)
    xe_amp = skip_artifacts(xe_amp,sample_freq, lower_cutoff)
    xe_phase = skip_artifacts(xe_phase, sample_freq, lower_cutoff)

    time = skip_artifacts(time, sample_freq, lower_cutoff)
    
    return time, he_amp, he_phase, xe_amp, xe_phase
    
def calc_freq_point(time, phase_data, f0, point_size = 3):
    '''Calculated frequency using different point numerical differentiation
    using phase data'''
    if point_size == 2:
        dy = two_point_diff(time,phase_data)
    elif point_size == 3:
        dy = three_point_diff(time,phase_data)
    elif point_size == 4:
        dy = four_point_diff(time,phase_data)
    elif point_size == 5:
        dy = five_point_diff(time,phase_data)     
    return np.array([f0 + d for d in dy])
    
def calc_freq_slope(time, phase_data, sample_freq, f0, technique = 'step',
						  time_step = 1, step = 1):
    '''Calculates frequency using specified time step blocks of phase data and
    finding slope to use as frequency. time_step is block size for phase data
    to find slope from and is in seconds.
    technique can be step or time. If step then step sized are used and if time
    then time blocks are used.'''
    if technique == 'step': step = step
    elif technique == 'time': step = int(time_step*sample_freq)
    freqs = []
    freq_time = []
    n = 0
    while n < len(phase_data):
        f = np.polyfit(time[n:n + step],phase_data[n:n + step],1)
        t = time[n]
        n+= (step + 1)
        freqs.append(f0 + f[0])
        freq_time.append(t)
    return freq_time, freqs
    
