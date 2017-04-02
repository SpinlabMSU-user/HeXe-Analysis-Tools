import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import json
import struct

import allantools as al
from DigWorker import *
from NumericalTools import *

def find_he_xe_freqs(x_data, y_data, sample_freq):
    '''Finds frequency of peak for xenon and helium. Does by finding peak of 
    fft of data where peak is around 14 for xe and 39 for he. Can make better
    by applying a fit around the guess area and finding peak of fit.
    
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

    freqs = np.fft.fftfreq(x_n,1.0/sample_freq) 
    #creates axis of frequencies corresponding to fft peaks

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
    
def fft_analysis(time_segment, signal_segment, sample_freq, kaiser_var = 14,
                 zero_pad_iterations = 1):
    '''FFT analysis that is iterated in blocked FFT analysis. works by 
    windowing and zero-padding data to perform precise FFT on signal. 
    Then performs a gaussian fitting to the FFT response to find frequency
    and amplitude of He-3 and Xe-129.
    
    -Input-
    time_segment: time data that corresponds to signal data (np.array)
    signal_segment: signal data that is to be analyzed (np.array)
    sample_freq: frequency at which data is sampled (float)
    kaiser_var: variable used for shape of kaiser windowing (float)
    zero_pad_iterations: number of times zero-padding is performed to data 
                         before FFT. Each iteration will
    zero-pad until data is of length equal to next power of 2. (int; 0-3)
    
    -Returns-
    both_freqs: frequency of He-3 and Xe-129 for signal inputed. (np.array;
                [he_freq, xe_freq])
    both_amps: Amplitudes of FFT response for He-3 and Xe-129 for signal
               inputed. (np.array; [he_amp, xe_amp])
    mid_time: middle time of inputed time segment. (float)
    he_gaus: FFT response curve of He-3 around. (np.array; [he_gaus_freqs,
             he_gaus_fourier_amplitudes])
    xe_gaus: FFT response curve of Xe-129. (np.array; [xe_gaus_freqs, 
             xe_gaus_fourier_amplitudes])
    he_g_fitting: Gaussian fitting to He-3 FFT curve. (np.array; [he_g_freqs,
                  he_g_fit])
    xe_g_fitting: Gaussian fitting to Xe-129 FFT curve. (np.array; 
                  [xe_g_freqs, xe_g_fit])
    sigmas: Standard deviations of Gaussian fitting for fits. (np.array;
            [he_sigma, xe_sigma])
    '''
    step = len(time_segment)
    
    kaiser_window = np.kaiser(len(time_segment), kaiser_var)
    signal_segment = signal_segment * kaiser_window

    time_segment = pad_to_next_2n(time_segment,zero_pad_iterations)
    signal_segment = pad_to_next_2n(signal_segment,zero_pad_iterations)

    max_he_freq, max_xe_freq, freqs, fourier = \
    find_he_xe_freqs(time_segment, signal_segment, sample_freq)
    
    fourier = renomalize_kaiser(fourier, kaiser_var)

    xe_loc = np.where(freqs == max_xe_freq)[0][0] 
    #finds index of Xe in frequencies for the FFT
    he_loc = np.where(freqs == max_he_freq)[0][0]

    #All numbers for these were found manually and may be looked into further
    if zero_pad_iterations == 0: 
        xe_gaus_max = xe_loc + 3
        xe_gaus_min = xe_loc - 2

        he_gaus_max = he_loc + 3
        he_gaus_min = he_loc - 2

    elif zero_pad_iterations == 1:
        xe_gaus_max = xe_loc + 6
        xe_gaus_min = xe_loc - 5

        he_gaus_max = he_loc + 6
        he_gaus_min = he_loc - 5

    if zero_pad_iterations == 2:
        xe_gaus_max = xe_loc + 11
        xe_gaus_min = xe_loc - 10

        he_gaus_max = he_loc + 11
        he_gaus_min = he_loc - 10

    if zero_pad_iterations == 3:
        xe_gaus_max = xe_loc + 26
        xe_gaus_min = xe_loc - 25

        he_gaus_max = he_loc + 26
        he_gaus_min = he_loc - 25

    #frequencies of FFT response curve for He
    he_gaus_freqs = freqs[he_gaus_min:he_gaus_max] 
    xe_gaus_freqs = freqs[xe_gaus_min:xe_gaus_max]

    #amplitudes of FFT response curve for He
    he_gaus_fourier = 2.0 / step * fourier[he_gaus_min:he_gaus_max]
    xe_gaus_fourier = 2.0 / step * fourier[xe_gaus_min:xe_gaus_max]

    try:
        he_offset, he_amp, he_freq, he_sigma =\
            gaussian_fit(he_gaus_freqs,he_gaus_fourier)

    except RuntimeError:
        he_offset, he_amp, he_freq, he_sigma = [0,0,0,0]
        he_sigma = he_sigma

    try:
        xe_offset, xe_amp, xe_freq, xe_sigma =\
            gaussian_fit(xe_gaus_freqs,xe_gaus_fourier)

    except RuntimeError:
        xe_offset, xe_amp, xe_freq, xe_sigma = [0,0,0,0]
        xe_sigma = xe_sigma

    #To show how close fit is to real FFT
    he_g_freqs = np.linspace(freqs[he_gaus_min], freqs[he_gaus_max], 50)
    he_g_fit = gaussian_curve(he_g_freqs, he_offset, he_amp, he_freq,he_sigma)

    xe_g_freqs = np.linspace(freqs[xe_gaus_min], freqs[xe_gaus_max], 50)
    xe_g_fit = gaussian_curve(xe_g_freqs, xe_offset, xe_amp, xe_freq,xe_sigma)
    
    mid_time = time_segment[int(len(time_segment)/2.0)]
    
    both_freqs = np.array([he_freq, xe_freq])
    both_amps = np.array([he_amp, xe_amp])
    he_gaus = np.array([he_gaus_freqs, he_gaus_fourier])
    xe_gaus = np.array([xe_gaus_freqs, xe_gaus_fourier])
    he_g_fitting = np.array([he_g_freqs, he_g_fit])
    xe_g_fitting = np.array([xe_g_freqs, xe_g_fit])
    sigmas = np.array([he_sigma, xe_sigma])
    offsets = np.array([he_offset, xe_offset])
    
    return both_freqs, both_amps, mid_time, he_gaus, xe_gaus, he_g_fitting,\
           xe_g_fitting, sigmas, offsets

def blocked_fft_analysis(input_time, input_signal, sample_freq,\
                         duo_signal = False, time_step = 1,\
                         truncated = True, zero_pad_iterations = 0,\
                         plots = True):
    '''Does analysis of signal by finding amplitude of signal or Xe-129 and
    He-3 in different time_step sizes. If truncated is true, then step size
    will be brought down to the closest power of 2 to make the fft quicker,
    making the time_step inputed to be approximte.
    
    Inputs:
    input_time: numpy array of time series
    input_signal: numpy array of signal(s) to be analyzed. either is a single
                  array or a list of two array. If 
                  list of two arrays, then the gradiometer of array_1 and
                  array_2 will be taken. Format will be [array_1, array_2]
    sample_freq: float of sampling frequency of experiment
    duo_signal: True or False. If True, then input_signal must be a list of 2
                channels. If False, then input_signal is just 1 array.
    time_step: float of approximate length of each block in seconds. This will
               be truncated so length of the block is equal to a power of 2
    truncated: If True, then time_step is truncated to a power of 2. If false,
               then time_step will not be truncated.
    zero_pad_iterations: int from 0-3, this will be the number of zero padding
                         iterations that are applied to the
                         time and signal arrays before applying the FFT. Each
                         iteration will zero-pad until the length of the array
                         is equal to a power of 2.
    plots: True or False. If True, then each data block that is analyzed will
           also display plots of analysis for that chunk. If True, then 
           analysis will be substantially slower.
    
    Returns:
    He-3 frequencies
    He-3 frequency amplitudes
    Xe-129 frequencies
    Xe-129 frequency amplitudes
    time corresponding to each amplitude'''  
    if duo_signal:
        z1 = input_signal[0]
        z2 = input_signal[1]
        signal = make_gradiometer(z1,z2)
        magnetometer = make_magnetometer(z1,z2)
    else:
        signal = input_signal
    
    step = int(time_step * sample_freq)

    if truncated:
        step = subtract_to_2n(step)
        step_size = step / sample_freq
        print 'step size: {} seconds'.format(step_size)
        print 'step length: {}'.format(step)
    
    xe_freqs = []
    he_freqs = []
    xe_freq_amps = []
    he_freq_amps = []
    xe_sigmas = []
    he_sigmas = []
    xe_offsets = []
    he_offsets = []
    av_mags = []
    times = []

    n = 0

    while n + step < len(input_time):
        if duo_signal:
            z1_segment = z1[n:n+step]
            z2_segment = z2[n:n+step]
            magnetometer_segment = magnetometer[n:n+step]
            av_mag = np.mean(magnetometer_segment)
        else:
            av_mag = 0
            
        time_segment1 = input_time[n:n+step]
        signal_segment1 = signal[n:n+step]
        
        both_freqs, both_amps, mid_time, he_gaus, xe_gaus, he_g_fitting,\
        xe_g_fitting, sigmas, offsets =\
        fft_analysis(time_segment1, signal_segment1, sample_freq,\ 
                     zero_pad_iterations = zero_pad_iterations,\
                     kaiser_var = 14)
        he_freq, xe_freq = both_freqs
        he_amp, xe_amp = both_amps
        he_gaus_freqs, he_gaus_fourier = he_gaus
        xe_gaus_freqs, xe_gaus_fourier = xe_gaus
        he_g_freqs, he_g_fit = he_g_fitting
        xe_g_freqs, xe_g_fit = xe_g_fitting
        he_sigma, xe_sigma = sigmas
        he_offset, xe_offset = offsets
        
        xe_freqs.append(xe_freq)
        he_freqs.append(he_freq)
        
        xe_freq_amps.append(xe_amp)
        he_freq_amps.append(he_amp)
        
        xe_sigmas.append(xe_sigma)
        he_sigmas.append(he_sigma)
        
        xe_offsets.append(xe_offset)
        he_offsets.append(he_offset)
        
        av_mags.append(av_mag)
        
        times.append(mid_time)
        
        if plots:
            fig, ax = plt.subplots(3,3, figsize = (14,8))
            
            #Gradiometer and Magnetometer
            ax[0,0].scatter(time_segment1, z1_segment, c = 'b', lw = 0,\
                            label = 'z1')
            ax[0,0].scatter(time_segment1, z2_segment, c = 'r', lw = 0,\
                            label = 'z2')
            ax[0,0].legend(loc = 7)
            ax[0,0].set_title('Each Z channel')
            
            ax[1,0].scatter(time_segment1, signal_segment1, lw = 0)
            ax[1,0].set_title('Gradiometer')
            
            ax[2,0].scatter(time_segment1, magnetometer_segment, lw = 0)
            ax[2,0].set_title('Magnetometer')
            
            #Gaussian Fits
            ax[0,1].scatter(he_gaus_freqs, he_gaus_fourier)
            ax[0,1].plot(he_g_freqs, he_g_fit)
            ax[0,1].set_title('He Fitting')
            ax[0,1].set_xlabel('Frequency (Hz)')
            ax[0,1].set_ylim([min(he_gaus_fourier),max(he_gaus_fourier)])
            
            he_fwhm = 2 * np.sqrt(2* np.log(2))*he_sigma
            he_g_info = \
            'A = {}\nFWHM = {}\nStDev = {}\nFreq = {}\nOffset = {}'\
            .format(round(he_amp,4),round(he_fwhm,4),round(he_sigma,4),\
                    round(he_freq,4),round(he_offset,4))
            ax[0,1].text(.6, .55, he_g_info, transform=ax[0,1].transAxes)
            
            ax[1,1].scatter(xe_gaus_freqs, xe_gaus_fourier)
            ax[1,1].plot(xe_g_freqs, xe_g_fit)
            ax[1,1].set_title('Xe Fitting')
            ax[1,1].set_xlabel('Frequency (Hz)')
            ax[1,1].set_ylim([min(xe_gaus_fourier), max(xe_gaus_fourier)])
            xe_fwhm = 2 * np.sqrt(2* np.log(2))*xe_sigma
            xe_g_info = \
            'A = {}\nFWHM = {}\nStDev = {}\nFreq = {}\nOffset = {}'\
            .format(round(xe_amp,4),round(xe_fwhm,4),round(xe_sigma,4),\
                    round(xe_freq,4),round(xe_offset,4))
            ax[1,1].text(.6, .55, xe_g_info, transform=ax[1,1].transAxes)
            #Amplitudes
            if he_freq_amps[0] > xe_freq_amps[0]:
                amp_max = he_freq_amps[0]
            else:
                amp_max = xe_freq_amps[0]
            
            ax[2,1].scatter(times, he_freq_amps, c='r', lw=0, label = 'He')
            ax[2,1].scatter(times, xe_freq_amps, c='b', lw=0, label = 'Xe')
            ax[2,1].set_ylim([0,amp_max])
            ax[2,1].set_xlim(left=min(times))
            ax[2,1].set_title('Amplitudes')
            ax[2,1].set_xlabel('Time (sec)')
            ax[2,1].legend(loc=6)
            
            if len(times) > 1:
                he_famp, he_t2, he_t2err = [round(x,4) for x in \
                                           fit_exp(times, he_freq_amps)]
                xe_famp, xe_t2, xe_t2err = [round(x,4) for x in \
                                           fit_exp(times, xe_freq_amps)]
            else:
                he_famp = he_freq_amps[0]
                xe_famp = xe_freq_amps[0]
                he_t2 = '?'
                xe_t2 = '?'
            
            if he_famp > xe_famp:
                max_famp = he_famp / 2.0
            else:
                max_famp = xe_famp / 2.0
            
            amp_info = 'He A0 = {}\nHe T2* = {}\nXe A0 = {}\nXe T2* = {}'\
                       .format(he_famp, he_t2, xe_famp, xe_t2)
            ax[2,1].text(.52, .63, amp_info, transform=ax[2,1].transAxes)

            #frequencies
            ax[0,2].scatter(times, xe_freqs)
            ax[0,2].set_title('Xe Freq')
            ax[0,2].set_xlabel('Time (sec)')
            ax[0,2].set_xlim(left=0)

            ax[1,2].scatter(times, he_freqs)
            ax[1,2].set_title('He Freq')
            ax[1,2].set_xlabel('Time (sec)')
            ax[1,2].set_xlim(left=0)
            
            #frequency comparison with gyromagnetic ratio
            ax[2,2].scatter(np.array(xe_freqs) / 11.777 ,\
                            np.array(he_freqs) / 32.434)
            ax[2,2].set_title('Frequency Comparisons')
            ax[2,2].set_xlabel('Xe freq/$\gamma$ Xe')
            ax[2,2].set_ylabel('He freq/$\gamma$ He')

            fig.tight_layout()
            display.clear_output(wait=True)
            display.display(plt.gcf())
            plt.close()
        n += step

    return he_freqs, he_freq_amps, he_sigmas, he_offsets, xe_freqs,\
           xe_freq_amps, xe_sigmas, xe_offsets, av_mags, times, step_size

#post-run analysis

def create_post_run_plots(time, block_size, he_freqs, xe_freqs, he_amps,\
                          xe_amps, he_sigmas, xe_sigmas, av_mags,\
                          he_offsets, xe_offsets, time_start = False):
    
    he_fwhm = [sigma * 2 * np.sqrt(2* np.log(2)) for sigma in he_sigmas]
    xe_fwhm = [sigma * 2 * np.sqrt(2* np.log(2)) for sigma in xe_sigmas]
    
    fig, ax = plt.subplots(3,3, figsize = (14,8))
    
    #amplitudes
    ax[0,0].scatter(time, he_amps, lw = 0, s=2)
    ax[0,0].set_title('He-3 Amplitude')
    ax[0,0].set_xlabel('Time (s)')
    ax[0,0].set_ylabel('Amplitude (nT)')
    ax[0,0].set_ylim([min(he_amps), max(he_amps)])
    ax[0,0].set_xlim([0,max(time)])
    if time_start: ax[0,0].set_xlim([min(time),max(time)])
    he_famp, he_t2, he_t2err = [round(x,4) for x in fit_exp(time, he_amps)]
    ax[0,0].text(.25, .8, 'Initial Amplitude: {} nT\nT2*:{} s'\
                 .format(he_famp, he_t2), transform = ax[0,0].transAxes)
    
    ax[1,0].scatter(time, xe_amps, lw=0, s=2)
    ax[1,0].set_title('Xe-129 Amplitude')
    ax[1,0].set_xlabel('Time (s)')
    ax[1,0].set_ylabel('Amplitude (nT)')
    ax[1,0].set_ylim([min(xe_amps), max(xe_amps)])
    ax[1,0].set_xlim([0,max(time)])
    if time_start: ax[1,0].set_xlim([min(time),max(time)])
    xe_famp, xe_t2, xe_t2err = [round(x,4) for x in fit_exp(time, xe_amps)]
    ax[1,0].text(.25, .8, 'Initial Amplitude: {} nT\nT2*:{} s'\
                 .format(xe_famp, xe_t2), transform = ax[1,0].transAxes)
    
    #frequencies
    he_freq_mean = np.mean(he_freqs)
    he_drift = [freq/he_freq_mean - 1 for freq in he_freqs]
    ax[0,1].scatter(time, he_drift, lw=0, s=2)
    ax[0,1].set_title('He-3 Frequency drift from {} (Hz)'\
                      .format(round(np.mean(he_freqs),3)))
    ax[0,1].set_xlabel('Time (s)')
    ax[0,1].set_ylabel('He freq / average He freq - 1 (Hz)')
    ax[0,1].set_xlim([0, max(time)])
    if time_start: ax[0,1].set_xlim([min(time),max(time)])
    ax[0,1].set_ylim([min(he_drift), max(he_drift)])
#    ax[0,1].set_ylim([-.0001, 0.0001])
    
    xe_freq_mean = np.mean(xe_freqs)
    xe_drift = [freq/xe_freq_mean - 1 for freq in xe_freqs]
    ax[1,1].scatter(time, xe_drift, lw = 0, s=2)
    ax[1,1].set_title('Xe-129 Frequency drift from {} (Hz)'\
                      .format(round(np.mean(xe_freqs),3)))
    ax[1,1].set_xlabel('Time (s)')
    ax[1,1].set_ylabel('Xe freq / average Xe freq - 1 (Hz)')
    ax[1,1].set_xlim([0,max(time)])
    if time_start: ax[1,1].set_xlim([min(time),max(time)])
    ax[1,1].set_ylim([min(xe_drift), max(xe_drift)])
    
    #OADs
    xe_he_gyro_ratio = 11.777/32.434
    xe_corr = [xe_freq - he_freq * xe_he_gyro_ratio for xe_freq,he_freq in\
               zip(xe_freqs, he_freqs)]
    xe_freq_taus, xe_freq_ad = freq_oad(xe_freqs, 1.0/block_size)
    he_freq_taus, he_freq_ad = freq_oad(he_freqs, 1.0/block_size)
    xe_corr_taus, xe_corr_ad = freq_oad(xe_corr, 1.0/block_size)
    
    ax[2,0].loglog(xe_freq_taus, xe_freq_ad, '-o', label = 'Xe AD')
    ax[2,0].loglog(xe_corr_taus, xe_corr_ad, '-o', label = 'Xe Corrected')
    ax[2,0].legend(loc='best')
    ax[2,0].set_title('Xe-129 Overlapping Allan Deviation')
    ax[2,0].set_xlabel('Integration time (sec)')
    ax[2,0].set_ylabel('Allan Deviation (Hz)')
    
    ax[2,1].loglog(xe_freq_taus, xe_freq_ad, '-o', label = 'Xe AD')
    ax[2,1].loglog(he_freq_taus, he_freq_ad, '-o', label = "He AD")
    ax[2,1].legend(loc='best')
    ax[2,1].set_title('He-3 and Xe-129 OADs')
    ax[2,1].set_xlabel('Integration time (sec)')
    ax[2,1].set_ylabel('Allan Deviation (Hz)')
    
    #Average Mag
    
    ax[0,2].scatter(time, av_mags, lw=0, s=2)
    ax[0,2].set_title('Average Z-Magnetometer Over Time')
    ax[0,2].set_xlabel('Time (sec)')
    ax[0,2].set_ylabel('Z-Magnetometer (nT)')
    ax[0,2].set_xlim([0, max(time)])
    if time_start: ax[0,2].set_xlim([min(time),max(time)])
    ax[0,2].set_ylim([min(av_mags), max(av_mags)])
    '''
    #Offsets
    ax[0,2].scatter(time, he_offsets, lw=0, c='r', label = 'He')
    ax[0,2].scatter(time, xe_offsets, lw=0, c='b', label = 'Xe')
    ax[0,2].legend(loc='best')
    ax[0,2].set_title('Offsets')
    ax[0,2].set_xlabel('Time (sec)')
    ax[0,2].set_ylabel('Gaussian Offset (nT)')
    ax[0,2].set_xlim([0,max(time)])
    ax[0,2].set_ylim([min(he_offsets), max(he_offsets)])
    '''
    #FWHM
    ax[1,2].scatter(time, he_fwhm, lw=0, c='r', s=2, label = 'He')
    ax[1,2].scatter(time, xe_fwhm, lw=0, c='b', s=2,  label = 'Xe')
    ax[1,2].legend(loc='best')
    ax[1,2].set_title('Gaussian Fit FWHM')
    ax[1,2].set_xlabel('Time (sec)')
    ax[1,2].set_ylabel('FWHM (Hz)')
    ax[1,2].set_xlim([0,max(time)])
    if time_start: ax[1,2].set_xlim([min(time),max(time)])
    
    if max(he_fwhm) > max(xe_fwhm): max_fwhm = max(he_fwhm)
    else: max_fwhm = max(xe_fwhm)
    if min(he_fwhm) > min(xe_fwhm): min_fwhm = min(he_fwhm)
    else: min_fwhm = min(xe_fwhm)
        
    ax[1,2].set_ylim([min_fwhm, max_fwhm])
    
    #Frequency comparison
    ax[2,2].scatter(np.array(xe_freqs) / 11.777 , np.array(he_freqs)\
                    / 32.434, lw=0, s=2)
    ax[2,2].set_title('Frequency Comparisons')
    ax[2,2].set_xlabel('Xe freq/$\gamma$ Xe')
    ax[2,2].set_ylabel('He freq/$\gamma$ He')
    ax[2,2].set_xlim([min(xe_freqs) / 11.777, max(xe_freqs) / 11.777])
    ax[2,2].set_ylim([min(he_freqs) / 32.434, max(he_freqs) / 32.434])
    
    fig.tight_layout()
    
def create_post_run_plots2(time, block_size, freqs, amps):
    fig, ax = plt.subplots(3,1, figsize = (14,8))
    
    #amplitudes
    ax[0].scatter(time, amps, lw = 0, s=2)
    ax[0].set_title('Amplitude')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Amplitude (nT)')
    ax[0].set_ylim([min(amps), max(amps)])
    ax[0].set_xlim([min(time),max(time)])
    famp, t2, t2err = [round(x,4) for x in fit_exp(time, amps)]
    ax[0].text(.25,.8,'Initial Amplitude: {}\nT2*: {} s'.format(famp, t2),\
               transform=ax[0].transAxes)
    
    #frequencies
    freq_mean = np.mean(freqs)
    drift = [freq/freq_mean - 1 for freq in freqs]
    ax[1].scatter(time, drift, lw=0, s=2)
    ax[1].set_title('Frequency drift from {}'.format(round(np.mean(freqs),3)))
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('He freq / average He freq - 1')
    ax[1].set_xlim([min(time), max(time)])
    ax[1].set_ylim([min(drift), max(drift)])
    
    #OADs
    freq_taus, freq_ad = freq_oad(freqs, 1.0/block_size)
    ad_slope = fit_exp_func(freq_taus,freq_ad)[1]
    
    ax[2].loglog(freq_taus, freq_ad, '-o')
    ax[2].set_title('Overlapping Allan Deviation')
    ax[2].set_xlabel('Integration time (sec)')
    ax[2].set_ylabel('Allan Deviation (Hz)')
    ax[2].text(.5,.8, 'slope = {}'.format(ad_slope),transform=ax[2].transAxes)
    
    fig.tight_layout()