# -*- coding: utf-8 -*-
# modified from pyephys.FunctionsLib...
"""
Created on Tue Oct 22 01:09:36 2013
makes use of scipi.signal
@author: Alessandro Scaglione
Contact: alessandro.scaglione@gmail.com

Version: 0
"""
from __future__ import division
from __future__ import print_function

import pdb
import numpy as np
import scipy.signal as signal

# LOGGER ----------------------------------------------------------------------

import logging
logger = logging.getLogger(__name__)


# def rem_artifacts(x):

#     # standardize the signal
#     x_std = (x - x.mean()) / x.std()

#     return x_std


# def rem_line(Data, Freqs=60 * np.arange(8), Fs=None):

#     if Fs is None:
#         Fs = int(getattr(Data, 'SamplingFreq', 30000))
#         logger.warn('rem_line:Sampling Frequency set to: %i' % (Fs))

#         # ======================================================================
#         # Fs = len(Data)
#         # if len(Data) > 80000:
#         #     Fs = 80000
#         # ======================================================================
#     cls = Data.__class__
#     Header = Data.Header

#     Fs = int(Fs)

#     def rem_line_Fs(Data, Freqs, Fs):

#         Npoints = range(-2, 3)
#         dft = np.fft.fft(Data, int(Fs))
#         for f in Freqs:
#             for n in Npoints:
#                 if (abs(dft[f + n - 1]) + abs(dft[f + n + len(Npoints)])) != 0:
#                     ratio = 2 * \
#                         abs(dft[f + n]) / (abs(dft[f + n - 1]) +
#                                            abs(dft[f + n + len(Npoints)]))
#                 else:
#                     ratio = 1

#                 # print(ratio)
#                 if ratio > 1 and f > abs(n):
#                     dft[f + n] = dft[f + n] / ratio
#                     dft[Fs - f - n] = dft[f + n].conjugate()
#         # print(np.fft.ifft(dft))

#         Data[:] = np.real(np.fft.ifft(dft))[0:len(Data)]

#     FsWindows = np.arange(np.ceil(len(Data) / Fs) + 1).astype(np.int)
#     for win in FsWindows:
#         # pdb.set_trace()
#         DataChunk = Data[win * Fs:(win + 1) * Fs]
#         rem_line_Fs(DataChunk, Freqs, Fs)

#     out = cls(Data)
#     out.Header = Header
#     return out


# def notch(data, sampfreq=1000, freq=60, bw=0.25, gpass=0.01, gstop=30, ftype='ellip'):
#     """Filter out frequencies in data centered on freq (Hz), of bandwidth +/- bw (Hz).

#     ftype: 'ellip', 'butter', 'cheby1', 'cheby2', 'bessel'
#     """
#     w = freq / \
#         (sampfreq / 2)  # fraction of Nyquist frequency == 1/2 sampling rate
#     bw = bw / (sampfreq / 2)
#     wp = [w - 2 * bw, w + 2 * bw]  # outer bandpass
#     ws = [w - bw, w + bw]  # inner bandstop
#     # using more extreme values for gpass or gstop seems to cause IIR filter instability.
#     # 'ellip' is the only one that seems to work
#     b, a = signal.iirdesign(
#         wp, ws, gpass=gpass, gstop=gstop, analog=False, ftype=ftype)
#     data = signal.lfilter(b, a, data)
#     return data, b, a


# def naivenotch(data, sampfreq=1000, freqs=60, bws=1):
#     """Filter out frequencies in data centered on freqs (Hz), of bandwidths bws (Hz).
#     Filtering out by setting components to 0 is probably naive"""
#     raise NotImplementedError("this doesn't seem to work right!")
#     nt = data.shape[1]
#     tres = 1 / sampfreq
#     dt = tres / 1e6  # in sec
#     f = np.fft.fftfreq(nt, dt)  # fft bin frequencies
#     f = f[:nt // 2]  # grab +ve freqs by splitting f in half
#     franges = []
#     freqs = np.atleast_1d(freqs)
#     bws = np.atleast_1d(bws)
#     if len(freqs) > 1 and len(bws) == 1:
#         bws = bws * len(freqs)  # make freqs and bw the same length
#     for freq, bw in zip(freqs, bws):
#         franges.append(freq - bw)
#         franges.append(freq + bw)
#     fis = f.searchsorted(franges)
#     fis = np.hstack([fis, -fis])  # indices for both +ve and -ve freq ranges
#     fis.shape = -1, 2  # reshape to 2 columns
#     fdata = np.fft.fft(data)
#     for f0i, f1i in fis:
#         fdata[:, f0i:f1i] = 0  # replace desired components with 0
#         # maybe try using complex average of freq bins just outside of freqs
#         # +/- bws
#     data = np.fft.ifft(fdata).real  # inverse FFT, leave as float
#     return data


# def design_filter(fs=30000, f0=210, f1=5000, fr=400, gpass=0.01, gstop=40, ftype='Elliptic'):
#     """Bandpass filter data on row indices chanis, between f0 and f1 (Hz), with filter
#     rolloff (?) fr (Hz).

#     ftype: 'ellip', 'butter', 'cheby1', 'cheby2', 'bessel'
#     """
#     FILTER_TYPE = {'Butterworth': 'butter', 'Bessel': 'bessel',
#                    'Chebyshev I': 'cheby1', 'Chebyshev II': 'cheby2',
#                    'Elliptic': 'ellip'}
#     ftype = FILTER_TYPE[ftype]
#     w0 = f0 / (fs / 2)  # fraction of Nyquist frequency == 1/2 sampling rate
#     w1 = f1 / (fs / 2)
#     wr = fr / (fs / 2)
#     if w0 == 0:
#         wp = w1
#         ws = w1 + wr
#     elif w1 == 0:
#         wp = w0
#         ws = w0 - wr
#     else:
#         wp = [w0, w1]
#         ws = [w0 - wr, w1 + wr]
#     b, a = signal.iirdesign(
#         wp, ws, gpass=gpass, gstop=gstop, analog=False, ftype=ftype)

#     return b, a


def design_filter_ord(FSampling=30000, LowCutOff=250, HighCutOff=5000, Order=4,
                      RipplePassBand=0.1, RippleStopBand=80,
                      FilterType='Band Pass', FilterFamily='Butterworth'):
    """Bandpass filter data by specifying filter Order and FilterType, instead of gpass and gstop.

    FilterType: 'lowpass', 'highpass', 'bandpass', 'bandstop'
    FilterFamily: 'ellip', 'butter', 'cheby1', 'cheby2', 'bessel'

    For 'ellip', need to also specify passband and stopband ripple with RipplePassBand and RippleStopBand.
    """
    FiltFamilyDict = {'Butterworth': 'butter', 'Bessel': 'bessel',
                      'Chebyshev I': 'cheby1', 'Chebyshev II': 'cheby2',
                      'Elliptic': 'ellip'}
    FiltTypeDict = {'Low Pass': 'lowpass', 'High Pass': 'highpass',
                    'Band Pass': 'bandpass', 'Band Stop': 'bandstop'}

    FilterFamily = FiltFamilyDict[FilterFamily]
    FilterType = FiltTypeDict[FilterType]

    if HighCutOff != None:
        fn = np.array([LowCutOff, HighCutOff])
    else:
        fn = LowCutOff
    wn = fn / (FSampling / 2)
    b, a = signal.iirfilter(Order, wn, rp=RipplePassBand, rs=RippleStopBand,
                            ftype=FilterFamily, analog=False, btype=FilterType,
                            output='ba')

    return b, a


def apply_filter(data, b, a):

    if type(data) is np.ndarray:
        return signal.filtfilt(b, a, data)

    out = signal.filtfilt(b, a, data)
    # out = data.__class__(signal.filtfilt(b, a, data))

    if hasattr(data, '_Header'):
        out._Header = data._Header.copy()
    return out


def design_and_filter(data, FSampling=None, LowCutOff=1, HighCutOff=60,
                      Order=4, RipplePassBand=0.1, RippleStopBand=100,
                      FilterType='Band Pass', FilterFamily='Butterworth', return_ab=False):
    '''

    :param data:
    :type data:
    :param FSampling:
    :type FSampling:
    :param LowCutOff:
    :type LowCutOff:
    :param HighCutOff:
    :type HighCutOff:
    :param Order:
    :type Order:
    :param RipplePassBand:
    :type RipplePassBand:
    :param RippleStopBand:
    :type RippleStopBand:
    :param FilterType:
    :type FilterType:
    :param FilterFamily: 'Butterworth', 'Bessel', 'Chebyshev I', 'Chebyshev II', 'Elliptic'
    :type FilterFamily: str
    '''

    # if FSampling == None:
    #     FSampling = getattr(data, 'SamplingFreq', 30000)
    #     logger.debug('Setting Sampling Freq to: %i', FSampling)

    b, a = design_filter_ord(FSampling, LowCutOff, HighCutOff, Order,
                             RipplePassBand, RippleStopBand, FilterType,
                             FilterFamily)

    filt_out = apply_filter(data, b, a)

    if return_ab:
        return filt_out, (b, a)
    return filt_out


def detectDisconnections(data, max_value, min_value):
    # detecting disconnections
    blackout_timepoints = []
    if max_value != 0:
        max_dig_value = max_value
        pos_disc_points = np.where(data >= max_dig_value)

        min_dig_value = min_value
        neg_disc_points = np.where(data <= min_dig_value)

        blackout_timepoints = np.hstack((pos_disc_points, neg_disc_points))

    return blackout_timepoints

# def hilbert(x):
#     """Return power (dB wrt 1 mV), phase (rad), energy, and amplitude of Hilbert transform of
#     data in x"""
#     hx = signal.hilbert(x)  # Hilbert transform of x
#     Ax = np.abs(hx)  # amplitude
#     Phx = np.angle(hx)  # phase
#     Ex = Ax ** 2  # energy == amplitude squared?
#     Px = 10 * np.log10(Ex)  # power in dB wrt 1 mV^2?
#     return Px, Phx, Ex, Ax


# def wavelet(data, wname="db4", maxlevel=6):
#     """Perform wavelet multi-level decomposition and reconstruction (WMLDR) on data.
#     See Wiltschko2008. Default to Daubechies(4) wavelet"""
#     import pywt

#     data = np.atleast_2d(data)
#     # filter data in place:
#     for i in range(len(data)):
#         # decompose the signal:
#         c = pywt.wavedec(data[i], wname, level=maxlevel)
#         # destroy the appropriate approximation coefficients:
#         c[0] = None
#         # reconstruct the signal:
#         data[i] = pywt.waverec(c, wname)

#     return data


# def BandPassFilter(lco=0.5, hco=5000, fs=30000, order=3):

#     b, a = signal.butter(
#         order, [lco * 1. / fs / 2, hco * 1. / fs / 2], btype='band')
#     return b, a


# def mfreqz(b, a=1, fs=None, FiltName=''):
#     from matplotlib import pylab as pyl

#     pyl.figure()
#     w, h = signal.freqz(b, a)
#     h_dB = 20 * np.log10(abs(h))
#     pyl.subplot(211)
#     x = w / max(w)
#     if fs:
#         x = x * fs * 1. / 2
#     pyl.plot(x, h_dB)
#     pyl.ylim(-150, 5)
#     pyl.ylabel('Magnitude (db)')
#     if fs:
#         pyl.xlabel(r'Frequency (Hz)')
#     else:
#         pyl.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
#     pyl.axis('tight')
#     pyl.title(r'Frequency response' + FiltName)
#     pyl.subplot(212)
#     h_Phase = np.unwrap(np.arctan2(np.imag(h), np.real(h)))
#     pyl.plot(x, h_Phase)
#     pyl.ylabel('Phase (radians)')
#     if fs:
#         pyl.xlabel(r'Frequency (Hz)')
#     else:
#         pyl.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
#     pyl.axis('tight')
#     pyl.title(r'Phase response')
#     pyl.subplots_adjust(hspace=0.5)
#     pyl.show()


# def impz(b, a=1, fs=None, FiltName='None'):
#     from matplotlib import pylab as pyl

#     npoints = 100000
#     pyl.figure()
#     impulse = np.repeat(0., npoints)
#     impulse[0] = 1.
#     x = np.arange(0, npoints)
#     if fs:
#         x = x * 1 / fs
#     response = signal.lfilter(b, a, impulse)
#     pyl.subplot(211)
#     pyl.plot(x, response, 'o')
#     pyl.ylabel('Amplitude')
#     if fs:
#         pyl.xlabel(r'Time (s)')
#     else:
#         pyl.xlabel(r'n (samples)')
#     pyl.axis('tight')
#     pyl.title(r'Impulse response' + FiltName)
#     pyl.subplot(212)
#     step = np.cumsum(response)
#     pyl.plot(x, response, 'o')
#     pyl.ylabel('Amplitude')
#     if fs:
#         pyl.xlabel(r'Time (s)')
#     else:
#         pyl.xlabel(r'n (samples)')
#     pyl.title(r'Step response')
#     pyl.axis('tight')
#     pyl.subplots_adjust(hspace=0.5)
#     pyl.show()


# def spectrum(Data, SamplingFreq=1.0, FFTPoints=None, Scaling='density', Type='periodogram', ax=None):

#     def dB(x, out=None):
#         if out is None:
#             return 10 * np.log10(x)
#         else:
#             np.log10(x, out)
#             np.multiply(out, 10, out)
#     if hasattr(Data, "SamplingFreq"):
#         SamplingFreq = Data.SamplingFreq
#     try:
#         import nitime.algorithms as tsa
#         f, psd_mt, nu = tsa.multi_taper_psd(Data, SamplingFreq, adaptive=False,
#                                             jackknife=False)

#         dB(psd_mt, psd_mt)
#         return f, psd_mt
#     except:
#         if Type == 'periodogram':
#             F, Pxx = signal.periodogram(Data, fs=1.0, window=None, nfft=FFTPoints,
#                                         detrend='constant', return_onesided=True,
#                                         scaling=Scaling, axis=-1)
#         # return F,Pxx

#         F = F * SamplingFreq
#         return F, Pxx


# def plot_spectrum(Data, SamplingFreq=1.0, FFTPoints=None, Scaling='density', Type='periodogram', ax=None):
#     from matplotlib import pylab as pyl
#     # removing mean
#     Data = Data - Data.mean()

#     def dB(x, out=None):
#         if out is None:
#             return 10 * np.log10(x)
#         else:
#             np.log10(x, out)
#             np.multiply(out, 10, out)

#     if hasattr(Data, "SamplingFreq"):
#         SamplingFreq = Data.SamplingFreq

#     try:
#         import nitime.algorithms as tsa
#         # ar
#         f, psd_mt, nu = tsa.multi_taper_psd(Data, SamplingFreq, adaptive=False,
#                                             jackknife=False)

#         dB(psd_mt, psd_mt)
#         return pyl.plot(f, psd_mt)
#         # title('Multi tapered spectrum')
#     except:

#         if Type == 'periodogram':
#             F, Pxx = signal.periodogram(Data, fs=1.0, window=None, nfft=FFTPoints,
#                                         detrend='constant', return_onesided=True,
#                                         scaling=Scaling, axis=-1)
#         # return F,Pxx

#         F = F * SamplingFreq
#         # pdb.set_trace()
#         dB(Pxx, Pxx)
#         graph = pyl.plot(F, Pxx)
#         return graph
#         pyl.axis('tight')
#         pyl.xlabel('Frequency (Hz)')
#         pyl.ylabel('PSD [V**2/Hz]')
#         # title('Power Spectral Density')
#         pyl.show()


# def decimate(x, q, n=None, ftype="iir", axis=-1):

#     return signal.decimate(x, q, n, ftype, axis)
