'''
Created on Jul 16, 2014

.. moduleauthor:: Alessandro Scaglione <alessandro.scaglione@gmail.com>

'''
# from _struct import unpack
import pdb
# =========================================================================
# BUILTIN
# =========================================================================

# =========================================================================
# PIP INSTALLABLES
# =========================================================================

import re

# from matplotlib import pylab as pyl

# from PyQt4 import QtGui
from numpy.fft import fft, ifft
import scipy.stats as stats
import tables

import numpy as np

import pdb


# from scipy.signal import fftconvolve

# LOGGER ----------------------------------------------------------------------

import logging
logger = logging.getLogger(__name__)


def plot_PSTH_and_Raster(TS, Event=None, time_window=(.2, .41), color='blue'):
    from matplotlib import pylab as pyl
    if Event is None:
        return
    # pyl.figure()
    x, y, M, x_ind = sparse_raster(TS,
                                   Event, TimeWindow=time_window, x_ind=True)
    time = np.arange(-time_window[0], time_window[1], 1. / 1000)
    PSTH_BA_N, CI = PSTH_from_PEM(
        M, background=(0, int(time_window[0] * 1000)), return_CI=True, kernel_size=10)
    pyl.plot(
        x, y, 'bo', markersize=3, alpha=0.3)
    pyl.plot(
        time, PSTH_BA_N, alpha=0.6, linewidth=4, color=color)
    pyl.xlim((-time_window[0], time_window[1]))


def PSTH_PEM(TS, Event=None, bin_size=1, time_window=(.2, .41), return_PEM=False, return_ind=False, **kwargs):
    '''

    :param TS:
    :type TS:
    :param Event:
    :type Event:
    :param time_window:
    :type time_window:
    :param return_PEM:
    :type return_PEM:
    '''

    _, _, M, ind = sparse_raster(TS,
                                 Event, TimeWindow=time_window, x_ind=True)
    # time = np.arange(-time_window[0], time_window[1], 1. / 1000)

    PSTH, _ = PSTH_from_PEM(
        M, bin_size=bin_size, background=(0, int(time_window[0] * 1000)), return_CI=True, **kwargs)

    if return_ind:
        return PSTH, M, ind
    if return_PEM:
        return PSTH, M

    return PSTH


def PSTH_from_PEM(M, in_units='counts', out_units='Hz', bin_size=1,
                  kernel='box', kernel_size=20, background_norm=False, background=None, return_CI=False):

    PSTH = M.sum(0)

    if out_units == 'probability':
        PSTH = PSTH / M.shape[0]
    if out_units in ['Hz', 'spikes/sec']:
        PSTH = M.sum(0) / (bin_size * 1. / 1000 * M.shape[0])

    if background_norm:
        BA = PSTH.__getslice__(*background)
        PSTH = PSTH * 1. / BA.mean()

    if kernel == 'box':
        kernel = np.ones(kernel_size) / kernel_size
        msg = "PSTH_from_PEM(): Kernel is {}".format(kernel)
        logger.debug(msg)

    if kernel is not None:
        kernel = kernel / kernel.sum()
        # pdb.set_trace()
        PSTH = np.correlate(PSTH, kernel, 'same')

    if return_CI:
        if background is None:
            raise Exception('no background selected for CI')

        # average number of events in NTrials
        BA_bin_trial = M[background[0]:background[1], :].sum(1).mean()

        CI = stats.poisson.interval(
            .95, BA_bin_trial)
        CI = np.array(CI) / (background[1] - background[0]) * M.shape[0]

        if out_units == 'probability':
            CI = CI / M.shape[0]
        if out_units in ['Hz', 'spikes/sec']:
            CI = CI / (bin_size * 1. / 1000 * M.shape[0])

        return PSTH, CI

    return PSTH


def plot_PSTH(TS, Event=None, time_window=(.2, .41), color='blue', return_PSTH=False, kernel_size=10):

    if Event is None:
        return
    # pyl.figure()
    from matplotlib import pylab as pyl
    x, y, M, x_ind = sparse_raster(TS,
                                   Event, TimeWindow=time_window, x_ind=True)
    time = np.arange(-time_window[0], time_window[1], 1. / 1000)
    PSTH_BA_N, CI = PSTH_from_PEM(
        M, background=(0, int(time_window[0] * 1000)), return_CI=True, kernel_size=kernel_size)
    pyl.plot(
        time, PSTH_BA_N, alpha=1, linewidth=3, color=color)
    pyl.xlim((-time_window[0], time_window[1]))
    if return_PSTH:
        return PSTH_BA_N


def plot_Raster(TS, Event=None, time_window=(.2, .41), color='blue'):

    if Event is None:
        return
    from matplotlib import pylab as pyl
    # pyl.figure()
    x, y, M, x_ind = sparse_raster(TS,
                                   Event, TimeWindow=time_window, x_ind=True)
    time = np.arange(-time_window[0], time_window[1], 1. / 1000)
    PSTH_BA_N, CI = PSTH_from_PEM(
        M, background=(0, int(time_window[0] * 1000)), return_CI=True, kernel_size=10)
    pyl.plot(
        x, y, 'bo', markersize=3, alpha=0.8, color=color)


def next_pow_of_two(n):
    n = np.array(n)
    return int(pow(2, np.ceil(np.log2(n))))


def ISI(x: np.ndarray, sampling_freq: int | float, bin_size=.1, t_max=.1,
        log_scale_y=False, normalized=True, add_ts=None) -> tuple[np.ndarray, np.ndarray]:
    """Compute the interspike interval distribution.

    Args:
        x (np.ndarray): timestamps
        sampling_freq (int | float): _description_
        bin_size (float, optional): _description_. Defaults to .1.
        t_max (float, optional): _description_. Defaults to .1.
        log_scale_y (bool, optional): _description_. Defaults to False.
        normalized (bool, optional): Normalize the isi array to % of total spikes. Defaults to True.
        add_ts (_type_, optional): _description_. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: The first array is the bins from 0ms to 1000ms, 
        the second array is the isi distribution.
    """

    # if not sampling_freq:
    #     sf = getattr(x, 'SamplingFreq', 30000) * 1.

    x = np.asarray(x)
    if add_ts is not None:
        x = np.hstack((x, add_ts)).flatten()
        x.sort()

    ISI = np.diff(x)
    ISI = ISI[ISI < t_max * sampling_freq]

    ISI = (ISI / sampling_freq)

    start = 0
    stop = t_max
    num = int(t_max * 1000 / bin_size + 1)
    # print(ISI)
    ISI, bins = np.histogram(ISI, np.linspace(start, stop, num))
    bins = bins[:-1]
    if normalized:
        # % of total spikes
        ISI = ISI * 1. / (x.size - 1)
    if log_scale_y:
        ISI = np.log10(ISI)

    return bins, ISI


def firing_rate(x: np.ndarray, sampling_freq: int | float, out_units='Hz') -> float:
    """Compute the firing rate.

    Args:
        x (np.ndarray): timestamps
        sampling_freq (int | float): _description_
        out_units (str, optional): _description_. Defaults to 'Hz'.

    Returns:
        float: firing rate
    """
    # if not sampling_freq:
    #     sf = getattr(x, 'SamplingFreq', 30000) * 1.

    if len(x) == 0:
        return 0

    return float(x.size / (x.max() / sampling_freq))


def xcorrelation(ts_unit1, ts_unit2=None, bin_size=10, method='ephys',
                 time_lag=[-100, 100]):

    if not np.any(ts_unit1):
        return

    if not isinstance(ts_unit2, (np.ndarray, list)) or len(ts_unit2) < 100:
        autocorr = True
        ts_unit2 = ts_unit1
    else:
        autocorr = False

    # transform the timestamps into an array of zeros and ones
    bin_size = int(bin_size)

    if method == 'ephys':
        xcor = []
        for ts in ts_unit1:
            t = ts_unit2 - ts
            xcor.extend(t[(t > time_lag[0]) & (t < time_lag[1])])

        xcor, lags = np.histogram(xcor, bins=int(np.diff(time_lag) / bin_size))

        if autocorr:
            xcor[np.flatnonzero(lags == 0)] = 0

    elif method == 'fft':

        nBins = max(ts_unit1[-1], ts_unit2[-1]) / bin_size
        train1 = np.zeros(nBins + 1, dtype=np.int16)
        train2 = np.zeros(nBins + 1, dtype=np.int16)

        for k in np.floor(ts_unit1 / bin_size).astype('int'):
            train1[k] = train1[k] + 1

        for k in np.floor(ts_unit2 / bin_size).astype('int'):
            train2[k] = train2[k] + 1

        N = next_pow_of_two(len(train1))

        fft_1 = fft(train1, N)
        fft_2 = fft(train2, N)

        xcor = np.real(ifft(fft_1 * np.conjugate(fft_2), N))
        xcor = np.concatenate([xcor[xcor.size / 2:], xcor[0:xcor.size / 2]])
        lags = np.linspace(-ts_unit1[-1] / 2, ts_unit1[-1] / 2, xcor.size)

        indx = np.flatnonzero((lags >= time_lag[0]) & (lags <= time_lag[1]))
        xcor = xcor[indx]
        lags = lags[indx]

    return xcor, lags


def fft_convolve(sig1, sig2):
    '''FFT convolution '''

    # transform into an array
    sig1 = np.array(sig1, ndmin=1)
    sig2 = np.array(sig2, ndmin=1)

    # pad with zeros
    if sig1.size > sig2.size:
        z = np.zeros(sig1.size - sig2.size)
        sig2 = np.concatenate((sig2, z))

    elif sig1.size < sig2.size:
        z = np.zeros(sig2.size - sig1.size)
        sig1 = np.concatenate((sig1, z))

    # calculate the fft convolution
    return np.abs(np.fft.ifft(np.fft.fft(sig1) * np.fft.fft(sig2)))


def PSTH(UnitTS, EventTS, TimeWindow, Method='convolution', BinSize=20.0,
         kernel='square', TSSF=None):
    '''Calculates PSTH using several methods.
    Inputs:
        UnitTS:     numpy array with the unit time stamps.
        EventTS:    numpy array with the timestamps of the events
        Method:     string, contains the Method used to calculate the psth.
                    Valid keys are: 'convolution' or 'histogram'
        binsize:    size of the bin in miliseconds. In the case of the
                     convolution
                    Method is the size of the kernel.
        kernel:     if Method == 'convolution', kernel is a string with the
                     type of kernel.
                    Valid values: 'hamming', anything else creates a square
                     kernel.
    Outputs:
        PSTH:
        xVector:    vector with the time values
    '''

    if TSSF is None:
        if hasattr(UnitTS, 'Header'):
            TSSF = UnitTS.Header.get('SamplingFreq', 1)
        else:
            TSSF = 1

    TimeWindow = (TSSF * TimeWindow[0] * 1e-3, TSSF * TimeWindow[1] * 1e-3)
    BinSize = TSSF * BinSize * 1e-3
    #    logger.warn(BinSize)

    EventTS = np.array(EventTS)
    BinSize = float(BinSize)
    spikeTS = []
    logger.debug(UnitTS)
    logger.debug(EventTS)
    # get the spike time stamps around a given event
    for e in EventTS:
        tmp = UnitTS[
            (UnitTS > (e - TimeWindow[0])) & (UnitTS < (e + TimeWindow[1]))] \
            - e
        logger.debug(tmp)
        spikeTS.extend(tmp)

    if Method == 'convolution':
        # create a point process from the spike train
        spikesIndx = np.array(spikeTS).astype(np.int32) + int(TimeWindow[0])

        # pool all the trials into a singel vector
        PSTH = np.zeros(np.sum(np.abs(TimeWindow)))
        for k in spikesIndx:
            PSTH[k] = PSTH[k] + 1

        # convolve spike train with a window of BinSize miliseconds
        if kernel == 'hamming':
            window = np.hamming(BinSize)
        else:
            window = np.ones(BinSize)
        PSTH = fft_convolve(PSTH, window)

        # create an xVector
        xVector = np.arange(-TimeWindow[0], TimeWindow[1])

    elif Method == 'histogram':
        nbins = int((TimeWindow[0] + TimeWindow[1]) / BinSize)
        h = np.histogram(spikeTS, nbins)
        PSTH = h[0]
        xVector = h[1][0:-1] / TSSF + np.diff(h[1]) / 2 / TSSF
        window = BinSize

    # divide by the number of trials and the size of the bin (in seconds)
    # to return the PSTH in spikes/second
    PSTH = (PSTH / float(EventTS.size)) / (BinSize / 30000)

    return PSTH, xVector


def get_PSTH_from_H5(h5file=None, events=None, units='all',
                     controlStim='toneL', tWin=[500.0, 1000.0],
                     bin_size=25.0, Normalize=False,
                     psthMethod='convolution', psthKernel='hamming'):
    '''
    Calculates the psths for several conditions.
    Inputs:
        h5file:      h5 file instance or string with the location of the file.
        events:      dictionary, each key holds.
        units:       either 'all' for all units, or a dictionary containing
                     channels in the keys and a list with units under
                     each channel.
        controlStim: stim used to calculate bursting index
        tWin:        temporal widow in miliseconds around each event.
        bin_size:     miliseconds.
        Normalize:   whether or not to normalize to the baseline.
        psthMethod:  method to obtain the PSTHs. See PSTH function help.
        psthKernel:  kernel type. See PSTH function help.

    Outputs:
        PSTH dictionary.
        Array with bursting units indices.
    '''
    from PyQt4 import QtGui
    if not h5file or type(h5file) == str:
        # get h5 file name and open it
        fname = str(QtGui.QFileDialog.getOpenFileName(filter='*.h5'))
        if not fname:
            return
        h5file = tables.open_file(fname, 'r')

    if not events:
        return

    if controlStim in events:
        controlStim = events.keys()[0]

    # Get the PSTHS for all the basal forebrain neurons

    # create a dictionary to hold the PSTHs
    psth = {}
    for k in events.keys():
        psth[k] = []
    unitsID = []

    # if we have to extract all the units from the h5file
    if units == 'all':
        # get the PSTHs for each unit
        for k in h5file.listNodes('/Spikes'):

            # get the timestamps of all the units on that channel
            TS = k.TimeStamp.read()

            # iterate over the group and check whether are there any units
            if 'Unit00' in k.__members__:

                for l in k:
                    if re.search('Unit', l._v_name) \
                       and l.isMultiunit.read() is False:

                        unitsID.append(k._v_name + '/' + l._v_name)
                        # get the timestamps indices of the unit
                        indx = l.Indx.read()
                        if len(indx) == 0:
                            continue
                        unitTS = TS[indx]

                        # calculate the PSTHs for each type of event
                        for e in events.keys():
                            tmp, xVector = PSTH(unitTS, events[e],
                                                TimeWindow=tWin,
                                                Method=psthMethod,
                                                BinSize=bin_size,
                                                kernel=psthKernel)
                            psth[e].append(tmp)

    # if a dictionary with units is provided
    elif type(units) == dict:
        unitsID = []
        for k in units.keys():
            if h5file.__contains__('/Spikes/Chan_%03d' % int(k)):
                TS = h5file.getNode(
                    '/Spikes/Chan_%03d' % int(k)).TimeStamp.read()
                for unit in units[k]:
                    if h5file.__contains__('/Spikes/Chan_%03d/Unit%02d'
                                           % (int(k), unit)):
                        unitsID.append('Chan_%03d/Unit%02d' % (int(k), unit))
                        indx = h5file.getNode(
                            '/Spikes/Chan_%03d' % int(k),
                            name='Unit%02d' % unit).Indx.read()
                        unitTS = TS[indx]

                        # calculate the PSTHs for each type of event
                        for e in events.keys():
                            tmp, xVector = PSTH(unitTS, events[e],
                                                TimeWindow=tWin,
                                                Method=psthMethod,
                                                BinSize=bin_size,
                                                kernel=psthKernel)
                            psth[e].append(tmp)
                    else:
                        continue
            else:
                continue
    else:
        return

    # transform the psth list into an array
    for k in psth.keys():
        psth[k] = np.array(psth[k])

    if Normalize:
        # Normalize using the 400 msec before stimulus as a baseline
        # create a dictionary to hold the normalized data

        # get the indices of the baseline data
        preIndx = np.flatnonzero((xVector > -500) & (xVector < 0))
        posIndx = np.flatnonzero((xVector > 100) & (xVector < 400))
        nPSTH = {}
        for k in psth.keys():
            # skip if the list is empty
            if len(psth[k]) == 0:
                continue

            # for each neuron zscore the firing rate to the 500 ms baseline
            # before stim
            m = psth[k][:, preIndx].mean(axis=1)
            psth[k][m < 0.1, :] = 0.1
            m = np.tile(m, (psth[k].shape[1], 1)).transpose()
            s = psth[k][:, preIndx].std(axis=1)
            s = np.tile(s, (psth[k].shape[1], 1)).transpose()
            s[s < 0.1] = 0.1
            nPSTH[k] = (psth[k] - m) / s

        # Calculate Bursting Index
        posStim = nPSTH[controlStim][:, posIndx].mean(axis=1)
        burstingUnits = np.flatnonzero(posStim > 3)

    else:
        nPSTH = psth
        # Calculate Bursting Index

        # get the indices of the baseline data
        preIndx = np.flatnonzero((xVector > -200) & (xVector < 0))
        posIndx = np.flatnonzero((xVector > 100) & (xVector < 400))

        preStimAvg = nPSTH[controlStim][:, preIndx].mean(axis=1)
        preStimStd = nPSTH[controlStim][:, preIndx].std(axis=1)
        posStimAvg = nPSTH[controlStim][:, posIndx].mean(axis=1)
        burstingUnits = np.flatnonzero(
            posStimAvg > (preStimAvg + 5 * preStimStd))

    return nPSTH, burstingUnits, xVector, unitsID


def dist_XY(x, y):
    ''' Obtain the minimum distances X-->Y between the events in two vectors of
    timestamps of different length. Note: Make sure that y happens after x'''

    import warnings
    warnings.warn('Deprecated use sparse distance')

    x.sort()
    y.sort()
    x = np.array(x, ndmin=1)
    y = np.array(y, ndmin=1)

    if 0:  # : Slow for sizes of vectors > 1000
        xx = np.tile(x, (np.size(y), 1))
        yy = np.tile(y, (np.size(x), 1)).transpose()
    #    Dif = np.round_(yy - xx, 3)
        Dif = yy - xx
        Dif[Dif < 0] = np.inf  # : Changed 1.e6 to np.inf
        iDif = Dif.argmin(0)
        Dif = Dif.min(0)
        e = Dif != np.inf
        Dif = Dif[e]
        iDif = iDif[e]
        print(Dif.size)
    if 1:
        Dif = np.zeros(x.size, dtype=x.dtype)
        iDif = np.zeros(x.size, dtype=np.int)
        bound = y[-1]
        el = 0
        for i in range(x.size):
            if bound >= x[i]:
                iDif[i] = np.where(y >= x[i])[0][0]
            else:
                el += 1
    #    print(el)
        if el > 0:
            iDif = iDif[0:-el]
            Dif = y[iDif] - x[0:-el]
        else:
            Dif = y[iDif] - x
    return Dif, iDif


def dist_YX(x, y):
    ''' Obtain the distances Y<--X between the events in two vectors of timestamps
    of different lenght. Note: Make sure that y happens before x'''

    x.sort()
    y.sort()
    x = np.array(x, ndmin=1)
    y = np.array(y, ndmin=1)

#    xx = np.tile(x, (np.size(y), 1))
#    yy = np.tile(y, (np.size(x), 1)).transpose()
#    Dif = np.round(xx - yy, 3)
    Dif = np.empty(y.size)
    iDif = np.empty(y.size, dtype=np.int)
    bound = max(x)
    el = 0
    for i, row in enumerate(y):
        if bound >= row:
            iDif[i] = np.where(x >= row)[0][0]
        else:
            el += 1
#    print(el)
    if el > 0:
        iDif = iDif[0:-el]
        Dif = x[iDif] - y[0:-el]
    else:
        Dif = x[iDif] - y
#            iDif[i] =
#    print(len(a))
#    Dif = xx - yy
# Dif[Dif < 0.00] = np.inf  #: Changed 1.e6 to np.inf
#    iDif = Dif.argmin(0)
#    Dif = Dif.min(0)
#    e = Dif != np.inf
#    Dif = Dif[e]
#    iDif = iDif[e]

    return Dif, iDif
#    return a


def dist_XY_back(x, y):
    ''' Obtain the distances between the events in two vectors of timestamps
    of different lenght. Note: Make sure that x happens after y'''
    x.sort()
    y.sort()
    x = np.array(x, ndmin=1)
    y = np.array(y, ndmin=1)

    Dif = np.zeros(x.size, dtype=x.dtype)
    iDif = np.zeros(x.size, dtype=np.int)
    bound = y[0]
    el = 0
    for i in range(x.size):
        if bound < x[i]:
            iDif[i] = np.where(y <= x[i])[0][-1]
            logger.info((x[i], iDif[i]))
        else:
            el += 1
            logger.info(el)
#    print(el)
    if el > 0:
        iDif = iDif[el - 1:]
        Dif = y[iDif] - x[el - 1:]
    else:
        Dif = y[iDif] - x

    return Dif, iDif


# def sparse_distance(x, y, direction='xy', maxTime=1e6):
#     '''Sparse calculation of minimum distance between two vectors
#     of different length.
#
#     Inputs:
#         x,y:
#             vectors of timestamps of different length
#         direction:
#             "xy" 'x-->y' if y happens after x "yx" 'y-->x' x happens after y
#         maxTime:
#             maximum time lag between the two events
#
#     Output:
#         Dif:
#             distances between the vectors.
#         xIndxDif:
#             indices of the first vector
#         yIndxDif:
#             indices of the second vector that give those differeces'''
#
#     x = np.array(x, ndmin=2)
#     y = np.array(y, ndmin=2)
#
#     if x.size == 0 or y.size == 0:
#         return np.array([]), np.array([]), np.array([])
#
#     xx = np.tile(x, (y.size, 1))
#     yy = np.tile(y, (x.size, 1)).transpose()
#
#     if direction == 'xy':
#         Dif = np.round(yy - xx, 3)
#     elif direction == 'yx':
#         Dif = np.round(xx - yy, 3)
#
#     Dif[Dif < 0.00] = maxTime
#     TValues = Dif < maxTime
#     yIndx, xIndx = np.where(TValues)
#
#     logger.debug((x.size, y.size))
#
#     if y.size <= x.size:
#         _, indx = np.unique(yIndx, True)
#         indx = np.roll(indx - 1, -1)
#     else:
#         _, indx = np.unique(xIndx, True)
#         indx = np.roll(indx - 1, -1)
#         logger.debug((xIndx, indx))
#
#     return Dif[TValues][indx], xIndx[indx], yIndx[indx]


def sparse_raster(x, y, TimeWindow=(2, 2), BinSize=1, SFx=30000, SFy=30000, x_ind=False):
    '''Returns a raster of all events in x referenced to y

    Inputs:
        x,y:
            vectors of timestamps of different length
        TimeWindow:
            tuple timewindow in seconds

    Output:
        Raster:
            the matrix with the rasterplot'''

    if len(x) == 0 or len(y) == 0:
        Out = np.empty(
            (2, np.arange(-TimeWindow[0], TimeWindow[1], BinSize * 1. / 1000).size))
        Out[:] = np.nan
        # pdb.set_trace()
        Trial = Time = np.asarray([])
        if x_ind:
            return Time, Trial, Out, np.asarray([])
        else:
            return Time, Trial, Out

    if hasattr(x, 'SamplingFreq'):
        SFx = x.SamplingFreq

    if hasattr(y, 'SamplingFreq'):
        SFy = y.SamplingFreq

    if isinstance(x[0], np.float):
        x = x * 1000
        SFx = 1000
    x = np.array(x.astype(np.int32))

    if isinstance(y[0], np.float):
        y = y * 1000
        SFy = 1000
    y = np.array(y.astype(np.int32))

    # print(x,y)

    if SFy < SFx:
        logger.warn('Ref Event has a lower sampling freq than the Timestapmps')

    OutSF = int(1000 / BinSize)
    # pdb.set_trace()
    TimePre = int(TimeWindow[0] * OutSF)
    TimePost = int(TimeWindow[1] * OutSF)

    Xind = np.where([(x > y.min() - TimePre) & (x < y.max() + TimePost)])[1]
    x = x[(x > y.min() - TimePre) & (x < y.max() + TimePost)]

    x = x / (SFx * 1. / OutSF)
    y = y / (SFy * 1. / OutSF)

    Out = np.zeros((y.size, TimePre + TimePost))
    IndOut = list()

    for row, ev in zip(Out, y):
        diff = x - ev
        IndOut.extend(Xind[(diff >= -TimePre) & (diff < TimePost)].tolist())
        diff = np.floor(
            diff[(diff >= -TimePre) & (diff < TimePost)] + int(TimePre))
        diff = diff.astype(np.int32)

        if diff.size > 0:
            row[0:int(np.max(diff)) +
                1] = np.bincount(diff).astype(np.int32)

    Trial, Time = np.where(Out)

    Time = Time * BinSize * 1. / (1000) - TimeWindow[0]
    Trial = Trial

    if x_ind:
        return Time, Trial, Out, np.array(IndOut)  # , Xind

    return Time, Trial, Out


def gcd(a, b):
    """Return greatest common divisor using Euclid's Algorithm."""
    while b:
        a, b = b, a % b
    return a


def lcm(a, b):
    """Return lowest common multiple."""
    return a * b // gcd(a, b)


def sparse_raster_v2(x, y, TimeWindow=(2, 2), BinSize=1, SFx=30000, SFy=30000, x_ind=False):
    '''Returns a raster of all events in x referenced to y

    Inputs:
        x,y:
            vectors of timestamps of different length
        TimeWindow:
            tuple timewindow in seconds

    Output:
        Raster:
            the matrix with the rasterplot'''

    if hasattr(x, 'SamplingFreq'):
        SFx = x.SamplingFreq

    if hasattr(y, 'SamplingFreq'):
        SFy = y.SamplingFreq

    # SFxy = lcm(SFx,SFy) # least common multiple sampling frequency
    OutSF = int(1000 / BinSize)
    TimePre = TimeWindow[0] * OutSF
    TimePost = TimeWindow[1] * OutSF

    if isinstance(y[0], np.float):
        # assuming that y is in seconds
        y_mult = OutSF
    else:
        y_mult = 1. / SFy * OutSF

    if isinstance(x[0], np.float):
        # assuming that x is in seconds
        x_mult = OutSF
    else:
        x_mult = 1. / SFx * OutSF

    TimePre_x = TimePre / x_mult
    TimePost_x = TimePost / x_mult
    y_x = (y * y_mult / x_mult).astype(x.dtype)
    # msg = "TimePre_x:{}|TimePost_x:{},y_x:{}".format(TimePre_x,TimePost_x,y_x)
    # print(msg)
    Xind = np.where(
        [(x >= y_x.min() - TimePre) & (x < y_x.max() + TimePost)])[1]
    x = x[Xind]

    y = y * y_mult
    x = x * x_mult
    # print(x,y)

    if SFy < SFx:
        logger.warn('Ref Event has a lower sampling freq than the Timestamps')

    # print(x,y)
    Out = np.zeros((y.size, TimePre + TimePost))
    IndOut = list()

    for row, ev in zip(Out, y):
        diff = x - ev

        IndOut.extend(Xind[(diff >= -TimePre) & (diff < TimePost)].tolist())
        diff = np.floor(diff[(diff >= -TimePre) & (diff < TimePost)] + TimePre)
        # print(diff)

        if diff.size > 0:
            row[0:np.max(diff) + 1] = np.bincount(diff)

    Trial, Time = np.where(Out)
    # Counts = Out[np.where(Out)]

    Time = (Time - TimePre) / OutSF
    Trial = Trial

    if x_ind:
        return Time, Trial, Out, np.array(IndOut)  # , Xind

    return Time, Trial, Out


def PSTH_Ale(x, y, TimeWindow=(2, 2), BinSize=1, SFx=30000, SFy=30000):
    '''Returns a raster of all events in x referenced to y

    Inputs:
        x,y:
            vectors of timestamps of different length
        TimeWindow:
            tuple timewindow in seconds

    Output:
        Raster:
            the matrix with the rasterplot'''

    if hasattr(x, 'SamplingFreq'):
        SFx = x.SamplingFreq

    if hasattr(y, 'SamplingFreq'):
        SFy = y.SamplingFreq

    if SFy < SFx:
        logger.warn('Ref Event has a lower sampling freq than the Timestapmps')

    OutSF = int(1000 / BinSize)

    TimePre = TimeWindow[0] * OutSF
    TimePost = TimeWindow[1] * OutSF
    x = x[(x > y[0] - TimePre) & (x < y[-1] + TimePost)]
    x = x / (SFx / OutSF)
    y = y / (SFy / OutSF)

    Out = np.zeros(TimePre + TimePost)

    for ev in y:
        diff = x - ev
        diff = diff[(diff >= -TimePre) & (diff < TimePost)] + int(TimePre)
        if diff.size > 0:
            Out[0:np.amax(diff) + 1] += np.bincount(diff)

    return Out


# Trying MultiProcesessing -----------------------------------------------


def strial_rast(TSEvOutTrial):
    return TSEvOutTrial


def sparse_raster_multi(x, y, TimeWindow=(2, 2), BinSize=1, SFx=30000,
                        SFy=30000):
    '''Returns a raster of all events in x referenced to y multiprocessed

    Inputs:
        x,y:
            vectors of timestamps of different length
        TimeWindow:
            tuple timewindow in seconds

    Output:
        Raster:
            the matrix with the rasterplot'''

    from multiprocessing import Pool
    #    from itertools import product

    if hasattr(x, 'SamplingFreq'):
        SFx = x.SamplingFreq

    if hasattr(y, 'SamplingFreq'):
        SFy = y.SamplingFreq

    if SFy < SFx:
        logger.warn('Ref Event has a lower sampling freq than the Timestapmps')

    OutSF = int(1000 / BinSize)

    TimePre = TimeWindow[0] * OutSF
    TimePost = TimeWindow[1] * OutSF
    x = x[(x > y[0] - TimePre) & (x < y[-1] + TimePost)]
    x = x / (SFx / OutSF)
    y = y / (SFy / OutSF)
    Out = np.zeros((y.size, TimePre + TimePost))

    #        TS = TSEvOutTrial[0]
    #        ev = TSEvOutTrial[1]
    #        Out = TSEvOutTrial[2]
    #        Trial = TSEvOutTrial[3]
    #
    #        diff = TS - ev
    #        diff = diff[(diff >= -TimePre) & (diff < TimePost)] + int(TimePre)
    #        if diff.size > 0:
    #            Out[Trial, 0:np.amax(diff) + 1] = np.bincount(diff)

    args = zip([x for _ in y], [ev for ev in y],
               [Out for _ in y], range(y.size))

    pool = Pool()
    pool.map(strial_rast, args)
    pool.close()

    return Out


def sparse_distance(x, y, direction='xy', MaxTime=1e9,
                    SFx=None, SFy=None, rep='all', return_indices=True):
    '''
    Given the vectors x and y returns the vector Dv whose elements, [Dv]i, are
    obtained by computing the minimal distance between each element of x, [x]i,
    and y, [Dv]i = min((y-[x]i)>0) when x precedes y, direction ='xy'. If y
    precedes x, direction = 'yx', then [Dv]i = min(([x]i-y)>0). Different [x]i
    can precede the same [y]i_r to generate distance to a repeat of y
    elements.The flag rep decides how to report these events, by default rep is
    set to 'all' which allows any repetition. Set rep to 'last' or 'first' to
    select only the reference [x]i with the shortest or longest distance to the
    [y]i_r. If return_indeces is True, then the function returns the indeces of
    x Xi and y Yi used to compute the distance.

    As an example of repetition consider the following vectors:

    x = [1, 2, 3, 4, 5]
    y = [6]

    Dv = [5, 4, 3, 2, 1] # for 'xy' the same y event is considered

    Dv = [5] #rep='first'
    Dv = [1] #rep ='last'

    :param x:
    :param y:
    :param direction:
    :param MaxTime:
    :param SFx:
    :param SFy:
    :param rep:
    :param return_indices:
    '''
    if direction not in ['xy', 'yx']:
        raise Exception('Direction `{}` not understood'.format(direction))

    # making sure x and y are numpy array
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    if x.size == 0 or y.size == 0:
        if return_indices:
            return np.array([]), np.array([]), np.array([])
        else:
            return np.array([])

    # deals with different sampling frequencies
    if SFx is None:
        SFx = getattr(x, 'SamplingFreq', 1)
        if SFx is None:
            SFx = 1

    if SFy is None:
        SFy = getattr(y, 'SamplingFreq', 1)
        if SFy is None:
            SFy = 1

    if direction == 'xy':
        #         pdb.set_trace()
        Dm = distance_matrix(y, x)
        Xi, Yi = np.where((Dm >= 0) & (Dm < MaxTime / 1000 * SFx))
        Xi, Yii = np.unique(Xi, return_index=True)
        Yi = Yi[Yii]

    elif direction == 'yx':
        Dm = -distance_matrix(y, x)
        Dm = Dm[:, ::-1]
        Xi, Yi = np.where((Dm >= 0) & (Dm < MaxTime / 1000 * SFx))
        Xi, Yii = np.unique(Xi, return_index=True)
        Yi = -Yi[Yii] + Dm.shape[1] - 1
        Dm[Dm == 0] = 0

    try:
        myinf = np.iinfo(x.dtype).max
    except:
        myinf = np.finfo(x.dtype).max
    # use the maximum for the given datatype with np.iinfo
    Dm[(Dm < 0) | (Dm > MaxTime / 1000 * SFx)] = myinf
    Dv = np.min(Dm, 1)
    Dv = Dv[Dv != myinf]

    if rep == 'last' and len(Yi) != 0:
        IndDiff = np.append(Yi, Yi.max() + 1)
        IndDiff = np.unique(IndDiff, return_inverse=True)[1]
        IndDiff = np.diff(IndDiff)

        Yi = Yi[IndDiff == 1]
        Xi = Xi[IndDiff == 1]
        Dv = Dv[IndDiff == 1]

    if rep == 'first' and len(Yi) != 0:

        IndDiff = np.unique(Yi, return_index=True)[1]

        Yi = Yi[IndDiff]
        Xi = Xi[IndDiff]
        Dv = Dv[IndDiff]

    if return_indices:
        return Dv, Xi, Yi
    else:
        return Dv


def distance_matrix(x, y, rounding=None):
    '''
    Create the distance matrix D Matrix that contains the relative differences
    between each element of x and y. D = [dij] where dij = yj - xi.

    :param x:
    :type x:
    :param y:
    :type y:
    '''
    # pdb.set_trace()

    # making sure x and y are numpy array
    # if not isinstance(x, np.ndarray):
    x = np.array(x)
    # if not isinstance(y, np.ndarray):
    y = np.array(y)

    # x = x.reshape(1, -1)
    # y = y.reshape(-1, 1)

    # check for possible memory problems
    mem = x.nbytes * y.nbytes
    if mem / 2**20 > 30000:
        msg = 'Too much memory will be used'
        logger.error(msg)
        raise Exception(msg)

    DiffM = x.reshape(1, -1) - y.reshape(-1, 1)

    if rounding is not None:
        DiffM = np.round(DiffM, rounding)

    if hasattr(DiffM, 'distance_matrix'):
        DiffM._Header['Name'] = DiffM.Name + '_distance'

    return DiffM
