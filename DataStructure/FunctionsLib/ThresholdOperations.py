# -*- coding: utf-8 -*-

"""
Library for threshold operations on continuous signals

"""
import logging
import pdb  # @UnusedImport
import time
import scipy as sp
import scipy.signal
import numpy as np
# import time

# import PyEphys.Classes.DiscreteClasses

THRESHOLD_MULTIPLIER = 4

# LOGGER ----------------------------------------------------------------------

logger = logging.getLogger(__name__)

# custom for this module
# logger.setLevel('INFO')
# logger.handlers[0].formatter.FORMATS[
#     20] = '--INFO:%(module)s:%(lineno)s\n%(message)s'
# uncomment below to have only the messages excluding module and lineno
# logger.handlers[0].formatter.FORMATS[20] = '%(message)s'


# def RMS(x, flag='normal'):
#     '''
#     Estimate the RMS of the signal accordingly with the type flag.

#     :param x: the signal
#     :type x: any compatible numpy type
#     :param flag: flag for the type of calculation see below
#     :type flag: str
#     '''

#     if flag == 'meansub':
#         return np.sqrt(np.sum((1. * x - x.mean()) ** 2) * 1. / x.size)
#     else:
#         return np.sqrt(np.sum((1. * x) ** 2) * 1. / x.size)


def find_local_maxima(x, DeadTime):
    '''
    finds any local maxima in the data

    :param x: the signal
    :type x: any compatible numpy type
    :param deadtime: the the deadtime for detecting spikes
    :type deadtime: int
    '''

    extrema = scipy.signal.argrelextrema(x, np.greater_equal)[0]
    # ExY=x[extrema[0]]
    apply_deadtime(extrema, DeadTime)
    return extrema  # ,ExY


def find_threshold_crossings(x, threshold, deadtime=0):
    '''
    find threshold crossing for the given threshold. It uses numpy.where
    with logical condition to find the threshold crossings. Note that the
    way in which the logical condition is set will find the first point before
    threshold crossing

    :param x: the signal
    :type x: any compatible numpy type
    :param threshold: the threshold value, if none then 4*RMS is used
    :type threshold: int
    :param deadtime: the the deadtime for detecting spikes
    :type deadtime: int
    '''

    # if threshold == None:
    #     # Quian Quiroga 2004
    #     # Threshold = 5 sn where sn = median(abs(x)/0.6745)
    #     sn = np.median(abs(x) / 0.6745)
    #     # multiplier = THRESHOLD_MULTIPLIER
    #     threshold = -multiplier * sn
    #     msg = 'Setting threshold:{:.3f} --> {} times estimated SD of the noise {:.3f}'.format(
    #         threshold, multiplier, sn)
    #     logger.info(msg)

    # print threshold
# =========================================================================
# Old way
#     t0 = time.time()
#     x = x.astype(np.float)
#     ThSig = x - threshold
#     if threshold > 0:
#         ThSig = -ThSig
#
#     crossing = scipy.where(((ThSig[:-1] * ThSig[1:]) < 0) & (ThSig[:-1] > 0))[0]
#     t1 = time.time()
#     print(t1-t0)
# =========================================================================

#     t0 = time.time()

    if threshold > 0:
        ThSig = x > threshold
    else:
        ThSig = x < threshold

    # first point after the crossing +1
    crossing = np.where(ThSig[:-1] ^ ThSig[1:])[0][::2] + 1

#     t1 = time.time()
#     print(t1-t0)
    # ThY=x[crossing[0]]
    # print crossing
    if crossing.size > 1:
        apply_deadtime(crossing, deadtime)
    return crossing, float(threshold)


def find_valleys_after_threshold_crossings(x, threshold, deadtime=48,
                                           deadtime_alg='keep_max'):
    '''
    finds valleys or peaks after threshold crossing for the given
    threshold. It uses numpy.where with logical condition to find the
    valleys or peaks. Note that the way in which the logical condition is set
    will find any valleys/peaks after threshold crossings.

    :param x: the signal
    :type x: any compatible numpy type
    :param threshold: the threshold value, if none then 4*RMS is used
    :type threshold: int
    :param deadtime: the the deadtime for detecting spikes
    :type deadtime: int
    '''
    # if threshold == None:
    #     # Quian Quiroga 2004
    #     # Threshold = 5 sn where sn = median(abs(x)/0.6745)
    #     sn = np.median(abs(x) / 0.6745)
    #     # multiplier = THRESHOLD_MULTIPLIER
    #     threshold = multiplier * sn
    #     msg = 'Setting threshold:{} --> {} times estimated SD of the noise {}'.format(
    #         threshold, multiplier, sn)
    #     logger.info(msg)

    # =========================================================================
    # t0 = time.time()
    # x = x * 1.
    # print x
    # ThSig = x - threshold
    # print ThSig
    # if threshold > 0:
    #     ThSig = -ThSig
    # crossing = scipy.where(((ThSig[:-1] * abs(ThSig[1:])) < 0) &
    #         (((ThSig[:-1] - np.roll(ThSig[:-1], 1)) < 0) & ((ThSig[:-1] - ThSig[1:]) < 0)))[0]
    # ThY=x[crossing[0]]
    # threshold = float(threshold)
    # t1 = time.time()
    # print(t1-t0)
    # =========================================================================

#     t0 = time.time()

    if threshold > 0:
        ThSig = x > threshold / 2
    else:
        ThSig = x < threshold / 2

    ind = np.where(ThSig)[0]

    ThSig = x[ind]

    if threshold > 0:
        crossing = (ind[1:-1])[(ThSig[1:-1] > ThSig[2:]) &
                               (ThSig[1:-1] > ThSig[0:-2])]
    else:
        crossing = (ind[1:-1])[(ThSig[1:-1] < ThSig[2:]) &
                               (ThSig[1:-1] < ThSig[0:-2])]

    # pdb.set_trace()

    ind = crossing

    if threshold > 0:
        crossing = x[crossing] >= threshold
    else:
        crossing = x[crossing] <= threshold

    crossing = ind[np.where(crossing)[0]]
    #: TODO: check for border effect

#     print(crossing.shape)
#     t1 = time.time()
#     print(t1-t0)
    if deadtime_alg == 'keep_max':
        crossing = apply_deadtime_keep_max(crossing, x[crossing], deadtime)
    else:
        crossing = apply_deadtime(crossing, deadtime)
    # print crossing
    return crossing, float(threshold)


def apply_deadtime(crossing, DeadTime=48):
    if DeadTime != float(0):
        deadtimed = [0]
        for i, el in enumerate(crossing[1:]):
            if el - crossing[deadtimed[-1]] > DeadTime:
                deadtimed.append(i + 1)
        # deadtimed=[0]+(scipy.where((crossing[1:]-crossing[:-1])>DeadTime)[0]+1).tolist()

        crossing = crossing[deadtimed]

    return crossing


def apply_deadtime_keep_max(crossing, values, DeadTime=48):
    values = abs(values)
    if DeadTime != float(0):
        deadtimed = []
        for i, el in enumerate(crossing[1:]):
            if i == 0:
                if el - crossing[0] <= DeadTime:
                    deadtimed.append(np.argmax(values[[0, 1]]))
                else:
                    deadtimed.extend([0, 1])
            else:
                if el - crossing[deadtimed[-1]] <= DeadTime:
                    if values[deadtimed[-1]] > values[i + 1]:
                        pass
                    else:
                        deadtimed.pop()
                        deadtimed.append(i + 1)
                else:
                    deadtimed.append(i + 1)
        # deadtimed=[0]+(scipy.where((crossing[1:]-crossing[:-1])>DeadTime)[0]+1).tolist()

        crossing = crossing[deadtimed]

    return crossing


def get_crossings(x, threshold, alg='Valley-Peak', deadtime=25,
                  dual_threshold=False, extract_after=0):
    '''
    generates the list of crossings accordingly with the chosen algorithm. It is
    basically a wrapper around all the possible threshold algs.

    :param x: the signal
    :type x: any compatible numpy type
    :param alg: flag for the type of alg to get crossings/valleys/peaks
    :type alg: str
    :param threshold: the threshold value, if none then 4*RMS is used
    :type threshold: int
    :param deadtime: the the deadtime for detecting spikes
    :type deadtime: int
    '''

    # if threshold is not None:
    # if threshold > 0:
    #     threshold_sign = '+'
    # else:
    #     threshold_sign = '-'
    x = x[extract_after:]
    if deadtime == 0:
        deadtime = 25

    msg = 'Extracting Waveforms: alg={}|threshold:{}|deadtime:{}'.format(
        alg, threshold, deadtime)
    logger.info(msg)

    # if threshold == None:
    #     # Quian Quiroga 2004
    #     # Threshold = 5 sn where sn = median(abs(x)/0.6745)
    #     if multiplier is None:
    #         multiplier = THRESHOLD_MULTIPLIER
    #     sn = float(np.median(abs(x) / 0.6745))
    #     threshold = multiplier * sn
    #     msg = 'Setting threshold:|{:.3f}| --> {} times estimated SD of the noise {:.3f}'.format(
    #         threshold, multiplier, sn)
    #     logger.info(msg)
    #     thresholds = [threshold, -threshold]
    #     # thresholds = [- threshold]
    #     msg = "Based on more number of crossings"
    # else:
    thresholds = [threshold]
    msg = ''
    crossings_list = []
    crossings_size_list = []

    # if threshold_sign is None:
    #     if sp.stats.skew(x) > 0:  # @UndefinedVariable
    #         thresholds = [np.abs(threshold)]
    #     else:
    #         thresholds = [-np.abs(threshold)]
    #     msg = "Based on signal skewness"
    # else:
    #     if threshold_sign == '+':
    #         thresholds = [np.abs(threshold)]
    #     if threshold_sign == '-':
    #         thresholds = [-np.abs(threshold)]
    #     msg = "Based on user threshold sign"
    for threshold in thresholds:
        if alg == 'Valley-Peak':
            crossings_list.append(find_valleys_after_threshold_crossings(
                x, threshold, deadtime))
        else:
            crossings_list.append(
                find_threshold_crossings(x, threshold, deadtime))
        crossings_size_list.append(len(crossings_list[-1][0]))
        # print(crossings_size_list)
    if dual_threshold:
        crossings = np.hstack((crossings_list[0][0], crossings_list[1][0]))
        crossings.sort()
        apply_deadtime(crossings, deadtime)
        _, Threshold = crossings_list[np.argmax(crossings_size_list)]
    else:
        crossings, Threshold = crossings_list[np.argmax(crossings_size_list)]

    # pdb.set_trace()
    msg = 'Threshold is {}.'.format(
        Threshold) + msg
    logger.info(msg)

    # if hasattr(x, '_Header'):
    #     header_field_name = 'DisconnectionTimeStamps'
    #     if header_field_name in x.Header:
    #         nsamples = 40
    #         msg = 'discarding timestamps that falls within {} samples of blackouts'.format(
    #             nsamples)
    #         logger.info(msg)
    #         # nsamples = int(x.SamplingFreq / 2)

    #         disc_rep = np.tile(
    #             x.Header['DisconnectionTimeStamps'], (2 * nsamples, 1))
    #         blackout_zone = (
    #             np.arange(0, 2 * nsamples) - nsamples).reshape(-1, 1)
    #         blackout_zone_matrix = np.tile(
    #             blackout_zone, (1, x.Header['DisconnectionTimeStamps'].size))
    #         bad_ts = disc_rep - blackout_zone_matrix
    #         bad_crossings = np.in1d(crossings, bad_ts)
    #         msg = "removing {} crossings".format(bad_crossings.sum())
    #         logger.info(msg)
    #         crossings = crossings[~bad_crossings]

    return crossings + extract_after, Threshold


def extract_waveforms(x, chan_ID, threshold, OffSet=0, wav_length=36, alg='Crossings',
                      deadtime=42, dual_threshold=False, amplitude_rejection=False, log_level='INFO',
                      ts_type='Spikes', extract_after=0):
    '''
    This function extract the waveforms from the given signal x with threshold
    threshold_value, with length wav_length and with threshold algorithm given
    by threshold_alg. It returns a dict, containing two main keys: Waveforms:
    dict containing all TimeStamps and Waveforms. The Waveforms are returned as
    an array and each one is separated by none in addition some extraction
    metadata are returned

    :param x: the signal
    :type x: any compatible numpy type
    :param wav_length: length of the waveforms to extract
    :type wav_length: int
    :param alg: flag for the type of alg to get crossings/valleys/peaks
    :type alg: str
    :param threshold: the threshold value, if none then 4*RMS is used
    :type threshold: int
    :param deadtime: the the deadtime for detecting spikes
    :type deadtime: int


    '''
    #     x = remove_artifacts(x)
    # print args,kwargs
    crossings, Threshold = get_crossings(
        x, threshold, alg, deadtime, dual_threshold=dual_threshold, extract_after=extract_after)

    points_before = int(wav_length / 4)
    if alg == 'Valley-Peak':
        points_before = int(wav_length * 3 / 8)

    points_after = wav_length - points_before
    # print crossings,Threshold
    # removing crossings that would lead to incomplete waveforms
    crossings = crossings[crossings > points_before]
    crossings = crossings[crossings + wav_length - wav_length / 4 < x.size]
    # SFC = getattr(x, 'SamplingFreq', 1)
    # OffSet = dict.get(x.Header, 'TimeFirstPoint', 0)  # * SFC
    # crossings = crossings + OffSet

    Waveforms = dict(Header={})
    Waveforms['Header']['WaveLength'] = wav_length
    Waveforms['Header']['NumRecords'] = crossings.size
    Waveforms['Header']['PointsBefore'] = points_before
    Waveforms['Header']['PointsAfter'] = points_after
    Waveforms['Header']['Threshold'] = Threshold

    Waveforms['Waveforms'] = np.empty((crossings.size, wav_length))
    Header = dict()
    # Header['Name'] = x.Header.get('Name', 'Unknown')
    Header['Name'] = 'Unknown'

    # Header['SamplingFreq'] = x.Header.get('SamplingFreq', 30000)
    Header['SamplingFreq'] = 30000

    Header['Threshold'] = Threshold
    Header['NumRecords'] = crossings.size

    if chan_ID <= 128:
        type_ = 'Spikes'
        Header['H5Location'] = '/Spikes/spike{:03d}'.format(chan_ID)
    elif chan_ID > 128:
        type_ = 'Events'
        Header['H5Location'] = '/Events/event{:03d}_offline'.format(chan_ID)
    else:
        type_ = 'Unknown'
        Header['H5Location'] = '/Unknown/unkwn{:03d}'.format(chan_ID)
    Header['H5Name'] = 'TimeStamps'

    Header['ID'] = chan_ID

    Header['Type'] = type_
    tokeep = []
    # pdb.set_trace()
    for i, ts in enumerate(crossings):
        # Waveforms['Waveforms'][i*(wav_length+1):(i+1)*(wav_length+1)] = \
        # x[ts-Waveforms['Header']['PointsBefore']:ts+Waveforms['Header']['PointsAfter']+1]
        # Waveforms['Waveforms'][(i+1)*(wav_length+1)-1]=None
        Waveforms['Waveforms'][i][:] = x[ts - Waveforms['Header']
                                         ['PointsBefore']:ts + Waveforms['Header']['PointsAfter']]
        # rejecting on amplitude:

        reject_threshold = 30000

        if (Waveforms['Waveforms'][i][:] > reject_threshold).any() or (Waveforms['Waveforms'][i][:] < -reject_threshold).any():
            pass
        else:
            tokeep.append(i)

    if amplitude_rejection:
        msg = "removing {} waves based on a rejection threshold of {}".format(
            crossings.size - len(tokeep), reject_threshold)
        # removing rejected wave
        logger.info(msg)
        crossings = crossings[tokeep]
        Waveforms['Waveforms'] = Waveforms['Waveforms'][tokeep, :]

        if Waveforms['Waveforms'].shape[0] > 10000000:
            # removing other wfs on distribution
            h_threshold = 5
            maxW = Waveforms['Waveforms'].max(1)
            # upper th
            counts, level = np.histogram(
                maxW, bins='fd')
            # argmax_counts = np.argmax(counts)
            supra_th = (counts >= h_threshold)
            first_after_supra = supra_th.size - supra_th[::-1].argmax()
            noise_level = level[first_after_supra]
            if noise_level.size > 0:
                max_threshold = noise_level.min()
            else:
                max_threshold = reject_threshold

            msg = "Waves with maximum > {} will be removed".format(
                max_threshold)
            # print(counts[argmax_counts:])
            # lower th

            minW = Waveforms['Waveforms'].min(1)
            counts, level = np.histogram(
                minW, bins='fd')
            # argmax_counts = np.argmax(counts)
            supra_th = (counts > h_threshold)
            last_before_supra = supra_th.argmax() - 1
            noise_level = level[last_before_supra]
            if noise_level.size > 0:
                min_threshold = noise_level.max()
            else:
                min_threshold = -reject_threshold
            # print(counts[0:argmax_counts])
            msg = msg + \
                "\nWaves with minimum < {} will be removed".format(
                    min_threshold)
            logger.info(msg)
            # print(counts)

            if not -min_threshold == max_threshold == reject_threshold:
                torem = (maxW >= max_threshold) | (minW <= min_threshold)
                msg = "removing {} waves based on histogram level counts threshold of {}".format(
                    torem.sum(), h_threshold)
                logger.info(msg)
                crossings = crossings[~torem]
                Waveforms['Waveforms'] = Waveforms['Waveforms'][~torem, :]

    # removing rejected wave
    # pdb.set_trace()
    crossings = crossings + OffSet
    crossings = (crossings * 30000 / Header['SamplingFreq']).astype(np.int32)
    Header['SamplingFreq'] = 30000
    # cHeader = x.Header.copy()

    # some header entry fixing
    Header['TimeFirstPoint'] = 0
    # h5_file_name = x._Header.get('_H5FileName', None)
    # if h5_file_name:
    #     Header['_H5FileName'] = h5_file_name

    # cHeader.update(Header)
    # pdb.set_trace()
    # if type_ in ['Events']:
    #     cHeader['Name'] = cHeader['Name'] + '_offline'
    # cHeader['Comment'] = "Extracted on {}".format(
    #     time.strftime('%Y-%B-%d', time.localtime()))
    return Waveforms, crossings
    Waveforms['TimeStamps'] = PyEphys.Classes.DiscreteClasses.DiscreteArrayCLS(
        crossings, header=cHeader, Prefix=type_)
    Waveforms['TimeStamps'].Units.reset()

    Waveforms['TimeStamps']._Waveforms = Waveforms['Waveforms']

    return Waveforms
