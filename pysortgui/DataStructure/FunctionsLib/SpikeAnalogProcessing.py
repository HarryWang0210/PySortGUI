# -*- coding: utf-8 -*-
"""
..
    Created on Tue Oct 22 01:09:36 2013
    makes use of scipi.signal
    @author: Alessandro Scaglione
    Contact: alessandro.scaglione@gmail.com

    Version: 0
"""
from __future__ import division
from __future__ import print_function


import numpy as np


# import scipy.signal as signal
# from pylab import *
# LOGGER ----------------------------------------------------------------------

import logging
logger = logging.getLogger(__name__)


def spike_triggered_average(x, EvList, TimeWindow=(.1, .1), SFC=None,
                            SFTs=None, ADC=None, RemoveMean=False, return_time=False, log_level='INFO'):
    '''
    Estimate the Spike Triggered Average for given a continuous signal x for a
    given timewindow

    :param x: The Signal
    :type x: numpy.ndarray or derived
    :param EvList: a sequence of timestamps
    :type EvList: any iterable
    :param TimeWindow: timewindow of the signal in seconds ex. (.1,.1)
        for 100 ms before and after the event
    :type TimeWindow: tuple
    :param RemoveMean: flag to normalize to pre-stimulus mean
    :type RemoveMean: boolean
    :param SFC: sampling frequency of the continuous
    :type SFC: numeral
    :param SFTs: sampling frequency of the timestamps
    :type SFTs: numeral

    .. note::
        if SFC or SFTs is not specified then the sampling rate of the
        Continuous and the Events is set to 1
    '''

    logger.debug('Setting additional parameters')
    if SFC is None:
        SFC = getattr(x, 'SamplingFreq', 2000)

    if SFTs is None:
        # TODO: implement common class for common attributes
        SFTs = getattr(EvList, 'SamplingFreq', 30000)

    if ADC is None:
        ADC = 1.
        ADC = getattr(x, 'ADC', 1)

    # * SFC Offset is in ticks
    OffSet = dict.get(x.Header, 'TimeFirstPoint', 0)
    logger.debug('Parameters set SFC:%s SFTs:%s ADC:%s OffSet:%s' %
                 (SFC, SFTs, ADC, OffSet))

    # TODO: Change implementation use reshape and boolean indexing to speed up
    # fixing sampling freq
    EvList = np.array(EvList)
    # pdb.set_trace()
    EvList = (EvList * (SFC / SFTs) * 1.0 - OffSet).astype(np.int)
    # print(EvList)

    # putting time window in seconds
    TimePre = int(TimeWindow[0] * SFC)
    TimePost = int(TimeWindow[1] * SFC)

    S = TimePre + TimePost
    if RemoveMean:
        Trials = [x[(SingleEvent - TimePre):(TimePost + SingleEvent)] -
                  np.mean(x[(SingleEvent - TimePre):(SingleEvent)])
                  for SingleEvent in EvList if
                  x[(SingleEvent - TimePre):(TimePost
                                             + SingleEvent)].size == S]
    else:
        Trials = [x[(SingleEvent - TimePre):(TimePost + SingleEvent)]
                  for SingleEvent in EvList if
                  x[(SingleEvent - TimePre):(TimePost
                                             + SingleEvent)].size == S]

    Trials = np.array(Trials)
    out = np.asarray(Trials * ADC).view(x.__class__)
    if hasattr(x, '_Header'):
        out._Header = x.Header

    if return_time:
        # pdb.set_trace()
        x_time = np.arange(
            Trials[0].size) * 1. / SFC - TimeWindow[0]
        return out, x_time
    return out
