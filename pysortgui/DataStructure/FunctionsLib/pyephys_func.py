'''
Created on Mar 12, 2016

@author: scaglionea
'''
import os
from subprocess import call
import pdb

import numpy as np
import scipy as sp
import pandas as pd

from PyEphys.FunctionsLib.FileLib import get_fields_from_filename
from PyEphys.FileIO import BlackRock
from PyEphys import PyEphysCLS
from PyEphys.Classes.ContinuousClasses import ContinuousArrayCLS
from PyEphys.Classes.DiscreteClasses import DiscreteArrayCLS


from PyEphys.FileIO.BlackRock import get_reference_channels
# LOGGER ----------------------------------------------------------------------

import logging
logger = logging.getLogger(__name__)

__updated__ = "2017-04-25"

# CONSTANTS --------------------------------------------------------------
SORTING_INFO = [
    'ID',
    'Name',
    'Unit',
    'Channel',
    'Channel_ID',
    'ISI_Error',
    'Date',
    'Subject',
    'Sorting',
    'NumSpikes',
    'Firing_Rate',
    'Single_Unit',
    'Cluster_KL']


def wav_corr_score(TS):

    avg_waves = []
    for unit in TS.Units:
        if unit.Waveforms.size > 0:
            avg_waves.append(
                np.reshape(np.asarray(unit.Waveforms.mean(0)), (1, -1)))
        else:
            avg_waves.append(np.zeros((1, unit.Waveforms.shape[1])))

    corr_mat = sp.signal.correlate2d(  # @UndefinedVariable
        TS.Waveforms, avg_waves[0], mode='valid')
    for avg_wav in avg_waves[1:]:
        tmp = sp.signal.correlate2d(  # @UndefinedVariable
            TS.Waveforms, avg_wav, mode='valid')
        corr_mat = np.hstack((corr_mat, tmp))

    return corr_mat


def cluster_score(TS, method='correlation'):

    DKL = []
    TS._Waveforms = TS.Waveforms * 1. / np.abs(TS.Waveforms).max()

    if method == 'correlation':
        corr_mat = wav_corr_score(TS)

    for i, unit in enumerate(TS.Units):
        unit_ind = np.in1d(np.arange(TS.size), list(unit))
        # pdb.set_trace()
        qk = np.histogram(
            corr_mat[unit_ind, i], 100, range=(-5, 5))[0]
        pk = np.histogram(
            corr_mat[~unit_ind, i], 100, range=(-5, 5))[0]
        qk = qk * 1. / sum(qk) + np.finfo(np.float).eps
        pk = pk * 1. / sum(pk) + np.finfo(np.float).eps
        # print(qk,pk)
        DKL.append(
            1. / 2 * sp.stats.entropy(pk, qk) + 1. / 2 * sp.stats.entropy(qk, pk))  # @UndefinedVariable

    return np.asarray(DKL)


def generate_session_sorting_log(data, cluster_metrics=True, chan=None):

    if not isinstance(data, PyEphysCLS):
        data = PyEphysCLS(data)

    msg = "generating session sorting log file"
    logger.info(msg)

    file_name = data.H5FileName
    file_name = os.path.splitext(file_name)[0] + '_sorting' + '.csv'
    if os.path.isfile(file_name):
        df = pd.DataFrame.from_csv(file_name)
        for col in SORTING_INFO:
            if col not in df.columns.tolist():
                df[col] = np.nan
    else:
        df = pd.DataFrame()
        for par in SORTING_INFO:
            df[par] = np.nan
    # this needs to be reimplemented in PyEphys

    file_format = "Date_Subject"
    file_info = get_fields_from_filename(file_name, file_format)
    if chan is not None:
        spike_chans = [data.Spikes.find(ID=chan)]
    else:
        spike_chans = data.Spikes

    for spike_chan in spike_chans:

        if cluster_metrics:
            DKL = cluster_score(spike_chan)
        else:
            DKL = np.zeros(len(spike_chan.Units))
        # pdb.set_trace()
        if df.size > 0:
            df = df[df['Channel'] != spike_chan.Name]

        df = df.reset_index(drop=True)
        for i, unit in enumerate(spike_chan.Units):
            loc = df.shape[0] + 1
            df.loc[loc, 'Unit'] = unit.ID
            df.loc[loc, 'Name'] = unit.Name
            df.loc[loc, 'Channel'] = unit.Parent.Name
            df.loc[loc, 'Channel_ID'] = unit.Parent.ID
            bins, ISI = unit.TimeStamps.ISI()
            r_error = ISI[bins < 0.005].sum()
            df.loc[loc, 'ISI_Error'] = r_error
            df.loc[loc, 'Date'] = file_info['date']
            df.loc[loc, 'Subject'] = file_info['subject']
            df.loc[loc, 'ID'] = unit.UID
            df.loc[loc, 'Sorting'] = 'Manual'
            df.loc[loc, 'NumSpikes'] = len(unit)
            df.loc[loc, 'Firing_Rate'] = unit.TimeStamps.rate()
            df.loc[loc, 'Single_Unit'] = not unit.Header.get(
                'MultiUnit', True)
            df.loc[loc, 'Cluster_KL'] = DKL[i]
        df = df.sort_values(['Channel_ID', 'Unit'])

    df.to_csv(file_name)


def auto_sort_session(file_full_name, chan=None,
                      overwrite=False, extract_only=False, debug=False, log_level='WARNING',
                      sorting='offline', ref_chan=None, threshold=None, only_enabled=True,
                      threshold_sign=None, merged_additional=False, sorting_alg='Valley-Peak'):

    pass


def pre_process_for_sorting(file_full_name=None, channel_id=None,
                            filt_raw=False, sorting=None,
                            ref_raw=True, log=True, ref_chan=None):

    # call('clear')

    # loading data
    data = PyEphysCLS(file_full_name)
    # setting sorting
    if sorting == 'offline':
        data.use_offline()
        # data.clear()
        #del data

    if sorting == 'online':
        data.use_online()

    try:
        ref_dict = get_reference_channels(ccf_info_or_file=file_full_name)
    except:
        ref_chan = -1

    # determing the spike channel
    if channel_id is None:
        chan_TS = data.Spikes[0]
        channel_id = chan_TS.ID
    else:
        chan_TS = data.Spikes.find(ID=channel_id)

    # this makes sure Waves are retained even if the data file closes
    chan_TS.Waveforms

    # checking if Raw exists
    # pdb.set_trace()
    # TODO: it needs to be fixed
    if data.Raws is None or ref_chan is None or ref_chan is -1:
        raw_chan = np.zeros(
            max(chan_TS) + chan_TS.Waveforms.shape[1], dtype=chan_TS.Waveforms.dtype) * np.nan
        #         for i, ts in enumerate(chan_TS):
        #             ts = int(ts)
        #             offset = 10
        #             raw_chan[
        # ts - offset:ts + chan_TS.Waveforms.shape[1] - offset] =
        # chan_TS.Waveforms[i]
        raw_chan = np.zeros(0)
        raw_chan = ContinuousArrayCLS(raw_chan)
        raw_chan.SamplingFreq = chan_TS.SamplingFreq
        raw_chan._Header['TimeFirstPoint'] = 0
        return chan_TS, raw_chan, DiscreteArrayCLS(), ContinuousArrayCLS()

    if not ref_chan:
        ref_id = ref_dict[channel_id]
    else:
        ref_id = ref_chan

    if ref_id == 0:
        # no reference selected for the channel
        # trying to find a good reference assuming
        # that the channels are grouped 16 by 16
        grouping = 16
        chan_group = np.arange(1 + int((channel_id - 1) / grouping) * grouping,
                               17 + grouping * int((channel_id - 1) / grouping))
        ref_vec = [ref_dict[id_] for id_ in chan_group]
        if len(np.bincount(ref_vec)[1:]) > 0:
            ref_id = np.argmax(np.bincount(ref_vec)[1:]) + 1

    if ref_chan < 0:
        return chan_TS, ContinuousArrayCLS(), DiscreteArrayCLS(), ContinuousArrayCLS()

    raw = data.Raws.find(ID=channel_id)
    if ref_raw:

        if ref_id != 0:
            ref = data.Raws.find(ID=ref_id)
            raw_ref = raw - ref
            del ref
        else:
            raw_ref = raw
    else:
        raw_ref = raw
    del raw

    if filt_raw:
        msg = "Filtering channel... "
        logger.info(msg)
        chan_raw = raw_ref.design_and_filter(
            LowCutOff=250, HighCutOff=5000, FilterFamily='Butterworth')
    else:
        chan_raw = raw_ref

    del raw_ref

    opt_stim_raw = data.Raws[-1]
    if opt_stim_raw.Name != 'OptStim':
        if data.Analogs is not None:
            opt_stim_raw = data.Analogs[0]
    if opt_stim_raw.Name == 'OptStim':
        opt_stim_TS = opt_stim_raw.extract_waveforms(
            threshold=1000, alg="Crossings")['TimeStamps']
    else:
        opt_stim_TS = chan_TS.__class__()
        opt_stim_raw = chan_raw.__class__()

    return chan_TS, chan_raw, opt_stim_TS, opt_stim_raw
