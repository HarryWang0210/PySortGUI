from __future__ import annotations
from numpy import ndarray, ufunc
import tables
import os
import pandas as pd
import numpy as np
from DataStructure.pyephysv3 import (loadPyephys, loadRaws, loadSpikes,
                                     saveSpikes, saveSpikesHeader)
# from pyephysv2 import loadPyephys, loadRaws, loadSpikes
from DataStructure.FunctionsLib.SignalProcessing import design_and_filter
from DataStructure.FunctionsLib.ThresholdOperations import extract_waveforms
from DataStructure.FunctionsLib.Sorting import auto_sort

from sklearn.decomposition import PCA
from sklearn.preprocessing import MaxAbsScaler

import logging
logger = logging.getLogger(__name__)


class SpikeSorterData(object):
    def __init__(self, filename, parent=None):
        super().__init__()
        self._filename = filename
        self._raws_dict = dict()
        self._channel_name_to_ID = dict()
        self._headers = loadPyephys(filename)
        self._createRawsData()
        self._createSpikesData()

    @property
    def filename(self):
        return self._filename

    @property
    def channel_IDs(self):
        return list(self._channel_name_to_ID.values())

    @property
    def raws_header(self):
        records = []
        for ID in self.channel_IDs:
            raw_object = self.getRaw(ID)
            records.append(raw_object.header)
        if len(records) < 1:
            return None
        return pd.DataFrame.from_records(records)

    @property
    def spikes_header(self):
        records = []
        for ID in self.channel_IDs:
            raw_object = self.getRaw(ID)
            for label in raw_object.spikes:
                spike_object = self.getSpike(ID, label)
                records.append(spike_object.header)
        if len(records) < 1:
            return None
        return pd.DataFrame.from_records(records)

    @property
    def events_header(self):
        records = []
        if len(records) < 1:
            return None
        return None

    def _createRawsData(self):
        raws_header = self._headers['RawsHeader'].to_dict('records')
        for header in raws_header:
            self._raws_dict[header['ID']] = ContinuousData(filename=self._filename,
                                                           header=header,
                                                           data_type='Raw')
            self._channel_name_to_ID[header['Name']] = header['ID']

    def _createSpikesData(self):
        if self._headers['SpikesHeader'] is None:
            return
        spikes_header = self._headers['SpikesHeader'].to_dict('records')
        for header in spikes_header:
            spike_object = DiscreteData(filename=self._filename,
                                        header=header,
                                        data_type='Spikes',
                                        _from_file=True)
            raw_object = self.getRaw(header['ID'])
            raw_object.setSpike(
                spike_object, label=header['Label'])

    def getRaw(self, channel: int | str, load_data: bool = False) -> ContinuousData | None:
        """_summary_

        Args:
            channel (int or string): channel ID or channel name.

        Returns:
            ContinuousData: Raw object.
        """
        if load_data:
            self.loadRaw(channel)
        chanID = self.validateChannel(channel)

        return self._raws_dict.get(chanID)

    def loadRaw(self, channel: int | str):
        """load Raw data, return nothing.

        Args:
            channel (int | str): channel ID or channel name.
        """
        raw_object = self.getRaw(channel)
        if raw_object is None:
            return
        raw_object._loadData()

    def getSpike(self, channel: int | str, label: str, load_data: bool = False) -> DiscreteData | None:
        """_summary_

        Args:
            channel (int | str): channel ID or channel name.
            label (str): spike label.

        Returns:
            _type_: Spike object
        """
        raw_object = self.getRaw(channel, load_data)
        if raw_object is None:
            return

        if not label in raw_object.spikes:
            logger.warning(f'No label {label} spike data in channel {channel}')
            return None

        if load_data:
            self.loadSpike(channel, label)
        return raw_object.getSpike(label)

    def loadSpike(self, channel: int | str, label: str):
        """load Spike data, return nothing.

        Args:
            channel (int | str): channel ID or channel name.
            label (str): spike label.
        """
        spike_object = self.getSpike(channel, label)
        if spike_object is None:
            return
        spike_object._loadData()

    def getEvent(self):
        # TODO
        logger.critical('Unimplemented function.')
        return

    def subtractReference(self, channel: int | str, reference: int | list) -> ContinuousData:
        ch_object = self.getRaw(channel, load_data=True)

        if isinstance(reference, int):
            referenceID = self.validateChannel(reference)
            ref_object = self.getRaw(referenceID, load_data=True)

        result = ch_object.subtractReference(ref_object.data, referenceID)

        return result

    def filt(self, channel: int | str, ref: list, use_median: bool = False, *args, **kargs) -> np.ndarray:
        # TODO
        """_summary_

        Args:
            channel (int or string): channel ID or channel name.
            ref (list): list of reference channel.
            use_median (bool): _description_
        """
        logger.critical('Unimplemented function.')
        return

    def sort(self, *args, **kargs):
        # TODO
        logger.critical('Unimplemented function.')
        return

    def saveChannel(self, channel):
        ch = self.getRaw(channel)
        records = []
        for label in ch.spikes:
            spike = ch.getSpike(label)
            spike._header['H5Name'] = 'TimeStamps'
            if label == 'default':
                spike._header['H5Location'] = f'/Spikes/spike{ch.channel_ID:03}'
            else:
                spike._header['H5Location'] = f'/Spikes/spike{ch.channel_ID:03}{label}'
            logger.debug(label)
            spike._unit_header['H5Location'] = spike._unit_header['ID'].apply(
                lambda ID: spike._header['H5Location'] + f'/Unit_{ID:02}')
            spike._unit_header['H5Name'] = 'Indxs'
            spike._unit_header['ParentID'] = ch.channel_ID
            spike._unit_header['ParentType'] = 'Spikes'
            spike._unit_header['Type'] = 'Unit'

            saveSpikes(self.filename, spike.header, spike.unit_header,
                       spike.unit_IDs, spike.timestamps, spike.waveforms)
            records.append(spike.header)
        spikes_header = self._headers['SpikesHeader']
        new_spikes_header = pd.DataFrame(
            spikes_header[spikes_header['ID'] != ch.channel_ID])
        # logger.debug(pd.DataFrame.from_records(records).dtypes)

        new_spikes_header = pd.concat([new_spikes_header, pd.DataFrame.from_records(records)],
                                      axis=0, ignore_index=True)
        # logger.debug(new_spikes_header.dtypes)
        self._headers['SpikesHeader'] = new_spikes_header
        # logger.debug(new_spikes_header)
        saveSpikesHeader(self.filename, self._headers['SpikesHeader'])
        # logger.debug(self.spikes_header)

    def validateChannel(self, channel: int | str) -> int:
        if isinstance(channel, str):
            channel = self._channel_name_to_ID.get(channel)
            if channel is None:
                logger.warning('Unknowed channel name.')
                return
            return channel

        elif isinstance(channel, int):
            if channel in self._channel_name_to_ID.values():
                return channel
            else:
                logger.warning('Unknowed channel ID.')
                return


class ContinuousData(object):
    # def __new__(cls, input_array=None, filename: str = '', header: dict = dict(), data_type: str = 'Filted'):
    #     """_summary_

    #     Args:
    #         input_array (array-like, optional): _description_. Defaults to [].
    #         filename (str, optional): _description_. Defaults to ''.
    #         header (dict, optional): _description_. Defaults to dict().
    #         data_type (str, optional): 'Raw' | 'Filted'. Defaults to 'Filted'.

    #     Returns:
    #         ContinuousData: _description_
    #     """
    #     # Input array is an already formed ndarray instance
    #     # We first cast to be our class type
    #     obj = np.asarray(input_array).view(cls)
    #     # add the new attribute to the created instance
    #     obj._header = header.copy()
    #     # obj._header = header.copy()
    #     obj._data_type = data_type
    #     obj._filename = filename
    #     obj._data_loaded = False
    #     if (not input_array is None) and (len(input_array) != 0):
    #         obj._data_loaded = True

    #     # Finally, we must return the newly created object:
    #     return obj

    def __init__(self, input_array: np.ndarray = [], filename: str = '', header: dict = dict(), data_type: str = 'Filted'):
        super().__init__()
        self._data = np.asarray(input_array)
        self._filename = filename
        self._header = header.copy()
        self._data_type = data_type
        self._data_loaded = False
        if len(input_array) > 0:
            self._data_loaded = True

        self._estimated_sd = None
        self._spikes = dict()

    @property
    def filename(self):
        return self._filename

    @property
    def channel_ID(self):
        if not isinstance(self._header['ID'], int):
            self._header['ID'] = int(self._header['ID'])
        return self._header['ID']

    @property
    def channel_name(self) -> int:
        return self._header['Name']

    @property
    def header(self):
        return self._header.copy()

    @property
    def data(self):
        return self._data.copy()

    @property
    def data_type(self):
        return self._data_type

    @property
    def fs(self):
        if not isinstance(self._header['SamplingFreq'], (int, float)):
            self._header['SamplingFreq'] = int(self._header['SamplingFreq'])
        return self._header['SamplingFreq']

    @property
    def reference(self) -> list | None:
        if self._header['ReferenceID'] == '':
            self._header['ReferenceID'] = None
        if isinstance(self._header['ReferenceID'], str):
            self._header['ReferenceID'] = int(self._header['ReferenceID'])
        return self._header['ReferenceID']

    @property
    def low_cutoff(self) -> int | float:
        if not isinstance(self._header['LowCutOff'], (int, float)):
            self._header['LowCutOff'] = float(self._header['LowCutOff'])
        return self._header['LowCutOff']

    @property
    def high_cutoff(self) -> int | float:
        if not isinstance(self._header['HighCutOff'], (int, float)):
            self._header['HighCutOff'] = float(self._header['HighCutOff'])
        return self._header['HighCutOff']

    @property
    def threshold(self) -> int | float:
        if not isinstance(self._header['Threshold'], (int, float)):
            self._header['Threshold'] = float(self._header['Threshold'])
        return self._header['Threshold']

    @property
    def estimated_sd(self):
        if isinstance(self._estimated_sd, (int, float)):
            return self._estimated_sd
        return self._estimatedSD()

    @property
    def spikes(self):
        return list(self._spikes.keys())

    def isLoaded(self) -> bool:
        return self._data_loaded

    def _loadData(self):
        if self._data_type != 'Raw' or self.isLoaded():
            logger.warning('No data to load.')
            return

        data = loadRaws(
            self._filename, self._header['H5Location'], self._header['H5Name'])

        self._data = np.asarray(data)

    def setSpike(self, spike_object, label: str = 'default'):
        self._spikes[label] = spike_object

    def getSpike(self, label: str) -> DiscreteData | None:
        return self._spikes.get(label)

    def _setReference(self, referenceID: int):
        # if isinstance(referenceID, int):
        #     referenceID = [referenceID]
        self._header['ReferenceID'] = referenceID

    def _setFilter(self, low: int | float | None = None, high: int | float | None = None):
        if isinstance(low, (int, float)):
            self._header['LowCutOff'] = low
        if isinstance(high, (int, float)):
            self._header['HighCutOff'] = high

    def _setThreshold(self, threshold: int | float):
        if isinstance(threshold, (int, float)):
            self._header['Threshold'] = threshold

    def subtractReference(self, array: np.ndarray, reference: int) -> ContinuousData | None:
        result = self.data - np.asarray(array)

        return self.createCopy(input_array=result, reference=reference)
        # if isinstance(result, self.__class__):
        #     result._setReference(referenceID)
        #     return result

    def bandpassFilter(self, low, high) -> ContinuousData | None:
        result = design_and_filter(
            self.data, FSampling=self.fs, LowCutOff=low, HighCutOff=high)

        return self.createCopy(input_array=result, low_cutoff=low, high_cutoff=high)
        # if isinstance(result, self.__class__):
        #     result._setFilter(low=low, high=high)
        # elif isinstance(result, np.ndarray):
        #     result = self.__class__(
        #         result, self._filename, self._header.copy(), data_type='Filted')
        #     result._setFilter(low=low, high=high)
        # return result

    def _estimatedSD(self) -> float:
        self._estimated_sd = float(np.median(np.abs(self._data) / 0.6745))
        return self.estimated_sd

    def extractWaveforms(self, threshold=None) -> DiscreteData:
        if threshold is None:
            threshold = self.threshold
        waveforms, timestamps = extract_waveforms(
            self.data, self.channel_ID, threshold, alg='Valley-Peak')

        self._setThreshold(threshold)
        # result = self.createCopy(threshold=threshold)

        unit_IDs = np.zeros(len(timestamps), dtype=int)

        header = self.header
        header['Comment'] = 'TODO'
        header['NumRecords'] = len(unit_IDs)
        header['Type'] = 'Spikes'

        spike = DiscreteData(filename=self.filename,
                             header=header,
                             unit_IDs=unit_IDs,
                             timestamps=timestamps,
                             waveforms=waveforms)
        return spike

    def createCopy(self,
                   input_array: np.ndarray = None,
                   header: dict = None,
                   reference=None,
                   low_cutoff=None,
                   high_cutoff=None,
                   threshold=None,
                   deep_copy_data=False):
        # args = ['input_array', 'header', 'reference',
        #         'low_cutoff', 'high_cutoff', 'threshold']

        # for k in kargs.keys():
        #     if k not in args:
        #         raise ValueError(
        #             'Invalid keyword argument: %s (allowed arguments are %s)' % (k, str(args)))

        data_ = self._data
        if not input_array is None:
            data_ = input_array.copy()
        elif self._data_type == 'Raw' or deep_copy_data:
            data_ = self._data.copy()

        header_ = self._header.copy()
        if not header is None:
            header_ = header.copy()

        new_object = self.__class__(
            data_, self._filename, header_, data_type='Filted')

        if not reference is None:
            new_object._setReference(reference)

        if not low_cutoff is None:
            new_object._setFilter(low=low_cutoff)

        if not high_cutoff is None:
            new_object._setFilter(high=high_cutoff)

        if not threshold is None:
            new_object._setThreshold(threshold)

        return new_object


class DiscreteData(object):
    def __init__(self, filename: str, header: dict, unit_header: pd.DataFrame | None = None,
                 unit_IDs: np.ndarray = [], timestamps: np.ndarray = [], waveforms: np.ndarray = [],
                 data_type: str = 'Spikes', _from_file=False):
        self._filename = filename
        self._header = header.copy()

        self._unit_IDs = np.array(unit_IDs)
        self._timestamps = timestamps
        self._waveforms = waveforms

        if len(self._timestamps) < 1:
            if _from_file and data_type == 'Spikes':
                self._data_loaded = False
            else:
                self._data_loaded = True
                logger.critical('No data input!!!')
                raise
        else:
            self._data_loaded = True
            self._header['NumUnits'] = len(np.unique(self.unit_IDs))

        if self._data_loaded == True:
            if unit_header is None:
                self._unit_header = self.createUnitHeader(unit_IDs=self._unit_IDs,
                                                          unsorted_unit_ID=0)
            else:
                self._unit_header = unit_header.copy()

        self._data_type = data_type

        self._unsorted_unit_ID: int | None = None
        self._invalid_unit_ID: int | None = None

    @property
    def filename(self):
        return self._filename

    @property
    def channel_ID(self) -> int:
        return self._header['ID']

    @property
    def channel_name(self) -> int:
        return self._header['Name']

    @property
    def label(self) -> str:
        return self._header['Label']

    def setLabel(self, label: str):
        self._header['Label'] = label

    @property
    def header(self) -> dict:
        return self._header.copy()

    @property
    def reference(self) -> int:
        return self._header['ReferenceID']

    @property
    def low_cutoff(self) -> int | float:
        return self._header['LowCutOff']

    @property
    def high_cutoff(self) -> int | float:
        return self._header['HighCutOff']

    @property
    def threshold(self) -> int | float:
        return self._header['Threshold']

    # @property
    # def unit_IDs(self) -> list:
    #     if self._unit_header.columns.isin(['ID']):
    #         return self._unit_header['ID'].to_list()
    #     return []

    @property
    def unsorted_unit_ID(self) -> int | None:
        if not 'ID' in self.unit_header.columns and not 'UnitType' in self.unit_header.columns:
            return None

        if 'Unsorted' in self.unit_header['UnitType']:
            self._unsorted_unit_ID = self.unit_header.loc[self.unit_header['UnitType'].isin(
                ['Unsorted']), 'ID'].values[0]

        return self._unsorted_unit_ID

    @property
    def invalid_unit_ID(self) -> int | None:
        if not 'ID' in self.unit_header.columns and not 'UnitType' in self.unit_header.columns:
            return None

        if 'Invalid' in self.unit_header['UnitType']:
            self._invalid_unit_ID = self.unit_header.loc[self.unit_header['UnitType'].isin(
                ['Invalid']), 'ID'].values[0]
        return self._invalid_unit_ID

    @property
    def unit_header(self) -> pd.DataFrame:
        return self._unit_header.copy()

    @property
    def timestamps(self) -> np.ndarray:
        return self._timestamps.copy()

    @property
    def unit_IDs(self) -> np.ndarray:
        return self._unit_IDs.copy()

    @property
    def waveforms(self) -> np.ndarray:
        return self._waveforms.copy()

    @property
    def data_type(self) -> str:
        return self._data_type

    def isLoaded(self) -> bool:
        return self._data_loaded

    def _loadData(self):
        if self.isLoaded():
            logger.warning('Data alreadly loaded.')
            return

        spike = loadSpikes(filename=self._filename,
                           path=self._header['H5Location'])

        if spike is None:
            logger.warning('No data to load.')
            return

        self._unit_header = spike.get('unitHeader')
        self._unit_IDs = spike.get('unitID')
        self._timestamps = spike.get('timestamps')
        self._waveforms = spike.get('waveforms')

    def setUnit(self, new_unit_IDs, new_unit_header: pd.DataFrame | None = None, unsorted_unit_ID: int | None = None, invalid_unit_ID: int | None = None) -> DiscreteData | None:
        if self._data_type != 'Spikes':
            logger.warning('Not spike type data.')
            return

        if len(new_unit_IDs) != len(self._timestamps):
            logger.warning('Length of unit id not match with timestamps.')
            return

        if new_unit_header is None:
            new_unit_header = self.createUnitHeader(new_unit_IDs,
                                                    unsorted_unit_ID,
                                                    invalid_unit_ID)

        # unit_header_name = ['H5Location', 'H5Name', 'ID', 'Name', 'NumRecords', 'ParentID',
        #                     'ParentType', 'Type', 'UnitType']

        # logger.critical('Unimplemented function.')
        return self.__class__(filename=self._filename,
                              header=self._header,
                              unit_header=new_unit_header,
                              timestamps=self._timestamps,
                              unit_IDs=new_unit_IDs,
                              waveforms=self._waveforms)

    def createUnitHeader(self, unit_IDs, unsorted_unit_ID: int | None = None, invalid_unit_ID: int | None = None) -> pd.DataFrame:
        values, counts = np.unique(unit_IDs, return_counts=True)

        new_unit_header = pd.DataFrame({'ID': values,
                                        'Name': [f'{self.channel_name}_Unit_{ID:02}' for ID in values],
                                        'NumRecords': counts,
                                        })
        new_unit_header['UnitType'] = 'Unit'

        if not unsorted_unit_ID is None:
            if unsorted_unit_ID in new_unit_header['ID'].to_list():
                new_unit_header.loc[new_unit_header['ID'] == unsorted_unit_ID, ['Name', 'UnitType']] = [
                    f'{self.channel_name}_Unit_{unsorted_unit_ID:02}_Unsorted', 'Unsorted']
            else:
                unsorted_unit_header = pd.DataFrame({'ID': [unsorted_unit_ID],
                                                     'Name': [f'{self.channel_name}_Unit_{unsorted_unit_ID:02}_Unsorted'],
                                                     'NumRecords': [0],
                                                     'UnitType': ['Unsorted']})
                new_unit_header = pd.concat([new_unit_header, unsorted_unit_header],
                                            axis=0)

        if not invalid_unit_ID is None:
            if invalid_unit_ID in new_unit_header['ID'].to_list():
                new_unit_header.loc[new_unit_header['ID'] == invalid_unit_ID, ['Name', 'UnitType']] = [
                    f'{self.channel_name}_Unit_{invalid_unit_ID:02}_Invalid', 'Invalid']
            else:
                invalid_unit_header = pd.DataFrame({'ID': [invalid_unit_ID],
                                                    'Name': [f'{self.channel_name}_Unit_{invalid_unit_ID:02}_Invalid'],
                                                    'NumRecords': [0],
                                                    'UnitType': ['Invalid']})
                new_unit_header = pd.concat([new_unit_header, invalid_unit_header],
                                            axis=0)

        new_unit_header.sort_values('ID', ignore_index=True, inplace=True)

        return new_unit_header

    def waveformsPCA(self, selected_unit_IDs: list = None, n_components: int = None, ignore_invalid: bool = False) -> np.ndarray:
        if selected_unit_IDs is None:
            selected_unit_IDs = np.unique(self._unit_IDs).tolist()
        elif not isinstance(selected_unit_IDs, list):
            selected_unit_IDs = list(selected_unit_IDs)

        if ignore_invalid and not self.invalid_unit_ID is None:
            if self.invalid_unit_ID in selected_unit_IDs:
                selected_unit_IDs.remove(self.invalid_unit_ID)

        mask = np.isin(self._unit_IDs, selected_unit_IDs)
        transformed_data = PCA(n_components).fit_transform(
            self.waveforms[mask])
        return transformed_data

    def autosort(self):
        """by default unsorted unit id is 0, and invalid unit id is the last.

        Returns:
            _type_: _description_
        """
        if self._data_type != 'Spikes':
            logger.warning('Not spike type data.')
            return

        feat = self.waveformsPCA(ignore_invalid=True)
        new_invalid_unit_ID = None
        if not self.invalid_unit_ID is None:
            ignored_mask = ~(self._unit_IDs == self.invalid_unit_ID)
            new_sort_unit_ID = auto_sort(self.filename, self.channel_ID, feat,
                                         self.waveforms[ignored_mask],
                                         self.timestamps[ignored_mask], sorting=None, re_sort=False)
            # by default invalid unit is last one
            new_invalid_unit_ID = np.max(new_unit_IDs) + 1
            new_unit_IDs = np.ones(len(self._unit_IDs)) * new_invalid_unit_ID
            new_unit_IDs[~ignored_mask] = new_sort_unit_ID

        else:
            new_unit_IDs = auto_sort(self.filename, self.channel_ID, feat,
                                     self.waveforms,
                                     self.timestamps, sorting=None, re_sort=False)
        # logger.critical('Unimplemented function.')
        return self.setUnit(new_unit_IDs=new_unit_IDs,
                            unsorted_unit_ID=0,
                            invalid_unit_ID=new_invalid_unit_ID)


if __name__ == '__main__':
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s][%(levelname)-5s] %(message)s (%(filename)s:%(lineno)d)",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    filename = "data/MX6-22_2020-06-17_17-07-48_no_ref.h5"
    # filename = "data/MD123_2022-09-07_10-38-00.h5"
    data = SpikeSorterData(filename)
    print(data.getRaw(1))
    print(data.getRaw('CH1'))
    print(data.getRaw(1.2))
    print(data.getRaw([1, 2, 3]))
    print(data.getRaw())
