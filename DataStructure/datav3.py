from __future__ import annotations
from numpy import ndarray, ufunc
import tables
import os
import pandas as pd
import numpy as np
from DataStructure.pyephysv3 import loadPyephys, loadRaws, loadSpikes
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
        chanID = self.validateChannel(channel)
        self._raws_dict[chanID] = raw_object._loadData()

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

    def subtractReference(self, channel: int | str, reference: list) -> ContinuousData:
        ch_object = self.getRaw(channel, load_data=True)

        if len(reference) == 1:
            referenceID = self.validateChannel(reference[0])
            ref_object = self.getRaw(referenceID, load_data=True)

        result = ch_object.subtractReference(
            ref_object, referenceID=referenceID)

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


class ContinuousData(np.ndarray):
    def __new__(cls, input_array=None, filename: str = '', header: dict = dict(), data_type: str = 'Filted'):
        """_summary_

        Args:
            input_array (array-like, optional): _description_. Defaults to [].
            filename (str, optional): _description_. Defaults to ''.
            header (dict, optional): _description_. Defaults to dict().
            data_type (str, optional): 'Raw' | 'Filted'. Defaults to 'Filted'.

        Returns:
            ContinuousData: _description_
        """
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj._header = header.copy()
        # obj._header = header.copy()
        obj._data_type = data_type
        obj._filename = filename
        obj._data_loaded = False
        if (not input_array is None) and (len(input_array) != 0):
            obj._data_loaded = True

        # Finally, we must return the newly created object:
        return obj

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._estimated_sd = None
        self._spikes = dict()

    @property
    def filename(self):
        return self._filename

    @property
    def channel_ID(self):
        return self._header['ID']

    @property
    def header(self):
        return self._header.copy()

    @property
    def data_type(self):
        return self._data_type

    @property
    def fs(self):
        return self._header['SamplingFreq']

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

    def _loadData(self) -> ContinuousData | None:
        if self._data_type != 'Raw' or self.isLoaded():
            logger.warning('No data to load.')
            return None

        data = loadRaws(
            self._filename, self._header['H5Location'], self._header['H5Name'])
        raw_object = self.__class__(
            data, self._filename, self._header, self._data_type)
        for label in self.spikes:
            raw_object.setSpike(self._spikes[label], label)
        return raw_object

    def setSpike(self, spike_object, label: str = 'default'):
        self._spikes[label] = spike_object

    def getSpike(self, label: str) -> DiscreteData | None:
        return self._spikes.get(label)

    def _setReference(self, referenceID: int):
        self._header['ReferenceID'] = referenceID

    def _setFilter(self, low: int | float | None = None, high: int | float | None = None):
        if isinstance(low, (int, float)):
            self._header['LowCutOff'] = low
        if isinstance(high, (int, float)):
            self._header['HighCutOff'] = high

    def _setThreshold(self, threshold: int | float):
        if isinstance(threshold, (int, float)):
            self._header['Threshold'] = threshold

    def subtractReference(self, array, referenceID: int) -> ContinuousData | None:
        result = self - array
        if isinstance(result, self.__class__):
            result._setReference(referenceID)
            return result

    def bandpassFilter(self, low, high) -> ContinuousData | None:
        result = design_and_filter(
            self, FSampling=self.fs, LowCutOff=low, HighCutOff=high)
        if isinstance(result, self.__class__):
            result._setFilter(low=low, high=high)
        elif isinstance(result, np.ndarray):
            result = self.__class__(
                result, self._filename, self._header.copy(), data_type='Filted')
            result._setFilter(low=low, high=high)
        return result

    def _estimatedSD(self) -> float:
        self._estimated_sd = float(np.median(np.abs(self) / 0.6745))
        return self.estimated_sd

    def extractWaveforms(self, threshold) -> tuple[ContinuousData, DiscreteData]:
        waveforms, timestamps = extract_waveforms(
            self, self.channel_ID, threshold, alg='Valley-Peak')
        result = self[:]
        if isinstance(result, self.__class__):
            result._setThreshold(threshold)
        unit_ID = np.zeros(len(timestamps))
        spike = DiscreteData(filename=result.filename, header=result._header,
                             unit_ID=unit_ID,
                             timestamps=timestamps,
                             waveforms=waveforms)
        return result, spike

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        if obj is None:
            return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(InfoArray).
        self._filename = getattr(obj, '_filename',  None)
        self._header = getattr(obj, '_header',  dict()).copy()
        self._data_type = 'Filted'
        self._data_loaded = True


class DiscreteData(object):
    def __init__(self, filename: str, header: dict, unit_header: pd.DataFrame = pd.DataFrame(),
                 unit_ID: np.ndarray = [], timestamps: np.ndarray = [], waveforms: np.ndarray = [], data_type: str = 'Spikes', _from_file=False):
        self._filename = filename
        self._header = header.copy()
        self._unit_header = unit_header.copy()
        self._unit_ID = np.array(unit_ID)
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
    def unit_ID(self) -> np.ndarray:
        return self._unit_ID.copy()

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
        self._unit_ID = spike.get('unitID')
        self._timestamps = spike.get('timestamps')
        self._waveforms = spike.get('waveforms')

    def setUnit(self, new_unit_ID, unsorted_unit_ID: int | None = None, invalid_unit_ID: int | None = None) -> DiscreteData | None:
        # TODO
        if self._data_type != 'Spikes':
            logger.warning('Not spike type data.')
            return
        unit_header_name = ['H5Location', 'H5Name', 'ID', 'Name', 'NumRecords', 'ParentID',
                            'ParentType', 'Type', 'UnitType']
        new_unit_header = []
        logger.critical('Unimplemented function.')
        return self.__class__(filename=self._filename,
                              header=self._header,
                              unit_header=new_unit_header,
                              timestamps=self._timestamps,
                              unit_ID=new_unit_ID,
                              waveforms=self._waveforms)

    def waveformsPCA(self, n_components: int = None, ignore_invalid: bool = False) -> np.ndarray:
        if ignore_invalid and not self.invalid_unit_ID is None:
            ignored_mask = ~(self.unit_ID == self.invalid_unit_ID)
            transformed_data = PCA(n_components).fit_transform(
                self.waveforms[ignored_mask])
        else:
            transformed_data = PCA(n_components).fit_transform(
                self.waveforms)
        return transformed_data

    def autosort(self):
        if self._data_type != 'Spikes':
            logger.warning('Not spike type data.')
            return

        feat = self.waveformsPCA(ignore_invalid=True)
        new_invalid_unit_ID = None
        if not self.invalid_unit_ID is None:
            ignored_mask = ~(self.unit_ID == self.invalid_unit_ID)
            new_sort_unit_ID = auto_sort(self.filename, self.channel_ID, feat,
                                         self.waveforms[ignored_mask],
                                         self.timestamps[ignored_mask], sorting=None, re_sort=False)
            new_invalid_unit_ID = np.max(new_unit_ID) + 1
            new_unit_ID = np.ones(len(self.unit_ID)) * new_invalid_unit_ID
            new_unit_ID[~ignored_mask] = new_sort_unit_ID

        else:
            new_unit_ID = auto_sort(self.filename, self.channel_ID, feat,
                                    self.waveforms,
                                    self.timestamps, sorting=None, re_sort=False)
        # logger.critical('Unimplemented function.')
        return self.setUnit(new_unit_ID, unsorted_unit_ID=0, invalid_unit_ID=new_invalid_unit_ID)


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
