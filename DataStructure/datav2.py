import tables
import os
import pandas as pd
import numpy as np
from DataStructure.pyephysv2 import loadPyephys, loadRaws, loadSpikes
# from pyephysv2 import loadPyephys, loadRaws, loadSpikes
from DataStructure.FunctionsLib.SignalProcessing import design_and_filter
from DataStructure.FunctionsLib.ThresholdOperations import extract_waveforms
from DataStructure.FunctionsLib.Sorting import auto_sort

from sklearn.decomposition import PCA
from sklearn.preprocessing import MaxAbsScaler


class SpikeSorterData():
    def __init__(self, filename, parent=None):
        # super().__init__(parent)
        self.filename = filename

        self.__chan_header_dict = loadPyephys(self.filename)
        self.__raw_data = dict()
        self.__spike_data = dict()

    @property
    def raws_header(self):
        raws_header = self.__chan_header_dict['RawsHeader']
        if isinstance(raws_header, pd.DataFrame):
            raws_header = raws_header.copy()
        return raws_header

    @property
    def spikes_header(self):
        spikes_header = self.__chan_header_dict['SpikesHeader']
        if isinstance(spikes_header, pd.DataFrame):
            spikes_header = spikes_header.copy()
        return spikes_header

    @property
    def events_header(self):
        events_header = self.__chan_header_dict['EventsHeader']
        if isinstance(events_header, pd.DataFrame):
            events_header = events_header.copy()
        return events_header

    def getRaw(self, chan_ID):
        if chan_ID not in self.__raw_data.keys():
            self.__raw_data[chan_ID] = loadRaws(self.filename, chan_ID)

        return self.__raw_data[chan_ID].copy()

    def getSpikes(self, chan_ID, label):
        if chan_ID not in self.__spike_data.keys():
            self.__spike_data[chan_ID] = loadSpikes(
                self.filename, chan_ID, label)
        return self.__spike_data[chan_ID].copy()

    def waveforms_pca(self, x, n_components=3):
        pca = PCA(n_components).fit(x)
        transformed_data = pca.transform(x)

        return transformed_data

    def spikeFilter(self, chan_ID, ref, low=1, high=60):
        data = self.getRaw(chan_ID)
        data = data - ref
        fs = self.raws_header[self.raws_header['ID']
                              == chan_ID]['SamplingFreq'].values[0]
        data = design_and_filter(data, FSampling=fs,
                                 LowCutOff=low, HighCutOff=high)
        return data

    def estimatedSD(self, x):
        return float(np.median(np.abs(x) / 0.6745))

    def test_extract_waveforms(self, x, chan_ID, threshold):
        waveforms, timestamps = extract_waveforms(
            x, chan_ID, threshold, alg='Valley-Peak')
        return waveforms, timestamps

    def test_auto_sort(self, chan_ID, waveforms, timestamps):
        feat = self.waveforms_pca(waveforms, n_components=None)
        unitID = auto_sort(self.filename, chan_ID, feat,
                           waveforms, timestamps, sorting=None, re_sort=False)
        return unitID


if __name__ == '__main__':
    filename = "data/MX6-22_2020-06-17_17-07-48_no_ref.h5"
    # filename = "data/MD123_2022-09-07_10-38-00.h5"
    data = SpikeSorterData(filename)
    ch1 = data.getRaw(1)
    ref = data.spikes_header[data.spikes_header['ID'] == 1]['ReferenceID']
    print(ch1, ref)
