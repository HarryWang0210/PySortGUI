import tables
import os
import pandas as pd
import numpy as np
from DataStructure.pyephys import loadPyephys, loadRaws, loadSpikes
# from pyephys import loadPyephys, loadRaws

from sklearn.decomposition import PCA
from sklearn.preprocessing import MaxAbsScaler


class SpikeSorterData():
    def __init__(self, filename, parent=None):
        # super().__init__(parent)
        self.filename = filename

        self.__chan_info = loadPyephys(self.filename)
        column_order = ['ID', 'Label', 'Name', 'NumUnits',
                        'ReferenceID', 'LowCutOff', 'HighCutOff', 'Threshold',
                        'NumRecords', 'SamplingFreq', 'SigUnits']
        self.__chan_info = self.__chan_info.reindex(columns=column_order)
        self.__chan_info.set_index(['ID', 'Label'], inplace=True)
        self.__chan_info.sort_index(inplace=True)

        # turn id to string
        self.__chan_info.index = self.__chan_info.index.set_levels(
            self.__chan_info.index.levels[0].astype('string'), level="ID")
        self.__chan_info["ReferenceID"] = self.__chan_info["ReferenceID"].astype(
            'string')

        self.__raw_data = dict()
        self.__spike_data = dict()

    @property
    def chan_info(self):
        chan_info = self.__chan_info.copy()
        # bytes to string
        byte_columns = chan_info.select_dtypes(include=[object]).columns
        chan_info[byte_columns] = chan_info[byte_columns].stack(
        ).str.decode('utf-8').unstack()
        chan_info.index = chan_info.index.set_levels(
            chan_info.index.levels[1].map(bytes.decode), level=1)

        # round float
        chan_info["Threshold"] = chan_info["Threshold"].round(3)
        # chan_info = chan_info.replace([np.NaN], [None])
        # print(chan_info.index[0][0], type(chan_info.index[0][0]))
        return chan_info

    def getRaw(self, chan_ID):
        if chan_ID not in self.__raw_data.keys():
            self.__raw_data[chan_ID] = loadRaws(self.filename, chan_ID)

        return self.__raw_data[chan_ID]

    def getSpikes(self, chan_ID, label):
        if chan_ID not in self.__spike_data.keys():
            self.__spike_data[chan_ID] = loadSpikes(
                self.filename, chan_ID, label)
        return self.__spike_data[chan_ID]

    def waveforms_pca(self, data):
        pca = PCA(n_components=3).fit(data)
        transformed_data = pca.transform(data)

        transformer = MaxAbsScaler().fit(transformed_data)
        transformed_data = transformer.transform(transformed_data)
        return transformed_data


if __name__ == '__main__':
    filename = "data/MX6-22_2020-06-17_17-07-48_no_ref.h5"
    # filename = "data/MD123_2022-09-07_10-38-00.h5"
    data = SpikeSorterData(filename)
    print(data.chan_info)
    print(data.getRaw(chan_ID=0))
