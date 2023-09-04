# import logging
# import traceback
# from importlib import import_module
# import pkgutil

# from functools import wraps

# _MODULES_NAME = ['openephys']
# _MODULES_DICT = {}
# _HEADER_FUNC_DICT = {}
# logger = logging.getLogger(__name__)

# for module in _MODULES_NAME:
#     # load input modules and save to  _MODULES_DICT
#     try:
#         prefix = __name__.rsplit('.', 1)[0] + '.'
#         _MODULES_DICT[module] = import_module(prefix + module)
#     except:
#         logger.error('Not able to import module {}'.format(module))

# def _execution_error(func):
#     '''
#     A wrapper that returns error if execution fails without raising an excetpion.
#     This decorator switches an exception with a logger.error
#     :param func: function
#     '''

#     @wraps(func)
#     def decorate(*args, **kws):
#         try:
#             return func(*args, **kws)
#         except Exception as _:
#             #exec_info = sys.exc_info()
#             logger.error('\t ******Execution halted for {}******\n{}'.format(
#                 func.__name__, traceback.format_exc()))

#     return decorate

# @_execution_error
# def _export_to_pyephys_single_file(file_name):
#     '''

#     :param file_name:
#     :type file_name:
#     '''

#     msg = "exporting {}".format(file_name)
#     logger.info(msg)

#     header = read_header(file_name)
#     data = read_data(file_name)

#     if header is None or header is None:
#         return

#     lib_pe.create_pyephys_datafile(header, data)

# # this decorator raises an exception if the given filename is not a file
# @_check_file_exist # 先確認file是存在再呼叫 read_header
# def read_header(file_name):

#     _, ext = os.path.splitext(file_name) # 分開檔案名, 擴展名
#     if ext.lower() in _HEADER_FUNC_DICT:
#         return _HEADER_FUNC_DICT[ext](file_name) # 使用該擴展名read header的function
#     else:
#         logger.warn('File extension not known or not implemented yet')


import tables
import os
import pandas as pd
import numpy as np
from DataStructure.pyephys import load_pyephys, load_raws, load_spikes
# from pyephys import load_pyephys, load_raws


class SpikeSorterData():
    def __init__(self, filename, parent=None):
        # super().__init__(parent)
        self.filename = filename

        self.__chan_info = load_pyephys(self.filename)
        column_order = ['ID', 'Label', 'Name', 'NumUnits',
                        'ReferenceID', 'LowCutOff', 'HighCutOff', 'Threshold',
                        'NumRecords', 'SamplingFreq', 'SigUnits']
        self.__chan_info = self.__chan_info.reindex(columns=column_order)
        self.__chan_info.set_index(['ID', 'Label'], inplace=True)
        self.__chan_info.sort_index(inplace=True)
        self.__raw_data = dict()
        self.__unit_data = dict()

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

        chan_info = chan_info.replace([np.NaN], [None])
        return chan_info

    def get_raw(self, chan_ID):
        if chan_ID not in self.__raw_data.keys():
            self.__raw_data[chan_ID] = load_raws(self.filename, chan_ID)

        return self.__raw_data[chan_ID]

    def get_spikes(self, chan_ID, label):
        if chan_ID not in self.__unit_data.keys():
            self.__unit_data[chan_ID] = load_spikes(
                self.filename, chan_ID, label)
        return self.__unit_data[chan_ID]


if __name__ == '__main__':
    filename = "data/MX6-22_2020-06-17_17-07-48_no_ref.h5"
    # filename = "data/MD123_2022-09-07_10-38-00.h5"
    data = SpikeSorterData(filename)
    print(data.chan_info)
    print(data.get_raw(chan_ID=0))
