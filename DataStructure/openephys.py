import logging
import os
import re
from datetime import datetime
from xml.etree import cElementTree as ET

import numpy as np
from pydantic import BaseModel, validator

# common Entries
_FILE_EXTENSIONS = [
    # '.spikes',
    '.continuous',
    '.events',
    '.xml',
]

_NON_DATA_FILES = [
    'settings.xml',
    'messages.events'
]

# __all__ = ['read_header']

logger = logging.getLogger(__name__)
# logger.setLevel('WARNING')
# __updated__ = '2017-05-24'

# constants
NUM_HEADER_BYTES = 1024
SAMPLES_PER_RECORD = 1024
# BYTES_PER_SAMPLE = 2
# RECORD_SIZE = 4 + 8 + SAMPLES_PER_RECORD * BYTES_PER_SAMPLE + \
#     10  # size of each continuous record in bytes
RECORD_MARKER = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 255])

# constants for pre-allocating matrices:
MAX_NUMBER_OF_SPIKES = int(1e6)
MAX_NUMBER_OF_RECORDS = int(1e6)
MAX_NUMBER_OF_EVENTS = int(1e6)

_CONTINUOUS_RECORD_DTYPE = np.dtype([
    ('timestamp', np.int64),
    ('nsamples', np.uint16),
    ('rnumber', np.uint16),
    ('records', ('>i2', SAMPLES_PER_RECORD)),
    ('marker', (np.uint8, 10)),
])

_EVENT_RECORD_DTYPE = np.dtype([
    ('timestamp', np.int64),
    ('position', np.int16),
    ('eventtype', np.uint8),
    ('processorid', np.uint8),
    ('eventid', np.uint8),
    ('channel', np.uint8),
    ('recnumber', np.uint16),
])

_SPIKE_RECORD_DTYPE = np.dtype([
    ('eventtype', np.uint8),
    ('timestamp', np.int64),
    ('softwaretimestamp', np.int64),
    ('sourceid', np.uint16),
    ('numchannels', np.uint16),
    ('numsample', np.uint16),
    ('sortedid', np.uint16),
    ('electrodeid', np.uint16),
    ('triggerchannel', np.uint16),
    ('color', (np.uint8, 3)),
    ('pcaproj', (np.float32, 2)),
])


class OpenEphysHeader(BaseModel):
    format: str  # 'Open Ephys Data Format'

    version: float  # 0.4

    header_bytes: int  # 1024

    description: str  # '(String describing the header)'

    date_created: datetime  # 'dd-mm-yyyy hhmmss'

    channel: str  # '(String with channel name)'

    channelType: str  # '(String describing the channel type)'

    sampleRate: int  # (integer sampling rate)

    blockLength: int  # 1024

    bufferSize: int  # 1024

    # (floating point value of microvolts/bit for headstage channels, or volts/bit for ADC channels)
    bitVolts: int | float

    # extend infomation from setting.xml
    lowCut: int | float = 0
    highCut: int | float = 0
    ID: int = 0

    @validator("date_created", pre=True)
    def parse_date_created(cls, value):
        return datetime.strptime(value, "%d-%b-%Y %H%M%S")

# def load(filepath, dtype=float):

#     # redirects to code for individual file types
#     if 'continuous' in filepath:
#         data = loadContinuous(filepath, dtype)
#     elif 'spikes' in filepath:
#         data = loadSpikes(filepath)
#     elif 'events' in filepath:
#         data = loadEvents(filepath)
#     else:
#         raise Exception(
#             "Not a recognized file type. Please input a .continuous, .spikes, or .events file")

#     return data


def readOpenEphysHeader(f) -> OpenEphysHeader:
    header = {}
    h = f.read(1024).decode().replace('\n', '').replace('header.', '')
    for item in h.split(';'):
        if '=' in item:
            splited_item = item.split(' = ')
            key = splited_item[0].strip()
            value = eval(splited_item[1])
            header[key] = value
    header = OpenEphysHeader.model_validate(header)
    return header


def loadContinuous(file_name, dtype=_CONTINUOUS_RECORD_DTYPE):

    # assert dtype in (float, np.int16), \
    #     'Invalid data type specified for loadContinous, valid types are float and np.int16'

    print("Loading continuous data...")

    data = {}

    with open(file_name, 'rb') as in_file:
        header = readOpenEphysHeader(in_file)
        in_file.seek(1024)
        continuous = np.fromfile(in_file, dtype=dtype)

        if re.findall(r'(^[0-9]+)_', os.path.basename(file_name)):
            pid = int(re.findall(r'(^[0-9]+)_',
                                 os.path.basename(file_name))[0])

        # get extend infomation
        xml_file = 'settings.xml'
        file_path = os.path.split(file_name)[0]
        xml_file = os.path.join(file_path, xml_file)

        root = None
        if os.path.isfile(xml_file):
            root = ET.parse(xml_file).getroot()
        #     else:
        #         return header

        if pid is not None:
            search = ".//PROCESSOR[@NodeId='{}']/EDITOR".format(pid)
            finfo = root.findall(search)[0].attrib
            header.lowCut = float(finfo['LowCut'])
            header.highCut = float(finfo['HighCut'])
            search = ".//PROCESSOR[@NodeId='{}']/CHANNEL_INFO/CHANNEL[@name='{}']".format(
                pid, header.channel)
            chinfo = root.findall(search)[0].attrib
            header.ID = int(chinfo['number'])

        data.update({
            'header': header,
            'data': continuous['records'].flatten()
        })
        # ch['timestamps'] = continuous['timestamp']
        # OR use downsample(samples,1), to save space
        # ch['recordingNumber'] = continuous['rnumber']

        # continuous = continuous['records'].flatten()
    return data


def loadEvents(file_name, dtype=_EVENT_RECORD_DTYPE):
    data = {}

    print('loading events...')
    time_first_point = _getTimeFirstPoint(file_name)

    with open(file_name, 'rb') as in_file:
        header = readOpenEphysHeader(in_file)
        in_file.seek(1024)
        events = np.fromfile(in_file, dtype=dtype)

        banks = events['processorid']

        # generating different events data for banks
        # for bank, header_bank in zip(np.unique(banks), header[1:]):
        for bank in np.unique(banks):
            units = []
            units_name = []
            # generating units for Event
            ev_per_bank = events[banks == bank]
            # getting  timestamps
            ts = ev_per_bank['timestamp'] - time_first_point
            # trying to compress data
            if ts.max() < 2**32 - 1:
                ts = ts.astype(np.int32)
            channels = ev_per_bank['channel']
            types = ev_per_bank['eventtype']
            stati = ev_per_bank['eventid']
            stati_word = ['Off', 'On']
            # separating units by channel, type and id (0 or 1)
            for channel in np.unique(channels):
                for type_ in np.unique(types):
                    for status in np.unique(stati):
                        units_ind = (channel == channels) & \
                            (type_ == types) & \
                            (status == stati)
                        units_ind = np.where(units_ind)[0]
                        if units_ind.size > 0:
                            units += [units_ind]
                            units_name += ["Chan_{:02d}_{}_{}".format(
                                channel, type_, stati_word[status])]

        # continuous = continuous['records'].flatten()
        data.update({
            'header': header,
            'timestamps': ts,
            'index': units,
            'units_name': units_name
        })
    return data


def _getTimeFirstPoint(file_name):
    '''
    Finds the time of first point for openephys data
    :param file_name:
    :type file_name:
    '''

    # trying to find messages.events
    location = os.path.split(file_name)[0]
    _, ext = os.path.splitext(file_name)

    msg_ev_file = os.path.join(location, 'messages.events')
    if os.path.isfile(msg_ev_file):
        with open(msg_ev_file, 'r') as text_file:
            search = re.search(r'start time: (\d+)@', text_file.read())
            if search is not None:
                return int(search.group(1))

    if os.stat(file_name).st_size <= 1024:
        return

    if ext == '.continuous':
        with open(file_name, 'r') as data_file:
            data_map = np.memmap(
                data_file, _CONTINUOUS_RECORD_DTYPE, 'r', 1024)
            return data_map[0]['timestamp']

    if ext == '.events':
        with open(file_name, 'r') as data_file:
            data_map = np.memmap(
                data_file, _EVENT_RECORD_DTYPE, 'r', 1024)
            # pdb.set_trace()
            return data_map[0][0]

    if ext == '.spikes':
        with open(file_name, 'r') as data_file:
            data_map = np.memmap(
                data_file, _SPIKE_RECORD_DTYPE, 'r', 1024)
            return data_map[0][1]


if __name__ == '__main__':
    data = loadEvents(
        r'C:\Users\harry\Desktop\Lab\Project_spikesorter\PySortGUI\data\RU01_2022-08-01_11-20-12\all_channels.events')
    # print(data)
    # data = loadContinuous(
    #     r'C:\Users\harry\Desktop\Lab\Project_spikesorter\PySortGUI\data\MX6-22_2020-06-17_17-07-48_no_ref\100_CH2.continuous')

    print(data)

    # print(len(v))
    # a = {'format': "'Open Ephys Data Format'", ' version': '0.4', ' header_bytes': '1024',
    #      'description': "'each record contains one 64-bit timestamp, one 16-bit sample count (N), 1 uint16 recordingNumber, N 16-bit samples, and one 10-byte record marker (0 1 2 3 4 5 6 7 8 255)'", ' date_created': "'17-Jun-2020 170748'", 'channel': "'CH1'", 'channelType': "'Continuous'", 'sampleRate': '30000', 'blockLength': '1024', 'bufferSize': '1024', 'bitVolts': '0.195'}
    # b = {'format': "'Open Ephys Data Format'"}
    # OpenEphysHeader.model_validate(b)
