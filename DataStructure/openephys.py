import logging
import os
import re
from datetime import datetime
from xml.etree import cElementTree as ET

import numpy as np
from pydantic import BaseModel, field_validator

from .header_class import (EventsHeader, FileHeader, RawsHeader,
                           SpikesHeader)

# common Entries
_FILE_EXTENSIONS = [
    '.spikes',
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

    @field_validator("date_created", mode="before")
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


def getFilesInFolder(dir_path: str):
    file_list = [os.path.join(dir_path, file_name)
                 for file_name in os.listdir(dir_path)]
    files = []
    for file_path in file_list:
        ext = os.path.splitext(file_path)[1]
        if ext in _FILE_EXTENSIONS:
            if not os.path.basename(file_path) in _NON_DATA_FILES:
                files.append(file_path)

    return files


def loadOpenephysHeader(file_list: list[str]):
    """Load header of given Openephys format files as Pyephys format

    Args:
        file_list (list[str]): file path list.

    Returns:
        dict: Pyephys format headers
        {
            'FileHeader': [FileHeader_dict, ...],
            'RawsHeader': [(file path, RawsHeader_dict), ...],
            'EventsHeader': [(file path, EventsHeader_dict), ...],
        }
    """
    file_headers: list[dict] = []
    raws_headers: list[tuple[str, dict]] = []
    events_headers: list[tuple[str, dict]] = []
    spikes_headers: list[tuple[str, dict]] = []
    for file_path in file_list:
        ext = os.path.splitext(file_path)[1]
        if ext in _FILE_EXTENSIONS:
            if os.path.basename(file_path) in _NON_DATA_FILES:
                continue

        if ext == '.continuous':
            file_header, raws_header = loadContinuousHeader(file_path)
            file_headers.append(file_header.model_dump())
            raws_headers.append((file_path, raws_header.model_dump()))

        if ext == '.events':
            file_header, events_header = loadEventsHeader(file_path)
            file_headers.append(file_header.model_dump())
            events_headers += [(file_path, header.model_dump())
                               for header in events_header]

        if ext == '.spikes':
            logger.warn('Not support load .spike files.')

    data = {}
    if file_headers:
        data['FileHeader'] = file_headers
    if raws_headers:
        data['RawsHeader'] = raws_headers
    if events_headers:
        data['EventsHeader'] = events_headers
    if spikes_headers:
        data['SpikesHeader'] = spikes_headers

    return data


def readOpenEphysHeader(file_name: str) -> OpenEphysHeader:
    """Read OpenEphysHeader

    Args:
        file_name (str): _description_

    Returns:
        OpenEphysHeader: _description_
    """
    header = {}
    with open(file_name, 'rb') as f:
        h = f.read(1024).decode().replace('\n', '').replace('header.', '')
        for item in h.split(';'):
            if '=' in item:
                splited_item = item.split(' = ')
                key = splited_item[0].strip()
                value = eval(splited_item[1])
                header[key] = value
    header = OpenEphysHeader.model_validate(header)
    return header


def loadContinuousHeader(file_name: str, dtype=_CONTINUOUS_RECORD_DTYPE) -> tuple[FileHeader, RawsHeader]:
    """Load header of given Openephys continuous format file, and convert to Pyephys format.

    Args:
        file_name (str): _description_
        dtype (_type_, optional): _description_. Defaults to _CONTINUOUS_RECORD_DTYPE.

    Returns:
        tuple[FileHeader, RawsHeader]: _description_
    """
    header = readOpenEphysHeader(file_name)

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

    if pid is not None:
        search = ".//PROCESSOR[@NodeId='{}']/EDITOR".format(pid)
        finfo = root.findall(search)[0].attrib
        header.lowCut = float(finfo['LowCut'])
        header.highCut = float(finfo['HighCut'])
        search = ".//PROCESSOR[@NodeId='{}']/CHANNEL_INFO/CHANNEL[@name='{}']".format(
            pid, header.channel)
        chinfo = root.findall(search)[0].attrib
        header.ID = int(chinfo['number'])

    # create FileHeader
    file_header = FileHeader(FullFileName=file_name,
                             DateTime=header.date_created,
                             RecordingSystem='OpenEphys',
                             HeaderLength=NUM_HEADER_BYTES,
                             FileMajorVersion=int(
                                 str(header.version).split('.')[0]),
                             FileMinorVersion=int(
                                 str(header.version).split('.')[1]),
                             NumChannels=1,
                             )

    if header.bitVolts == 0.195:
        SigUnits = 'uV'
        MaxAnalogValue = 5000
        MinAnalogValue = -5000
        MaxDigValue = 2**15 - 1
        MinDigValue = -2**15
    else:
        SigUnits = 'V'
        MaxAnalogValue = 5
        MinAnalogValue = -5
        MaxDigValue = 2**15 - 1
        MinDigValue = -2**15

    # try:
    #     sample_data = np.memmap(file_name, offset=NUM_HEADER_BYTES,
    #                             dtype=dtype)
    # except:
    #     logger.warning(
    #         'Size of available data is not a multiple of the data-type size')
    #     n_records = int(os.stat(file_name).st_size /
    #                     dtype.itemsize)
    #     sample_data = np.memmap(file_name, offset=NUM_HEADER_BYTES,
    #                             dtype=dtype, shape=(n_records, 1))

    raws_header = RawsHeader(ADC=header.bitVolts,
                             Bank=pid,
                             ID=header.ID,
                             Pin=header.ID,
                             Name=header.channel,
                             SigUnits=SigUnits,

                             HighCutOff=header.highCut,
                             LowCutOff=header.lowCut,
                             MaxAnalogValue=MaxAnalogValue,
                             MinAnalogValue=MinAnalogValue,
                             MaxDigValue=MaxDigValue,
                             MinDigValue=MinDigValue,
                             #  NumRecords=len(data),
                             SamplingFreq=header.sampleRate,
                             )
    return (file_header, raws_header)


def loadEventsHeader(file_name: str, dtype=_EVENT_RECORD_DTYPE) -> tuple[FileHeader, list[EventsHeader]]:
    """Load header of given Openephys events format file, and convert to Pyephys format.

    Args:
        file_name (str): _description_
        dtype (_type_, optional): _description_. Defaults to _EVENT_RECORD_DTYPE.

    Returns:
        tuple[FileHeader, list[EventsHeader]]: _description_
    """
    header = readOpenEphysHeader(file_name)
    events = np.memmap(file_name, dtype=dtype, offset=NUM_HEADER_BYTES)

    # events = np.fromfile(in_file, dtype=dtype)

    banks = events['processorid']
    del events
    file_header = FileHeader(FullFileName=file_name,
                             DateTime=header.date_created,
                             RecordingSystem='OpenEphys',
                             HeaderLength=NUM_HEADER_BYTES,
                             FileMajorVersion=int(
                                 str(header.version).split('.')[0]),
                             FileMinorVersion=int(
                                 str(header.version).split('.')[1]),
                             NumChannels=1,
                             )
    events_headers: list[EventsHeader] = []
    # generating different events data for banks
    for bank in np.unique(banks):
        events_header = EventsHeader(ADC=header.bitVolts,
                                     Bank=bank,
                                     ID=bank,
                                     Name=header.channel,
                                     SigUnits='ticks',

                                     NumRecords=np.sum(banks == bank),
                                     #  NumUnits=len(units_name),
                                     SamplingFreq=header.sampleRate,
                                     )
        events_headers.append(events_header)

    return (file_header, events_headers)


def loadContinuous(file_name: str, dtype=_CONTINUOUS_RECORD_DTYPE) -> np.ndarray:
    """Load data of given Openephys continuous format file.

    Args:
        file_name (str): _description_
        dtype (_type_, optional): _description_. Defaults to _CONTINUOUS_RECORD_DTYPE.

    Returns:
        np.ndarray: data
    """

    logger.info("Loading continuous data...")

    with open(file_name, 'rb') as in_file:
        in_file.seek(NUM_HEADER_BYTES)
        continuous = np.fromfile(in_file, dtype=dtype)
        data = continuous['records'].flatten()

    return data


def loadEvents(file_name: str, dtype=_EVENT_RECORD_DTYPE) -> dict[int, list[tuple[str, np.ndarray, np.ndarray]]]:
    """Load data of given Openephys events format file.

    Args:
        file_name (str): _description_
        dtype (_type_, optional): _description_. Defaults to _EVENT_RECORD_DTYPE.

    Returns:
        dict[int, list[tuple[str, np.ndarray, np.ndarray]]]: (unit name, timestamps, index) of each event unit by bank.
        {
            bank0: [(unit name, timestamps, index), ...], ...
        }
    """
    logger.info('loading events...')
    data = {}
    time_first_point = _getTimeFirstPoint(file_name)

    with open(file_name, 'rb') as in_file:
        in_file.seek(NUM_HEADER_BYTES)
        events = np.fromfile(in_file, dtype=dtype)

        banks = events['processorid']

        # generating different events data for banks
        for bank in np.unique(banks):
            index = []
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
                            index += [units_ind]
                            units_name += ["Chan_{:02d}_{}_{}".format(
                                channel, type_, stati_word[status])]

            data[bank] = list(zip(units_name, ts, index))

    return data


def _getTimeFirstPoint(file_name: str) -> int | float:
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

    if os.stat(file_name).st_size <= NUM_HEADER_BYTES:
        return

    if ext == '.continuous':
        with open(file_name, 'r') as data_file:
            data_map = np.memmap(
                data_file, _CONTINUOUS_RECORD_DTYPE, 'r', NUM_HEADER_BYTES)
            return data_map[0]['timestamp']

    if ext == '.events':
        with open(file_name, 'r') as data_file:
            data_map = np.memmap(
                data_file, _EVENT_RECORD_DTYPE, 'r', NUM_HEADER_BYTES)
            # pdb.set_trace()
            return data_map[0][0]

    if ext == '.spikes':
        with open(file_name, 'r') as data_file:
            data_map = np.memmap(
                data_file, _SPIKE_RECORD_DTYPE, 'r', NUM_HEADER_BYTES)
            return data_map[0][1]


if __name__ == '__main__':
    pass
    # data = loadEvents(
    #     r'C:\Users\harry\Desktop\Lab\Project_spikesorter\PySortGUI\data\RU01_2022-08-01_11-20-12\all_channels.events')
    # print(data)
    # data = loadContinuousHeader(
    #     r'C:\Users\harry\Desktop\Lab\Project_spikesorter\PySortGUI\data\MX6-22_2020-06-17_17-07-48_no_ref\100_CH2.continuous')
    # file_list = getFilesInFolder(
    #     r'C:\Users\harry\Desktop\Lab\Project_spikesorter\PySortGUI\data\MX6-22_2020-06-17_17-07-48_no_ref')
    # data = loadOpenephysHeader(file_list)
    # file_header, events_headers, ts, index, units_name = data

    # print(len(v))
    # a = {'format': "'Open Ephys Data Format'", ' version': '0.4', ' header_bytes': '1024',
    #      'description': "'each record contains one 64-bit timestamp, one 16-bit sample count (N), 1 uint16 recordingNumber, N 16-bit samples, and one 10-byte record marker (0 1 2 3 4 5 6 7 8 255)'", ' date_created': "'17-Jun-2020 170748'", 'channel': "'CH1'", 'channelType': "'Continuous'", 'sampleRate': '30000', 'blockLength': '1024', 'bufferSize': '1024', 'bitVolts': '0.195'}
    # b = {'format': "'Open Ephys Data Format'"}
    # OpenEphysHeader.model_validate(b)
