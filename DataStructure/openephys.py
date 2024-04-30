import logging

import numpy as np
from pydantic import BaseModel

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

    date_created: str  # 'dd-mm-yyyy hhmmss'

    channel: str  # '(String with channel name)'

    channelType: str  # '(String describing the channel type)'

    sampleRate: int  # (integer sampling rate)

    blockLength: int  # 1024

    bufferSize: int  # 1024

    # (floating point value of microvolts/bit for headstage channels, or volts/bit for ADC channels)
    bitVolts: float | int


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


def readHeader(f):
    header = {}
    h = f.read(1024).decode().replace('\n', '').replace('header.', '')
    for i, item in enumerate(h.split(';')):
        if '=' in item:
            header[item.split(' = ')[0].lstrip().rstrip()] = item.split(
                ' = ')[1].lstrip("'").rstrip("'")
    return header


def loadContinuous(file_name, dtype=_CONTINUOUS_RECORD_DTYPE):

    # assert dtype in (float, np.int16), \
    #     'Invalid data type specified for loadContinous, valid types are float and np.int16'

    print("Loading continuous data...")

    ch = {}

    with open(file_name, 'rb') as in_file:
        header = OpenEphysHeader.model_validate(readHeader(in_file))
        in_file.seek(1024)
        continuous = np.fromfile(in_file, dtype=dtype)

        ch['header'] = header
        ch['timestamps'] = continuous['timestamp']
        # OR use downsample(samples,1), to save space
        ch['data'] = continuous['records'].flatten()
        ch['recordingNumber'] = continuous['rnumber']

        # continuous = continuous['records'].flatten()
    return ch
    # # read in the data
    # f = open(filepath, 'rb')

    # fileLength = os.fstat(f.fileno()).st_size

    # # calculate number of samples
    # recordBytes = fileLength - NUM_HEADER_BYTES
    # if recordBytes % RECORD_SIZE != 0:
    #     raise Exception(
    #         "File size is not consistent with a continuous file: may be corrupt")
    # nrec = recordBytes // RECORD_SIZE
    # nsamp = nrec * SAMPLES_PER_RECORD
    # # pre-allocate samples
    # samples = np.zeros(nsamp, dtype)
    # timestamps = np.zeros(nrec)
    # recordingNumbers = np.zeros(nrec)
    # indices = np.arange(0, nsamp + 1, SAMPLES_PER_RECORD, np.dtype(np.int64))

    # header = readHeader(f)

    # recIndices = np.arange(0, nrec)

    # for recordNumber in recIndices:

    #     timestamps[recordNumber] = np.fromfile(f, np.dtype(
    #         '<i8'), 1)  # little-endian 64-bit signed integer
    #     # little-endian 16-bit unsigned integer
    #     N = np.fromfile(f, np.dtype('<u2'), 1)[0]

    #     # print index

    #     if N != SAMPLES_PER_RECORD:
    #         raise Exception(
    #             'Found corrupted record in block ' + str(recordNumber))

    #     # big-endian 16-bit unsigned integer
    #     recordingNumbers[recordNumber] = (np.fromfile(f, np.dtype('>u2'), 1))

    #     if dtype == float:  # Convert data to float array and convert bits to voltage.
    #         # big-endian 16-bit signed integer, multiplied by bitVolts
    #         data = np.fromfile(f, np.dtype('>i2'), N) * \
    #             float(header['bitVolts'])
    #     else:  # Keep data in signed 16 bit integer format.
    #         # big-endian 16-bit signed integer
    #         data = np.fromfile(f, np.dtype('>i2'), N)
    #     samples[indices[recordNumber]:indices[recordNumber+1]] = data

    #     marker = f.read(10)  # dump

    # # print recordNumber
    # # print index

    # ch['header'] = header
    # ch['timestamps'] = timestamps
    # ch['data'] = samples  # OR use downsample(samples,1), to save space
    # ch['recordingNumber'] = recordingNumbers
    # f.close()
    return ch


if __name__ == '__main__':
    data = loadContinuous(
        r'C:\Users\harry\Desktop\Lab\Project_spikesorter\PySortGUI\data\MX6-22_2020-06-17_17-07-48_no_ref\100_CH1.continuous')
    # print(data)

    for k, v in data.items():
        print(k)
        print(v)
        print(type(v))
        # print(len(v))
    # a = {'format': "'Open Ephys Data Format'", ' version': '0.4', ' header_bytes': '1024',
    #      'description': "'each record contains one 64-bit timestamp, one 16-bit sample count (N), 1 uint16 recordingNumber, N 16-bit samples, and one 10-byte record marker (0 1 2 3 4 5 6 7 8 255)'", ' date_created': "'17-Jun-2020 170748'", 'channel': "'CH1'", 'channelType': "'Continuous'", 'sampleRate': '30000', 'blockLength': '1024', 'bufferSize': '1024', 'bitVolts': '0.195'}
    # b = {'format': "'Open Ephys Data Format'"}
    # OpenEphysHeader.model_validate(b)
