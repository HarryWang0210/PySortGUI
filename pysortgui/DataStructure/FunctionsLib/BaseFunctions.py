'''
This module provides some Basic functions to work with basic classes defined

.. sectionauthor:: Alessandro Scaglione

.. codeauthor: Alessandro Scaglione
'''
from __future__ import division
import pdb
'''
Created on Mar 17, 2014

@author: Alessandro Scaglione
@email: alessandro.scaglione@gmail.com
'''
#=========================================================================
# BUITLIN
#=========================================================================
import logging

#=========================================================================
# PIP
#=========================================================================
#import psutil
import numpy as np

#=========================================================================
# PYEPHYS
#=========================================================================


#------------------------------------------------------------------------ Logger
logger = logging.getLogger(__name__)


def maxpoints_in_memory(NumBytesPerPoint=2, NChannels=None, FileNumPoints=None):
    import psutil
    #: Uses at least 200Mb of free ram for the import
    MINMEMORY = 2 ** 20 * 1500

    #: Uses at Most 1.5G of free ram for the import
    MAXMEMORY = 2 ** 20 * 3000

    AvailableMemory = psutil.phymem_usage().free  # @UndefinedVariable

    if AvailableMemory < MINMEMORY:
        AvailableMemory = MINMEMORY

    if AvailableMemory > MAXMEMORY:
        AvailableMemory = MAXMEMORY

    MemChunk = AvailableMemory  # * 3 / 4
    MaxNumPoints = (MemChunk / (NumBytesPerPoint))

    if NChannels is None or FileNumPoints is None:
        return long(MaxNumPoints)

    MaxNumPoints = MaxNumPoints - MaxNumPoints % NChannels
    if (MaxNumPoints < FileNumPoints):
        PointsToRead = MaxNumPoints
        # print('There is not enough free memory on the system to load the file'
        #      ' in memory')
    else:
        PointsToRead = FileNumPoints
        # print('There is enough free memory on the system to load the file'
        #      ' in memory')
    return long(PointsToRead)


def print_in_color(Color, InputString):
    """

    prints strings in color.

    :param Color: string with the color
    :type Color: str
    :param InputString: the string containing the text
    :type InputString: str

    Example::

        >>> print_in_color('red', 'color %s' %('red'))
    .. raw:: html

        <font color="red">color red</font>

    """

    ColorToNum = {'red': 31,
                  'green': 32,
                  'yellow': 33,
                  'blue': 34,
                  'cyan': 36,
                  'redB': 31 + 10,
                  'greenB': 32 + 10,
                  'yellowB': 33 + 10,
                  'blueB': 34 + 10,
                  'cyanB': 36 + 10,
                  }

    CSI = "\x1b["
    color = CSI + str(ColorToNum[Color]) + 'm'
    reset = CSI + '0m'

    print(color + InputString + reset)


def dict_to_recarray(Dicts, Key=None):
    """dict_to_recarray(Dict, [Key])

    Converts a dictionary or a list or dictionaries into a numpy recarray. Given
    a dictionary returns a recarray whose fields are keys in the original dict.
    If al list of dictionary is given, the a recarray of n rows where n is the
    lenght of the dictionary list is returned. Please make sure that each key of
    the dictionary contains only **one variable**. Also, that there are the
    **same number** of keys and that **each key name is the same** for each
    dictionary of the list, order is not important.

    If Key is specified then the function is applied to the key of the given
    dictionary or list of dictionaries. Convenient when the recarray is created
    by subdict in the class

    :param Dict: input dictionary or list of dictionaries
    :type Dict: dictionary or list
    :param Value: Key of the dictionary
    :rtype: str

    .. note::
        by default it excludes keys that begin with _ 'obfuscation'
    """

    if not isinstance(Dicts, list):
        Dicts = [Dicts]

    # make a copy of the list of dictionaries
    # #can be made as a generator but then index below is now working
    Dicts = [Dict.copy() for Dict in Dicts]

    for Dict in Dicts:

        Keys = Dict.keys()
        # removes key starting with _
        for RKey in Keys:
            if RKey[0] == '_':
                Dict.pop(RKey)

        if not isinstance(Dict, dict):
            raise ValueError('The input must be a dictionary')
        # check if Dict dict has a Dict subdict and select that for the
        # conversion
        if Key in Dict:
            Dict = Dict[Key]
        # print [np.isscalar(y) for (x, y) in Dict.items()]
        Test = [
            np.isscalar(y) if y is not None else True for y in Dict.values()]
        if False in Test:
            keys = Dict.keys()
            for item, key in zip(Test, keys):
                if item is False:
                    # check if it is of quantity class
                    String = "Field {} is not scalar. Removing it from rec_array".format(
                        Dicts.index(Dict), key)

                    logging.warn(String)

                    Dict.pop(key)

        # Warning

        # tries to sort in alphabetical order this is why we don't use the two
        # lines below
        # Data.append(tuple(Dict.values()))
        # Names=Dict.keys()
        # Data.append(tuple([y for (x, y) in sorted(Dict.items())]))  # @UnusedVariable
        # Names = sorted(Dict.keys())

    Names = sorted(set([item for Dict in Dicts for item in Dict.keys()]))
    Data = [[item[key] if (key in item and item[key] is not None) else np.nan for key in Names]
            for item in Dicts]
    Data = [[item if type(item) is not unicode else str(item)
             for item in row] for row in Data]
    #print(Data, Names)
    # print(Names)
    return np.rec.fromrecords(Data, names=Names)


def recarray_to_dict(RecArray, Class=dict(), Exclude=(None)):
    """
    given a list of recarray or a recarray return a dict with Header subdict
    of the respective recarray. If a class instance is given insted of dict
    return the Header subdict in the corresponding class

    :param RecArray: the recarray to render as a dict
    :type RecArray: numpy.recarray
    :param Class: return the recarray in a particular class instance
    :type Class: any derived dict class
    """
    if RecArray is None:
        return []

    if not hasattr(RecArray, 'dtype'):
        raise ValueError('Input not numpy recarray :-o')
    if RecArray.dtype.names == None:
        raise ValueError('Input not numpy Recarray :-o')

    if not isinstance(Class, dict):
        raise ValueError('Class must be of type dict or derived :-o')

    if isinstance(Exclude, str) or Exclude is None:
        Exclude = [Exclude]
    else:
        Exclude = list(Exclude)
    # Header=Class
    # if not Class.has_key('Header'):
    #    Header['Header']={}

    # Headers=[Header] * len(RecArray)

    Headers = [type(Class)() for _ in RecArray]
    for Header, row in zip(Headers, RecArray):
        # print row
        # print(row.dtype.names)
        keys = row.dtype.names
        keys = [key for key in keys if key not in Exclude]
        for key in keys:
            value = row[key]
            if value is np.nan:
                value = None
            if not 'Header' in Header:
                # print(np.isscalar(row[key]))
                Header[key] = value

            else:
                Header['Header'][key] = value

    return Headers


def class_patch(clsi, methods):
    """
    Creates a patch for a method of the given clsi class instance. It
    works by dinamically creating a new classes on which the given methods is
    patched and is then assigned to the instance

    :param clsi: class instance
    :type clsi: a class containing the given methods
    :param methods: a dictionary of the method to change with the name and the
                    method name
    :type methods: dict

    example:
    >>>class_patch(dict(),{'__getitem__':__getitem__}
    this will change the special function __getitem__ for the instance created
    by dict()
    """

    oldtype = type(clsi)
    newtype = type(oldtype.__name__ + '_mpatch_', (oldtype,), methods)
    clsi.__class__ = newtype


def class_patch_nonheap(clsi, methods):
    """
    Creates a patch for a method of the given clsi class instance. It
    works by dinamically creating a new classes on which the given methods is
    patched and is then assigned to the instance

    :param clsi: class instance
    :type clsi: a class containing the given methods
    :param methods: a dictionary of the method to change with the name and the
                    method name
    :type methods: dict

    example:
    >>>class_patch(<non heap class>,{'__getitem__':__getitem__}
    this will change the special function __getitem__ for the instance created
    by dict()
    """

    oldtype = type(clsi)
    newtype = type(oldtype.__name__ + 'mpatch', (oldtype,), methods)
    clsi = newtype(**clsi)
    return clsi


def file_read_iterator(fid, dtype=np.int16, points=100 * 2 ** 20 / 2):
    """
    file_read_iterator(fid,dtype=np.int16,points=100*2**20/2)

    utility function that return an iterator on the raw data files in order to
    avoid memory size problems

    :param fid: File identifier
    :type fid: file <type>
    :param dtype: the type of data based on numpy
    :type dtype: numpy.dtype <type>
    :param points: number of points to read of given dtype
    :type points: int
    """

    while True:
        Data = np.fromfile(fid, dtype=dtype, count=points)
        if len(Data) != 0:
            yield Data
        else:
            return


if __name__ == '__main__':
    pass
