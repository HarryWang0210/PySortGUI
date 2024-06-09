"""
This module provides utility functions to deal with files



.. sectionauthor:: Alessandro Scaglione

.. codeauthor: Alessandro Scaglione

---------
"""

from __future__ import print_function


import os
import sys
import re
import pdb  # @UnusedImport
import itertools
import PyEphys.Constants as lib_c
# LOGGER ----------------------------------------------------------------------

import logging
logger = logging.getLogger(__name__)

__updated__ = "2016-04-03"


def find_mount_point(path):
    path = os.path.abspath(path)
    while not os.path.ismount(path):
        path = os.path.dirname(path)
    return path


def print_and_delete(string_, history=[]):
    pass


def progress_bar(obj, iterable):

    level = logger.getEffectiveLevel()
    logger.setLevel('ERROR')
    p_per = (iterable.index(obj) + 1) * 1. / len(iterable) * 100
    msg = "\r{:.2f}%".format(p_per)
    sys.stdout.write(msg)
    sys.stdout.flush()
    if iterable.index(obj) == len(iterable) - 1:
        logger.setLevel(level)
        msg = "\r{:>10s}".format(' ')
        sys.stdout.write(msg)
        sys.stdout.flush()
        msg = "\r{:.0f}%".format(p_per)
        sys.stdout.write(msg)
        sys.stdout.flush()


def spinning_wheel(seq=["|", "/", "-", "\\"], index=[], last=False):

    if len(index) == 0:
        sys.stdout.write(' ')
        sys.stdout.flush()
        index.append(0)
    else:
        index[0] = index[0] + 1
        if index[0] == len(seq):
            index[0] = 0

    msg = "\r"
    sys.stdout.write(msg)
    sys.stdout.flush()
    if last:
        index.pop()
        return
    msg = seq[index[0]]
    sys.stdout.write(msg)
    sys.stdout.flush()


def hash_data_file(file_full_name, alg='CRC32', partial_read=False, block_size=1 * 2 ** 20, verbose=False):

    import hashlib
    import zlib

    fun_dict = {'SHA1': hashlib.sha1(),
                'SHA224': hashlib.sha224(),
                'CRC32': zlib.crc32,
                'MD5': hashlib.md5()}

    alg = alg.upper()
    if alg not in fun_dict:
        raise ValueError('hash_data_file type not known')

    if partial_read:
        string_ = 'partial'
    else:
        string_ = 'full'
        block_size = 10 * block_size
    if type(file_full_name) is str:
        msg = "|Computing {} {} for {} ".format(string_, alg, file_full_name)
    else:
        msg = "|Computing {} {}".format(string_, alg)
    if len(msg) > 60:
        msg = "|Computing {} {} for ...{} ".format(
            string_, alg, file_full_name[-38:])
    if verbose:
        sys.stdout.write(msg)
        sys.stdout.flush()
        # block_size = 4 * 2 ** 20  # 1-megabyte blocks
        spinning_wheel()
    out = 0
    if type(file_full_name) is str:
        with open(file_full_name, 'rb') as bin_file:
            while True:
                if verbose:
                    spinning_wheel()
                data_block = bin_file.read(block_size)
                if not data_block:
                    break
                if alg == 'CRC32':
                    out = fun_dict[alg](data_block, out)
                else:
                    fun_dict[alg].update(data_block)
    else:
        if verbose:
            spinning_wheel()
        data_block = file_full_name
        if alg == 'CRC32':
            out = fun_dict[alg](data_block, out)
        else:
            fun_dict[alg].update(data_block)

    msg = "\b \b\n" * len(msg)
    if verbose:
        sys.stdout.write(msg)
        sys.stdout.flush()

    if alg == 'CRC32':
        out = "{:x}".format(out & 0xFFFFFFFF)
    else:
        out = fun_dict[alg].hexdigest()

    return out


def hash_sha1(file_full_name, partial_read=False, block_size=1 * 2 ** 20):

    import hashlib
    if partial_read:
        string_ = 'partial'
    else:
        string_ = 'full'
        block_size = 10 * block_size

    sha = hashlib.sha1()
    msg = "|Computing {} SHA1 for {} ".format(string_, file_full_name)
    if len(msg) > 60:
        msg = "|Computing {} SHA1 for ...{} ".format(
            string_, file_full_name[-38:])
    sys.stdout.write(msg)
    sys.stdout.flush()
    # block_size = 4 * 2 ** 20  # 1-megabyte blocks
    spinning_wheel()
    if not os.path.isfile(file_full_name):
        return 'File Not Found'
    with open(file_full_name, 'rb') as bin_file:
        bin_file.seek(0, 2)
        fsize = bin_file.tell()
        bin_file.seek(0, 0)
        while True:
            spinning_wheel()

            # partial file reading
            if partial_read == True:
                if fsize <= 3 * block_size:
                    data_block = bin_file.read()
                else:
                    data_block = bin_file.read(block_size)
                    sha.update(data_block)
                    bin_file.seek(int(fsize / 2) - block_size, 0)
                    data_block = bin_file.read(block_size)
                    sha.update(data_block)
                    bin_file.seek(-block_size, 2)
                    data_block = bin_file.read(block_size)
                sha.update(data_block)
                break

            data_block = bin_file.read(block_size)
            if not data_block:
                break
            sha.update(data_block)
        spinning_wheel(last=True)
        msg = "\r" # * len(msg)
        sys.stdout.write(msg)
        sys.stdout.flush()
        return sha.hexdigest()


def hash_crc32(file_full_name, partial_read=False, block_size=1 * 2 ** 20):

    import hashlib
    import zlib
    alg = 'CRC32'
    if partial_read:
        string_ = 'partial'
    else:
        string_ = 'full'
        block_size = 10 * block_size

    sha = hashlib.md5()
    msg = "|Computing {} {} for {} ".format(string_, alg, file_full_name)
    if len(msg) > 60:
        msg = "|Computing {} {} for ...{} ".format(
            string_, alg, file_full_name[-38:])
    sys.stdout.write(msg)
    sys.stdout.flush()
    # block_size = 4 * 2 ** 20  # 1-megabyte blocks
    spinning_wheel()
    prev = 0
    with open(file_full_name, 'rb') as bin_file:
        while True:
            spinning_wheel()
            data_block = bin_file.read(block_size)
            if not data_block:
                break
            prev = zlib.crc32(data_block, prev)
    msg = "\b \b" * len(msg)
    return "%X" % (prev & 0xFFFFFFFF)

    with open(file_full_name, 'rb') as bin_file:
        bin_file.seek(0, 2)
        fsize = bin_file.tell()
        bin_file.seek(0, 0)
        while True:
            spinning_wheel()

            # partial file reading
            if partial_read == True:
                if fsize <= 3 * block_size:
                    data_block = bin_file.read()
                else:
                    data_block = bin_file.read(block_size)
                    sha.update(data_block)
                    bin_file.seek(int(fsize / 2) - block_size, 0)
                    data_block = bin_file.read(block_size)
                    sha.update(data_block)
                    bin_file.seek(-block_size, 2)
                    data_block = bin_file.read(block_size)
                sha.update(data_block)
                break

            data_block = bin_file.read(block_size)
            if not data_block:
                break
            sha.update(data_block)
        spinning_wheel(last=True)
        msg = "\b \b" * len(msg)
        sys.stdout.write(msg)
        sys.stdout.flush()
        return sha.hexdigest()


def files_in_dirs(Dir=None, Exts=None, isData=False):

    import os
    import pathlib

    if Dir is None:
        logger.warn('A starting path is needed to search for files')
        return

    if Exts is None:
        logger.warn(
            'No file extensions are given using the default values for BlackRock file types')
        Exts = {'.ns6', '.ns3', '.nev', '.ns2', '.ns5', '.ns4'}  # - {'.ns6'}

    Files = []
    for subdirs, _, files in os.walk(Dir):
        Files.extend([pathlib.Path(subdirs) / ffile for ffile in files
                      if os.path.splitext(ffile)[1] in Exts])

    if isData is False:
        return Files


def find_files_in_dir(start_dir='.', filters=['*'], recursive=True, return_dirs=False):

    import fnmatch
    files_list = []
    
    if isinstance(filters, str):
        filters = [filters]
    elif isinstance(filters, unicode):
        filters = [filters]

    if not type(start_dir) is list or tuple:
        start_dir = [start_dir]

    start_dirs = map(os.path.abspath, start_dir)
    # pdb.set_trace()
    for start_dir in start_dirs:
        if recursive:
            for subdirs, _, files in os.walk(start_dir):
                files_list.extend([os.path.join(subdirs, file_)
                                   for file_ in files if not file_.startswith('.')])
        else:
            files_list.extend([os.path.join(start_dir, file_) for file_ in os.walk(
                start_dir).next()[2] if not file_.startswith('.')])

    # removing duplicates
    files_list = list(set(files_list))
    filt_list = []

    for filter_ in list(filters):
        filt_list.extend(fnmatch.filter(files_list, filter_))
    filt_list = sorted(filt_list)
    if return_dirs:
        dirs = [os.path.split(path)[0] for path in filt_list]
        dirs = set(dirs)
        dirs = list(dirs)
        return filt_list, dirs
    filt_list = [os.path.abspath(file) for file in filt_list]
    return filt_list


def get_fields_from_filename(FileName, Format='Date_Subject_Group_Experiment', prefix='', sep='-', lower=True):

    FileName = os.path.splitext(FileName)[0]
    _, Name = os.path.split(FileName)
    # print(Name)
    Names = re.split('[_]?', Format)
    if lower:
        Names = [re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name).lower()
                 for name in Names]
    else:
        Names = [re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
                 for name in Names]
    Fields = re.split('[_]?', Name)
    #Names = map(str.capitalize, Names)
    unique_names = set(Names)

    if len(unique_names) != len(Names):
        NNames = list(set(Names))
        NFields = list()
        for name in NNames:
            NFields.append(
                [bel for bel, al in zip(Fields, Names) if al == name])

        Names = NNames
        Fields = NFields
        Fields = map(sep.join, Fields)

    Names = [prefix + name for name in Names]
    if len(Fields) != len(Names):

        #logger.warn('get_fields_from_filename:Number of Fields mismatch')
        pass

    return dict(zip(Names, Fields))


def is_data(file_name, reg_ex=lib_c.DATA_FILE_REGEX):

    _, ext = os.path.splitext(file_name)
    if re.match(reg_ex, ext):
        return True
    else:
        return False


def get_next_non_null_line(open_file):

    curr_file_pos = open_file.tell()
    open_file.seek(0, 2)
    f_size = open_file.tell()
    open_file.seek(curr_file_pos)

    line = ''
    while len(line) == 0 and open_file.tell() != f_size:
        line = open_file.readline().strip()
    return line
