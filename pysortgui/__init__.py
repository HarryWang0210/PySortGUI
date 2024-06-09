'''
PyEphys is a module that allows the manipulation of Neurophysiological data in
Python.

.. todo::
    make a tutorial

'''
import logging

_DEBUG = 1

# non specific imports defaults to none
__all__ = []
# spikesortergl root logger
logger = logging.getLogger(__name__)
# adding null handler [see logging documentation]
if len(logger.handlers) == 0:
    logger.addHandler(logging.NullHandler())


class _StreamFormatter(logging.Formatter):

    '''Class that implements custom formatters for each type of level'''

    FORMATS = {logging.DEBUG: "\033[1m%(message)-72s\033[37m%(levelname)-8s\n\t%(module)s: %(lineno)d:\033[0m",
               logging.INFO: "\033[1m%(message)-72s\033[34m%(levelname)-8s\n\t%(module)s:line %(lineno)s\033[0m",
               logging.WARN: "\033[1m%(message)-72s\033[33m%(levelname)-8s\n\t%(module)s:line %(lineno)s-%(funcName)-15s()\033[0m",
               logging.ERROR: "\033[1m%(message)-72s\033[31m%(levelname)-8s\n\t%(module)s:line %(lineno)s-%(funcName)-15s()[%(pathname)s]\033[0m",
               logging.CRITICAL: "\033[1m%(message)-72s\033[31m%(levelname)-8s\n\t%(module)s:line %(lineno)s-%(funcName)-15s()[%(pathname)s]\033[0m",
               'DEFAULT': "\033[1m%(message)-84s\033[0m"}

    def format(self, record):
        import sys
        import logging  # @Reimport
        if sys.version_info < (3, 0, 0):
            self._fmt = self.FORMATS.get(
                record.levelno, self.FORMATS['DEFAULT'])
        else:
            self._style._fmt = self.FORMATS.get(
                record.levelno, self.FORMATS['DEFAULT'])
        return logging.Formatter.format(self, record)


class _FileFormatter(logging.Formatter):

    '''Class that implements custom formatters for each type of level'''

    FORMATS = {logging.DEBUG: "%(asctime)s - %(levelname)-8s : %(message)s | %(module)s:line %(lineno)s-%(funcName)-15s()[%(pathname)s]",
               logging.INFO: "%(asctime)s - %(levelname)-8s : %(message)s | %(module)s:line %(lineno)s",
               logging.WARN: "%(asctime)s - %(levelname)-8s :%(message)s | %(module)s:line %(lineno)s-%(funcName)-15s() ",
               logging.ERROR: "%(asctime)s - %(levelname)-8s :%(message)s | %(module)s:line %(lineno)s-%(funcName)-15s()[%(pathname)s]",
               logging.CRITICAL: "%(asctime)s - %(levelname)-8s :%(message)s | %(module)s:line %(lineno)s-%(funcName)-15s()[%(pathname)s]",
               'DEFAULT': "%(message)s"}

    def format(self, record):
        import sys
        import logging  # @Reimport
        if sys.version_info < (3, 0, 0):
            self._fmt = self.FORMATS.get(
                record.levelno, self.FORMATS['DEFAULT'])
        else:
            self._style._fmt = self.FORMATS.get(
                record.levelno, self.FORMATS['DEFAULT'])
        return logging.Formatter.format(self, record)


def _get_logger_file_name():

    import os
    import time

    if os.path.isdir(os.path.expanduser('~/Library/Logs/')):
        log_file = os.path.expanduser('~/Library/Logs/')
    else:
        log_file = os.path.expanduser('~')

    try:
        log_file = os.path.join(log_file, __name__ + '.log')
        # if not os.path.isdir(log_file):
        #    os.mkdir(log_file)

        # remove log older than n days
        n_days = 7
        if os.stat(log_file).st_ctime < time.time() - n_days * 24 * 60 * 60:
            os.remove(log_file)

        # old remove code for folder
        # rem_list = []
        # for f in os.listdir(log_file):
        #    if os.stat(os.path.join(log_file, f)).st_mtime < time.time() - 30 * 86400:
        #        rem_list.append(f)

        # if len(rem_list) > 0:
        #    [os.remove(file_) for file_ in rem_list]

    except:
        log_file = None

    return log_file


def _attach_handler(file_handler=False, level=logging.NOTSET):
    '''Attaches spikesortergl predef handler to logger'''

    import logging  # @Reimport
    import sys
    # standard output handler
    logger = logging.getLogger(__name__)
    logger.propagate = False
    std_hnd = logging.StreamHandler(sys.stdout)
    std_hnd.setFormatter(_StreamFormatter())
    std_hnd.setLevel(level)

    # attaching handler
    logger.addHandler(std_hnd)

    # file handler
    if file_handler:
        log_file = _get_logger_file_name()
        if log_file is not None:
            file_hnd = logging.FileHandler(log_file, 'a')
            file_hnd.setFormatter(_FileFormatter())
            file_hnd.setLevel(level)

            # attaching handler
            logger.addHandler(file_hnd)


def _remove_handler():
    '''Removes all handlers attached to the logger'''

    import logging  # @Reimport
    logger = logging.getLogger(__name__)
    logger.propagate = True
    hnds = logger.handlers
    for handler in hnds[::-1]:
        if handler == logging.NullHandler():
            continue
        hnds.pop(handler)
        handler.close()


if _DEBUG:
    _attach_handler(file_handler=True)
    logger.setLevel('INFO')
    # FileLogger logs complete execution
    logger.handlers[-1].setLevel(logging.DEBUG)

else:
    _attach_handler(file_handler=False)
    logger.setLevel('INFO')

del logging
