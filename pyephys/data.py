import logging
import traceback
from importlib import import_module
import pkgutil

from functools import wraps

_MODULES_NAME = ['openephys']
_MODULES_DICT = {}
_HEADER_FUNC_DICT = {}
logger = logging.getLogger(__name__)

for module in _MODULES_NAME:
    # load input modules and save to  _MODULES_DICT
    try:
        prefix = __name__.rsplit('.', 1)[0] + '.'
        _MODULES_DICT[module] = import_module(prefix + module)
    except:
        logger.error('Not able to import module {}'.format(module))

def _execution_error(func):
    '''
    A wrapper that returns error if execution fails without raising an excetpion.
    This decorator switches an exception with a logger.error
    :param func: function
    '''

    @wraps(func)
    def decorate(*args, **kws):
        try:
            return func(*args, **kws)
        except Exception as _:
            #exec_info = sys.exc_info()
            logger.error('\t ******Execution halted for {}******\n{}'.format(
                func.__name__, traceback.format_exc()))

    return decorate

@_execution_error
def _export_to_pyephys_single_file(file_name):
    '''

    :param file_name:
    :type file_name:
    '''

    msg = "exporting {}".format(file_name)
    logger.info(msg)

    header = read_header(file_name)
    data = read_data(file_name)

    if header is None or header is None:
        return

    lib_pe.create_pyephys_datafile(header, data)

# this decorator raises an exception if the given filename is not a file
@_check_file_exist # 先確認file是存在再呼叫 read_header 
def read_header(file_name):

    _, ext = os.path.splitext(file_name) # 分開檔案名, 擴展名
    if ext.lower() in _HEADER_FUNC_DICT:
        return _HEADER_FUNC_DICT[ext](file_name) # 使用該擴展名read header的function
    else:
        logger.warn('File extension not known or not implemented yet')