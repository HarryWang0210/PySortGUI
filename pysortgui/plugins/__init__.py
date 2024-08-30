# modified from pyephys.plugins.__init__.py
"""Plugin system.
This module deals with the plugin system to add functionalities to the PyEphys package.
The code used here is based upon:
1) http://eli.thegreenplace.net/2012/08/07/fundamental-concepts-of-plugin-infrastructures  # noqa
2) phy.utils.plugin package

"""


from six import with_metaclass
# import imp
import logging
# import os
# import pdb

logger = logging.getLogger(__name__)


class IPluginRegistry(type):
    plugins = []

    def __init__(cls, name, bases, attrs):
        if name != 'IPlugin':
            IPluginRegistry.plugins.append(cls)


class IPlugin(with_metaclass(IPluginRegistry)):
    """A class deriving from IPlugin can implement the following methods:

    * `attach_to_cli(cli)`: called when the CLI is created.

    """
    pass


# # ------------------------------------------------------------------------------
# # Plugins discovery
# # ------------------------------------------------------------------------------

# def _iter_plugin_files(dirs):
#     for plugin_dir in dirs:
#         plugin_dir = os.path.realpath(os.path.expanduser(plugin_dir))
#         if not os.path.exists(plugin_dir):
#             continue
#         for subdir, dirs, files in os.walk(plugin_dir, followlinks=True):
#             # Skip test folders.
#             base = os.path.basename(subdir)
#             if 'test' in base or '__' in base:  # pragma: no cover
#                 continue
#             logger.debug("Scanning `%s`.", subdir)
#             for filename in files:
#                 if (filename.startswith('__') or
#                         not filename.endswith('.py') or filename.startswith('cli.py')):
#                     continue  # pragma: no cover
#                 logger.debug("Found plugin module `%s`.", filename)
#                 yield os.path.join(subdir, filename)


# def discover_plugins(dirs=None):
#     """Discover the plugin classes contained in Python files.

#     Parameters
#     ----------

#     dirs : list
#         List of directory names to scan.

#     Returns
#     -------

#     plugins : list
#         List of plugin classes.

#     """

#     c = load_config()
#     for module in c.Imports.module_list:
#         try:
#             mod = __import__(module)
#         except:
#             pass

#     if dirs is None:
#         dirs = _gen_plugin_paths()
#     # pdb.set_trace()
#     # Scan all subdirectories recursively.
#     for path in _iter_plugin_files(dirs):
#         filename = os.path.basename(path)
#         subdir = os.path.dirname(path)
#         modname, ext = os.path.splitext(filename)
#         file, path, descr = imp.find_module(modname, [subdir])
#         if file:
#             # Loading the module registers the plugin in
#             # IPluginRegistry.
#             try:
#                 mod = imp.load_module(modname, file, path, descr)  # noqa
#             except Exception as e:  # pragma: no cover
#                 logger.exception(e)
#             finally:
#                 file.close()
#     return IPluginRegistry.plugins


# def _gen_plugin_paths():
#     out = [os.path.split(__file__)[0]]
#     # add also a folder in the home folder
#     out += [os.path.expanduser('~/.pyephys')]

#     return out
