# modified form pyephys.plugins.data_utils.py
import logging
import os

import click

from pysortgui.DataStructure.datav3 import SpikeSorterData
# from PyEphys.FileIO.FileIO import export_to_pyephys
# from PyEphys import PyEphysCLS
from pysortgui.plugins import IPlugin

# import pprint

# import pdb
logger = logging.getLogger(__name__)

# PLUGIN -----------------------------------------------------------------


class DataConvert(IPlugin):
    def attach_to_cli(self, cli):

        @cli.group()
        def data():
            '''
            Data convert group
            '''
            pass
        # import command

        @data.command('convert')
        @click.argument('paths', type=click.Path(exists=True, file_okay=True, dir_okay=True), nargs=-1)
        @click.option('-i', '--input_format', 'input_format', type=str, nargs=1,
                      default='openephys', show_default=True,
                      help='Input file/folder format')
        @click.option('-o', '--output_format', 'output_format', type=str, nargs=1,
                      default='pyephys', show_default=True,
                      help='Output file/folder format')
        @click.help_option('-h', '--help')
        @click.pass_context
        def convert_files(ctx, paths, input_format, output_format):
            """Convert the file or folder from input_format to output_format.
            Can convert multiple files."""
            if len(paths) == 0:
                paths = '.'

            for path in paths:
                data_object = SpikeSorterData(file_or_folder=path,
                                              data_format=input_format)

                new_filename = data_object.path
                if output_format == 'pyephys':
                    new_filename = os.path.splitext(new_filename)[0] + '.h5'

                data_object.export(new_filename=new_filename,
                                   data_format=output_format)
