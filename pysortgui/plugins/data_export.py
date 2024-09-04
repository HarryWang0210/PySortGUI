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
        @click.help_option('-h', '--help')
        def data():
            '''
            Data utilities group
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
            """Convert files or folders from input_format to output_format.
            Supports conversion of multiple files."""
            if len(paths) == 0:
                paths = '.'

            for path in paths:
                data_object = SpikeSorterData(file_or_folder=path,
                                              data_format=input_format)
                data_object.export(data_format=output_format)

        @data.command('create_ref')
        @click.argument('channels', nargs=-1, type=click.INT)
        @click.option('--path', 'path',
                      type=click.Path(
                          exists=True, file_okay=True, dir_okay=True),
                      nargs=1, default='.',
                      help='Path to the data file or folder. Uses the current path by default.')
        @click.option('-i', '--input_format', 'input_format', type=click.STRING, nargs=1,
                      default='pyephys', show_default=True,
                      help='Input file/folder format. If the output format does not match the input format, the data will be exported in the output format.')
        @click.option('-o', '--output_format', 'output_format', type=click.STRING, nargs=1,
                      default='pyephys', show_default=True,
                      help='Output file/folder format. If the output format does not match the input format, the data will be exported in the output format.')
        @click.option('-n', '--name', 'name', type=click.STRING, nargs=1,
                      help='New channel name.')
        @click.option('--comment', type=click.STRING, nargs=1)
        @click.option('-m', '--method', type=click.STRING, nargs=1,
                      default='median', show_default=True)
        @click.help_option('-h', '--help')
        @click.pass_context
        def create_reference(ctx, channels, path, input_format, output_format,
                             name, comment, method):
            """Create a common reference channel from the given channels."""

            if method != 'median':
                logger.error(
                    f'Can not create reference channel by {method} method!')
                return
            channels = set(channels)
            data_object = SpikeSorterData(file_or_folder=path,
                                          data_format=input_format)
            channel_text = ', '.join([str(i) for i in channels])
            if name is None:
                name = f'{method}Ref({channel_text})'

            if comment is None:
                comment = f'This channel is a {method.lower()} reference channel made from ' +\
                    f'{channel_text}'

            new_ref_object = data_object.createMedianReference(
                list(channels), name, comment)

            if output_format != input_format:
                data_object.export(data_format=output_format)
            else:
                logger.info(f'Save result to {path}...')
                data_object.saveReference(new_ref_object.channel_ID)
