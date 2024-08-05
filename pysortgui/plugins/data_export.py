#!/usr/bin/env python

'''
Created on May 4, 2017

@author: scaglionea
'''

import os

import click

from pysortgui.DataStructure.datav3 import SpikeSorterData
# from PyEphys.FileIO.FileIO import export_to_pyephys
# from PyEphys import PyEphysCLS
from pysortgui.plugins import IPlugin

# import pprint

# import pdb

# PLUGIN -----------------------------------------------------------------


class DataConvert(IPlugin):
    def attach_to_cli(self, cli):

        @cli.group()
        def data():
            '''
            Data info and import group :pyephys:
            '''
            pass
        # import command

        @data.command('convert')
        @click.argument('paths', type=click.Path(exists=True, file_okay=True, dir_okay=True), nargs=-1)
        @click.argument('input_format', nargs=1)
        @click.argument('output_format', nargs=1)
        @click.help_option('-h', '--help')
        @click.pass_context
        def convert_files(ctx, paths, input_format, output_format):
            """Convert the file or folder into output_format. 
            Can convert multiple files."""
            if len(paths) == 0:
                paths = '.'

            for path in paths:
                data_object = SpikeSorterData(file_or_folder=path,
                                              data_format=input_format)
                file = data_object.path
                if output_format == 'pyephys':
                    new_filename = os.path.splitext(file)[0] + '.h5'
                    data_object.export(new_filename=new_filename,
                                       data_format=output_format)
                else:
                    pass
