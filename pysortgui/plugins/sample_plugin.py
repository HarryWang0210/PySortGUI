#!/usr/bin/env python

'''
Created on May 4, 2017

@author: scaglionea

Sample plugin file for pyephys, copy this file and change extension to .py 
to add new functionalities
'''

# this import is mandatory
import click
from pysortgui.plugins import IPlugin

# import the function that you want to be applied to the data
# from PyEphys import my_super_fun

# PLUGIN -----------------------------------------------------------------


class SamplePlugin(IPlugin):
    """Class used for the plugin system."""

    def attach_to_cli(self, cli):

        # sample group below
        @cli.group('sample_group')
        def group():
            pass

        # sample command, if group left on top the command will be part of the above group
        # Add new commands here
        @cli.command('sample_command')
        @click.help_option('-h', '--help')
        # modify options here you may want to keep the following path variable
        @click.argument('arg1', type=click.Path(exists=True))
        # add other options like the following
        @click.argument('arg2', default=0)
        @click.pass_context
        def sample_command(ctx, arg1, arg2):
            """Export the file in the pyephys format"""

            # my_super_fun(path, channel)
