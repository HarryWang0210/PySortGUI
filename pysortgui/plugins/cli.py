import logging

import click

from pysortgui.plugins.autosort import SortPlugin
from pysortgui.plugins.data_export import DataConvert

logger = logging.getLogger(__name__)


@click.group()
# @click.version_option(version=__version_git__)
@click.option('-d', '--debug', is_flag=True, default=False)
@click.help_option('-h', '--help')
@click.pass_context
def pysortgui_cli(ctx, debug):
    """
    pysortgui command line interface.

    \b
    For help on entries listed below:
    >>>pysortgui-cli `entry` --help"""
    if debug:
        root_logger = logging.getLogger('pysortgui')
        for handler in root_logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Enable debug logger")

# @click.option('-t', '--to', 'to', help='To who')
# def greeting(to):
#     '''Say hello to someone'''
#     print(f'Hello, {to or "stranger"}!')


def launch_cli():
    DataConvert().attach_to_cli(pysortgui_cli)
    SortPlugin().attach_to_cli(pysortgui_cli)

    pysortgui_cli()


if __name__ == '__main__':
    launch_cli()
