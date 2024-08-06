import click
from pysortgui.plugins.data_export import DataConvert


@click.group()
# @click.version_option(version=__version_git__)
@click.help_option('-h', '--help')
@click.pass_context
def pysortgui_cli(ctx, pdb=None, ipython=None, prof=None, lprof=None):
    """
    pysortgui command line interface.

    \b
    For help on entries listed below:
    >>>pysortgui-cli `entry` --help"""
    pass


# @click.command()
# @click.option('-t', '--to', 'to', help='To who')
# def greeting(to):
#     '''Say hello to someone'''
#     print(f'Hello, {to or "stranger"}!')

def launch_cli():
    DataConvert().attach_to_cli(pysortgui_cli)
    pysortgui_cli()


if __name__ == '__main__':
    launch_cli()
