import click
from pysortgui.plugins.data_export import DataConvert


@click.group()
# @click.version_option(version=__version_git__)
@click.help_option('-h', '--help')
@click.pass_context
def pyephys(ctx, pdb=None, ipython=None, prof=None, lprof=None):
    """
    pyephys command line interface.

    \b
    For help on entries listed below:
    >>>pyephys `entry` --help"""
    pass


# @click.command()
# @click.option('-t', '--to', 'to', help='To who')
# def greeting(to):
#     '''Say hello to someone'''
#     print(f'Hello, {to or "stranger"}!')

def launch_cli():
    DataConvert().attach_to_cli(pyephys)
    pyephys()


if __name__ == '__main__':
    launch_cli()
