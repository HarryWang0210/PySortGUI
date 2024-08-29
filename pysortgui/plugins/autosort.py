import logging
import os

import click

from pysortgui.DataStructure.datav3 import SpikeSorterData
from pysortgui.plugins import IPlugin

# LOGGER -----------------------------------------------------------------
logger = logging.getLogger(__name__)


class SortPlugin(IPlugin):
    """Create the `phy cluster-manual` command for Kwik files."""

    def attach_to_cli(self, cli):

        @cli.group()
        @click.help_option('-h', '--help')
        def sorting():
            """
            Sorting utilities group
            """
            pass

        @sorting.command('extract_wav')
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
        @click.option('-r', '--ref', 'reference', type=click.INT, nargs=1,
                      default=-1, show_default=True,
                      help='Channel ID to sutract from. -1 means no reference.')
        @click.option('-t', '--thr', 'threshold_setting', type=(click.STRING, click.FLOAT), nargs=2,
                      default=('mad', -3), show_default=True,
                      help="""Threshold type and value. There are two optional types: [mad, const]. \n\b
                      - For the "mad" type, specify the multiplier of MAD as the value. \n\b
                      - For the "const" type, input the threshold directly as the value. \n\b
                      e.g. -t mad -3""")
        @click.option('-f', '--filter', 'filter_setting', type=(click.FLOAT, click.FLOAT), nargs=2,
                      default=(250, 3000), show_default=True,
                      help="""Bandpass filter values, including low cutoff and high cutoff. \n\b
                      e.g. -f 250 3000""")
        @click.help_option('-h', '--help')
        @click.pass_context
        def extract_waveforms(ctx, channels, path, input_format, output_format,
                              reference, threshold_setting, filter_setting):
            """Extract waveforms from the specified channel IDs."""
            data_object = SpikeSorterData(file_or_folder=path,
                                          data_format=input_format)

            for channel_ID in channels:
                filted_object = data_object.subtractReference(
                    channel_ID, reference)
                filted_object = filted_object.bandpassFilter(*filter_setting)
                if threshold_setting[0] == 'mad':
                    thr = filted_object.estimated_sd * threshold_setting[1]
                elif threshold_setting[0] == 'const':
                    thr = threshold_setting[1]
                spike_object = filted_object.extractWaveforms(thr)

                # generate label name
                raw_object = data_object.getRaw(channel_ID)
                if len(raw_object.spikes) > 0:
                    i = len(raw_object.spikes)
                    while True:
                        new_label = f'label{i}'
                        if new_label in raw_object.spikes:
                            i += 1
                        else:
                            break
                else:
                    new_label = 'default'

                spike_object.setLabel(new_label)
                raw_object.setSpike(spike_object, new_label)

            logger.info(f'Waveforms extraction finish. ')

            if output_format != input_format:
                new_filename = data_object.path
                if output_format == 'pyephys':
                    new_filename = os.path.splitext(new_filename)[0] + '.h5'
                    logger.info(f'Export and save result to {new_filename}...')
                    data_object.export(new_filename=new_filename,
                                       data_format=output_format)
            else:
                logger.info(f'Save result to {path}...')
                data_object.saveAll()
                # logger.debug('Unimplemented')

        @sorting.command('autosort')
        @click.argument('spikes', nargs=-1, type=click.STRING)
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
        @click.option('--overwrite', is_flag=True, default=False,
                      show_default=True,
                      help='Overwrite spike data. If False, create a new label.')
        @click.help_option('-h', '--help')
        @click.pass_context
        def autosort(ctx, spikes, path, input_format, output_format, overwrite):
            """
            Perform Klustakwik sorting on the specified spike data.

            \b
            The spikes argument must be provided in pairs.
            Input format: 'channelID1 label1 channelID2 label2 ...'

            Explanation:
            Each pair consists of a channel ID and a corresponding spike label.
            For example, '0 default 1 label1' means:
            Channel ID 0 with spike label 'default'.
            Channel ID 1 with spike label 'label1'.
            """
            if len(spikes) % 2 != 0:
                logger.error("Must provide channel ID and label pairs.")
                return

            data_object = SpikeSorterData(file_or_folder=path,
                                          data_format=input_format)

            for i in range(0, len(spikes), 2):
                channel_ID = int(spikes[i])
                spike_label = spikes[i+1]
                spike_object = data_object.getSpike(channel_ID, spike_label,
                                                    load_data=True)
                new_spike_object = spike_object.autosort()

                new_label = spike_label
                if not overwrite:
                    # generate label name
                    raw_object = data_object.getRaw(channel_ID)
                    if len(raw_object.spikes) > 0:
                        i = len(raw_object.spikes)
                        while True:
                            new_label = f'label{i}'
                            if new_label in raw_object.spikes:
                                i += 1
                            else:
                                break
                    else:
                        new_label = 'default'

                new_spike_object.setLabel(new_label)
                raw_object.setSpike(new_spike_object, new_label)

            logger.info(f'Waveforms extraction finish. ')

            if output_format != input_format:
                new_filename = data_object.path
                if output_format == 'pyephys':
                    new_filename = os.path.splitext(new_filename)[0] + '.h5'
                    logger.info(f'Export and save result to {new_filename}...')
                    data_object.export(new_filename=new_filename,
                                       data_format=output_format)
            else:
                logger.info(f'Save result to {path}...')
                data_object.saveAll()

        @sorting.group()
        @click.help_option('-h', '--help')
        def pipeline():
            """Processing pipeline group: Combine multiple steps into one step."""
            pass

        @pipeline.command('autosort')
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
        @click.option('-r', '--ref', 'reference', type=click.INT, nargs=1,
                      default=-1, show_default=True,
                      help='Channel ID to sutract from. -1 means no reference.')
        @click.option('-t', '--thr', 'threshold_setting', type=(click.STRING, click.FLOAT), nargs=2,
                      default=('mad', -3), show_default=True,
                      help="""Threshold type and value. There are two optional types: [mad, const]. \n\b
                      - For the "mad" type, specify the multiplier of MAD as the value. \n\b
                      - For the "const" type, input the threshold directly as the value. \n\b
                      e.g. -t mad -3""")
        @click.option('-f', '--filter', 'filter_setting', type=(click.FLOAT, click.FLOAT), nargs=2,
                      default=(250, 3000), show_default=True,
                      help="""Bandpass filter values, including low cutoff and high cutoff. \n\b
                      e.g. -f 250 3000""")
        @click.help_option('-h', '--help')
        @click.pass_context
        def autosort_pipeline(ctx, channels, path, input_format, output_format,
                              reference, threshold_setting, filter_setting):
            """Perform Klustakwik sorting after waveform extraction."""
            data_object = SpikeSorterData(file_or_folder=path,
                                          data_format=input_format)

            for channel_ID in channels:
                filted_object = data_object.subtractReference(
                    channel_ID, reference)
                filted_object = filted_object.bandpassFilter(*filter_setting)
                if threshold_setting[0] == 'mad':
                    thr = filted_object.estimated_sd * threshold_setting[1]
                elif threshold_setting[0] == 'const':
                    thr = threshold_setting[1]
                spike_object = filted_object.extractWaveforms(thr)
                spike_object = spike_object.autosort()

                # generate label name
                raw_object = data_object.getRaw(channel_ID)
                if len(raw_object.spikes) > 0:
                    i = len(raw_object.spikes)
                    while True:
                        new_label = f'label{i}'
                        if new_label in raw_object.spikes:
                            i += 1
                        else:
                            break
                else:
                    new_label = 'default'

                spike_object.setLabel(new_label)
                raw_object.setSpike(spike_object, new_label)

            logger.info(f'Waveforms extraction finish. ')

            if output_format != input_format:
                new_filename = data_object.path
                if output_format == 'pyephys':
                    new_filename = os.path.splitext(new_filename)[0] + '.h5'
                    logger.info(f'Export and save result to {new_filename}...')
                    data_object.export(new_filename=new_filename,
                                       data_format=output_format)
            else:
                logger.info(f'Save result to {path}...')
                data_object.saveAll()
