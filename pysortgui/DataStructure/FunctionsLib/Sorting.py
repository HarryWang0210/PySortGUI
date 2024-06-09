import numpy as np
import os
import platform

from subprocess import call

import logging
logger = logging.getLogger(__name__)
KLUSTA_SUFFIX = '_klusta_sort'


__version__ = 0.1
__date__ = '2015-10-15'
__updated__ = '2017-05-24'

DEBUG = 0
TESTRUN = 0
PROFILE = 0


def _sort_util_create_klusta_files(data_file, chan_ID=None, sorting='offline'):

    file_full_name = os.path.abspath(data_file)

    if not os.path.isfile(file_full_name):
        return

    if chan_ID is None:
        return

    # chan_ID = data.Raws.find(ID=chan_ID).ID

    file_full_name = os.path.abspath(file_full_name)
    file_path, data_file_name = os.path.split(file_full_name)
    data_file_name, _ = os.path.splitext(data_file_name)
    if sorting == 'offline' or sorting is None:
        klusta_base_file = os.path.join(
            file_path, data_file_name + KLUSTA_SUFFIX)
        klusta_clu_file = os.path.join(
            file_path, data_file_name + KLUSTA_SUFFIX + '.clu.{}'.format(chan_ID))
        klusta_fet_file = os.path.join(
            file_path, data_file_name + KLUSTA_SUFFIX + '.fet.{}'.format(chan_ID))
        klusta_klg_file = os.path.join(
            file_path, data_file_name + KLUSTA_SUFFIX + '.klg.{}'.format(chan_ID))
        klusta_temp_file = os.path.join(
            file_path, data_file_name + KLUSTA_SUFFIX + '.temp.clu.{}'.format(chan_ID))

        klusta_base_file_indir = os.path.join(
            klusta_base_file, data_file_name + KLUSTA_SUFFIX)
        klusta_clu_file_indir = os.path.join(
            klusta_base_file, data_file_name + KLUSTA_SUFFIX + '.clu.{}'.format(chan_ID))
        klusta_fet_file_indir = os.path.join(
            klusta_base_file, data_file_name + KLUSTA_SUFFIX + '.fet.{}'.format(chan_ID))
        klusta_klg_file_indir = os.path.join(
            klusta_base_file, data_file_name + KLUSTA_SUFFIX + '.klg.{}'.format(chan_ID))
        klusta_temp_file_indir = os.path.join(
            klusta_base_file, data_file_name + KLUSTA_SUFFIX + '.temp.clu.{}'.format(chan_ID))

    if sorting == 'online':

        klusta_base_file = os.path.join(
            file_path, data_file_name + KLUSTA_SUFFIX + '_online')
        klusta_clu_file = os.path.join(
            file_path, data_file_name + KLUSTA_SUFFIX + '_online.' + 'clu.{}'.format(chan_ID))
        klusta_fet_file = os.path.join(
            file_path, data_file_name + KLUSTA_SUFFIX + '_online.' + 'fet.{}'.format(chan_ID))
        klusta_klg_file = os.path.join(
            file_path, data_file_name + KLUSTA_SUFFIX + '_online.' + 'klg.{}'.format(chan_ID))
        klusta_temp_file = os.path.join(
            file_path, data_file_name + KLUSTA_SUFFIX + '_online.' + 'temp.clu.{}'.format(chan_ID))

        klusta_base_file_indir = os.path.join(
            klusta_base_file, data_file_name + KLUSTA_SUFFIX + '_online')
        klusta_clu_file_indir = os.path.join(
            klusta_base_file, data_file_name + KLUSTA_SUFFIX + '_online.' + 'clu.{}'.format(chan_ID))
        klusta_fet_file_indir = os.path.join(
            klusta_base_file, data_file_name + KLUSTA_SUFFIX + '_online.' + 'fet.{}'.format(chan_ID))
        klusta_klg_file_indir = os.path.join(
            klusta_base_file, data_file_name + KLUSTA_SUFFIX + '_online.' + 'klg.{}'.format(chan_ID))
        klusta_temp_file_indir = os.path.join(
            klusta_base_file, data_file_name + KLUSTA_SUFFIX + '_online.' + 'temp.clu.{}'.format(chan_ID))

    # checking if file already auto_sorted

    if not os.path.isdir(klusta_base_file):
        os.mkdir(klusta_base_file)

    if os.path.isfile(klusta_clu_file):
        os.rename(klusta_clu_file, klusta_clu_file_indir)
    if os.path.isfile(klusta_fet_file):
        os.rename(klusta_fet_file, klusta_fet_file_indir)
    if os.path.isfile(klusta_klg_file):
        os.rename(klusta_klg_file, klusta_klg_file_indir)
    if os.path.isfile(klusta_temp_file):
        os.rename(klusta_temp_file, klusta_temp_file_indir)

    if os.path.isfile(klusta_clu_file + '.png'):
        os.rename(
            klusta_clu_file + '.png', klusta_clu_file_indir + '.png')

    klusta_base_file = klusta_base_file_indir
    klusta_clu_file = klusta_clu_file_indir
    klusta_fet_file = klusta_fet_file_indir

    return klusta_base_file, klusta_clu_file, klusta_fet_file, klusta_clu_file + '.png'


def auto_sort(
        data_file, chan_ID, feat, waveforms, timestamps, sorting=None,
        debug=DEBUG, re_sort=False
):

    klusta_base_file, klusta_clu_file, klusta_fet_file, klusta_png_file = _sort_util_create_klusta_files(
        data_file, chan_ID, sorting
    )

    if os.path.isfile(klusta_png_file) and not re_sort:
        msg = "Channel already sorted. Skipping..."
        logger.info(msg)
        return

    # data = PyEphysCLS(data_file)

    # if data.Spikes is not None:
    #     spk = data.Spikes.find(ID=chan_ID)
    # else:
    #     spk = None

    # if spk is None:
    #     msg = "No spikes channels with ID {} have been found".format(chan_ID)
    #     logger.info(msg)
    #     data.clear()
    #     del data
    #     detect_waveforms(
    #         data_file, chan_ID, ref_chan=ref_chan, sorting=sorting,
    #         debug=debug
    #     )
    #     data = PyEphysCLS(data_file)
    #     spk = data.Spikes.find(ID=chan_ID)

    msg = "Autosorting channel {}".format(chan_ID)
    logger.info(msg)

    msg = "Computing PCA feature and saving fet file... "
    logger.info(msg)
    # PCAs feature
    if waveforms.shape[0] < 42:
        msg = 'Few Waveforms. Skipping'
        return

    feat = feat[:, 0:4]

    # adding max and min to the features
    add_feat_min = waveforms.min(1).reshape(-1, 1)
    add_feat_max = waveforms.max(1).reshape(-1, 1)
    add_feat = np.hstack((add_feat_min, add_feat_max))

    feat = np.hstack((feat, add_feat))
    num_features = '{:d}'.format(feat.shape[1])

    np.savetxt(
        klusta_fet_file, feat, header=num_features, comments='')
    msg = "{} features for {} waveforms".format(
        num_features, feat.shape[0])
    logger.info(msg)

    msg = "AutoSort in progress..."
    logger.info(msg)
    call_list = [_KlustaKwik_executable(), '{}'.format(klusta_base_file), '{}'.format(chan_ID),
                 '-UseDistributional', '0',
                 '-MinClusters', '4',
                 '-MaxClusters', '7',
                 '-MaxPossibleClusters', '20',
                 '-MaxIter', '500',
                 '-Screen', '0',
                 '-PenaltyKLogN', '2'
                 ]
    # print(call_list)
    if call(call_list):
        raise IOError('KlustaKwik failed')
    indexes = np.loadtxt(klusta_clu_file, dtype=np.uint8)
    msg = "Done!. Autosort found {} units".format(indexes[0])
    logger.info(msg)

    # spk.Units.Indexes = indexes[1:]
    # spk.Units.save()
    unitID = indexes[1:]
    return unitID
    # unit_list = np.unique(unitID)

    # BA_string = ''
    # msg = 'generating figures'
    # import pylab
    # logger.info(msg)

    # for unit in spk.Units[1:]:
    #     Wf_unit = unit.Waveforms
    #     BA_string += 'unit{}-{:.3f}Hz|'.format(
    #         unit.ID, Wf_unit.shape[0] * 1. / (spk.TimeStamps[-1] / spk.SamplingFreq))
    #     if spk.Units[1:].index(unit) % 3 == 0:
    #         BA_string += '\n'
    #     # print(Wf_unit.shape[0], rec_duration)
    #     fig = pylab.figure(0)
    #     if Wf_unit.shape[0] > 0:
    #         sample = np.random.randint(0, Wf_unit.shape[0], 100)
    #         Wf = Wf_unit[sample, :]
    #         St = np.repeat(np.nan, Wf.shape[0]).reshape(-1, 1)
    #         Wf = np.hstack((St, Wf))
    #         # print(sample)
    #         wfx = np.tile(np.arange(
    #             Wf.shape[1]), sample.size)
    #         wfy = Wf.reshape(-1)
    #         pylab.plot(wfx, wfy)
    # # print(BA_string)
    # msg = "klustaKwikSorting for Channel:{}|ID:{} \n units BA |{}".format(
    #     spk.Name, chan_ID + 1, BA_string)
    # pylab.title(msg)
    # fig.tight_layout()
    # pylab.savefig(klusta_png_file)
    # pylab.draw()
    # fig.clear()


def _KlustaKwik_executable():
    print(platform.system())
    if platform.system() == 'Windows':
        relative_path = '../../External/bins/KlustaKwik.exe'

    elif platform.system() == 'Linux':
        relative_path = '../../External/bins/KlustaKwiklinux'

    else:
        relative_path = '../../External/bins/KlustaKwik'

    path = os.path.split(__file__)[0] + os.path.sep
    return os.path.abspath(path + relative_path)
