import logging
import os

import numpy as np
import pandas as pd
import tables

logger = logging.getLogger(__name__)


def loadPyephys(filename):
    basename, extname = os.path.splitext(filename)
    if extname == "h5raw":
        filename = ".".join(basename, "h5")

    with tables.open_file(filename, mode="r") as file:
        # FileHeader = file.get_node("/FileHeader")
        if "/RawsHeader" in file.root:
            raws_header = file.get_node("/RawsHeader").read()
            df_raws_header = pd.DataFrame(raws_header)

            # convert to string
            df_raws_header = df_raws_header.applymap(lambda x: x.decode(
                'utf-8') if isinstance(x, bytes) else x)

            df_raws_header = df_raws_header.sort_values('ID')
        else:
            logger.critical('Can not load RawsHeader.')

        if "/SpikesHeader" in file.root:
            spikes_header = file.get_node("/SpikesHeader").read()
            df_spikes_header = pd.DataFrame(spikes_header)

            # convert to string
            df_spikes_header = df_spikes_header.applymap(lambda x: x.decode(
                'utf-8') if isinstance(x, bytes) else x)

            # add Label field if not have
            if "Label" not in df_spikes_header.columns:
                df_spikes_header['Label'] = "default"

            # ========== Test Label ==========
            # new_row = df_spikes_header.iloc[0:2, :].copy()
            # new_row['Label'] = 'test_label'
            # df_spikes_header = pd.concat(
            #     [df_spikes_header, new_row], axis=0, ignore_index=True)
            # ================================
            # logger.debug(df_spikes_header.dtypes)
            df_spikes_header = df_spikes_header.sort_values('ID')

        else:
            df_spikes_header = None
            logger.info('Can not load SpikesHeader.')

        if "/EventsHeader" in file.root:
            spikes_header = file.get_node("/EventsHeader").read()
            df_events_header = pd.DataFrame(spikes_header)

            # convert to string
            df_events_header = df_events_header.applymap(lambda x: x.decode(
                'utf-8') if isinstance(x, bytes) else x)

            df_events_header = df_events_header.sort_values('ID')

        else:
            df_events_header = None
            logger.info('Can not load EventsHeader.')

    return {'RawsHeader': df_raws_header,
            'SpikesHeader': df_spikes_header,
            'EventsHeader': df_events_header}


def loadRaws(filename: str, path: str, name: str):
    # chan_ID = int(chan_ID)
    basename, extname = os.path.splitext(filename)
    if extname == "h5raw":
        filename = ".".join(basename, "h5")

    with tables.open_file(filename, mode="r") as file:
        if "/Raws" in file.root:
            raws = file.get_node(path)
            with tables.open_file(raws(mode='r')._v_file.filename, mode="r") as rawsfile:
                data = rawsfile.get_node('/'.join([path, name]))
                return data[:]
        else:
            raise


def loadSpikes(filename, path):
    # chan_ID = int(chan_ID)
    basename, extname = os.path.splitext(filename)
    if extname == "h5raw":
        filename = ".".join(basename, "h5")

    with tables.open_file(filename, mode="r") as file:
        if path not in file.root:
            return

        spike_chan = file.get_node(path)

        df_units_header = pd.DataFrame(
            spike_chan._f_get_child("UnitsHeader").read())

        # convert to string
        df_units_header = df_units_header.applymap(lambda x: x.decode(
            'utf-8') if isinstance(x, bytes) else x)

        if 'UnitType' not in df_units_header.columns:
            df_units_header['UnitType'] = 'Unit'

            unsorted_pattern = r'(?i)unsorted'
            matches = df_units_header['Name'].str.contains(
                unsorted_pattern, regex=True)
            df_units_header.loc[matches, 'UnitType'] = 'Unsorted'

            invalid_pattern = r'(?i)invalid'
            matches = df_units_header['Name'].str.contains(
                invalid_pattern, regex=True)
            df_units_header.loc[matches, 'UnitType'] = 'Invalid'

        timestamps = spike_chan._f_get_child("TimeStamps").read()
        waveforms = spike_chan._f_get_child("Waveforms").read()

        # get units id
        unitID = np.zeros(len(timestamps))
        not_zero_units = df_units_header[df_units_header["NumRecords"] > 0]
        for unit in not_zero_units.index:
            unit_h5_name = "/".join([not_zero_units.loc[unit, "H5Location"],
                                    not_zero_units.loc[unit, "H5Name"]])
            ind = file.get_node(unit_h5_name).read()
            unitID[ind] = not_zero_units.loc[unit, "ID"]

        return {"unitHeader": df_units_header,
                "unitID": unitID,
                "timestamps": timestamps,
                "waveforms": waveforms}


def saveSpikes(filename, header: dict, unit_header: pd.DataFrame | None = None,
               unit_IDs: np.ndarray = [], timestamps: np.ndarray = [], waveforms: np.ndarray = []):
    basename, extname = os.path.splitext(filename)
    # if extname == "h5raw":
    #     filename = ".".join(basename, "h5")
    path = header['H5Location']
    path_split = path.split('/')
    with tables.open_file(filename, mode="a") as file:
        if path in file.root:
            spike_chan = file.get_node(path)
            spike_chan._f_remove(force=True)
        file.create_group('/'+path_split[-2], path_split[-1])
        file.create_array(path, "TimeStamps", timestamps)
        file.create_array(path, "Waveforms", waveforms)

        object_cols = unit_header.dtypes[unit_header.dtypes == 'object'].index
        string_len = unit_header[object_cols].applymap(len)
        max_length = string_len.max()
        max_length = max_length.apply(lambda x: f'S{x}' if x > 0 else 'S1')
        unit_table = unit_header.to_records(
            index=False, column_dtypes=max_length.to_dict())
        file.create_table(path, "UnitsHeader", unit_table)

        not_zero_unit = unit_header.loc[unit_header['NumRecords'] > 0, 'ID']
        for unit_ID in not_zero_unit:
            ID_group = file.create_group(path, f'Unit_{unit_ID:02}')
            file.create_array(ID_group, 'Indxs',
                              (unit_IDs == unit_ID).nonzero()[0])


def saveSpikesHeader(filename, header: pd.DataFrame | None = None):
    path = '/SpikesHeader'
    with tables.open_file(filename, mode="a") as file:
        if path in file.root:
            spike_chan = file.get_node(path)
            spike_chan._f_remove()

        object_cols = header.dtypes[header.dtypes == 'object'].index
        # # logger.debug(header.dtypes)
        # # logger.debug(header)

        string_len = header[object_cols].applymap(len)
        max_length = string_len.max()
        # # logger.debug(max_length)
        max_length = max_length.apply(lambda x: f'S{x}' if x > 0 else 'S1')
        spike_table = header.to_records(index=False,
                                        column_dtypes=max_length.to_dict())
        # logger.debug(spike_table)
        # logger.debug(spike_table.dtype)
        file.create_table('/', 'SpikesHeader', spike_table)


if __name__ == '__main__':
    filename = "data/MX6-22_2020-06-17_17-07-48_no_ref.h5"
    headers = loadPyephys(filename)
    spike_header = pd.DataFrame(headers['SpikesHeader']).to_dict('records')[0]
    spike = loadSpikes(filename, spike_header['H5Location'])
    print(spike is dict)
